"""ctypes wrapper over libplayweb.so (playweb.cpp) and the network agent that plays one seat.

The network side reuses research/card_repr/selfplay/league/common.py as it is (load_net,
act_rows): the checkpoint's options are detected from its state dict, as match.play_nets
does, and the agent plays greedy (argmax). Inference is CPU-only.
"""
import ctypes
import json
import os
import sys

import numpy as np

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')   # never touch the GPU (a training run owns it)

REPO_WSL = os.path.expanduser(os.environ.get('RIFTBOUND_WSL_COPY', '~/riftbound/alpharune'))
LIB_PATH = os.environ.get('PLAYWEB_LIB', os.path.join(REPO_WSL, 'build_pic', 'libplayweb.so'))
SUBJECTS = os.path.join(REPO_WSL, 'research/card_repr/C_engine/out/subjects.txt')
# The research tree (selfplay code, vocab, checkpoints) lives in the Windows checkout.
CARD_REPR = os.environ.get('CARD_REPR', '/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr')
SELFPLAY = os.path.join(CARD_REPR, 'selfplay')
VOCAB_PATH = os.path.join(SELFPLAY, 'rbenv', 'vocab.json')

_I32 = np.ctypeslib.ndpointer(np.int32, flags='C_CONTIGUOUS')
_U8 = np.ctypeslib.ndpointer(np.uint8, flags='C_CONTIGUOUS')
_lib = None
_dims = None


def lib():
    global _lib, _dims
    if _lib is not None:
        return _lib
    L = ctypes.CDLL(LIB_PATH)
    cp, ci, c64, vp, cl = ctypes.c_char_p, ctypes.c_int, ctypes.c_int64, ctypes.c_void_p, ctypes.c_long
    L.pw_init.argtypes = [cp, cp, cp, ci]; L.pw_init.restype = ci
    L.pw_dims.argtypes = [_I32]; L.pw_dims.restype = None
    L.pw_create.argtypes = [cp, cp, ci, c64, cp, ci]; L.pw_create.restype = vp
    L.pw_destroy.argtypes = [vp]; L.pw_destroy.restype = None
    L.pw_status.argtypes = [vp, _I32]; L.pw_status.restype = None
    L.pw_observe.argtypes = [vp, _I32, _I32, _U8, _I32, _U8, _I32, _I32]; L.pw_observe.restype = None
    L.pw_step.argtypes = [vp, ci, cp, ci]; L.pw_step.restype = ci
    L.pw_view.argtypes = [vp, ci, cl, cp, ci]; L.pw_view.restype = ci
    err = ctypes.create_string_buffer(4096)
    if L.pw_init(SUBJECTS.encode(), VOCAB_PATH.encode(), err, 4096) != 0:
        raise RuntimeError(err.value.decode())
    d = np.zeros(6, np.int32)
    L.pw_dims(d)
    _dims = dict(G=int(d[0]), T=int(d[1]), F=int(d[2]), A=int(d[3]), K=int(d[4]), V=int(d[5]))
    _lib = L
    return L


def dims():
    lib()
    return _dims


class Game:
    """One game. Seat numbers are the engine's (1 = Player1, 2 = Player2)."""

    def __init__(self, deck_p1, deck_p2, human_seat, seed):
        L = lib()
        err = ctypes.create_string_buffer(4096)
        self._h = L.pw_create(deck_p1.encode(), deck_p2.encode(), int(human_seat), int(seed), err, 4096)
        if not self._h:
            raise RuntimeError(err.value.decode())
        self.human = int(human_seat)
        self._buf = ctypes.create_string_buffer(1 << 20)

    def status(self):
        o = np.zeros(6, np.int32)
        lib().pw_status(self._h, o)
        return dict(done=bool(o[0]), winner=int(o[1]), need=bool(o[2]), seat=int(o[3]), nlegal=int(o[4]), pk=int(o[5]))

    def observe(self):
        D = _dims
        o = dict(glob=np.empty((1, D['G']), np.int32), tok=np.empty((1, D['T'], D['F']), np.int32),
                 tok_mask=np.empty((1, D['T']), np.uint8), act=np.empty((1, D['A'], D['K']), np.int32),
                 act_mask=np.empty((1, D['A']), np.uint8), pk=np.empty(1, np.int32), psrc=np.empty(1, np.int32))
        lib().pw_observe(self._h, o['glob'], o['tok'], o['tok_mask'], o['act'], o['act_mask'], o['pk'], o['psrc'])
        o['tok_mask'] = o['tok_mask'].view(np.bool_)
        o['act_mask'] = o['act_mask'].view(np.bool_)
        return o

    def step(self, k):
        err = ctypes.create_string_buffer(1024)
        if lib().pw_step(self._h, int(k), err, 1024) != 0:
            raise ValueError(err.value.decode())

    def view(self, seat, log_since=0):
        n = lib().pw_view(self._h, int(seat), int(log_since), self._buf, len(self._buf))
        if n < 0:
            self._buf = ctypes.create_string_buffer(-n + 1024)
            n = lib().pw_view(self._h, int(seat), int(log_since), self._buf, len(self._buf))
        return json.loads(self._buf.raw[:n].decode('utf-8', 'replace'))

    def close(self):
        if self._h:
            lib().pw_destroy(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class NetAgent:
    """A self-play checkpoint, greedy, on CPU (league/common.load_net + act_rows)."""

    def __init__(self, ckpt):
        import torch
        torch.set_num_threads(int(os.environ.get('PLAYWEB_TORCH_THREADS', '2')))
        league = os.path.join(SELFPLAY, 'league')
        if league not in sys.path:
            sys.path.insert(0, league)
        import common  # noqa: E402  (research/card_repr/selfplay/league/common.py, used read-only)
        self._common = common
        self.ckpt = ckpt
        self.net = common.load_net(ckpt, dims(), None, 'cpu')

    def choose(self, obs, nlegal):
        import torch
        with torch.no_grad():
            logp, v = self.net(self._common.to_dev(obs, np.array([0]), 'cpu'))
        a = int(logp.argmax(-1)[0])
        A = _dims['A']
        if not (0 <= a < min(nlegal, A)) or not obs['act_mask'][0, a]:
            a = int(np.argmax(obs['act_mask'][0]))   # never seen; keeps the game legal
        value = float(v.reshape(-1)[0]) if v is not None else None
        return a, value
