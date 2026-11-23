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
# Card vocabulary v2 (every card its own key), as the research tree uses by default; the AZ networks need it.
# RB_VOCAB=v1: the old 51-card map the G1 / G2 / ov_main networks were trained with.
if os.environ.get('RB_VOCAB', 'v2') == 'v2':
    SUBJECTS = os.path.join(CARD_REPR, 'cards_v2', 'subjects_v2.txt')
    VOCAB_PATH = os.path.join(CARD_REPR, 'cards_v2', 'vocab_v2.json')

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
    L.pw_observe_hist.argtypes = [vp, _I32]; L.pw_observe_hist.restype = None
    L.pw_search_seat.argtypes = [vp, ci, ci]; L.pw_search_seat.restype = None
    L.rbenv_hist_dims.argtypes = [_I32]; L.rbenv_hist_dims.restype = None
    L.pw_set_manual_runes.argtypes = [ci]; L.pw_set_manual_runes.restype = None
    L.pw_has_manual_runes.argtypes = []; L.pw_has_manual_runes.restype = ci
    L.pw_step.argtypes = [vp, ci, cp, ci]; L.pw_step.restype = ci
    L.pw_view.argtypes = [vp, ci, cl, cp, ci]; L.pw_view.restype = ci
    err = ctypes.create_string_buffer(4096)
    if L.pw_init(SUBJECTS.encode(), VOCAB_PATH.encode(), err, 4096) != 0:
        raise RuntimeError(err.value.decode())
    d = np.zeros(6, np.int32)
    L.pw_dims(d)
    hd = np.zeros(2, np.int32)
    L.rbenv_hist_dims(hd)
    _dims = dict(G=int(d[0]), T=int(d[1]), F=int(d[2]), A=int(d[3]), K=int(d[4]), V=int(d[5]), H=int(hd[0]), HF=int(hd[1]))
    _lib = L
    return L


def dims():
    lib()
    return _dims


def engine_tag():
    """Which engine rules the library plays by: saved games replay only on the same one."""
    return 'float1' if lib().pw_has_manual_runes() else 'base'


class Game:
    """One game. Seat numbers are the engine's (1 = Player1, 2 = Player2)."""

    def __init__(self, deck_p1, deck_p2, human_seat, seed, manual_runes=False):
        L = lib()
        L.pw_set_manual_runes(1 if manual_runes else 0)
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
        o['hist'] = np.zeros((1, D['H'], D['HF']), np.int32)
        lib().pw_observe_hist(self._h, o['hist'])
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
        self.last_prob = float(logp.exp()[0, a])
        self.last_probs = logp.exp()[0, :min(nlegal, A)].numpy().astype(float)   # every option's probability
        return a, value


class SearchAgent:
    """The network with search (research/card_repr/search, 64 simulations as in its evaluations), one game.

    The search harness owns a second copy of the game (same decks, seed and start, so the same game). The human sits on
    its non-searching seat: a decision of that seat comes back as a 'greedy' row, answered with the human's choice. The
    searched decisions are read from the harness's decision log and applied to the displayed game by the caller. The
    search samples the human's hidden cards from what its seat knows; it is never given them.
    """

    def __init__(self, ckpt, deck_p1, deck_p2, seed, seat, sims, manual_runes=False):
        import torch
        self.manual_runes = manual_runes
        torch.set_num_threads(int(os.environ.get('PLAYWEB_TORCH_THREADS', '2')))
        for p in (os.path.join(CARD_REPR, 'search'), os.path.join(SELFPLAY, 'rbenv')):
            if p not in sys.path:
                sys.path.insert(0, p)
        lib()
        import rbenv
        rbenv.LIB_PATH, rbenv.SUBJECTS, rbenv.VOCAB_PATH = LIB_PATH, SUBJECTS, VOCAB_PATH   # the one library of this process
        import rbsearch
        self.S = rbsearch.Search(1, deck_p1, seed, sims=sims, leaves=4, threads=1, seat_mode='balanced', deck_b=deck_p2)
        lib().pw_set_manual_runes(1 if manual_runes else 0)
        lib().pw_search_seat(self.S.h, 0, seat)
        self.seat, self.sims, self.ckpt = seat, sims, ckpt
        self.A = self.S.dims['A']
        self.ev = rbsearch.Evaluator(rbsearch.load_net(ckpt, self.S.dims, 'cpu'), 'cpu', self.A)
        L = self.S.L
        L.rbs_declog.argtypes = [ctypes.c_void_p, ctypes.c_int, np.ctypeslib.ndpointer(np.float32, flags='C'), ctypes.c_int]
        L.rbs_declog.restype = ctypes.c_int
        self._dl = np.zeros((4000, 18), np.float32)
        self._has_root = hasattr(L, 'rbs_root')
        if self._has_root:
            L.rbs_root.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, np.ctypeslib.ndpointer(np.float32, flags='C'), ctypes.c_int]
            L.rbs_root.restype = ctypes.c_int
        self._rt = np.zeros((self.A, 4), np.float32)
        self.waiting = False      # a greedy row (the human's decision) is pending
        self._nl = 0

    def log(self):
        n = self.S.L.rbs_declog(self.S.h, 0, self._dl, len(self._dl))
        return self._dl[:n]

    def root(self, d):
        """Every option of decision d of the decision log: rows (live index, prior, visits, mean value for the decider,
        -2 unvisited); None if this library does not keep them."""
        if not self._has_root:
            return None
        n = self.S.L.rbs_root(self.S.h, 0, int(d), self._rt, len(self._rt))
        return None if n < 0 else self._rt[:n].copy()

    def advance(self):
        """Search until the human is asked or the game ends."""
        while True:
            B = self.S.collect()
            if B == 0:
                self.waiting = False
                r = self.S.results()[0]
                if r['error']:
                    raise RuntimeError('search: ' + r['error'])
                return
            rows = self.S.rows(B)
            if rows['leaf'][0] == 0:
                if int(rows['seat'][0]) == self.seat:
                    raise RuntimeError('search harness asked for a greedy move of the searching seat')
                self.waiting = True
                self._nl = int(rows['nlegal'][0])
                return
            self.S.feed(*self.ev(rows, B))

    def human(self, k):
        if not self.waiting:
            raise RuntimeError('search harness is not waiting for the human')
        if k >= self.A:
            raise ValueError(f'the search opponent supports the first {self.A} options of a decision only')
        lp = np.full((1, self.A), -1e9, np.float32)
        lp[0, k] = 0.0
        self.waiting = False
        self.S.feed(lp, np.zeros(1, np.float32))

    def close(self):
        self.S.close()
