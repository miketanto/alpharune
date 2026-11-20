"""Gate 4 playout harness: ctypes wrapper over the rbenv gate-4 API (rbenv.cpp, end of file;
source research/card_repr/gate4/rbenv_gate4_ext.cpp) plus network loading and batched playouts.

A position = (game seed, applied-choice history). Po(seed, hist) replays it; add_variant(edit) applies one
edit through the engine's editing path and lets resume + Cleanup recompute derived state; play(jobs) runs
playouts with the network on both seats (GREEDY by default) and returns per-job winners.
"""
import ctypes, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SP = os.path.normpath(os.path.join(HERE, '..', 'selfplay'))
sys.path.insert(0, os.path.join(SP, 'rbenv'))
sys.path.insert(0, os.path.join(SP, 'learner'))
import rbenv  # noqa: E402
from rbenv import RBEnv, resolve_deck, _I32, _U8, _I64  # noqa: E402

_ext = False


def lib():
    global _ext
    L = rbenv._load()
    if _ext:
        return L
    cp, ci, c64, vp = ctypes.c_char_p, ctypes.c_int, ctypes.c_int64, ctypes.c_void_p
    L.rbenv_hist.argtypes = [vp, ci, _I64, _I32, ci]; L.rbenv_hist.restype = ci
    L.rbenv_resumable.argtypes = [vp, ci]; L.rbenv_resumable.restype = ci
    L.rbenv_tok_obj.argtypes = [vp, ci, _I64]; L.rbenv_tok_obj.restype = None
    L.rbenv_po_create.argtypes = [ci, cp, cp, c64, _I32, ci, cp, ci]; L.rbenv_po_create.restype = vp
    L.rbenv_po_describe.argtypes = [vp, cp, ci]; L.rbenv_po_describe.restype = ci
    L.rbenv_po_add_variant.argtypes = [vp, cp, cp, ci]; L.rbenv_po_add_variant.restype = ci
    L.rbenv_po_variant_info.argtypes = [vp, ci, cp, ci]; L.rbenv_po_variant_info.restype = ci
    L.rbenv_po_root_obs.argtypes = [vp, ci, _I32, _I32, _U8, _I32, _U8, _I32, _I64]; L.rbenv_po_root_obs.restype = None
    L.rbenv_po_start.argtypes = [vp, _I32, _I64, ci]; L.rbenv_po_start.restype = None
    L.rbenv_po_busy.argtypes = [vp]; L.rbenv_po_busy.restype = ci
    L.rbenv_po_results.argtypes = [vp, _I64, ci]; L.rbenv_po_results.restype = ci
    _ext = True
    return L


def env_hist(env, i):
    L = lib()
    buf = np.zeros(20000, np.int32)
    sd = np.zeros(1, np.int64)
    n = L.rbenv_hist(env._h, i, sd, buf, len(buf))
    return int(sd[0]), buf[:n].copy()


def env_resumable(env, i):
    return bool(lib().rbenv_resumable(env._h, i))


def _str(fn, *a):
    cap = 1 << 16
    while True:
        b = ctypes.create_string_buffer(cap)
        n = fn(*a, b, cap)
        if n < cap:
            return b.value.decode()
        cap = n + 16


class Po:
    def __init__(self, seed, hist, n_envs=256, deck='rengar', variant=None):
        L = lib()
        err = ctypes.create_string_buffer(4096)
        if L.rbenv_init(rbenv.SUBJECTS.encode(), rbenv.VOCAB_PATH.encode(), (variant or '').encode(), err, 4096) != 0:
            raise RuntimeError(err.value.decode())
        d = np.zeros(6, np.int32)
        L.rbenv_dims(d)
        self.dims = dict(G=int(d[0]), T=int(d[1]), F=int(d[2]), A=int(d[3]), K=int(d[4]), V=int(d[5]))
        h = np.ascontiguousarray(np.asarray(hist, np.int32))
        dk = resolve_deck(deck).encode()
        self._h = L.rbenv_po_create(n_envs, dk, dk, int(seed), h, len(h), err, 4096)
        if not self._h:
            raise RuntimeError(err.value.decode())
        self.n_envs = n_envs
        self.L = L
        self._L = L

    def describe(self):
        return json.loads(_str(self.L.rbenv_po_describe, self._h))

    def add_variant(self, edit):
        err = ctypes.create_string_buffer(2048)
        v = self.L.rbenv_po_add_variant(self._h, edit.encode(), err, 2048)
        return (v, None) if v >= 0 else (None, err.value.decode())

    def info(self, v):
        return json.loads(_str(lambda *a: self.L.rbenv_po_variant_info(self._h, v, *a)))

    def root_obs(self, v):
        D = self.dims
        o = dict(glob=np.empty((1, D['G']), np.int32), tok=np.empty((1, D['T'], D['F']), np.int32),
                 tok_mask=np.empty((1, D['T']), np.uint8), act=np.empty((1, D['A'], D['K']), np.int32),
                 act_mask=np.empty((1, D['A']), np.uint8))
        meta = np.empty(4, np.int32)
        to = np.empty(D['T'], np.int64)
        self.L.rbenv_po_root_obs(self._h, v, o['glob'], o['tok'], o['tok_mask'], o['act'], o['act_mask'], meta, to)
        o['tok_mask'] = o['tok_mask'].view(np.bool_)
        o['act_mask'] = o['act_mask'].view(np.bool_)
        o['seat'] = meta[:1].copy(); o['pk'] = meta[1:2].copy(); o['psrc'] = meta[2:3].copy()
        o['nlegal'] = int(meta[3]); o['tok_obj'] = to
        return o

    # RBEnv-compatible observe / step
    observe = RBEnv.observe
    step = RBEnv.step

    def start(self, vars_, seeds):
        v = np.ascontiguousarray(np.asarray(vars_, np.int32))
        s = np.ascontiguousarray(np.asarray(seeds, np.int64))
        self.L.rbenv_po_start(self._h, v, s, len(v))

    def busy(self):
        return self.L.rbenv_po_busy(self._h)

    def results(self):
        out = []
        buf = np.zeros(5 * 4096, np.int64)
        while True:
            n = self.L.rbenv_po_results(self._h, buf, 4096)
            out += [tuple(int(x) for x in buf[5 * i:5 * i + 5]) for i in range(n)]
            if n < 4096:
                return out

    def close(self):
        if getattr(self, '_h', None):
            self.L.rbenv_destroy(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def load_net(ck, rep, dev='cpu'):
    import torch
    from model import Net, load_vocab
    sd = torch.load(ck, map_location=dev)
    sd = sd.get('net', sd.get('model', sd)) if isinstance(sd, dict) else sd
    dims = dict(G=72, T=96, F=24, A=96, K=61, V=len(load_vocab()[0]))
    net = Net(dims, load_vocab()[0], rep, ask_head='W_ask.weight' in sd, card_norm='card_ln.weight' in sd,
              ask_opt='W_ask_opt.weight' in sd, opt_cost='W_oc.weight' in sd, opt_refs='W_mg.weight' in sd).to(dev)
    net.load_state_dict(sd)
    net.eval()
    return net


def net_eval(net, obs, rows, dev):
    """(probs over options [n, A], value [n]) for rows of an observe()-like dict."""
    import torch
    from selfplay import to_dev
    with torch.no_grad():
        b = to_dev(obs, rows, dev)
        logp, v = net(b)
    p = logp.exp().cpu().numpy()
    out = np.zeros((len(rows), obs['act_mask'].shape[1]), np.float32)
    out[:, :p.shape[1]] = p
    return out, v.cpu().numpy()


def play(po, net, jobs, dev, greedy=True):
    """jobs: list of (variant, seed). Returns {(variant, seed): winner} (winner 1/2, 0 draw, -1 error)."""
    import torch
    from selfplay import act_rows
    po.start([j[0] for j in jobs], [j[1] for j in jobs])
    res = {}
    while len(res) < len(jobs):
        obs = po.observe()
        rows = np.nonzero(obs['need'])[0]
        ch = np.zeros(po.n_envs, np.int64)
        if len(rows):
            a, _, _ = act_rows(net, obs, rows, dev, greedy)
            ch[rows] = a
        elif po.busy() == 0:
            for r in po.results():
                res[(r[0], r[1])] = -1 if r[4] else r[2]
            if len(res) < len(jobs):
                raise RuntimeError('playouts stalled: %d of %d' % (len(res), len(jobs)))
            break
        po.step(ch)
        for r in po.results():
            res[(r[0], r[1])] = -1 if r[4] else r[2]
    return res
