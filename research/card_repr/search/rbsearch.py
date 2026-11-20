"""ctypes wrapper of the search harness in librbenv.so (rbsearch.cpp, included by rbenv.cpp).

    S = Search(n_games, deck, seed, sims=64, leaves=4, threads=14, cpuct=2.0)
    while not S.finished():
        B = S.collect()                  # advance every game to its next network request
        rows = S.rows(B)                 # dict of numpy arrays, one row per request
        logp, v = evaluate(rows)         # (B, A) float32 log-probs over options, (B,) value for the row's decider
        S.feed(logp, v)
    S.results()

Game i: seed = seed + i // 2; seat_mode 'balanced': search seat 1 for even i, 2 for odd i
(each seed is played once with the search on each seat); 'none': nobody searches.
"""
import ctypes
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SP = os.path.normpath(os.path.join(HERE, '..', 'selfplay'))
sys.path.insert(0, os.path.join(SP, 'rbenv'))
sys.path.insert(0, os.path.join(SP, 'learner'))
import rbenv  # noqa: E402

_I32 = np.ctypeslib.ndpointer(np.int32, flags='C_CONTIGUOUS')
_U8 = np.ctypeslib.ndpointer(np.uint8, flags='C_CONTIGUOUS')
_I64 = np.ctypeslib.ndpointer(np.int64, flags='C_CONTIGUOUS')
_F32 = np.ctypeslib.ndpointer(np.float32, flags='C_CONTIGUOUS')

STAT_NAMES = ['sims', 'leaves', 'terminals', 'fails', 'reexpand', 'fallbacks', 'searches', 'greedy', 'depth_sum',
              'collisions', 'ns_build', 'ns_edit', 'ns_descend', 'ns_encode', 'ns_teardown', 'ns_snap',
              'replay_steps', 'unk_key', 'trunc_tok', 'trunc_act', 'changed', 'dl_fallback']
LEAK_NAMES = ['dec_main', 'dec_other', 'worlds_main', 'worlds_other', 'mismatch_main', 'mismatch_other',
              'worlds_differ', 'check_problems', 'build_fail', 'opp_hidden_compared', 'opp_hidden_redrawn', 'control_opp_view_differs']


def _lib():
    L = rbenv._load()
    if getattr(L, '_rbs', False):
        return L
    cp, ci, c64, vp = ctypes.c_char_p, ctypes.c_int, ctypes.c_int64, ctypes.c_void_p
    L.rbs_create.argtypes = [ci, cp, cp, c64, _I32, _F32, cp, cp, ci]; L.rbs_create.restype = vp
    L.rbs_destroy.argtypes = [vp]; L.rbs_destroy.restype = None
    L.rbs_collect.argtypes = [vp]; L.rbs_collect.restype = ci
    L.rbs_rows.argtypes = [vp, _I32, _I32, _U8, _I32, _U8, _I32]; L.rbs_rows.restype = None
    L.rbs_feed.argtypes = [vp, _F32, _F32]; L.rbs_feed.restype = None
    L.rbs_n_done.argtypes = [vp]; L.rbs_n_done.restype = ci
    L.rbs_results.argtypes = [vp, _I64]; L.rbs_results.restype = None
    L.rbs_error.argtypes = [vp, ci, cp, ci]; L.rbs_error.restype = ci
    L.rbs_stats.argtypes = [vp, _I64]; L.rbs_stats.restype = None
    L.rbs_leaktest.argtypes = [vp, ci, _I64, cp, ci]; L.rbs_leaktest.restype = None
    L.rbs_set_strict_vis.argtypes = [ci]; L.rbs_set_strict_vis.restype = None
    L.rbs_get_strict_vis.argtypes = []; L.rbs_get_strict_vis.restype = ci
    L._rbs = True
    return L


def init(variant=None):
    """Process-wide card table / vocabulary (rbenv_init). Returns dims."""
    L = _lib()
    err = ctypes.create_string_buffer(4096)
    if L.rbenv_init(rbenv.SUBJECTS.encode(), rbenv.VOCAB_PATH.encode(), (variant or '').encode(), err, 4096) != 0:
        raise RuntimeError(err.value.decode())
    d = np.zeros(6, np.int32)
    L.rbenv_dims(d)
    return dict(G=int(d[0]), T=int(d[1]), F=int(d[2]), A=int(d[3]), K=int(d[4]), V=int(d[5]))


class Search:
    def __init__(self, n_games, deck, seed, sims=64, leaves=4, threads=14, cpuct=2.0, fpu_red=0.0, vl=1.0,
                 perfect=False, seat_mode='balanced', deck_b=None, strict=True, deck_model='pool', opponent=None):
        self.dims = init()
        self.L = _lib()
        self.L.rbs_set_strict_vis(1 if strict else 0)   # encoder strict visibility (README), both seats
        self.n = n_games
        ci = np.array([sims, leaves, threads, 1 if perfect else 0, 0 if seat_mode == 'balanced' else 1,
                       {'pool': 0, 'decklist': 1}[deck_model]], np.int32)
        cf = np.array([cpuct, fpu_red, vl], np.float32)
        err = ctypes.create_string_buffer(4096)
        self.h = self.L.rbs_create(n_games, rbenv.resolve_deck(deck).encode(),
                                   rbenv.resolve_deck(deck_b or deck).encode(), int(seed), ci, cf, (opponent or '').encode(), err, 4096)
        if not self.h:
            raise RuntimeError(err.value.decode())
        cap = n_games * max(1, leaves)
        D = self.dims
        self._buf = dict(glob=np.zeros((cap, D['G']), np.int32), tok=np.zeros((cap, D['T'], D['F']), np.int32),
                         tok_mask=np.zeros((cap, D['T']), np.uint8), act=np.zeros((cap, D['A'], D['K']), np.int32),
                         act_mask=np.zeros((cap, D['A']), np.uint8), meta=np.zeros((cap, 6), np.int32))

    def collect(self):
        return int(self.L.rbs_collect(self.h))

    def rows(self, B):
        b = self._buf
        self.L.rbs_rows(self.h, b['glob'], b['tok'], b['tok_mask'], b['act'], b['act_mask'], b['meta'])
        m = b['meta'][:B]
        return dict(glob=b['glob'][:B], tok=b['tok'][:B], tok_mask=b['tok_mask'][:B].view(np.bool_),
                    act=b['act'][:B], act_mask=b['act_mask'][:B].view(np.bool_), pk=m[:, 0].copy(),
                    psrc=m[:, 1].copy(), nlegal=m[:, 2].copy(), seat=m[:, 3].copy(), game=m[:, 4].copy(),
                    leaf=m[:, 5].copy())

    def feed(self, logp, v):
        self.L.rbs_feed(self.h, np.ascontiguousarray(logp, np.float32), np.ascontiguousarray(v, np.float32))

    def n_done(self):
        return int(self.L.rbs_n_done(self.h))

    def finished(self):
        return self.n_done() == self.n

    def results(self):
        out = np.zeros((self.n, 9), np.int64)
        self.L.rbs_results(self.h, out)
        err = ctypes.create_string_buffer(2048)
        res = []
        for i, r in enumerate(out):
            e = ''
            if r[7]:
                self.L.rbs_error(self.h, i, err, 2048)
                e = err.value.decode()
            res.append(dict(game=i, seed=int(r[0]), search_seat=int(r[1]), done=bool(r[2]), winner=int(r[3]),
                            turns=int(r[4]), n_search=int(r[5]), n_greedy=int(r[6]), error=e, sims=int(r[8])))
        return res

    def stats(self):
        out = np.zeros(22, np.int64)
        self.L.rbs_stats(self.h, out)
        return dict(zip(STAT_NAMES, (int(x) for x in out)))

    def leaktest(self, worlds, acc=None):
        out = np.zeros(12, np.int64)
        msg = ctypes.create_string_buffer(4096)
        self.L.rbs_leaktest(self.h, int(worlds), out, msg, 4096)
        if acc is None:
            acc = {k: 0 for k in LEAK_NAMES}
        for k, x in zip(LEAK_NAMES, out):
            acc[k] += int(x)
        return acc, msg.value.decode()

    def close(self):
        if getattr(self, 'h', None):
            self.L.rbs_destroy(self.h)
            self.h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def load_net(ck, dims, dev):
    """As viz/play_viz_nn.py load_net: options detected from the state dict, repr B2."""
    import torch
    from model import Net, load_vocab
    sd = torch.load(ck, map_location='cpu')
    sd = sd.get('net', sd.get('model', sd)) if isinstance(sd, dict) else sd
    net = Net(dims, load_vocab()[0], 'B2', ask_head='W_ask.weight' in sd, card_norm='card_ln.weight' in sd,
              ask_opt='W_ask_opt.weight' in sd, opt_cost='W_oc.weight' in sd, opt_refs='W_mg.weight' in sd)
    net.load_state_dict(sd)
    net.eval()
    return net.to(dev)


class Evaluator:
    """Batched network evaluation of rows (trimmed to the used T / A, chunks of `chunk`)."""

    def __init__(self, net, dev, A, chunk=2048):
        self.net, self.dev, self.A, self.chunk = net, dev, A, chunk

    def __call__(self, rows, B):
        import torch
        logp = np.full((B, self.A), -1e9, np.float32)
        v = np.zeros(B, np.float32)
        with torch.no_grad():
            for s in range(0, B, self.chunk):
                e = min(B, s + self.chunk)
                tm = rows['tok_mask'][s:e]
                am = rows['act_mask'][s:e]
                T = int(tm.any(0).nonzero()[0].max()) + 1 if tm.any() else 1
                Aa = int(am.any(0).nonzero()[0].max()) + 1 if am.any() else 1
                b = dict(glob=rows['glob'][s:e], tok=rows['tok'][s:e, :T], tok_mask=tm[:, :T],
                         act=rows['act'][s:e, :Aa], act_mask=am[:, :Aa], pk=rows['pk'][s:e], psrc=rows['psrc'][s:e])
                b = {k: torch.from_numpy(np.ascontiguousarray(x)).to(self.dev, non_blocking=True) for k, x in b.items()}
                lp, vv = self.net(b)
                logp[s:e, :Aa] = lp.float().cpu().numpy()
                v[s:e] = vv.float().cpu().numpy()
        return logp, v


def dummy_eval(rows, B, A):
    """Uniform prior over the legal options, value 0 (benchmark)."""
    am = rows['act_mask'][:B]
    n = np.maximum(am.sum(1), 1)
    logp = np.where(am, -np.log(n)[:, None], -1e9).astype(np.float32)
    return np.ascontiguousarray(logp), np.zeros(B, np.float32)
