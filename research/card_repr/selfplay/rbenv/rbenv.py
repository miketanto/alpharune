"""rbenv: vectorised Riftbound environment (research/card_repr/SELFPLAY.md,
"Python interface contract"). ctypes + numpy over librbenv.so (rbenv.cpp).

Deviations from / additions to the contract (all additive):
  * observe() also returns 'psrc' int32 (N,) (token index of the object asking
    the prompt, -1 none; the exporter's psrc) and 'nlegal' int32 (N,) (number of
    legal options before truncation to A).
  * results() dicts also carry 'n_ext' (external decisions) and 'turns'.
    'length' = all engine decisions of the game (auto-answered ones included).
  * extra methods: counters() (truncations, unknown card keys, decision counts),
    close().
  * variant: None, a name, or a list of names (all registered at once; a deck may mix
    them). One variant set per process (the card table is process-global).
  * step() raises ValueError for a choice outside the legal mask (that game
    is then recorded with an error and restarted).
  * Game seeds: env i starts with seed+i; every new game takes the next
    unused seed (seed+n_envs, ...), in the order games finish.
Field layouts: README.txt.
"""
import ctypes
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_WSL = os.path.expanduser('~/riftbound/alpharune')   # WSL build copy
LIB_PATH = os.path.join(REPO_WSL, 'build_pic', 'librbenv.so')
SUBJECTS = os.path.join(REPO_WSL, 'research/card_repr/C_engine/out/subjects.txt')
VOCAB_PATH = os.path.join(HERE, 'vocab.json')
DECK_DIR = os.path.join(HERE, '..', '..', 'smoke', 'data', 'decks')

_lib = None
_I32 = np.ctypeslib.ndpointer(np.int32, flags='C_CONTIGUOUS')
_U8 = np.ctypeslib.ndpointer(np.uint8, flags='C_CONTIGUOUS')
_I64 = np.ctypeslib.ndpointer(np.int64, flags='C_CONTIGUOUS')


def _load():
    global _lib
    if _lib is not None:
        return _lib
    L = ctypes.CDLL(LIB_PATH)
    cp, ci, c64, vp = ctypes.c_char_p, ctypes.c_int, ctypes.c_int64, ctypes.c_void_p
    L.rbenv_init.argtypes = [cp, cp, cp, cp, ci]; L.rbenv_init.restype = ci
    L.rbenv_dims.argtypes = [_I32]; L.rbenv_dims.restype = None
    L.rbenv_create.argtypes = [ci, cp, cp, c64, ci, cp, cp, ci]; L.rbenv_create.restype = vp
    L.rbenv_destroy.argtypes = [vp]; L.rbenv_destroy.restype = None
    L.rbenv_observe.argtypes = [vp, _I32, _I32, _U8, _I32, _U8, _U8, _I32, _I32, _I32, _I32]
    L.rbenv_observe.restype = None
    L.rbenv_step.argtypes = [vp, _I32, cp, ci]; L.rbenv_step.restype = ci
    L.rbenv_results_count.argtypes = [vp]; L.rbenv_results_count.restype = ci
    L.rbenv_result_get.argtypes = [vp, ci, _I64, cp, ci]; L.rbenv_result_get.restype = None
    L.rbenv_results_clear.argtypes = [vp]; L.rbenv_results_clear.restype = None
    L.rbenv_counters.argtypes = [vp, _I64]; L.rbenv_counters.restype = None
    L.rbenv_verify.argtypes = [vp, _I64]; L.rbenv_verify.restype = None
    L.rbenv_schema_version.argtypes = []; L.rbenv_schema_version.restype = ci
    _lib = L
    return L


def resolve_deck(d):
    """A deck name ('rengar', a variant deck basename) or a path. Base decks
    live in the repo's decks/, variant decks in smoke/data/decks/."""
    for c in (d, os.path.join(REPO_WSL, d), os.path.join(REPO_WSL, 'decks', d + '.txt'),
              os.path.join(DECK_DIR, d), os.path.join(DECK_DIR, d + '.txt')):
        if os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError(d)


def load_vocab():
    with open(VOCAB_PATH) as f:
        return json.load(f)


class RBEnv:
    def __init__(self, n_envs, deck_a, deck_b, seed, variant=None, autopay=True, opponent=None):
        L = _load()
        err = ctypes.create_string_buffer(4096)
        if isinstance(variant, (list, tuple)):
            variant = ','.join(sorted(variant))
        if L.rbenv_init(SUBJECTS.encode(), VOCAB_PATH.encode(), (variant or '').encode(), err, 4096) != 0:
            raise RuntimeError(err.value.decode())
        d = np.zeros(6, np.int32)
        L.rbenv_dims(d)
        self.dims = dict(G=int(d[0]), T=int(d[1]), F=int(d[2]), A=int(d[3]), K=int(d[4]), V=int(d[5]))
        self.n_envs = n_envs
        self._h = L.rbenv_create(n_envs, resolve_deck(deck_a).encode(), resolve_deck(deck_b).encode(),
                                 int(seed), 1 if autopay else 0, (opponent or '').encode(), err, 4096)
        if not self._h:
            raise RuntimeError(err.value.decode())
        self._L = L

    def observe(self):
        N, D = self.n_envs, self.dims
        o = dict(glob=np.empty((N, D['G']), np.int32), tok=np.empty((N, D['T'], D['F']), np.int32),
                 tok_mask=np.empty((N, D['T']), np.uint8), act=np.empty((N, D['A'], D['K']), np.int32),
                 act_mask=np.empty((N, D['A']), np.uint8), need=np.empty(N, np.uint8),
                 seat=np.empty(N, np.int32), pk=np.empty(N, np.int32), psrc=np.empty(N, np.int32),
                 nlegal=np.empty(N, np.int32))
        self._L.rbenv_observe(self._h, o['glob'], o['tok'], o['tok_mask'], o['act'], o['act_mask'],
                              o['need'], o['seat'], o['pk'], o['psrc'], o['nlegal'])
        for k in ('tok_mask', 'act_mask', 'need'):
            o[k] = o[k].view(np.bool_)
        return o

    def step(self, choices):
        c = np.ascontiguousarray(np.asarray(choices).astype(np.int32, copy=False))
        if c.shape != (self.n_envs,):
            raise ValueError('choices must have shape (n_envs,)')
        err = ctypes.create_string_buffer(1024)
        rc = self._L.rbenv_step(self._h, c, err, 1024)
        if rc == -1:
            raise ValueError(err.value.decode())
        if rc != 0:
            raise RuntimeError(err.value.decode())

    def results(self):
        n = self._L.rbenv_results_count(self._h)
        out = []
        row = np.zeros(6, np.int64)
        err = ctypes.create_string_buffer(2048)
        for i in range(n):
            self._L.rbenv_result_get(self._h, i, row, err, 2048)
            out.append(dict(env=int(row[0]), game_seed=int(row[1]), winner=int(row[2]), length=int(row[3]),
                            n_ext=int(row[4]), turns=int(row[5]), error=err.value.decode()))
        self._L.rbenv_results_clear(self._h)
        return out

    def counters(self):
        c = np.zeros(6, np.int64)
        self._L.rbenv_counters(self._h, c)
        return dict(trunc_tok=int(c[0]), trunc_act=int(c[1]), trunc_list=int(c[2]), unk_key=int(c[3]),
                    ext_decisions=int(c[4]), auto_decisions=int(c[5]))

    def verify(self, acc=None):
        """Schema v2 field check of the pending decisions (rbenv.cpp verifyEnv); adds into acc (23 int64)."""
        if acc is None:
            acc = np.zeros(23, np.int64)
        self._L.rbenv_verify(self._h, acc)
        return acc

    def schema_version(self):
        return int(self._L.rbenv_schema_version())

    def close(self):
        if getattr(self, '_h', None):
            self._L.rbenv_destroy(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
