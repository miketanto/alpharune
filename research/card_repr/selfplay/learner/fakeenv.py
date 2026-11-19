"""FakeEnv: same interface/shapes as rbenv.RBEnv (SELFPLAY.md contract) with random content.
Random tokens and legal masks, random game lengths, random winners. For developing the learner."""
import numpy as np


class FakeEnv:
    def __init__(self, n_envs, deck_a=None, deck_b=None, seed=0, variant=None, autopay=True, opponent=None,
                 V=80, T=64, A=96, K=46, F=17, G=72, learnable=False):
        # learnable=True: the winner is the seat that more often picked option 0 (a sanity task for PPO)
        self.learnable = learnable
        self.n = n_envs
        self.rng = np.random.default_rng(seed)
        self.next_seed = seed * 100003 + 1
        self.opponent = opponent
        self.dims = dict(G=G, T=T, F=F, A=A, K=K, V=V)
        self.seed_of = np.zeros(n_envs, np.int64)
        self.left = np.zeros(n_envs, np.int64)
        self.length = np.zeros(n_envs, np.int64)
        self.seat = np.ones(n_envs, np.int32)
        self._done = []
        for e in range(n_envs):
            self._reset(e)
        self._make_obs()

    def _reset(self, e):
        self.seed_of[e] = self.next_seed; self.next_seed += 1
        self.left[e] = self.rng.integers(20, 150)
        self.length[e] = 0
        if not hasattr(self, 'z0'):
            self.z0 = np.zeros((self.n, 3)); self.c0 = np.zeros((self.n, 3))
        self.z0[e] = 0; self.c0[e] = 0

    def _make_obs(self):
        r, n, d = self.rng, self.n, self.dims
        T, A, F, K = d['T'], d['A'], d['F'], d['K']
        nt = r.integers(3, T + 1, n)
        na = r.integers(1, A + 1, n)
        tok = r.integers(0, 6, (n, T, F)).astype(np.int32)
        tok[..., 0] = r.integers(1, 4, (n, T))
        tok[..., 1] = r.integers(1, d['V'], (n, T))
        self.tok_mask = np.arange(T)[None] < nt[:, None]
        tok[~self.tok_mask] = 0
        act = np.full((n, A, K), -1, np.int32)
        act[..., 0] = r.integers(0, 22, (n, A))
        act[..., 1:] = r.integers(-1, 3, (n, A, K - 1))
        refc = [1, 2] + list(range(4, 18)) + list(range(18, 30, 2)) + [33]
        refs = (r.random((n, A, len(refc))) * nt[:, None, None]).astype(np.int32)
        refs[r.random(refs.shape) < 0.7] = -1
        act[..., refc] = refs
        act[..., 34] = np.where(r.random((n, A)) < 0.05, r.integers(2, d['V'], (n, A)), 0)
        self.act_mask = np.arange(A)[None] < na[:, None]
        self.tok, self.act = tok, act
        self.glob = r.integers(0, 10, (n, d['G'])).astype(np.int32)
        self.pk = r.integers(0, 37, n).astype(np.int32)
        self.glob[:, 68] = self.pk
        self.psrc = np.where(r.random(n) < 0.5, (r.random(n) * nt).astype(np.int32), -1).astype(np.int32)
        self.nlegal = na.astype(np.int32)
        self.need = r.random(n) < 0.95
        self.seat = r.integers(1, 3, n).astype(np.int32)
        if self.opponent is not None:  # seat 2 played "in C++": only seat 1 is external
            self.seat[:] = 1

    def observe(self):
        return dict(glob=self.glob, tok=self.tok, tok_mask=self.tok_mask, act=self.act, act_mask=self.act_mask,
                    need=self.need.copy(), seat=self.seat.copy(), pk=self.pk, psrc=self.psrc, nlegal=self.nlegal)

    def step(self, choices):
        choices = np.asarray(choices)
        for e in np.nonzero(self.need)[0]:
            assert self.act_mask[e, choices[e]], 'illegal choice'
        for e in np.nonzero(self.need)[0]:
            s = self.seat[e]
            self.z0[e, s] += choices[e] == 0; self.c0[e, s] += 1
        if self.opponent is not None:  # internal seat-2 player: uniform over the legal options
            na = self.act_mask.sum(1)
            self.z0[:, 2] += self.rng.random(self.n) < 1.0 / na; self.c0[:, 2] += 1
        self.left -= 1
        self.length += 1
        for e in np.nonzero(self.left <= 0)[0]:
            # weak signal: vs a scripted opponent, choosing index 0 a lot wins more (for sanity checks)
            w = int(self.rng.integers(1, 3))
            if self.learnable:
                f = self.z0[e] / np.maximum(self.c0[e], 1)
                if f[1] != f[2]:
                    w = 1 if f[1] > f[2] else 2
            self._done.append(dict(env=int(e), game_seed=int(self.seed_of[e]), winner=w,
                                   length=int(self.length[e]), error=''))
            self._reset(e)
        self._make_obs()

    def results(self):
        out, self._done = self._done, []
        return out
