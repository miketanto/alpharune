"""Opponent sampling for self-play / league training (league/README.txt).

All samplers share one API, callable per game or per batch:
    s.add(member_id)                 add a frozen opponent (a registry id, or any hashable key)
    s.sample(n=1, rng=None) -> list  opponent ids; SELF ('self') = the current learner (self-play)
    s.update(opp_id, outcome)        outcome for the LEARNER: 1 win, 0 loss, 0.5 draw
    s.probs() -> {id: probability}   the distribution sample() draws from (SELF included), for logging
    s.state_dict() / s.load_state_dict(d)   JSON-serialisable; keep it in latest.pt to resume
Win statistics (WinStats) are per opponent: an exponential moving average with a Beta(1,1)-style start, so a
new opponent starts at p = 0.5 (configurable).

Samplers
  Uniform(p_self=0.5, window=20)
      Today's behaviour (learner/selfplay.py): with probability p_self the current policy, otherwise one of the
      last `window` snapshots uniformly. (The trainer today draws ONE pool net per update, i.e. per batch.)
  PFSP(p_self=0.0, weighting='squared', window=None)
      Prioritised fictitious self-play (AlphaStar, Vinyals et al. 2019, Nature 575, Methods "Prioritized
      fictitious self-play"): P(B) ~ f(P[learner beats B]), with
        'squared'  f(p) = (1 - p)^2   (AlphaStar's f_hard, default: focus on opponents the learner loses to)
        'hard'     f(p) = (1 - p)^k   (k = `power`, default 2; k -> inf = only the hardest)
        'linear'   f(p) = 1 - p
        'variance' f(p) = p (1 - p)   (AlphaStar's f_var: opponents of about equal strength)
        'even'     f(p) = 1           (= uniform FSP)
      `floor` (default 1e-3) keeps every opponent at a small probability so its win rate stays measured.
  OSFP(p_self=0.6, optimism=1.0, xi=0.55, max_periods=6, period_samples=..., f=None)
      Optimistic Smooth Fictitious Play as used for Hearthstone (Xiao et al. 2023, arXiv 2303.05197, Sec. II-C and
      Appendix VII-4 Algorithm 1, Table IV) from Legends of Code and Magic (Xi et al. 2023, arXiv 2303.04096,
      Sec. 2.1 Eq. 6, Sec. 2.2 Eq. 16, Appendix B Algorithm 1). What is implemented:
        * Smooth FSP mixture: with probability p_self (0.6 in Table IV) the opponent is the current learner (the
          "latest policy"); otherwise a historical model H[i].
        * Optimism (LoCM Eq. 6/16): the best response is taken to F~_k + F_k, i.e. the newest model counts twice:
          alpha_i = 1 for i < k, alpha_k = 2.  Here alpha_k = 1 + `optimism` (optimism = 1 reproduces the paper;
          0 = plain smooth FSP over the history).  P(H[i]) ~ alpha_i * f(G[i], C[i]).
        * f: the papers write "i ~ f(..., (G[i], C[i]), ...)" without defining f. Default f = 1 (only the alpha
          weights); f='pfsp' multiplies by PFSP 'squared' on the period's win rate (our choice, not the paper's).
        * Learning periods (LP): G[i] += g (g = +1 win / -1 loss), C[i] += 1 per game vs H[i]; both reset at each
          LP. At the end of an LP the learner is added to H if G[i]/C[i] > xi for EVERY i, or if `count` > c LPs
          passed without adding (c = max_periods = 6). xi = 0.55 in the Hearthstone paper (Table IV); the LoCM
          paper's Table 7 reportedly says 0.7. Note G/C = 2*winrate - 1, so xi = 0.55 means win rate > 77.5%.
        * LP length: 3.2e8 samples in the paper; ours is `period_samples` decisions (default 160k = ~10 updates
          of the current trainer) -- the caller reports decisions with s.tick(n) and asks s.end_of_period().
      The RL step itself (PPO with clipped V-trace-free GAE here; V-trace + PPO-clip in the paper, Sec. IV-F
      Eq. 7) is unchanged: OSFP only decides who the opponent is and when to snapshot.
  RoleSampler(role, registry-like members ...)  AlphaStar league roles, see league_mix() below.

Helpers
  pfsp_weights(ps, weighting, power) -> np.array
"""
import math

import numpy as np

SELF = 'self'


# --------------------------------------------------------------------------- statistics
class WinStats:
    """Per-opponent learner win rate: EMA with prior (p0, n0 pseudo-games) for unseen opponents."""

    def __init__(self, decay=0.99, p0=0.5, n0=2.0):
        self.decay, self.p0, self.n0 = decay, p0, n0
        self.w = {}   # discounted wins
        self.n = {}   # discounted games
        self.games = {}

    def update(self, k, outcome):
        self.w[k] = self.w.get(k, 0.0) * self.decay + float(outcome)
        self.n[k] = self.n.get(k, 0.0) * self.decay + 1.0
        self.games[k] = self.games.get(k, 0) + 1

    def p(self, k):
        return (self.w.get(k, 0.0) + self.p0 * self.n0) / (self.n.get(k, 0.0) + self.n0)

    def state_dict(self):
        return dict(decay=self.decay, p0=self.p0, n0=self.n0, w=self.w, n=self.n, games=self.games)

    def load_state_dict(self, d):
        self.decay, self.p0, self.n0 = d['decay'], d['p0'], d['n0']
        self.w, self.n, self.games = dict(d['w']), dict(d['n']), dict(d['games'])


def pfsp_weights(ps, weighting='squared', power=2.0):
    p = np.clip(np.asarray(ps, float), 0.0, 1.0)
    if weighting == 'squared':
        f = (1 - p) ** 2
    elif weighting == 'hard':
        f = (1 - p) ** power
    elif weighting == 'linear':
        f = 1 - p
    elif weighting == 'variance':
        f = p * (1 - p)
    elif weighting == 'even':
        f = np.ones_like(p)
    else:
        raise ValueError(weighting)
    return f


def _draw(rng, keys, probs, n):
    probs = np.asarray(probs, float)
    probs = probs / probs.sum()
    idx = rng.choice(len(keys), size=n, p=probs)
    return [keys[i] for i in idx]


# --------------------------------------------------------------------------- samplers
class Sampler:
    def __init__(self, seed=0, decay=0.99):
        self.pool = []                       # frozen opponents, oldest first
        self.stats = WinStats(decay)
        self.rng = np.random.default_rng(seed)

    def add(self, k):
        if k not in self.pool:
            self.pool.append(k)

    def remove(self, k):
        if k in self.pool:
            self.pool.remove(k)

    def update(self, k, outcome):
        self.stats.update(k, outcome)

    def probs(self):
        raise NotImplementedError

    def sample(self, n=1, rng=None):
        pr = self.probs()
        keys = list(pr)
        return _draw(rng or self.rng, keys, [pr[k] for k in keys], n)

    def win_rates(self):
        return {k: round(self.stats.p(k), 3) for k in self.pool}

    def state_dict(self):
        return dict(cls=type(self).__name__, pool=list(self.pool), stats=self.stats.state_dict(),
                    cfg={k: v for k, v in self.__dict__.items() if k not in ('pool', 'stats', 'rng') and
                         isinstance(v, (int, float, str, bool, type(None), dict, list))})

    def load_state_dict(self, d):
        self.pool = list(d['pool'])
        self.stats.load_state_dict(d['stats'])
        for k, v in d['cfg'].items():
            setattr(self, k, v)


class Uniform(Sampler):
    """Today's trainer: p_self current policy, else uniform over the last `window` snapshots."""

    def __init__(self, p_self=0.5, window=20, seed=0):
        super().__init__(seed)
        self.p_self, self.window = p_self, window

    def probs(self):
        pool = self.pool[-self.window:] if self.window else self.pool
        if not pool:
            return {SELF: 1.0}
        out = {SELF: self.p_self}
        for k in pool:
            out[k] = (1 - self.p_self) / len(pool)
        return out


class PFSP(Sampler):
    def __init__(self, p_self=0.0, weighting='squared', power=2.0, window=None, floor=1e-3, seed=0, decay=0.99):
        super().__init__(seed, decay)
        self.p_self, self.weighting, self.power, self.window, self.floor = p_self, weighting, power, window, floor

    def probs(self, candidates=None):
        pool = candidates if candidates is not None else (self.pool[-self.window:] if self.window else self.pool)
        if not pool:
            return {SELF: 1.0}
        f = pfsp_weights([self.stats.p(k) for k in pool], self.weighting, self.power) + self.floor
        f = f / f.sum()
        out = {SELF: self.p_self} if self.p_self > 0 else {}
        for k, x in zip(pool, f):
            out[k] = (1 - self.p_self) * float(x)
        return out


class OSFP(Sampler):
    """Optimistic smooth fictitious play (arXiv 2303.05197 App. VII-4 Alg. 1; arXiv 2303.04096 Eq. 6, 16)."""

    def __init__(self, p_self=0.6, optimism=1.0, xi=0.55, max_periods=6, period_samples=160_000, f=None,
                 seed=0, decay=0.99):
        super().__init__(seed, decay)
        self.p_self, self.optimism, self.xi = p_self, optimism, xi
        self.max_periods, self.period_samples, self.f = max_periods, period_samples, f
        self.G, self.C = {}, {}          # this learning period's score sums and game counts per H[i]
        self.count = 0                   # periods since the learner was last added
        self.samples = 0                 # decisions in the current period

    def probs(self):
        if not self.pool:
            return {SELF: 1.0}
        k = len(self.pool)
        alpha = np.ones(k)
        alpha[-1] = 1.0 + self.optimism
        if self.f == 'pfsp':
            alpha = alpha * (pfsp_weights([self.period_winrate(h) for h in self.pool], 'squared') + 1e-3)
        alpha = alpha / alpha.sum()
        out = {SELF: self.p_self}
        for h, a in zip(self.pool, alpha):
            out[h] = (1 - self.p_self) * float(a)
        return out

    def period_winrate(self, h):
        c = self.C.get(h, 0)
        return 0.5 if c == 0 else 0.5 * (1 + self.G[h] / c)

    def update(self, h, outcome):
        super().update(h, outcome)
        if h == SELF:
            return
        g = 1.0 if outcome > 0.5 else (-1.0 if outcome < 0.5 else 0.0)
        self.G[h] = self.G.get(h, 0.0) + g
        self.C[h] = self.C.get(h, 0) + 1

    def tick(self, n_samples):
        """Report learner decisions (training samples) collected; returns True when the period is over."""
        self.samples += int(n_samples)
        return self.samples >= self.period_samples

    def end_of_period(self, min_games=1):
        """Close the learning period. Returns True if the CALLER should add the current learner to H now
        (then call s.add(new_id)). Rule: G[i]/C[i] > xi for every i in H (with >= min_games games), or more
        than max_periods periods without an addition. H empty -> True (the first snapshot)."""
        if not self.pool:
            ok = True
        else:
            beats_all = all(self.C.get(h, 0) >= min_games and self.G[h] / self.C[h] > self.xi for h in self.pool)
            ok = beats_all or self.count > self.max_periods
        self.G, self.C, self.samples = {}, {}, 0
        if ok:
            self.count = 0
        else:
            self.count += 1
        return ok


class RoleSampler(Sampler):
    """AlphaStar league matchmaking (Vinyals et al. 2019, Methods "League composition"), per learner role.

    members: dict id -> dict(role='main'|'main-exploiter'|'league-exploiter'|'anchor'|'historical',
             origin=<role of the agent that produced the snapshot>, current=bool (a learner's live policy))
    role 'main':            35% self-play, 50% PFSP(squared) vs every frozen member, 15% PFSP(squared) vs
                            "forgotten" main snapshots (p < forgotten_p) and main-exploiter snapshots; if there
                            are none, that 15% is self-play.
    role 'main-exploiter':  vs the CURRENT main agent(s). If its win rate vs them is < 0.2, half of the games are
                            PFSP(variance) vs the main agents' snapshots (AlphaStar). Never self-play.
    role 'league-exploiter': PFSP(squared) vs every frozen member. Never self-play.
    Anchors (C++ agents) are members too, so the league can include random-player / ISMCTS.
    """

    def __init__(self, role, p_sp=0.35, p_pfsp=0.5, forgotten_p=0.3, seed=0, decay=0.99):
        super().__init__(seed, decay)
        self.role, self.p_sp, self.p_pfsp, self.forgotten_p = role, p_sp, p_pfsp, forgotten_p
        self.members = {}

    def add(self, k, role='historical', origin=None, current=False):
        super().add(k)
        self.members[k] = dict(role=role, origin=origin or role, current=current)

    def _pf(self, keys, weighting):
        if not keys:
            return {}
        f = pfsp_weights([self.stats.p(k) for k in keys], weighting) + 1e-3
        f = f / f.sum()
        return {k: float(x) for k, x in zip(keys, f)}

    def probs(self):
        M = self.members
        frozen = [k for k in self.pool if not M[k]['current']]
        out = {}

        def mix(d, w):
            for k, v in d.items():
                out[k] = out.get(k, 0.0) + w * v

        if self.role == 'main':
            forgotten = [k for k in frozen if (M[k]['origin'] == 'main' and self.stats.p(k) < self.forgotten_p)
                         or M[k]['origin'] == 'main-exploiter']
            p_sp = self.p_sp + (0 if forgotten else 1 - self.p_sp - self.p_pfsp)
            if not frozen:
                return {SELF: 1.0}
            mix({SELF: 1.0}, p_sp)
            mix(self._pf(frozen, 'squared'), self.p_pfsp)
            if forgotten:
                mix(self._pf(forgotten, 'squared'), 1 - self.p_sp - self.p_pfsp)
        elif self.role == 'main-exploiter':
            cur = [k for k in self.pool if M[k]['current'] and M[k]['role'] == 'main']
            snaps = [k for k in frozen if M[k]['origin'] == 'main']
            if not cur and not snaps:
                raise ValueError('main exploiter needs a main agent in the pool')
            p_cur = np.mean([self.stats.p(k) for k in cur]) if cur else 0.0
            if cur and (p_cur >= 0.2 or not snaps):
                mix({k: 1 / len(cur) for k in cur}, 1.0)
            else:
                if cur:
                    mix({k: 1 / len(cur) for k in cur}, 0.5)
                mix(self._pf(snaps, 'variance'), 0.5 if cur else 1.0)
        elif self.role == 'league-exploiter':
            if not frozen:
                raise ValueError('league exploiter needs frozen members')
            mix(self._pf(frozen, 'squared'), 1.0)
        else:
            raise ValueError(self.role)
        s = sum(out.values())
        return {k: v / s for k, v in out.items()}


class OSFPMix(Sampler):
    """Main agent for the overnight league (league/overnight.py): OSFP over the learner's OWN history, mixed with
    PFSP over other league members (exploiter snapshots, fixed baselines).

      P(opponent) = p_osfp * OSFP.probs()  +  (1 - p_osfp) * PFSP(weighting).probs()
    The OSFP part keeps the paper's settings (p_self 0.6 of its share is the current learner; optimism 1 = newest
    history model counted twice; xi 0.55; max_periods 6); the OSFP learning periods (tick / end_of_period) and the
    history H only see the learner's own snapshots. If the PFSP pool is empty its share goes to the OSFP part.
    add(k, origin): origin 'main' -> OSFP history H; anything else -> the PFSP pool.
    """

    def __init__(self, p_osfp=0.8, osfp=None, pfsp=None, seed=0, decay=0.99):
        super().__init__(seed, decay)
        self.p_osfp = p_osfp
        self.osfp = OSFP(**(osfp or {}), seed=seed, decay=decay)
        pf = dict(pfsp or {})
        pf.setdefault('p_self', 0.0)
        self.pfsp = PFSP(**pf, seed=seed + 1, decay=decay)
        self.origin = {}

    @property
    def period_samples(self):
        return self.osfp.period_samples

    def add(self, k, origin=None):
        if k in self.pool:
            return
        super().add(k)
        self.origin[k] = origin or 'league'
        (self.osfp if self.origin[k] == 'main' else self.pfsp).add(k)

    def remove(self, k):
        super().remove(k)
        self.osfp.remove(k); self.pfsp.remove(k)
        self.osfp.G.pop(k, None); self.osfp.C.pop(k, None)
        self.origin.pop(k, None)

    def update(self, k, outcome):
        super().update(k, outcome)
        if k in self.osfp.pool:
            self.osfp.update(k, outcome)
        elif k in self.pfsp.pool:
            self.pfsp.update(k, outcome)

    def tick(self, n):
        return self.osfp.tick(n)

    def end_of_period(self, min_games=1):
        return self.osfp.end_of_period(min_games)

    def probs(self):
        po = self.osfp.probs()
        if not self.pfsp.pool:
            return po
        pp = self.pfsp.probs()
        out = {}
        for d, w in ((po, self.p_osfp), (pp, 1 - self.p_osfp)):
            for k, v in d.items():
                out[k] = out.get(k, 0.0) + w * v
        return out

    def state_dict(self):
        return dict(cls='OSFPMix', pool=list(self.pool), stats=self.stats.state_dict(),
                    cfg=dict(p_osfp=self.p_osfp, origin=dict(self.origin)),
                    osfp=dict(self.osfp.state_dict(), G=dict(self.osfp.G), C=dict(self.osfp.C)),
                    pfsp=self.pfsp.state_dict())

    def load_state_dict(self, d):
        super().load_state_dict(dict(pool=d['pool'], stats=d['stats'], cfg=d['cfg']))
        self.osfp.load_state_dict(d['osfp'])
        self.osfp.G, self.osfp.C = dict(d['osfp'].get('G', {})), dict(d['osfp'].get('C', {}))
        self.pfsp.load_state_dict(d['pfsp'])


def make_sampler(cfg):
    """cfg: dict(kind='uniform'|'pfsp'|'osfp'|'role', **kwargs) (as in league/configs/*.json 'opponents')."""
    cfg = dict(cfg)
    kind = cfg.pop('kind')
    return {'uniform': Uniform, 'pfsp': PFSP, 'osfp': OSFP, 'role': RoleSampler, 'osfp_mix': OSFPMix}[kind](**cfg)


def _selftest():
    rng = np.random.default_rng(0)
    # learner strength 0; opponents of Elo -400..+400
    true = {f'h{i}': e for i, e in enumerate([-400, -200, 0, 200, 400])}
    for S in (Uniform(0.5, 20), PFSP(0.0, 'squared'), PFSP(0.0, 'variance'), OSFP(0.6, 1.0)):
        for k in true:
            S.add(k)
        for _ in range(3000):
            o = S.sample(1, rng)[0]
            p = 0.5 if o == SELF else 1 / (1 + 10 ** (true[o] / 400))
            S.update(o, float(rng.random() < p))
        print(type(S).__name__, getattr(S, 'weighting', ''), {k: round(v, 3) for k, v in S.probs().items()})
    o = OSFP(0.6, 1.0, xi=0.55, max_periods=2)
    print('OSFP add on empty H:', o.end_of_period())
    o.add('h0')
    for _ in range(50):
        o.update('h0', 1.0)
    print('beats all -> add:', o.end_of_period())
    o.add('h1')
    print('no games this period -> add:', [o.end_of_period() for _ in range(4)])
    r = RoleSampler('main')
    r.add('m@10', 'historical', 'main'); r.add('mx@5', 'historical', 'main-exploiter'); r.add('anchor:ismcts', 'anchor')
    print('main role', {k: round(v, 3) for k, v in r.probs().items()})
    e = RoleSampler('main-exploiter')
    e.add('main', 'main', current=True); e.add('m@10', 'historical', 'main')
    for _ in range(30):
        e.update('main', 0.0)
    print('main-exploiter losing', {k: round(v, 3) for k, v in e.probs().items()})


if __name__ == '__main__':
    _selftest()
