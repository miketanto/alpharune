"""Results store and Bradley-Terry / Elo ratings for the league (league/README.txt).

ResultStore: an append-only JSONL file (default league/results.jsonl) of match records (match.py). It is the
incremental results matrix: new games are appended, never rewritten; matrix() sums every record of a pair
(either order), so adding 50 more games to a pair later just adds a line.

Model (Bradley-Terry, Elo scale):  P(i beats j) = 1 / (1 + 10^((R_j - R_i) / 400)).
  * A draw counts as half a win for each side (draws are rare: games end by score).
  * Anchor: R(anchor:random-player) = 0 by default (documented choice; `anchor=` changes it). Everything else is
    relative to the C++ random-player (pass=0.2), so the numbers are comparable across league snapshots as long
    as that anchor stays in the matrix.
  * Prior: each PLAYED pair gets `prior_games` (default 1.0) virtual game split as a draw. Without it a pair with
    100-0 (every net vs random-player) has an infinite MLE; with it, a 100-0 pair alone gives about +920 Elo
    (a 100.5 : 0.5 split). Ratings driven only by perfect scores are lower bounds; the table flags them.
  * Fit: Newton's method on the penalised log-likelihood (the virtual draws), anchor fixed.
  * Uncertainty: Laplace / Fisher information (95% = +-1.96 sd from the inverse Hessian, with the anchor fixed,
    so each CI is the uncertainty of R_i - R_anchor), and optionally a parametric bootstrap (each pair's wins
    redrawn ~ Binomial(n, p_observed), refit, 2.5/97.5 percentiles).
  * Members not connected to the anchor through played games get no rating (reported as such).

API
  st = ResultStore(path); st.append(rec); recs = st.load()
  names, W, N = st.matrix(names=None)       # W[i,j] = wins of i over j (+0.5 per draw), N[i,j] = games
  fit = fit_bt(names, W, N, anchor='anchor:random-player', prior_games=1.0, boot=0)
        -> dict(names, elo, sd, lo, hi, games, connected, boot_lo, boot_hi, perfect)
  ratings_table(fit) / winrate_matrix(names, W, N, order) -> str
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE  # noqa: E402

ELO = 400.0 / math.log(10.0)


class ResultStore:
    def __init__(self, path=None):
        self.path = path or os.path.join(HERE, 'results.jsonl')

    def append(self, rec):
        with open(self.path, 'a') as f:
            f.write(json.dumps(rec) + '\n')

    def load(self):
        if not os.path.exists(self.path):
            return []
        return [json.loads(l) for l in open(self.path) if l.strip()]

    def pair_games(self):
        """{(a, b) sorted: games} over all records (to skip pairs already played)."""
        out = {}
        for r in self.load():
            k = tuple(sorted((r['a'], r['b'])))
            out[k] = out.get(k, 0) + int(r['games'])
        return out

    def matrix(self, names=None, mode=None):
        recs = [r for r in self.load() if mode is None or r.get('mode') == mode]
        if names is None:
            names = sorted({r['a'] for r in recs} | {r['b'] for r in recs})
        ix = {n: i for i, n in enumerate(names)}
        n = len(names)
        W = np.zeros((n, n))
        N = np.zeros((n, n))
        for r in recs:
            if r['a'] not in ix or r['b'] not in ix:
                continue
            i, j = ix[r['a']], ix[r['b']]
            d = 0.5 * r.get('draws', 0)
            W[i, j] += r['a_wins'] + d
            W[j, i] += r['b_wins'] + d
            N[i, j] += r['games']
            N[j, i] += r['games']
        return names, W, N


def _components(N, start):
    n = len(N)
    seen = {start}
    st = [start]
    while st:
        i = st.pop()
        for j in np.nonzero(N[i] > 0)[0]:
            if j not in seen:
                seen.add(int(j))
                st.append(int(j))
    return seen


def _newton(W, N, a, prior, iters=100):
    """Maximise sum_ij W_ij log s(r_i - r_j) with virtual draws; r[a] = 0. Natural-log units."""
    n = len(W)
    Wp = W + 0.5 * prior * (N > 0)
    Np = N + prior * (N > 0)
    r = np.zeros(n)
    free = np.array([i for i in range(n) if i != a])
    for _ in range(iters):
        d = r[:, None] - r[None, :]
        p = 1 / (1 + np.exp(-d))
        g = (Wp - Np * p).sum(1)                  # dL/dr_i (each pair once per row: W_ij - N_ij p_ij)
        h = Np * p * (1 - p)
        H = -np.diag(h.sum(1)) + h                # Hessian
        Hf = H[np.ix_(free, free)]
        step = np.linalg.solve(Hf, -g[free])
        step = np.clip(step, -2, 2)
        r[free] += step
        if np.abs(step).max() < 1e-9:
            break
    d = r[:, None] - r[None, :]
    p = 1 / (1 + np.exp(-d))
    h = Np * p * (1 - p)
    H = -np.diag(h.sum(1)) + h
    cov = np.zeros((n, n))
    cov[np.ix_(free, free)] = np.linalg.inv(-H[np.ix_(free, free)])
    return r, cov


def fit_bt(names, W, N, anchor='anchor:random-player', prior_games=1.0, boot=0, seed=0):
    names = list(names)
    if anchor not in names:
        raise ValueError(f'anchor {anchor} has no games')
    a0 = names.index(anchor)
    comp = sorted(_components(N, a0))
    sub = np.ix_(comp, comp)
    Ws, Ns = W[sub], N[sub]
    a = comp.index(a0)
    r, cov = _newton(Ws, Ns, a, prior_games)
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    out = dict(names=names, anchor=anchor, prior_games=prior_games, elo={}, sd={}, lo={}, hi={}, games={},
               connected={}, perfect={}, boot_lo={}, boot_hi={})
    for k, i in enumerate(comp):
        nm = names[i]
        out['elo'][nm] = float(r[k] * ELO)
        out['sd'][nm] = float(sd[k] * ELO)
        out['lo'][nm] = float((r[k] - 1.96 * sd[k]) * ELO)
        out['hi'][nm] = float((r[k] + 1.96 * sd[k]) * ELO)
    for i, nm in enumerate(names):
        out['games'][nm] = int(N[i].sum())
        out['connected'][nm] = i in comp
        played = N[i] > 0
        out['perfect'][nm] = bool(played.any() and (np.all(W[i][played] >= N[i][played]) or np.all(W[i][played] <= 0)))
    if boot:
        rng = np.random.default_rng(seed)
        P = np.where(Ns > 0, Ws / np.maximum(Ns, 1), 0)
        samples = []
        iu = np.triu_indices(len(comp), 1)
        for _ in range(boot):
            Wb = np.zeros_like(Ws)
            wins = rng.binomial(Ns[iu].astype(int), P[iu])
            Wb[iu] = wins
            Wb.T[iu] = Ns[iu] - wins
            rb, _ = _newton(Wb, Ns, a, prior_games, iters=60)
            samples.append(rb * ELO)
        S = np.array(samples)
        for k, i in enumerate(comp):
            out['boot_lo'][names[i]] = float(np.percentile(S[:, k], 2.5))
            out['boot_hi'][names[i]] = float(np.percentile(S[:, k], 97.5))
    return out


def residuals(fit, names, W, N, top=10):
    """Fit check: observed vs predicted win rate per played pair, z = (obs - pred) / sqrt(pred (1 - pred) / n).
    Large |z| = the pair does not fit one rating line (intransitivity, seat effects, or style matchups)."""
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if N[i, j] == 0 or a not in fit['elo'] or b not in fit['elo']:
                continue
            pred = 1 / (1 + 10 ** ((fit['elo'][b] - fit['elo'][a]) / 400))
            obs = W[i, j] / N[i, j]
            z = (obs - pred) / math.sqrt(max(pred * (1 - pred), 1e-6) / N[i, j])
            out.append((abs(z), a, b, obs, pred, int(N[i, j]), z))
    out.sort(reverse=True)
    rows = [f"{'pair':62s} {'obs':>5s} {'pred':>5s} {'n':>4s} {'z':>6s}"]
    for _, a, b, o, p_, n, z in out[:top]:
        rows.append(f"{a + ' vs ' + b:62s} {o:5.2f} {p_:5.2f} {n:4d} {z:6.2f}")
    return '\n'.join(rows)


def ratings_table(fit, extra=None):
    """extra: {name: str} appended per row (e.g. repr / schema)."""
    rows = [f"{'rank':>4s}  {'member':34s} {'Elo':>7s} {'95% CI (Laplace)':>19s}"
            + (f" {'95% CI (bootstrap)':>19s}" if fit['boot_lo'] else '') + f" {'games':>6s}  note"]
    order = sorted(fit['elo'], key=lambda k: -fit['elo'][k])
    for rk, nm in enumerate(order, 1):
        s = f"{rk:4d}  {nm:34s} {fit['elo'][nm]:7.0f} [{fit['lo'][nm]:7.0f},{fit['hi'][nm]:7.0f}]"
        if fit['boot_lo']:
            s += f" [{fit['boot_lo'][nm]:7.0f},{fit['boot_hi'][nm]:7.0f}]"
        note = []
        if nm == fit['anchor']:
            note.append('anchor (fixed 0)')
        if fit['perfect'][nm]:
            note.append('all-win/all-loss: bound set by prior')
        if extra and nm in extra:
            note.append(extra[nm])
        rows.append(s + f" {fit['games'][nm]:6d}  {'; '.join(note)}")
    for nm in fit['names']:
        if not fit['connected'][nm]:
            rows.append(f"   -  {nm:34s}   (not connected to the anchor: no rating)")
    return '\n'.join(rows)


def winrate_matrix(names, W, N, order=None, short=None):
    """Row player's win rate vs column player (draws half), '-' = not played. Plus games per cell."""
    idx = [names.index(n) for n in (order or names)]
    lab = [(short or {}).get(names[i], names[i])[:12] for i in idx]
    head = f"{'':14s}" + ''.join(f"{l:>13s}" for l in lab)
    rows = ['win rate of ROW vs COLUMN (draws = 1/2); games in parentheses', head]
    for i in idx:
        s = f"{(short or {}).get(names[i], names[i])[:13]:14s}"
        for j in idx:
            if i == j:
                s += f"{'.':>13s}"
            elif N[i, j] == 0:
                s += f"{'-':>13s}"
            else:
                s += f"{W[i, j] / N[i, j]:7.2f}({int(N[i, j]):4d})"
        rows.append(s)
    return '\n'.join(rows)


def _selftest():
    """Synthetic check: recover known ratings from simulated games."""
    rng = np.random.default_rng(1)
    true = np.array([0, 300, 500, 800, 1200.0])
    names = [f'p{i}' for i in range(5)]
    n = 5
    W = np.zeros((n, n)); N = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p = 1 / (1 + 10 ** ((true[j] - true[i]) / 400))
            w = rng.binomial(400, p)
            W[i, j], W[j, i], N[i, j], N[j, i] = w, 400 - w, 400, 400
    f = fit_bt(names, W, N, anchor='p0', prior_games=1.0, boot=50)
    print(ratings_table(f))
    print('true', true.tolist())
    print(winrate_matrix(names, W, N))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'selftest':
        _selftest()
    else:
        st = ResultStore(sys.argv[1] if len(sys.argv) > 1 else None)
        names, W, N = st.matrix()
        f = fit_bt(names, W, N, boot=200)
        print(ratings_table(f))
        print()
        print(winrate_matrix(names, W, N, sorted(f['elo'], key=lambda k: -f['elo'][k])))
