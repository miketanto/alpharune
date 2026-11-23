"""How much of a checkpoint's policy goes to rune taps (never trained on them), as a function of tap_bias.

  RB_TAP_INIT=0 RB_WSL_COPY=... python3 az/tap_prior.py <ckpt> [games]
Greedy games (argmax among the moves that are not taps), taps offered to both seats.
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('rbenv', 'league', 'learner'):
    sys.path.insert(0, os.path.join(HERE, '..', 'selfplay', d))
import torch
import rbenv
from common import load_net, to_dev
torch.set_num_threads(4)
V = rbenv.load_vocab(); V = V if isinstance(V, list) else V['vocab']
RUNE = np.array([k.endswith('_rune') for k in V])
games = int(sys.argv[2]) if len(sys.argv) > 2 else 60
M = []   # (mass on exhaust, mass on recycle, n taps, n moves, main phase?)
for a, b in (('azir', 'azir'), ('azir', 'rengar')):
    E = rbenv.RBEnv(games, a, b, 31337, runes=3)
    net = load_net(sys.argv[1], E.dims, device='cpu')
    done = set()
    while len(done) < games:
        o = E.observe(); rows = np.nonzero(o['need'])[0]
        with torch.no_grad():
            lp, v = net(to_dev(o, rows, 'cpu'))
        p = lp.exp().numpy()
        ch = np.zeros(games, np.int64)
        for j, i in enumerate(rows):
            A = p.shape[1]
            src = o['act'][i, :A, 2]; ok = o['act_mask'][i, :A] & (src >= 0)
            t = np.zeros(A, bool); t[ok] = RUNE[o['tok'][i, src[ok], 1]]
            if t.any() and i not in done:
                ai = o['act'][i, :A, 3]
                M.append((p[j][t & (ai == 0)].sum(), p[j][t & (ai == 1)].sum(), t.sum(), (o['act_mask'][i, :A] & ~t).sum(), o['pk'][i]))
            q = p[j].copy(); q[t] = -1; q[~o['act_mask'][i, :A]] = -1
            ch[i] = q.argmax()
        E.step(ch)
        for r in E.results():
            done.add(r['env'])
    E.close()
M = np.array(M, float)
print(f'{len(M)} decisions offering taps, mean taps {M[:,2].mean():.1f}, moves {M[:,3].mean():.1f}; tap_bias now {net.tap_bias.data.tolist()}')
m = M[:, 0] + M[:, 1]
print(f'policy mass on taps: mean {m.mean():.3f} (exhaust {M[:,0].mean():.3f}, recycle {M[:,1].mean():.3f}), median {np.median(m):.3f}')
for k in np.unique(M[:, 4]):
    s = M[:, 4] == k; print('  prompt kind', int(k), 'n', s.sum(), 'mass', round(m[s].mean(), 3))
for b in (-1, -2, -3, -4, -5):
    x = m * np.exp(b) / (1 - m + m * np.exp(b)); print('  with bias', b, 'mean mass', round(x.mean(), 3), 'over 0.2:', round((x > .2).mean(), 3))
