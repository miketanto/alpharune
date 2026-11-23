"""Are the taps in a buffer good moves by the search's own values, or noise? python3 az/taps_buf.py <buf> [chunks]"""
import glob, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'selfplay', 'rbenv'))
import rbenv
V = rbenv.load_vocab(); V = V if isinstance(V, list) else V['vocab']
RUNE = np.array([k.endswith('_rune') for k in V])
fs = sorted(glob.glob(os.path.join(sys.argv[1], 'c_*.npz')))
k = int(sys.argv[2]) if len(sys.argv) > 2 else 12


def look(files, name):
    rows = {0: [], 1: []}; best = []; tot = 0; share = []; pshare = []
    for f in files:
        z = np.load(f)
        act, tok, am, N, Q, P = (z[x] for x in ('act', 'tok', 'act_mask', 'N', 'Q', 'P'))
        N, Q, P = N.astype(float), Q.astype(float), P.astype(float)
        src = act[:, :, 2]; ok = am.astype(bool) & (src >= 0)
        key = np.take_along_axis(tok[:, :, 1], np.clip(src, 0, tok.shape[1] - 1), 1)
        t = ok & RUNE[key]
        for i in np.nonzero(t.any(1))[0]:
            tot += 1
            mv = am[i].astype(bool) & ~t[i]
            n = N[i].sum(); share.append(N[i][t[i]].sum() / max(n, 1)); pshare.append(P[i][t[i]].sum())
            best.append(bool(t[i][N[i].argmax()]))
            vm = mv & (N[i] >= 4)
            if not vm.any():
                continue
            qb = Q[i][mv][N[i][mv].argmax()]         # value of the most visited move that is not a tap
            for a in (0, 1):
                s = t[i] & (act[i, :, 3] == a) & (N[i] >= 4)
                for j in np.nonzero(s)[0]:
                    rows[a].append(Q[i][j] - qb)
    print(f'{name}: decisions offering taps {tot}; prior on taps (noise included) {np.mean(pshare):.3f}, visits {np.mean(share):.3f}, a tap most visited {np.mean(best):.3f}')
    for a, nm in ((0, 'exhaust'), (1, 'recycle')):
        r = np.array(rows[a])
        if len(r):
            print(f'  {nm}: n {len(r)} (4+ visits), value minus the main non-tap move: mean {r.mean():+.3f}, median {np.median(r):+.3f}, better {np.mean(r > 0):.2f}, better by 0.05 {np.mean(r > .05):.2f}')


look(fs[:k], 'first chunks'); look(fs[-k:], 'last chunks')
