"""Trade-off: cap the share of the slot-signature codes in the typical squared distance (they are
identity-like and cost cross-card generalisation, eval T2-direction / T3) and re-optimise the other
weights; prints min rel (raw) per cap. Usage: python tune_sigcap.py [cap ...]"""
import sys, json, os
import numpy as np
caps = [float(x) for x in sys.argv[1:]] or [0.4, 0.3, 0.2, 0.1]
sys.argv = ['x']
import tune_struct as T
G = T.GROUPS
SIG = [G.index(g) for g in G if g.startswith('sig_')]


def shares(alpha, w):
    coef = np.array([alpha ** 2 * w[g] ** 2 for g in G])
    tot = T.typ_rows[:, 0] + T.typ_rows[:, 1:] @ coef
    sig = T.typ_rows[:, 1:][:, SIG] @ coef[SIG]
    return float(np.mean(sig / tot))


def search(cap, share=float(os.environ.get('HASH_SHARE', 0.5)), iters=2500):
    rng = np.random.default_rng(1)
    w = json.load(open(os.path.join(T.HERE, 'tune_struct.json')))['weights']
    w = dict(w)
    def sc(w):
        a = T.alpha_for(w, share)
        pen = max(0, shares(a, w) - cap) * 20
        return T.score(a, w) - pen
    bs = sc(w); step = 0.5
    for it in range(iters):
        w2 = dict(w); g = G[rng.integers(len(G))]
        w2[g] = float(np.clip(w2[g] * np.exp(rng.normal(0, step)), 0.02, 20))
        s = sc(w2)
        if s > bs: w, bs = w2, s
        if it % 600 == 599: step *= 0.6
    a = T.alpha_for(w, share)
    return a, w


res = {}
for cap in caps:
    a, w = search(cap)
    r = T.rels(a, w)
    res[str(cap) + ('' if os.environ.get('HASH_SHARE') is None else '_h' + os.environ['HASH_SHARE'])] = {'alpha': a, 'w': w, 'min_rel': float(r.min()), 'sig_share': shares(a, w)}
    print('cap %.2f sig share %.3f min rel %.3f #<.25 %d  lowest %s' % (cap, shares(a, w), r.min(), (r < .25).sum(),
          [T.items[i][-1][:35] for i in np.argsort(r)[:3]]), flush=True)
old = json.load(open(os.path.join(T.HERE, 'tune_sigcap.json'))) if os.path.exists(os.path.join(T.HERE, 'tune_sigcap.json')) else {}
old.update(res)
json.dump(old, open(os.path.join(T.HERE, 'tune_sigcap.json'), 'w'), indent=1)
