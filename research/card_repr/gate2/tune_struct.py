"""Choose the structured block's group weights (struct_block.WEIGHTS) and ALPHA against the hash.

Distances decompose exactly: d^2 = d_hash^2 + ALPHA^2 * sum_g w_g^2 d_g^2 (groups use disjoint dims).
rel = d / median d over the 26 real cards (raw space; the Gaussian projection keeps ratios, gate2.py
then checks rel at init). Items: every pair of the 50 gate-2 vectors with different mechanics and every
C2 edit of gate2.py. Objective: maximise the minimum rel over items (a soft-min), with a small penalty
for weights far from 1 (keep all groups readable). Writes tune_struct.json.
"""
import copy, json, os, re, sys, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'gate2.py'), encoding='utf-8').read()
head = src[:src.index('# ---------------------------------------------------------------- pipeline stages')]
head = chr(10).join(l for l in head.split(chr(10)) if not l.startswith(('VR = ', 'rebuild_mismatch')))
edits_src = src[src.index('def walk_effects'):src.index('def facts')] + \
    src[src.index('def first_ab'):src.index('def structural')]
ns = {'__name__': 'tune', '__file__': os.path.join(HERE, 'gate2.py')}
exec(compile(head.replace("VR_ = {k: np.array(B.vectorize(G[k]), dtype=np.float64) for k in KEYS}", ""), 'gate2_head', 'exec'), ns)
exec(compile(edits_src, 'gate2_edits', 'exec'), ns)
B, G, KEYS, REAL = ns['B'], ns['G'], ns['KEYS'], ns['REAL']
SB = B.SB
GROUPS = sorted(SB.WEIGHTS)


def decomp(g):
    h = np.array(B.hash_vectorize(g))
    gv = SB.group_vectors(g)
    n = SB.size()
    out = {}
    for gr in GROUPS:
        v = np.zeros(n)
        for i, x in gv.get(gr, {}).items():
            v[i] = x
        out[gr] = v
    return h, out


D = {k: decomp(G[k]) for k in KEYS}


def sq(a, b):
    return np.array([((a[0] - b[0]) ** 2).sum()] + [((a[1][g] - b[1][g]) ** 2).sum() for g in GROUPS])


typ_rows = np.array([sq(D[a], D[b]) for a, b in itertools.combinations(REAL, 2)])
items, rows = [], []
for a, b in itertools.combinations(KEYS, 2):
    if json.dumps(G[a], sort_keys=True) == json.dumps(G[b], sort_keys=True):
        continue
    items.append(('pair', a, b)); rows.append(sq(D[a], D[b]))
for name, fam, base, g in ns['edits']():
    items.append(('edit', fam, name)); rows.append(sq(D[base], decomp(g)))
rows = np.array(rows)


def rels(alpha, w):
    coef = np.concatenate([[1.0], alpha ** 2 * np.array([w[g] ** 2 for g in GROUPS])])
    t = np.sqrt(np.median(typ_rows @ coef))
    return np.sqrt(rows @ coef) / t


def score(alpha, w):
    r = rels(alpha, w)
    soft = -np.log(np.exp(-40 * r).sum()) / 40
    return soft + 0.3 * r.min() - 0.002 * sum(np.log(w[g]) ** 2 for g in GROUPS)


def hash_share(alpha, w):
    """Share of the hash in the squared distance, averaged over the 325 real-card pairs."""
    coef = np.array([alpha ** 2 * w[g] ** 2 for g in GROUPS])
    hb = typ_rows[:, 0]
    return float(np.mean(hb / (hb + typ_rows[:, 1:] @ coef)))


def alpha_for(w, share):
    lo, hi = 1e-4, 1e4
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        if hash_share(mid, w) > share:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


def search(share, iters=3000):
    """Weights maximising the soft-min rel with ALPHA set so the hash keeps `share` of typical d^2."""
    rng = np.random.default_rng(0)
    w = {g: 1.0 for g in GROUPS}
    bs = score(alpha_for(w, share), w)
    step = 0.5
    for it in range(iters):
        w2 = dict(w)
        g = GROUPS[rng.integers(len(GROUPS))]
        w2[g] = float(np.clip(w2[g] * np.exp(rng.normal(0, step)), 0.05, 20))
        s = score(alpha_for(w2, share), w2)
        if s > bs:
            w, bs = w2, s
        if it % 750 == 749:
            step *= 0.6
    return alpha_for(w, share), w


if __name__ == '__main__':
    r0 = rels(1e-9, {g: 1.0 for g in GROUPS})
    r1 = rels(1.0, {g: 1.0 for g in GROUPS})
    curve = {}
    for share in (0.75, 0.6, 0.5, 0.4, 0.3, 0.2):
        a_, w_ = search(share, 1500)
        rr = rels(a_, w_)
        curve[share] = {'min_rel': float(rr.min()), 'below_025': int((rr < .25).sum()), 'alpha': a_, 'w': w_}
        print('hash share %.2f: min rel %.3f  #<.25 %d' % (share, rr.min(), (rr < .25).sum()), flush=True)
    SHARE = float(os.environ.get('HASH_SHARE', 0.5))
    alpha, w = search(SHARE)
    # normalise so the weights are relative to 'verb' = 1 (cosmetic)
    k = w['verb']
    w = {g: round(w[g] / k, 3) for g in GROUPS}; alpha = round(alpha * k, 3)
    r = rels(alpha, w)
    print('hash only  min %.3f  #<.25 %d' % (r0.min(), (r0 < .25).sum()))
    print('uniform    min %.3f  #<.25 %d' % (r1.min(), (r1 < .25).sum()))
    print('tuned (hash share %.2f -> %.3f) min %.3f  #<.25 %d  alpha %s' % (SHARE, hash_share(alpha, w), r.min(), (r < .25).sum(), alpha), w)
    order = np.argsort(r)
    for i in order[:25]:
        print('  %.3f (hash-only %.3f) %s' % (r[i], r0[i], items[i]))
    json.dump({'hash_share': SHARE, 'curve': {str(k): {kk: vv for kk, vv in v.items()} for k, v in curve.items()},
               'alpha': alpha, 'weights': w, 'min_rel_raw': float(r.min()),
               'n_items': len(items), 'below_025': int((r < .25).sum()),
               'lowest': [(float(r[i]), float(r0[i]), items[i]) for i in order[:40]]},
              open(os.path.join(HERE, 'tune_struct.json'), 'w'), indent=1)
