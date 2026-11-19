"""Gate 2 fix estimate: how far would IDF reweighting (or x3 on decisive leaves: side, duration, chooser, optional, neg, relative_to, conditions) of the hashed paths lift the at-risk ratios?
(path weight *= log(N/df) over the in-scope cards + base cards; raw space, since init ~ raw).
Usage: python fix_estimate.py  -> fix_estimate.json"""
import json, os, sys, math
import numpy as np
sys.argv = [sys.argv[0]]
import gate2 as G2
B = G2.B
allg = dict(G2.G)
for k in G2.BASE_KEYS:
    if k not in allg:
        allg[k] = G2.X.strip(json.load(open(os.path.join(G2.CR, 'B2_code', 'cards', k + '.json'), encoding='utf-8')))
paths = {k: B.graph_paths(g) for k, g in allg.items()}
N = len(paths)
df = {}
for p in paths.values():
    for q in p:
        df[q] = df.get(q, 0) + 1
DECISIVE_LEAVES = ('side=', 'duration=', 'chooser=', 'optional', 'neg', 'relative_to=')
def vec(pp, idf):
    v = np.zeros(B.DIM)
    for q, val in pp.items():
        i, s = B.h(q)
        leaf = q.split('/')[-1]
        if idf == 'family':
            w = 3.0 if leaf.startswith(DECISIVE_LEAVES) or leaf.startswith('c:') else 1.0
        elif idf:
            w = math.log(N / df.get(q, 1))
        else:
            w = 1.0
        v[i] += s * val * w
    return v
def typ(V):
    ks = G2.REAL
    return np.median([np.linalg.norm(V[a] - V[b]) for i, a in enumerate(ks) for b in ks[i + 1:]])
out = {}
for idf in (False, True, 'family'):
    V = {k: vec(paths[k], idf) for k in G2.KEYS}
    t = typ(V)
    res = {}
    for a, b in [('synthetic:punch_first__ctrl_scores', 'synthetic:punch_first__opp_of_ctrl_scores'),
                 ('body_rune', 'fury_rune'), ('pit_rookie', 'synthetic:pit_rookie__might3'),
                 ('punch_first', 'synthetic:punch_first__amount4'), ('sabotage', 'synthetic:sabotage__unit')]:
        res[f'{a} | {b}'] = float(np.linalg.norm(V[a] - V[b]) / t)
    fam = {}
    for name, f, base, g in G2.edits():
        ve = vec(B.graph_paths(g), idf)
        fam.setdefault(f, []).append(float(np.linalg.norm(ve - V[base]) / t))
    res['c2_family_mean'] = {f: float(np.mean(v)) for f, v in fam.items()}
    out[{False: 'plain', True: 'idf', 'family': 'family_x3'}[idf]] = res
json.dump(out, open(os.path.join(G2.HERE, 'fix_estimate.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
