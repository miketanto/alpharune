"""Per-family tracking on the CF data's VALIDATION positions (never trained on): r(dv, dwin), sign agreement on
clear pairs (|dwin| > 2 se and > 0.1), mean dv / dwin, for several nets.

usage: python3 cf_val_eval.py "cf_data/w*.npz" --nets B2=ck,id=ck,B2_cf=ck,id_cf=ck [--split val]
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import po as P  # noqa: E402
from cf_train import CFData  # noqa: E402  (learner on sys.path via po)
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data'); ap.add_argument('--nets', required=True); ap.add_argument('--split', default='val')
    ap.add_argument('--out', default='')
    ap.add_argument('--val-frac', type=float, default=0.1, help='1.0 = every position (a test-only set)')
    a = ap.parse_args()
    torch.set_num_threads(4)
    cf = CFData(a.data, a.val_frac)
    G = cf.groups[a.split]
    rows = []
    for o, e in G:
        for j in e:
            rows.append((o, j, cf.meta[j]['fam'], cf.d[j], cf.meta[j]['dse']))
    res = {}
    for spec in a.nets.split(','):
        name, ck = spec.split('=')
        net = P.load_net(ck, name.split('_')[0], 'cpu')
        v = {}
        with torch.no_grad():
            for i in range(0, len(G), 32):
                idx, _ = cf.batch(0, a.split, G[i:i + 32])
                _, vv = net(cf.tensors(idx, 'cpu'))
                v.update(zip(idx.tolist(), vv.numpy().tolist()))
        res[name] = {r[:2]: (v[r[1]] - v[r[0]]) / 2 for r in rows}
    fams = sorted({r[2] for r in rows}) + ['ALL']
    L = ['CF %s split: %d positions, %d edits' % (a.split, len(G), len(rows)),
         '%-15s %4s %4s %8s | ' % ('family', 'n', 'clr', 'dwin') + ' | '.join('%-6s r     sign  dv    ' % n for n in res)]
    out = {}
    for f in fams:
        rr = [r for r in rows if f == 'ALL' or r[2] == f]
        t = np.array([r[3] for r in rr]); se = np.array([r[4] for r in rr])
        clr = (np.abs(t) > 2 * se) & (np.abs(t) > 0.1)
        line = '%-15s %4d %4d %+8.3f | ' % (f, len(rr), clr.sum(), t.mean())
        out[f] = dict(n=len(rr), clear=int(clr.sum()), dwin=float(t.mean()))
        parts = []
        for n in res:
            m = np.array([res[n][r[:2]] for r in rr])
            r_ = float(np.corrcoef(m, t)[0, 1]) if len(rr) > 2 and m.std() > 0 and t.std() > 0 else float('nan')
            sa = float((np.sign(m[clr]) == np.sign(t[clr])).mean()) if clr.any() else float('nan')
            parts.append('%+.2f %5.2f %+.3f' % (r_, sa, m.mean()))
            out[f][n] = dict(r=r_, sign=sa, dv=float(m.mean()), dv_abs=float(np.abs(m).mean()))
        L.append(line + ' | '.join('%-22s' % p for p in parts))
    print('\n'.join(L))
    if a.out:
        open(a.out, 'w').write('\n'.join(L) + '\n')
        json.dump(out, open(a.out.replace('.txt', '.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
