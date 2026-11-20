"""Gate 4 phase C: per (mechanic, context) cell tables from run.py's JSONL output(s).

usage: python3 analyze.py results.jsonl [more.jsonl ...] --json results.json --txt tables.txt

Per (position, edit), oriented so that + = "mechanic on" (orient in the edit plan):
  truth  = mean over paired seeds of (win(edited) - win(original)), decider's perspective; se from the paired diffs
  model  = (v(edited) - v(original)) / 2   (value head in [-1, 1] -> win-probability units)
  tv     = total variation between the root policies (options matched by the engine's option text)
Cell: n positions, mean truth with 95% CI (across positions), Pearson / Spearman r(model, truth) with a bootstrap
95% CI, sign agreement on positions where |truth| > 2 se (and > 0.05), mean |model|, mean tv.
"""
import argparse, json, math, os, sys
from collections import defaultdict
import numpy as np


def tv(p, q, lp, lq):
    a = defaultdict(float); b = defaultdict(float)
    for k, x in zip(lp, p): a[k] += x
    for k, x in zip(lq, q): b[k] += x
    return 0.5 * sum(abs(a[k] - b[k]) for k in set(a) | set(b))


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else float('nan')


def pearson(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def boot_r(x, y, B=2000, seed=0):
    if len(x) < 4:
        return [float('nan')] * 2
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x))
        r = pearson(x[i], y[i])
        if not math.isnan(r):
            rs.append(r)
    return [round(float(np.percentile(rs, 2.5)), 3), round(float(np.percentile(rs, 97.5)), 3)] if rs else [float('nan')] * 2


LEVEL = []   # (original win rate, {net: value head as a win probability}) per position


def rows_from(files, nets):
    rows, skipped, positions, greedy = [], 0, 0, 0
    se_orig = []
    seen = set()
    for fn in files:
        if not os.path.exists(fn):
            continue
        for line in open(fn):
            r = json.loads(line)
            key = (r['seed'], len(r['hist']))
            if key in seen:      # a position played by two shard sets: keep the first
                continue
            seen.add(key)
            if r.get('skipped'):
                skipped += 1
                continue
            positions += 1
            o = r['variants'][0]
            wo = np.array(o['wins'])
            ok_o = wo >= 0
            LEVEL.append((float(wo[ok_o].mean()), {nm: (o['model'][nm]['v'] + 1) / 2 for nm in nets if nm in o['model']}))
            se_orig.append(1.96 * math.sqrt(max(wo[ok_o].mean() * (1 - wo[ok_o].mean()), 1e-9) / ok_o.sum()))
            for v in r['variants'][1:]:
                if not v['ok'] or 'wins' not in v:
                    continue
                we = np.array(v['wins'])
                m = ok_o & (we >= 0)
                d = (we[m] - wo[m]) * v['orient']
                row = dict(cell=v['cell'], edit=v['edit'], note=v['note'], seed=r['seed'], turn=r['turn'],
                           n=int(m.sum()), orig=float(wo[m].mean()), truth=float(d.mean()),
                           se=float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float('nan'))
                for nm in nets:
                    mo, me = o['model'][nm], v['model'][nm]
                    row['model_' + nm] = v['orient'] * (me['v'] - mo['v']) / 2
                    row['tv_' + nm] = tv(mo['p'], me['p'], o['legal'], v['legal'])
                rows.append(row)
    return rows, skipped, positions, se_orig


def cell_stats(rs, nets):
    t = np.array([r['truth'] for r in rs]); se = np.array([r['se'] for r in rs])
    out = dict(n=len(rs), truth_mean=round(float(t.mean()), 4),
               truth_ci=round(float(1.96 * t.std(ddof=1) / math.sqrt(len(t))), 4) if len(t) > 1 else None,
               pos_ci_median=round(float(np.median(1.96 * se)), 4),
               clear=int(((np.abs(t) > 2 * se) & (np.abs(t) > 0.05)).sum()))
    clear = (np.abs(t) > 2 * se) & (np.abs(t) > 0.05)
    for nm in nets:
        m = np.array([r['model_' + nm] for r in rs]); tvv = np.array([r['tv_' + nm] for r in rs])
        sa = float((np.sign(m[clear]) == np.sign(t[clear])).mean()) if clear.any() else float('nan')
        out[nm] = dict(r=round(pearson(m, t), 3), r_ci=boot_r(m, t), rho=round(spearman(m, t), 3),
                       sign_agree=round(sa, 3), model_mean=round(float(m.mean()), 4),
                       model_abs=round(float(np.abs(m).mean()), 4), tv=round(float(tvv.mean()), 4))
    return out


def verdict(c, nets, is_null, ref=None):
    """Mechanic cell: PASS = bootstrap r CI above 0 and sign agreement >= 0.7 on clear positions (n >= 8, >= 4
    clear); WEAK = r > 0.2 and sign agreement > 0.5. Null cell: NULL-OK = mean |dv| at most half the pooled
    mechanic cells' mean |dv| for that net and |mean dv| <= 0.01."""
    res = {}
    for nm in nets:
        s = c[nm]
        if is_null:
            ok = s['model_abs'] <= 0.5 * ref[nm] and abs(s['model_mean']) <= 0.01
            res[nm] = 'NULL-OK' if ok else 'NULL-FAIL'
            continue
        if c['n'] < 8 or c['clear'] < 4:
            res[nm] = 'TOO-FEW'
            continue
        ok = (not math.isnan(s['r'])) and s['r_ci'][0] > 0 and s['sign_agree'] >= 0.7
        weak = (not math.isnan(s['r'])) and s['r'] > 0.2 and s['sign_agree'] > 0.5
        res[nm] = 'PASS' if ok else ('WEAK' if weak else 'FAIL')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--nets', default='B2,id')
    ap.add_argument('--json', default='')
    ap.add_argument('--txt', default='')
    a = ap.parse_args()
    nets = a.nets.split(',')
    rows, skipped, npos, se_orig = rows_from(a.files, nets)
    by = defaultdict(list)
    for r in rows:
        by[r['cell']].append(r)
    cells = {}
    L = []
    L.append('positions played out: %d (skipped by the screen: %d); edits: %d' % (npos, skipped, len(rows)))
    L.append('original win rate 95%% CI half-width per position: median %.3f, max %.3f' % (
        float(np.median(se_orig)), float(np.max(se_orig))))
    for nm in nets:
        w = np.array([x[0] for x in LEVEL if nm in x[1]]); v = np.array([x[1][nm] for x in LEVEL if nm in x[1]])
        if len(w) > 3:
            L.append('level check, %s: r(value head at the original root, playout win rate) = %.3f over %d positions; '
                     'mean |value - win| = %.3f' % (nm, pearson(v, w), len(w), float(np.abs(v - w).mean())))
    hdr = '%-42s %3s %4s %15s %6s' % ('cell', 'n', 'clr', 'truth mean+-CI', 'posCI')
    for nm in nets:
        hdr += ' | %-6s %-15s %5s %5s %7s %6s %6s %-9s' % (nm + ' r', '[r CI]', 'sign', 'rho', 'mean dv', '|dv|', 'TV', 'verdict')
    L.append(hdr)
    order = sorted(by, key=lambda c: (c.startswith('null'), c))
    pooled_mech = []
    allm = [r for r in rows if not r['cell'].startswith('null')]
    ref = {nm: float(np.mean([abs(r['model_' + nm]) for r in allm])) if allm else 1.0 for nm in nets}
    for c in order:
        st = cell_stats(by[c], nets)
        st['verdict'] = verdict(st, nets, c.startswith('null'), ref)
        cells[c] = st
        if not c.startswith('null'):
            pooled_mech += by[c]
        line = '%-42s %3d %4d %+7.3f+-%-6.3f %6.3f' % (c, st['n'], st['clear'], st['truth_mean'], st['truth_ci'] or 0, st['pos_ci_median'])
        for nm in nets:
            s = st[nm]
            line += ' | %+6.2f [%+.2f,%+.2f] %5.2f %+5.2f %+7.3f %6.3f %6.3f %-9s' % (s['r'], s['r_ci'][0], s['r_ci'][1], s['sign_agree'], s['rho'], s['model_mean'], s['model_abs'], s['tv'], st['verdict'][nm])
        L.append(line)
    if pooled_mech:
        st = cell_stats(pooled_mech, nets)
        cells['ALL non-null edits (pooled)'] = st
        line = '%-42s %3d %4d %+7.3f+-%-6.3f %6.3f' % ('ALL non-null edits (pooled)', st['n'], st['clear'], st['truth_mean'], st['truth_ci'] or 0, st['pos_ci_median'])
        for nm in nets:
            s = st[nm]
            line += ' | %+6.2f [%+.2f,%+.2f] %5.2f %+5.2f %+7.3f %6.3f %6.3f' % (s['r'], s['r_ci'][0], s['r_ci'][1], s['sign_agree'], s['rho'], s['model_mean'], s['model_abs'], s['tv'])
        L.append(line)
    txt = '\n'.join(L)
    print(txt)
    if a.txt:
        open(a.txt, 'w').write(txt + '\n')
    if a.json:
        json.dump(dict(cells=cells, n_positions=npos, n_skipped=skipped, rows=rows), open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
