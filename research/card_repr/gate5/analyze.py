"""Gate 5 test 1 analysis: card-swap counterfactuals, B2 vs id, per pool (seen / unseen / synth).

truth  = playout dwin of the swap (decider's win rate, swap - original), paired over common seeds:
         'avg' = mean of the B2-policy and id-policy dwins (N = 256 + 256), 'B2pol' / 'idpol' = one policy;
model  = dv = (v(swap) - v(orig)) / 2 at the root, per net.
r      = Pearson r(dv, truth) across swaps, 95% bootstrap CI over POSITIONS (cluster bootstrap);
sign   = sign agreement on clear swaps (|truth| > 2 se and > 0.05);
slope  = OLS slope of dv on truth;  mae = mean |dv - truth|.
B2 - id: paired cluster-bootstrap CI of the difference in r (same swaps).

usage: python3 analyze.py res_*.jsonl [--nets B2,id] [--json out.json]
"""
import argparse, glob, json, sys
from collections import defaultdict
import numpy as np


def load(files):
    rows = []
    for f in files:
        for line in open(f):
            r = json.loads(line)
            if r.get('skipped'):
                continue
            vs = r['variants']
            o = vs[0]
            if not o['ok']:
                continue
            for v in vs[1:]:
                if not v['ok']:
                    continue
                rec = dict(pos='%d/%d' % (r['seed'], len(r['hist'])), deck=r['deck'], kind=v['kind'], pool=v['pool'],
                           new=v['new'], old=v['old'], side=v['side'])
                for pol in ('B2', 'id'):
                    a = np.array(v['wins_' + pol], float); b = np.array(o['wins_' + pol], float)
                    ok = (a >= 0) & (b >= 0)
                    d = a[ok] - b[ok]
                    rec['t_' + pol] = d.mean(); rec['se_' + pol] = d.std(ddof=1) / np.sqrt(len(d))
                    rec['w0_' + pol] = b[ok].mean()
                rec['t_avg'] = 0.5 * (rec['t_B2'] + rec['t_id'])
                rec['se_avg'] = 0.5 * np.sqrt(rec['se_B2'] ** 2 + rec['se_id'] ** 2)
                for net, m in v['model'].items():
                    rec['dv_' + net] = (m['v'] - o['model'][net]['v']) / 2
                    rec['tv_' + net] = tv(o['model'][net]['p'], m['p'], o['legal'], v['legal'])
                    rec['pc_' + net] = card_mass(m['p'], v['legal'], v.get('src_name'))
                    rec['pc0_' + net] = card_mass(o['model'][net]['p'], o['legal'], o_name(o, v))
                    po_ = m.get("pobj", {}).get(str(v["obj"])); pb_ = o["model"][net].get("pobj", {}).get(str(v["obj"]))
                    rec["pa_" + net] = po_; rec["pb_" + net] = pb_
                rows.append(rec)
    return rows


def o_name(o, v):
    return v.get('old_name')


def tv(p, q, lp, lq):
    a = defaultdict(float); b = defaultdict(float)
    for k, x in zip(lp, p): a[k] += x
    for k, x in zip(lq, q): b[k] += x
    return 0.5 * sum(abs(a[k] - b[k]) for k in set(a) | set(b))


def card_mass(p, legal, name):
    if not name:
        return None
    return float(sum(x for x, s in zip(p, legal) if name in s))


def stats(rows, net, tkey='t_avg', sekey='se_avg', B=1000, seed=0):
    if len(rows) < 3:
        return None
    t = np.array([r[tkey] for r in rows]); se = np.array([r[sekey] for r in rows])
    dv = np.array([r['dv_' + net] for r in rows])
    pos = np.array([r['pos'] for r in rows])
    up = np.unique(pos)
    idx = {p: np.nonzero(pos == p)[0] for p in up}
    rng = np.random.default_rng(seed)
    r0 = corr(dv, t)
    bs = []
    for _ in range(B):
        s = np.concatenate([idx[p] for p in rng.choice(up, len(up))])
        bs.append(corr(dv[s], t[s]))
    bs = np.array([x for x in bs if np.isfinite(x)])
    clear = (np.abs(t) > 2 * se) & (np.abs(t) > 0.05)
    sign = float(np.mean(np.sign(dv[clear]) == np.sign(t[clear]))) if clear.sum() else float('nan')
    slope = float(np.polyfit(t, dv, 1)[0]) if np.std(t) > 0 else float('nan')
    return dict(n=len(rows), npos=len(up), clear=int(clear.sum()), truth=float(t.mean()),
                truth_ci=float(1.96 * t.std(ddof=1) / np.sqrt(len(t))), r=r0,
                r_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if len(bs) else [np.nan, np.nan],
                sign=sign, dv=float(dv.mean()), absdv=float(np.abs(dv).mean()), slope=slope,
                mae=float(np.abs(dv - t).mean()))


def corr(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def rdiff(rows, n1, n2, tkey='t_avg', B=2000, seed=1):
    t = np.array([r[tkey] for r in rows]); a = np.array([r['dv_' + n1] for r in rows]); b = np.array([r['dv_' + n2] for r in rows])
    pos = np.array([r['pos'] for r in rows]); up = np.unique(pos)
    idx = {p: np.nonzero(pos == p)[0] for p in up}
    rng = np.random.default_rng(seed)
    d0 = corr(a, t) - corr(b, t)
    bs = []
    for _ in range(B):
        s = np.concatenate([idx[p] for p in rng.choice(up, len(up))])
        bs.append(corr(a[s], t[s]) - corr(b[s], t[s]))
    bs = np.array([x for x in bs if np.isfinite(x)])
    m0 = np.abs(a - t).mean() - np.abs(b - t).mean()
    bm = []
    for _ in range(B):
        s = np.concatenate([idx[p] for p in rng.choice(up, len(up))])
        bm.append(np.abs(a[s] - t[s]).mean() - np.abs(b[s] - t[s]).mean())
    return dict(dr=d0, dr_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                p_dr_le0=float(np.mean(bs <= 0)), dmae=float(m0), dmae_ci=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))])


def fmt(s):
    if s is None:
        return '%-44s' % 'n/a'
    return '%+.2f [%+.2f,%+.2f] %4.2f %+.3f %5.2f' % (s['r'], s['r_ci'][0], s['r_ci'][1], s['sign'], s['dv'], s['slope'])


def table(rows, nets, title, groups, tkey='t_avg', sekey='se_avg', out=None):
    print('\n' + title)
    print('%-28s %4s %4s %4s %-15s | ' % ('cell', 'n', 'pos', 'clr', 'truth') +
          ' | '.join('%-44s' % (n + ': r [CI]            sign  dv     slope') for n in nets))
    res = {}
    for name, f in groups:
        sub = [r for r in rows if f(r)]
        ss = {n: stats(sub, n, tkey, sekey) for n in nets}
        s0 = next((s for s in ss.values() if s), None)
        if s0 is None:
            continue
        print('%-28s %4d %4d %4d %+.3f+-%.3f | ' % (name, s0['n'], s0['npos'], s0['clear'], s0['truth'], s0['truth_ci']) +
              ' | '.join(fmt(ss[n]) for n in nets))
        res[name] = ss
    if out is not None:
        out[title] = res
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+'); ap.add_argument('--nets', default='B2,id'); ap.add_argument('--json', default='')
    a = ap.parse_args()
    files = sorted(set(sum([glob.glob(f) for f in a.files], [])))
    rows = load(files)
    nets = a.nets.split(',')
    out = {}
    print('swaps', len(rows), 'positions', len({r['pos'] for r in rows}), 'files', files)
    G = [('ALL', lambda r: True)]
    for pool in ('seen', 'unseen', 'synth'):
        G.append((pool, lambda r, p=pool: r['pool'] == p))
    for pool in ('seen', 'unseen', 'synth'):
        for kind in ('board', 'hand'):
            G.append(('%s:%s' % (pool, kind), lambda r, p=pool, k=kind: r['pool'] == p and r['kind'] == k))
    for pool in ('seen', 'unseen', 'synth'):
        for deck in ('rengar', 'yi'):
            G.append(('%s@%s' % (pool, deck), lambda r, p=pool, d=deck: r['pool'] == p and r['deck'] == d))
    G.append(('synth(base seen)', lambda r: r['pool'] == 'synth' and any(k in r['new'] for k in ('pit_rookie', 'punch_first', 'sabotage'))))
    G.append(('synth(base unseen)', lambda r: r['pool'] == 'synth' and any(k in r['new'] for k in ('lonely_poro', 'discipline', 'defy'))))
    table(rows, nets, 'MAIN: truth = mean of B2- and id-policy playouts (N 256+256)', G, out=out)
    table(rows, nets, 'B2-policy truth only (N 256)', G[:4], 't_B2', 'se_B2', out=out)
    table(rows, nets, 'id-policy truth only (N 256)', G[:4], 't_id', 'se_id', out=out)
    # own-policy truth: each net against its own policy's playouts
    print('\nOWN-POLICY truth (each net vs its own policy\'s playouts):')
    for pool in ('seen', 'unseen', 'synth'):
        sub = [r for r in rows if r['pool'] == pool]
        line = '%-8s' % pool
        for n in nets:
            if n in ('B2', 'id'):
                s = stats(sub, n, 't_' + n, 'se_' + n)
                line += ' | %s(own): %s' % (n, fmt(s))
        print(line)
    print('\nB2 - id (paired cluster bootstrap over positions; truth = avg):')
    out['diff'] = {}
    for name, f in G[:4] + [('seen+synth', lambda r: r['pool'] != 'unseen'), ('unseen+synth', lambda r: r['pool'] != 'seen')]:
        sub = [r for r in rows if f(r)]
        if len(sub) < 5 or not all('dv_' + n in sub[0] for n in nets[:2]):
            continue
        d = rdiff(sub, nets[0], nets[1])
        out['diff'][name] = d
        print('  %-14s dr = %+.3f [%+.3f, %+.3f]  P(dr<=0) %.2f   dMAE = %+.4f [%+.4f, %+.4f]' % (
            name, d['dr'], d['dr_ci'][0], d['dr_ci'][1], d['p_dr_le0'], d['dmae'], d['dmae_ci'][0], d['dmae_ci'][1]))
    # WITHIN TARGET: the three swaps of one target share "the old card is gone"; demeaning truth and dv over the
    # three pools of that target leaves the part due to WHICH new card arrives (the card-identity test).
    print('\nWITHIN-TARGET (truth and dv demeaned over the 3 pools of each (position, target)): r of residuals,')
    print('and the pairwise contrast unseen - seen / synth - seen on the same target (sign on clear contrasts):')
    tg = defaultdict(dict)
    for r in rows:
        tg[(r['pos'], r['kind'])][r['pool']] = r
    full = [g for g in tg.values() if len(g) == 3]
    out['within'] = {}
    res_rows = []
    for g in full:
        mt = np.mean([g[p]['t_avg'] for p in g])
        for p, r in g.items():
            x = dict(pos=r['pos'], pool=p, kind=r['kind'], deck=r['deck'], t_avg=r['t_avg'] - mt, se_avg=r['se_avg'])
            for n in nets:
                x['dv_' + n] = r['dv_' + n] - np.mean([g[q]['dv_' + n] for q in g])
            res_rows.append(x)
    WG = [('ALL', lambda r: True)] + [(p, lambda r, p=p: r['pool'] == p) for p in ('seen', 'unseen', 'synth')] + \
         [('%s:%s' % (p, k), lambda r, p=p, k=k: r['pool'] == p and r['kind'] == k) for p in ('unseen', 'synth') for k in ('board', 'hand')]
    table(res_rows, nets, 'WITHIN-TARGET residuals (%d targets)' % len(full), WG, out=out)
    for a_, b_ in (('unseen', 'seen'), ('synth', 'seen'), ('unseen', 'synth')):
        ct, cd = [], {n: [] for n in nets}
        cse, cpos = [], []
        for k, g in tg.items():
            if a_ in g and b_ in g:
                ct.append(g[a_]['t_avg'] - g[b_]['t_avg'])
                # se of the contrast (same seeds; conservative: independent)
                cse.append(np.sqrt(g[a_]['se_avg'] ** 2 + g[b_]['se_avg'] ** 2))
                cpos.append(k[0])
                for n in nets:
                    cd[n].append(g[a_]['dv_' + n] - g[b_]['dv_' + n])
        crow = [dict(pos=p, t_avg=t, se_avg=s, **{'dv_' + n: cd[n][i] for n in nets}) for i, (p, t, s) in enumerate(zip(cpos, ct, cse))]
        table(crow, nets, 'CONTRAST %s - %s on the same target' % (a_, b_), [('all', lambda r: True)], out=out)
        if len(crow) >= 5:
            d = rdiff(crow, nets[0], nets[1])
            out['diff']['contrast %s-%s' % (a_, b_)] = d
            print('  B2 - id: dr = %+.3f [%+.3f, %+.3f], dMAE %+.4f [%+.4f, %+.4f]' % (d['dr'], d['dr_ci'][0], d['dr_ci'][1], d['dmae'], d['dmae_ci'][0], d['dmae_ci'][1]))
    for name, f in WG[:4]:
        sub = [r for r in res_rows if f(r)]
        if len(sub) >= 5:
            d = rdiff(sub, nets[0], nets[1])
            out['diff']['within ' + name] = d
            print('  within %-8s B2 - id: dr = %+.3f [%+.3f, %+.3f]  P(dr<=0) %.2f' % (name, d['dr'], d['dr_ci'][0], d['dr_ci'][1], d['p_dr_le0']))
    # diff-in-diff: is B2's within-target advantage larger on unseen / synth cards than on seen ones?
    if len(nets) >= 2:
        pos_all = np.unique([r['pos'] for r in res_rows])
        rng = np.random.default_rng(7)
        def dd(rr, pa, pb):
            def rr_(p, n):
                s = [r for r in rr if r['pool'] == p]
                return corr(np.array([r['dv_' + n] for r in s]), np.array([r['t_avg'] for r in s]))
            return (rr_(pa, nets[0]) - rr_(pa, nets[1])) - (rr_(pb, nets[0]) - rr_(pb, nets[1]))
        byp = defaultdict(list)
        for r in res_rows:
            byp[r['pos']].append(r)
        out['diff']['did'] = {}
        for pa in ('unseen', 'synth'):
            d0 = dd(res_rows, pa, 'seen')
            bs = []
            for _ in range(2000):
                s = sum((byp[p] for p in rng.choice(pos_all, len(pos_all))), [])
                bs.append(dd(s, pa, 'seen'))
            bs = np.array([x for x in bs if np.isfinite(x)])
            out['diff']['did'][pa] = [d0, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            print('  diff-in-diff within-target (B2-id)_%s - (B2-id)_seen = %+.3f [%+.3f, %+.3f]' % (pa, d0, np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
    # compact per-net summary (for the fine-tune budgets)
    print('\nCOMPACT per net: raw r(dv, truth) and within-target r, per pool (truth avg; [95% CI])')
    out['compact'] = {}
    for n in nets:
        line = '%-10s' % n
        out['compact'][n] = {}
        for pool in ('seen', 'unseen', 'synth'):
            s1 = stats([r for r in rows if r['pool'] == pool], n)
            s2 = stats([r for r in res_rows if r['pool'] == pool], n)
            out['compact'][n][pool] = dict(raw=s1, within=s2)
            line += ' | %s raw %+.2f [%+.2f,%+.2f] within %+.2f [%+.2f,%+.2f] |dv| %.3f' % (
                pool, s1['r'], s1['r_ci'][0], s1['r_ci'][1], s2['r'], s2['r_ci'][0], s2['r_ci'][1], s1['absdv'])
        print(line)
    # per card (new card), unseen + synth
    print('\nPER NEW CARD (truth avg; n >= 6):')
    byc = defaultdict(list)
    for r in rows:
        byc[(r['pool'], r['new'])].append(r)
    out['per_card'] = {}
    print('%-8s %-36s %3s %-15s ' % ('pool', 'card', 'n', 'truth') + ' '.join('%-22s' % (n + ': dv   r') for n in nets))
    for (pool, card), sub in sorted(byc.items()):
        if len(sub) < 6:
            continue
        t = np.array([r['t_avg'] for r in sub])
        line = '%-8s %-36s %3d %+.3f+-%.3f ' % (pool, card, len(sub), t.mean(), 1.96 * t.std(ddof=1) / np.sqrt(len(t)))
        out['per_card']['%s|%s' % (pool, card)] = {}
        for n in nets:
            dv = np.array([r['dv_' + n] for r in sub])
            line += '%+.3f %+.2f          ' % (dv.mean(), corr(dv, t))
            out['per_card']['%s|%s' % (pool, card)][n] = dict(n=len(sub), truth=float(t.mean()), dv=float(dv.mean()), r=corr(dv, t))
        print(line)
    # policy shift
    print('\nPOLICY SHIFT at the root (mean TV orig->swap; mean prob. mass on options naming the swapped card after the swap):')
    for pool in ('seen', 'unseen', 'synth'):
        for kind in ('board', 'hand'):
            sub = [r for r in rows if r['pool'] == pool and r['kind'] == kind]
            if not sub:
                continue
            line = '  %-7s %-5s n %3d' % (pool, kind, len(sub))
            for n in nets:
                pa = [r["pa_" + n] for r in sub if r.get("pa_" + n) is not None]; pb = [r["pb_" + n] for r in sub if r.get("pb_" + n) is not None]
                line += " | %s TV %.3f P(card) %.3f -> %.3f" % (n, np.mean([r["tv_" + n] for r in sub]), np.mean(pb) if pb else float("nan"), np.mean(pa) if pa else float("nan"))
            print(line)
    if a.json:
        json.dump(dict(out, rows=rows), open(a.json, 'w'), default=float)


if __name__ == '__main__':
    main()
