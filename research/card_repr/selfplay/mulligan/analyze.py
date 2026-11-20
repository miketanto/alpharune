"""Analysis of mull_check.py results (paired by deal)."""
import json, math, os, sys
import numpy as np
D = sys.argv[1] if len(sys.argv) > 1 else 'results'
def load(a):
    return {r['deal']: r for r in map(json.loads, open(os.path.join(D, a + '.jsonl')))
            if not r['error'] and r['winner'] in (1, 2)}
net, keep, samp = load('net'), load('keep'), load('sample')
def ci(w, n):
    p = w / n; h = 1.96 * math.sqrt(p * (1 - p) / n); return f'{p:.3f} ±{h:.3f} (n={n})'
print('== 1. seat / first player (net arm)')
rs = list(net.values())
print('first player wins', ci(sum(r['winner'] == r['first'] for r in rs), len(rs)))
print('P1 wins', ci(sum(r['winner'] == 1 for r in rs), len(rs)))
print('== 2. mulligan behaviour (net arm, per seat-decision)')
d = [(m, r) for r in rs for m in r['mull']]
pk = np.array([m['p_keep'] for m, _ in d]); sa = np.array([m['set_aside'] for m, _ in d])
print('decisions', len(d), ' set aside 0/1/2:', [round(float(np.mean(sa == k)), 3) for k in (0, 1, 2)])
print('P(keep) quantiles 10/25/50/75/90:', np.round(np.quantile(pk, [.1, .25, .5, .75, .9]), 3))
mx = np.array([max(m['p_keep'], m['p_best_mull']) for m, _ in d])
print('policy confidence (max option prob) median', round(float(np.median(mx)), 3), ' share >0.9:', round(float(np.mean(mx > .9)), 3))
print('first vs second player mulligan rate:',
      [round(float(np.mean([m['set_aside'] > 0 for m, r in d if (m['seat'] == r['first']) == f])), 3) for f in (True, False)])
print('== 3. does the pre-mulligan value predict the result? (net arm, within first/second player)')
for f, name in ((True, 'first'), (False, 'second')):
    xs = [(m['v'], r['winner'] == m['seat']) for m, r in d if (m['seat'] == r['first']) == f]
    v = np.array([x[0] for x in xs]); w = np.array([x[1] for x in xs], float)
    r_ = np.corrcoef(v, w)[0, 1]
    q = np.quantile(v, [.25, .5, .75])
    bins = np.digitize(v, q)
    print(f' {name}: n={len(v)} corr(v, win)={r_:+.3f}  win rate by v quartile:',
          [round(float(w[bins == b].mean()), 3) for b in range(4)], ' mean v', round(float(v.mean()), 3))
print('== 4. does the network mulligan beat always-keep? (paired deals)')
def seatres(arms_a, arms_b, sel):
    """Win rate of seat s in arm A vs arm B over (deal, seat) chosen by sel(record_in_A, mull)."""
    wa = wb = n = 0
    for k, ra in arms_a.items():
        rb = arms_b.get(k)
        if rb is None: continue
        for m in ra['mull']:
            if sel(ra, m):
                n += 1; wa += ra['winner'] == m['seat']; wb += rb['winner'] == m['seat']
    return wa, wb, n
for label, sel in (('seat mulliganed (any)', lambda r, m: m['set_aside'] > 0),
                   ('only this seat mulliganed', lambda r, m: m['set_aside'] > 0 and all(x['set_aside'] == 0 for x in r['mull'] if x['seat'] != m['seat'])),
                   ('seat kept', lambda r, m: m['set_aside'] == 0)):
    wa, wb, n = seatres(net, keep, sel)
    print(f' {label}: net-mulligan wr {wa/n:.3f} vs keep wr {wb/n:.3f}  diff {100*(wa-wb)/n:+.1f} pts ±{196*math.sqrt(.5/n):.1f} (n={n})')
same = sum(net[k]['winner'] == keep[k]['winner'] for k in net if k in keep)
print(' deals with identical winner net vs keep:', same, '/', len(net))
print('== 5. sampled mulligan (training behaviour) vs greedy network mulligan')
wa, wb, n = seatres(net, samp, lambda r, m: True)
print(f' greedy-mulligan seat wr {wa/n:.3f} vs sampled-mulligan seat wr {wb/n:.3f}  (n={n}, both seats; 0.5 each by symmetry)')
ds = [m for r in samp.values() for m in r['mull']]
print(' sampled arm mulligan rate', round(float(np.mean([m['set_aside'] > 0 for m in ds])), 3))
