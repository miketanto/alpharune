"""Before (results_v1.json, hash only) / after (results.json, hash + structured block + card LayerNorm) table."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
v1 = json.load(open(os.path.join(HERE, 'results_v1.json')))
v2 = json.load(open(os.path.join(HERE, 'results.json')))
out = []
P = lambda x: 'PASS' if x >= 0.25 else ('WEAK' if x >= 0.05 else 'FAIL')
e1 = {s['edit']: s for s in v1['c2_structure']}
out.append('C2 STRUCTURAL EDITS: rel at init (before = hash only, no card LN; after = v2 model card_emb incl. card LN)')
out.append('| family | edit | before init | after init_lin | after init | after ln | after trained* | verdict |')
out.append('|---|---|---|---|---|---|---|---|')
worst = 9
for s in sorted(v2['c2_structure'], key=lambda s: (s['family'], s['init_rel'])):
    b = e1.get(s['edit'])
    worst = min(worst, s['init_rel'])
    out.append('| %s | %s | %s | %.3f | %.3f | %.3f | %s | %s |' % (
        s['family'], s['edit'], '%.3f' % b['init_rel'] if b else 'new', s.get('init_lin_rel', float('nan')),
        s['init_rel'], s['ln_rel'], '%.3f' % s['trained_rel'] if s['trained_rel'] is not None else '-', P(s['init_rel'])))
out.append('min after init over C2 edits: %.3f' % worst)
pw2 = {}
for p in v2['c1_at_risk']:
    pw2[(p['a'], p['b'])] = p
out.append('')
out.append('C1 AT-RISK PAIRS (v1 lowest 12 at init), rel at init')
out.append('| pair | before init | after init | after trained* | verdict |')
out.append('|---|---|---|---|---|')
# v2 at-risk list holds only the 25 lowest; recompute the rest from results.json pairs is not stored -> use v2 list or mark >
lo2 = max(p['init'] for p in v2['c1_at_risk'])
for p in v1['c1_at_risk'][:12]:
    q = pw2.get((p['a'], p['b']))
    out.append('| %s / %s | %.3f | %s | %s | %s |' % (p['a'], p['b'], p['init'], '%.3f' % q['init'] if q else '>%.3f' % lo2,
                                                   '%.3f' % q['trained'] if q else '-', P(q['init']) if q else 'PASS'))
c1 = v2['c1_summary']
out.append('')
out.append('C1 all pairs: before %s; after %s; after min init %.3f (v2 lowest pair %s / %s)' % (
    v1['c1_summary']['counts_by_verdict'], c1['counts_by_verdict'], v2['c1_at_risk'][0]['init'],
    v2['c1_at_risk'][0]['a'], v2['c1_at_risk'][0]['b']))
out.append('init norms before %s / after %s' % (v1['init_norms'], v2['init_norms']))
out.append('token share before %s / after %s' % (v1['token_share_at_init'], v2['token_share_at_init']))
out.append('')
out.append('C3 NUMBERS (LOO R2 / Spearman), raw | init | trained*: before -> after')
for n in v2['c3_numbers']:
    a, b = v1['c3_numbers'].get(n), v2['c3_numbers'][n]
    f = lambda d, g: '%.2f/%.2f' % (d[g]['loo_r2'], d[g]['loo_spearman']) if d and g in d else '-'
    out.append('  %-18s %s | %s | %s  ->  %s | %s | %s' % (n, f(a, 'raw'), f(a, 'init'), f(a, 'trained'),
                                                         f(b, 'raw'), f(b, 'init'), f(b, 'trained')))
out.append('')
out.append('ATOM FAMILIES LOO balanced acc at init: before -> after')
for fam, d in v2['atom_families'].items():
    a = v1['atom_families'].get(fam, {})
    g = lambda x, k: '%.2f' % x[k] if x.get(k) is not None else '-'
    out.append('  %-12s %s -> %s   (raw %s -> %s)' % (fam, g(a, 'loo_init'), g(d, 'loo_init'), g(a, 'loo_raw'), g(d, 'loo_raw')))
out.append('* trained = W_card of the old B2 checkpoints (no card LN) applied to the new R: approximate.')
open(os.path.join(HERE, 'compare_v1_v2.txt'), 'w').write('\n'.join(out))
print('\n'.join(out))
