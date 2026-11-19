"""Target-prompt analysis (--tgt runs): the card is played (first card action),
the decider's prompts before the target prompt get their first option, and the
teacher searches the target prompt. Best target = q-argmax of q averaged over seeds."""
import json, os, glob, collections, statistics as st
H = os.path.dirname(os.path.abspath(__file__))
def qm(t, idx=None):
    se = [x for x in t['searches'] if x.get('vis')]
    if idx is not None: se = [se[i] for i in idx if i < len(se)]
    if not se: return None
    return [st.mean(x['q'][i] for x in se) for i in range(len(t['act']))]
def best(t, q):
    i = max(range(len(q)), key=lambda k: q[k]); ch = t['act'][i].get('ch') or [None]
    side = t['tok'][ch[0]][2] if ch[0] is not None else None
    return i, side
out = {}; sides = collections.defaultdict(dict)
for f in sorted(glob.glob(os.path.join(H, 'raw_tgt', '*.jsonl'))):
    rows = [json.loads(l) for l in open(f)]; rows = [r for r in rows if 'versions' in r]
    var = rows[0]['variant']; c = collections.Counter()
    for r in rows:
        o, s = r['versions']['orig'].get('tgt'), r['versions']['swap'].get('tgt')
        c['n'] += 1
        if not o or not s or 'error' in o or 'error' in s: c['tgt_error_or_missing'] += 1; continue
        if not (o.get('found') and s.get('found')): c['no_target_prompt'] += 1; continue
        if len(o['act']) < 2 or len(s['act']) < 2: c['single_target'] += 1; continue
        if [a.get('ch') for a in o['act']] != [a.get('ch') for a in s['act']]: c['target_lists_differ'] += 1; continue
        qo, qs = qm(o), qm(s)
        if qo is None or qs is None: c['no_search'] += 1; continue
        c['multi'] += 1
        bo, bs = best(o, qo), best(s, qs)
        c['tgt_flip'] += bo[0] != bs[0]
        both = any(o['tok'][a['ch'][0]][2] == 2 for a in o['act'] if a.get('ch')) and any(o['tok'][a['ch'][0]][2] == 1 for a in o['act'] if a.get('ch'))
        if both:
            c['both_sides'] += 1; c['both_sides_side_flip'] += bo[1] != bs[1]
            c['both_sides_swap_best_enemy'] += bs[1] == 2; c['both_sides_orig_best_enemy'] += bo[1] == 2
            sides[(r['gid'], r['seq'])][var] = bs[1]; sides[(r['gid'], r['seq'])]['orig'] = bo[1]
        c['side_flip'] += bo[1] != bs[1]
        c[f'orig_best_side_{bo[1]}'] += 1; c[f'swap_best_side_{bs[1]}'] += 1
        c['has_enemy_option'] += any(o['tok'][a['ch'][0]][2] == 2 for a in o['act'] if a.get('ch'))
        c['has_friendly_option'] += any(o['tok'][a['ch'][0]][2] == 1 for a in o['act'] if a.get('ch'))
        K = len(o['searches']); A, B = list(range(K // 2)), list(range(K // 2, K))
        na, nb = best(o, qm(o, A)), best(o, qm(o, B))
        c['noise_tgt_flip'] += na[0] != nb[0]; c['noise_side_flip'] += na[1] != nb[1]
    m = max(1, c['multi'])
    out[var] = {'n_states': c['n'], 'n_multi_target': c['multi'], 'counts': dict(c),
                'target_flip_rate': c['tgt_flip'] / m, 'target_side_flip_rate': c['side_flip'] / m,
                'noise_target_flip_rate_seed_halves': c['noise_tgt_flip'] / m,
                'noise_side_flip_rate_seed_halves': c['noise_side_flip'] / m,
                'n_both_sides_available': c['both_sides'],
                'side_flip_rate_both_sides_available': c['both_sides_side_flip'] / max(1, c['both_sides']),
                'orig_best_enemy_rate_both_sides': c['both_sides_orig_best_enemy'] / max(1, c['both_sides']),
                'variant_best_enemy_rate_both_sides': c['both_sides_swap_best_enemy'] / max(1, c['both_sides'])}
    print(var, json.dumps(out[var]))
vv = {}
vs = sorted({v for d in sides.values() for v in d if v != 'orig'})
for i in range(len(vs)):
    for j in range(i + 1, len(vs)):
        pr = [(d[vs[i]], d[vs[j]]) for d in sides.values() if vs[i] in d and vs[j] in d]
        vv[f'{vs[i]} | {vs[j]}'] = {'n_both_sides_states': len(pr), 'best_target_side_flip_rate': sum(a != b for a, b in pr) / max(1, len(pr))}
        print('VV', vs[i], vs[j], vv[f'{vs[i]} | {vs[j]}'])
out['_variant_vs_variant'] = vv
json.dump(out, open(os.path.join(H, 'analysis_tgt_raw.json'), 'w'), indent=1)
