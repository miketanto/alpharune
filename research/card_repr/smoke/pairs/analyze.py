"""Flip analysis for the play-changing pairs (logic follows audit/audit_swap.py,
but on q averaged over seeds). Deterministic."""
import json, os, glob, collections, statistics as st, sys
H = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-9
RAW = sys.argv[1] if len(sys.argv) > 1 else 'raw_s200'
def qmean(v, idx=None):
    se = [x for x in v['searches'] if 'q' in x and sum(x['vis']) > 0]
    if idx is not None: se = [se[i] for i in idx if i < len(se)]
    if not se: return None
    return [st.mean(x['q'][i] for x in se) for i in range(len(v['act']))]
def decide(v, q):
    """('noplay'|'play'|'illegal', best card act key, margin)"""
    ca = v['card_acts']
    if not ca: return 'illegal', None, 0.0
    other = [q[i] for i in range(len(q)) if i not in ca]
    bc = max(ca, key=lambda i: q[i])
    mo = max(other) if other else -9
    play = q[bc] > mo + EPS
    a = v['act'][bc]
    return ('play' if play else 'noplay'), (a.get('t'), a.get('dst'), json.dumps(a.get('tg'))), q[bc] - mo
def own_units(v, bf_only=False):
    return sum(1 for t in v['tok'] if t[0] == 'c' and t[2] == 1 and t[9] > 0 and
               ((t[5] >= 1) if bf_only else t[4] in (3, 8)))
COND = {  # condition of the conditional variants, read from the observation (proxy: unit = board card with Might > 0)
    'synthetic:punch_first__score_nounits': lambda v: own_units(v) == 0,
    'synthetic:pit_rookie__score_alone': lambda v: own_units(v) == 0,
    'synthetic:punch_first__score_bfunit': lambda v: own_units(v, True) > 0,
    'synthetic:punch_first__score_behind': lambda v: v['g'][5] > v['g'][4],
    'synthetic:sabotage__score_opphand4': lambda v: v['g'][10] >= 4,
}
res = {}; orig_dec = collections.defaultdict(dict); swap_dec = collections.defaultdict(dict)
for f in sorted(glob.glob(os.path.join(H, RAW, '*.jsonl'))):
    rows = [json.loads(l) for l in open(f)]
    rows = [r for r in rows if 'versions' in r]
    var = rows[0]['variant']; sims = rows[0]['sims']
    c = collections.Counter(); mq = []
    for r in rows:
        o, s = r['versions']['orig'], r['versions']['swap']
        qo, qs = qmean(o), qmean(s)
        if qo is None or qs is None: c['no_search'] += 1; continue
        c['n'] += 1
        do, ds = decide(o, qo), decide(s, qs)
        orig_dec[(r['gid'], r['seq'])][var] = do[0]
        swap_dec[(r['base'], r['gid'], r['seq'])][var] = ds[0]
        c['orig_play'] += do[0] == 'play'; c['swap_play'] += ds[0] == 'play'
        c['orig_qties_all'] += max(qo) - min(qo) < EPS
        if ds[0] == 'illegal' and do[0] != 'illegal': c['swap_illegal'] += 1
        if len(o['card_acts']) != len(s['card_acts']): c['card_acts_count_changed'] += 1
        if do[0] != 'illegal' and ds[0] != 'illegal':
            c['both_legal'] += 1
            c['play_flip'] += do[0] != ds[0]
            c['flip_to_play'] += do[0] == 'noplay' and ds[0] == 'play'
            c['flip_to_noplay'] += do[0] == 'play' and ds[0] == 'noplay'
            if do[0] == ds[0] == 'play':
                c['both_play'] += 1; c['dst_flip'] += do[1] != ds[1]
            mq.append(ds[2] - do[2])
            if var in COND:
                cv = COND[var](s)
                c['cond_true' if cv else 'cond_false'] += 1
                c['play_cond_true' if cv else 'play_cond_false'] += ds[0] == 'play'
        # noise floor: two halves of the seeds, orig only
        K = len(o['searches']); a, b = list(range(K // 2)), list(range(K // 2, K))
        qa, qb = qmean(o, a), qmean(o, b)
        na, nb = decide(o, qa), decide(o, qb)
        if na[0] != 'illegal':
            c['noise_n'] += 1; c['noise_play_flip'] += na[0] != nb[0]
            if na[0] == nb[0] == 'play': c['noise_both_play'] += 1; c['noise_dst_flip'] += na[1] != nb[1]
    bl = max(1, c['both_legal'])
    res[var] = {'sims': sims, 'seeds': K, 'n_states': c['n'], 'counts': dict(c),
        'play_flip_rate': c['play_flip'] / bl,
        'flip_to_play_rate': c['flip_to_play'] / bl, 'flip_to_noplay_rate': c['flip_to_noplay'] / bl,
        'orig_play_rate': c['orig_play'] / max(1, c['n']), 'swap_play_rate': c['swap_play'] / max(1, c['n']),
        'dst_flip_rate_given_both_play': (c['dst_flip'] / c['both_play']) if c['both_play'] else None,
        'noise_play_flip_rate_seed_halves': c['noise_play_flip'] / max(1, c['noise_n']),
        'noise_dst_flip_rate_seed_halves': (c['noise_dst_flip'] / c['noise_both_play']) if c['noise_both_play'] else None,
        'swap_no_legal_play_rate': c['swap_illegal'] / max(1, c['n']),
        'swap_play_rate_cond_true': (c['play_cond_true'] / c['cond_true']) if c['cond_true'] else None,
        'swap_play_rate_cond_false': (c['play_cond_false'] / c['cond_false']) if c['cond_false'] else None,
        'mean_delta_play_margin': st.mean(mq) if mq else None}
# second noise check: orig is re-searched in every variant process of the same base
dis = tot = 0
for k, d in orig_dec.items():
    vals = list(d.values())
    if len(vals) >= 2: tot += 1; dis += len(set(vals)) > 1
res['_orig_disagreement_across_processes'] = {'states': tot, 'disagree': dis, 'rate': dis / max(1, tot)}
# variant vs variant (same base, same states): both swap versions searched with the same seeds
vv = {}
bases = collections.defaultdict(set)
for (b, g, q), d in swap_dec.items(): bases[b].update(d)
for b, vs in bases.items():
    vs = sorted(vs)
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            n = fl = 0
            for (bb, g, q), d in swap_dec.items():
                if bb != b or vs[i] not in d or vs[j] not in d: continue
                if 'illegal' in (d[vs[i]], d[vs[j]]): continue
                n += 1; fl += d[vs[i]] != d[vs[j]]
            vv[f'{vs[i]} | {vs[j]}'] = {'n': n, 'play_flip_rate': fl / max(1, n)}
res['_variant_vs_variant'] = vv
json.dump(res, open(os.path.join(H, f'analysis_{RAW}.json'), 'w'), indent=1)
for v, r in res.items():
    if v == '_variant_vs_variant':
        for k, x in r.items(): print('  VV', k, x)
        continue
    if v.startswith('_'): print(v, r); continue
    print(f"{v:42s} n={r['n_states']:3d} flip={r['play_flip_rate']:.2f} (->play {r['flip_to_play_rate']:.2f}) noise={r['noise_play_flip_rate_seed_halves']:.2f} "
          f"orig_play={r['orig_play_rate']:.2f} swap_play={r['swap_play_rate']:.2f} dst={r['dst_flip_rate_given_both_play']} illegal={r['swap_no_legal_play_rate']:.2f} condT={r['counts'].get('cond_true')}:{r['swap_play_rate_cond_true']} condF={r['counts'].get('cond_false')}:{r['swap_play_rate_cond_false']}")
