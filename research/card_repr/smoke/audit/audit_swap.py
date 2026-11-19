"""Swap-test teacher stability. Deterministic (no RNG)."""
import json, os, math, collections, statistics as st
H = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(H, '..', 'data', 'swap', 'swap_tests.jsonl')
orig_by_state = collections.defaultdict(list)
rows = collections.defaultdict(lambda: collections.defaultdict(list))
for line in open(D):
    r = json.loads(line)
    o, s = r['versions']['orig'], r['versions']['swap']
    def summ(v):
        se = [x for x in v['searches'] if sum(x['vis']) > 0] or [{'vis':[1],'q':[0],'pi':0,'rv':0.0}]; ca = set(v['card_acts'])
        vm = [sum(x['vis'][i] for i in ca) / sum(x['vis']) for x in se]
        play = [1 if int(x['pi']) in ca else 0 for x in se]
        rv = [x['rv'] for x in se]
        # q-argmax is a card action?
        qplay = [1 if max(range(len(x['q'])), key=lambda i: x['q'][i]) in ca else 0 for x in se] if ca else [0]*len(se)
        return vm, play, rv, qplay
    vo, po, ro, qo = summ(o); vs, ps, rs, qs = summ(s)
    orig_by_state[(r['gid'], r['seq'], r['sims'])].append((r['variant'], ro, po))
    legal_same = len(o['card_acts']) == len(s['card_acts']) and len(o['act']) == len(s['act'])
    R = rows[(r['variant'], r['sims'])]
    R['legal_same'].append(legal_same)
    R['orig_play_frac'].append(sum(po) / len(po))
    R['orig_play_unanimous'].append(sum(po) in (0, len(po)))
    R['orig_qplay_unanimous'].append(sum(qo) in (0, len(qo)))
    R['qplay_flip_majority'].append((sum(qo) > len(qo) / 2) != (sum(qs) > len(qs) / 2))
    R['sd_rv_orig'].append(st.pstdev(ro))
    R['drv'].append(st.mean(rs) - st.mean(ro))
    R['card_acts_orig'].append(len(o['card_acts'])); R['card_acts_swap'].append(len(s['card_acts']))
out = {}
for (v, sims), R in sorted(rows.items()):
    n = len(R['legal_same'])
    out[f'{v}|{sims}'] = {
        'n': n, 'legal_same': sum(R['legal_same']),
        'n_card_acts_changed': sum(a != b for a, b in zip(R['card_acts_orig'], R['card_acts_swap'])),
        'frac_states_teacher_play_choice_unanimous_over_8_seeds': sum(R['orig_play_unanimous']) / n,
        'frac_states_qargmax_card_unanimous_over_8_seeds': sum(R['orig_qplay_unanimous']) / n,
        'frac_states_majority_qargmax_play_flips_on_swap': sum(R['qplay_flip_majority']) / n,
        'mean_seed_sd_rv_orig': st.mean(R['sd_rv_orig']), 'mean_abs_drv': st.mean(abs(x) for x in R['drv'])}
# orig re-searched in different variant processes with same seeds: identical?
diff = collections.Counter()
for k, lst in orig_by_state.items():
    if len(lst) < 2: continue
    ref = lst[0]
    same = all(x[1] == ref[1] and x[2] == ref[2] for x in lst[1:])
    diff['identical' if same else 'differs'] += 1
    if not same:
        diff['max_rv_gap_x1e4'] = max(diff['max_rv_gap_x1e4'], int(1e4 * max(abs(st.mean(x[1]) - st.mean(ref[1])) for x in lst)))
out['orig_across_variant_processes'] = dict(diff)
json.dump(out, open(os.path.join(H, 'audit_swap.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
