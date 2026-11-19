"""Adversarial audit: data-level checks (split, leakage, labels). Seed-free (deterministic full scans)."""
import json, gzip, os, glob, math, collections
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
out = {}
man = json.load(open(os.path.join(D, 'manifest.json')))
games = man['games']
gid_split = {}
bad_split = []; dup = []
for g in games:
    exp = {0:'train',1:'train',2:'train',3:'train',4:'train',5:'train',6:'train',7:'val',8:'test',9:'test'}[g['seed'] % 10]
    if g['split'] != exp: bad_split.append(g['gid'])
    if g['gid'] in gid_split: dup.append(g['gid'])
    gid_split[g['gid']] = g['split']
seeds = collections.Counter(g['seed'] for g in games)
out['split'] = {'n_games': len(games), 'split_mismatch': len(bad_split), 'dup_gid': len(dup),
                'dup_seed': sum(1 for s, c in seeds.items() if c > 1),
                'dup_engine_seed': sum(1 for s, c in collections.Counter(g['engine_seed'] for g in games).items() if c > 1)}
# swap states
sw_gids = set(); sw_bad = []
for line in open(os.path.join(D, 'swap', 'swap_tests.jsonl')):
    r = json.loads(line)
    sw_gids.add(r['gid'])
gm = {g['gid']: g for g in games}
out['swap_gids'] = {'n': len(sw_gids), 'splits': dict(collections.Counter(gm[x]['split'] for x in sw_gids)),
                    'groups': dict(collections.Counter(gm[x]['group'] for x in sw_gids))}
# decision scan
VARKEYS = None
stats = collections.Counter()
ent = []; zcheck = collections.Counter()
tok_len = collections.Counter(); act_fields = collections.Counter()
nc_vals = collections.Counter()
play_kind = collections.defaultdict(collections.Counter)   # card key -> intent type of actions it is src of
legal_pk = collections.defaultdict(collections.Counter)
qbest_unvisited = 0; qbest_inf = 0; pi_not_argmax = 0; n_srch = 0; unvisited_opts = 0; n_opts = 0
base_train_keys = collections.Counter()
for fn in sorted(glob.glob(os.path.join(D, 'out', '*teacher*.decisions.jsonl.gz'))):
    grp = os.path.basename(fn).split('__s')[0]
    for line in gzip.open(fn, 'rt'):
        d = json.loads(line)
        sp = gid_split[d['gid']]
        g = gm[d['gid']]
        # z vs winner
        w = g['winner']; seat = d['seat']
        expz = 0 if w == 0 else (1 if w == seat else -1)
        zcheck['ok' if d['z'] == expz else 'bad'] += 1
        for t in d['tok']:
            tok_len[(t[0], len(t))] += 1
            if grp == 'base_teacher' and sp == 'train':
                base_train_keys[t[1]] += 1
        for a in d['act']:
            for k in a: act_fields[k] += 1
            if 'nc' in a: nc_vals[a['nc']] += 1
            if 'src' in a and 0 <= a['src'] < len(d['tok']):
                k = d['tok'][a['src']][1]
                play_kind[k][a['t']] += 1
                legal_pk[k][d['pk']] += 1
        if d.get('searched') and d.get('vis'):
            n_srch += 1
            v = d['vis']; q = d['q']; s = sum(v)
            if s > 0 and len(v) > 1:
                p = [x / s for x in v if x > 0]
                ent.append(-sum(x * math.log(x) for x in p) / math.log(len(v)))
            mv = max(v)
            if v[d['pi']] != mv: pi_not_argmax += 1
            n_opts += len(v); unvisited_opts += sum(1 for x in v if x == 0)
            mq = max(q)
            if mq != min(q):
                qbest_inf += 1
                if any(q[i] == mq and v[i] == 0 for i in range(len(v))): qbest_unvisited += 1
ent.sort()
out['z_vs_winner'] = dict(zcheck)
out['token_shapes'] = {f'{k[0]}:{k[1]}': c for k, c in tok_len.items()}
out['action_fields'] = dict(act_fields)
out['nc_values_top'] = nc_vals.most_common(10)
out['teacher'] = {'searched': n_srch, 'pi_not_argmax_vis': pi_not_argmax,
                  'unvisited_option_frac': unvisited_opts / max(1, n_opts),
                  'q_informative': qbest_inf, 'qbest_is_unvisited_option': qbest_unvisited,
                  'entropy_mean': sum(ent) / len(ent), 'entropy_p10': ent[len(ent) // 10], 'entropy_median': ent[len(ent) // 2]}
cards = json.load(open(os.path.join(D, '..', '..', 'cards.json')))
pairs = [('punch_first', 'primal_strength')] + [(s['base'], 'synthetic:' + s['id']) for s in cards['t2_minimal_pairs_synthetic']]
out['intent_type_by_card'] = {k: dict(play_kind[k]) for p in pairs for k in p}
out['prompt_kind_by_card'] = {k: dict(legal_pk[k]) for p in pairs for k in p}
out['variant_keys_in_base_train_tokens'] = {v: base_train_keys.get(v, 0) for _, v in pairs}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audit_data.json'), 'w'), indent=1)
print(json.dumps(out, indent=1)[:6000])
