"""Statistics over the exported decisions and swap tests.
Writes decision_stats.json and swap/swap_summary.json (data/)."""
import json, os, glob, gzip, math, collections, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from pipeline import split_of, group_of  # noqa


def decision_stats():
    man = json.load(open(os.path.join(DATA, 'manifest.json')))
    info = {g['gid']: g for g in man['games']}
    appear = collections.defaultdict(collections.Counter)     # (set|split) -> card -> decisions where a token has the key
    source = collections.defaultdict(collections.Counter)     # ... card is the src/abs of some legal action
    chosen = collections.defaultdict(collections.Counter)     # ... chosen action's src/abs is the card
    pk = collections.defaultdict(collections.Counter)
    ent = collections.defaultdict(list)                        # pk -> normalised visit entropy (teacher, searched)
    top_share = collections.defaultdict(list)
    qspread = collections.defaultdict(list)
    n = collections.Counter()
    for f in sorted(glob.glob(os.path.join(DATA, 'out', '*.decisions.jsonl.gz'))):
        with gzip.open(f, 'rt') as fh:
            for line in fh:
                r = json.loads(line)
                g = info[r['gid']]
                key = f"{g['set']}|{g['split']}"
                n[key] += 1
                pk[g['set']][r['pk']] += 1
                toks = r['tok']
                keys = set(t[1] for t in toks)
                for k in keys:
                    appear[key][k] += 1
                srcs = set()
                for a in r['act']:
                    for fld in ('src', 'abs'):
                        if fld in a and a[fld] >= 0:
                            srcs.add(toks[a[fld]][1])
                for k in srcs:
                    source[key][k] += 1
                a = r['act'][r['pi']]
                for fld in ('src', 'abs'):
                    if fld in a and a[fld] >= 0:
                        chosen[key][toks[a[fld]][1]] += 1
                if g['set'] == 'teacher' and r.get('searched'):
                    v = r['vis']; t = sum(v)
                    if t > 0 and len(v) > 1:
                        p = [x / t for x in v if x > 0]
                        ent[r['pk']].append(-sum(x * math.log(x) for x in p) / math.log(len(v)))
                        top_share[r['pk']].append(max(v) / t)
                        qspread[r['pk']].append(max(r['q']) - min(r['q']))
    def summ(xs):
        xs = sorted(xs)
        if not xs:
            return None
        return {'n': len(xs), 'mean': round(sum(xs) / len(xs), 4), 'p10': round(xs[len(xs) // 10], 4),
                'median': round(xs[len(xs) // 2], 4), 'p90': round(xs[9 * len(xs) // 10], 4)}
    allent = [x for v in ent.values() for x in v]
    out = {'decisions_per_set_split': dict(n),
           'decisions_per_card': {k: {'appears': dict(sorted(appear[k].items())),
                                      'is_action_source': dict(sorted(source[k].items())),
                                      'is_chosen_source': dict(sorted(chosen[k].items()))} for k in sorted(appear)},
           'prompt_kinds': {k: dict(v.most_common()) for k, v in pk.items()},
           'teacher_visits': {'normalised_entropy_all': summ(allent),
                              'normalised_entropy_by_pk': {k: summ(v) for k, v in ent.items()},
                              'top_visit_share_by_pk': {k: summ(v) for k, v in top_share.items()},
                              'q_spread_by_pk': {k: summ(v) for k, v in qspread.items()},
                              'note': 'entropy of root visits / log(n options); 1.0 = uniform'}}
    json.dump(out, open(os.path.join(DATA, 'decision_stats.json'), 'w'), indent=1)
    print(json.dumps(out['decisions_per_set_split']), json.dumps(out['teacher_visits']['normalised_entropy_all']))
    return out


def swap_summary():
    path = os.path.join(DATA, 'swap', 'swap_tests.jsonl')
    if not os.path.exists(path):
        return
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    errors = collections.Counter()
    for line in open(path):
        r = json.loads(line)
        v = f"{r['variant']}|sims{r['sims']}"
        if 'error' in r:
            errors[v] += 1; continue
        o, s = r['versions']['orig'], r['versions']['swap']
        P = per[v]
        P['fp_ok'].append(r['fp_ok']); P['legal_same'].append(o.get('legal_same_as_game', 0))
        P['n_card_acts_orig'].append(len(o['card_acts'])); P['n_card_acts_swap'].append(len(s['card_acts']))
        def stats(V):
            ms, rv = [], []
            for x in V['searches']:
                if 'error' in x:
                    errors[v] += 1; continue
                t = sum(x['vis']) or 1
                ms.append(sum(x['vis'][i] for i in V['card_acts']) / t)
                rv.append(x['rv'])
            return ms, rv
        mo, ro = stats(o); ms, rs = stats(s)
        if mo and ms:
            mean = lambda a: sum(a) / len(a)
            P['card_mass_orig'].append(mean(mo)); P['card_mass_swap'].append(mean(ms))
            P['rv_orig'].append(mean(ro)); P['rv_swap'].append(mean(rs))
            P['rv_diff'].append(mean(rs) - mean(ro))
            # seed-to-seed noise of orig (sd) for scale
            m = mean(ro); P['rv_orig_seed_sd'].append((sum((x - m) ** 2 for x in ro) / max(1, len(ro) - 1)) ** 0.5)
            P['card_playable_orig'].append(1 if o['card_acts'] else 0)
            P['card_playable_swap'].append(1 if s['card_acts'] else 0)
    out = {}
    for v, P in sorted(per.items()):
        mean = lambda a: round(sum(a) / len(a), 4) if a else None
        out[v] = {'states': len(P['fp_ok']), 'fp_ok': sum(P['fp_ok']), 'legal_same_as_game': sum(P['legal_same']),
                  'card_playable_orig': sum(P['card_playable_orig']), 'card_playable_swap': sum(P['card_playable_swap']),
                  'mean_card_visit_mass_orig': mean(P['card_mass_orig']), 'mean_card_visit_mass_swap': mean(P['card_mass_swap']),
                  'mean_root_value_orig': mean(P['rv_orig']), 'mean_root_value_swap': mean(P['rv_swap']),
                  'mean_root_value_diff': mean(P['rv_diff']),
                  'mean_abs_root_value_diff': mean([abs(x) for x in P['rv_diff']]),
                  'mean_orig_root_value_seed_sd': mean(P['rv_orig_seed_sd'])}
    out['_errors'] = dict(errors)
    json.dump(out, open(os.path.join(DATA, 'swap', 'swap_summary.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    what = sys.argv[1:] or ['decisions', 'swap']
    if 'decisions' in what:
        decision_stats()
    if 'swap' in what:
        swap_summary()
