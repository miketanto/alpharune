"""Aggregates raw/jobs.csv + raw/games/*.games.jsonl (speed_export runs) into raw/games_summary.json.
Per config (tag prefix without _r<k>[_k<j>]): per repeat games/hour (games / job wall), decisions,
searches, per-prompt-kind decide time; then median and min/max over the repeats."""
import csv, json, os, re, glob, statistics as st, collections
HERE = os.path.dirname(os.path.abspath(__file__)); RAW = os.path.join(HERE, '..', 'raw')
RUNE = ('rune for energy', 'rune for power')


def med(xs):
    return {'median': round(st.median(xs), 4), 'min': round(min(xs), 4), 'max': round(max(xs), 4), 'n': len(xs)}


jobs = list(csv.DictReader(open(os.path.join(RAW, 'jobs.csv'))))
games = {}
for j in jobs:
    gs = [json.loads(l) for l in open(os.path.join(RAW, 'games', j['tag'] + '.games.jsonl'), encoding='utf-8')]
    games[j['tag']] = gs


def game_stats(gs):
    """decisions / searches / time split for a list of games."""
    out = collections.Counter(); kinds = collections.defaultdict(lambda: [0.0] * 5)
    for g in gs:
        out['games'] += 1; out['secs'] += g['secs']; out['n_seq'] += g['n_seq']; out['n_multi'] += g['n_multi']
        out['turns'] += g['turns']; out['errors'] += g['term'].startswith('error')
        for k, a in g['timing'].items():
            kk = kinds[k]
            for i in range(5):
                kk[i] += a[i]
            out['searches'] += a[1]; out['t_decide'] += a[2]; out['t_encode'] += a[3]
            if k in RUNE:
                out['rune_multi'] += a[0]; out['rune_searches'] += a[1]; out['t_rune'] += a[2]
    return out, kinds


summary = {}
# --- sweep / autopay / random: one job = one repeat
by_cfg = collections.defaultdict(list)
for j in jobs:
    m = re.match(r'(sw_\w+?|ap_\w+?)_r(\d+)$', j['tag'])
    if m:
        by_cfg[m.group(1)].append(j)
for cfg, js in sorted(by_cfg.items()):
    reps = []
    for j in js:
        o, kinds = game_stats(games[j['tag']])
        wall = float(j['wall_s']); n = o['games']
        reps.append({'tag': j['tag'], 'cpu': j['cpu'], 'games': n, 'wall_s': round(wall, 2),
                     'games_per_hour': round(3600 * n / wall, 2), 'games_per_s': round(n / wall, 3),
                     'secs_per_game': round(o['secs'] / n, 4),
                     'decisions_per_game': round(o['n_seq'] / n, 1), 'multi_option_per_game': round(o['n_multi'] / n, 1),
                     'searches_per_game': round(o['searches'] / n, 1), 'turns_per_game': round(o['turns'] / n, 1),
                     'secs_per_search': round(o['t_decide'] / max(1, o['searches']), 5),
                     'decide_frac_of_game': round(o['t_decide'] / o['secs'], 4),
                     'encode_frac_of_game': round(o['t_encode'] / o['secs'], 4),
                     'rune_frac_of_multi': round(o['rune_multi'] / max(1, o['n_multi']), 4),
                     'rune_frac_of_searches': round(o['rune_searches'] / max(1, o['searches']), 4),
                     'rune_frac_of_decide_time': round(o['t_rune'] / max(1e-9, o['t_decide']), 4),
                     'rune_frac_of_game_time': round(o['t_rune'] / o['secs'], 4),
                     'decisions_bytes_per_game': round(int(j['decisions_bytes']) / n),
                     'errors': o['errors'], 'loadavg_start': j['loadavg_start'], 'loadavg_end': j['loadavg_end']})
    agg = {k: med([r[k] for r in reps]) for k in reps[0] if isinstance(reps[0][k], (int, float)) and k != 'errors'}
    # per prompt kind (pooled over repeats)
    pk = collections.defaultdict(lambda: [0.0] * 5)
    for j in js:
        for k, a in game_stats(games[j['tag']])[1].items():
            for i in range(5):
                pk[k][i] += a[i]
    ng = sum(r['games'] for r in reps)
    kinds = {k: {'per_game': round(a[0] / ng, 2), 'searched_per_game': round(a[1] / ng, 2),
                 'mean_decide_ms': round(1000 * a[2] / max(1, a[0]), 3),
                 'decide_secs_per_game': round(a[2] / ng, 4), 'mean_options': round(a[4] / max(1, a[0]), 2)}
             for k, a in sorted(pk.items(), key=lambda x: -x[1][2])}
    summary[cfg] = {'repeats': reps, 'aggregate': agg, 'by_prompt_kind': kinds}

# --- scaling: sc_P<P>_r<r>_k<k>: per repeat, aggregate games/hour over the P concurrent processes
sc = collections.defaultdict(lambda: collections.defaultdict(list))
for j in jobs:
    m = re.match(r'sc_P(\d+)_r(\d+)_k(\d+)$', j['tag'])
    if m:
        sc[int(m.group(1))][int(m.group(2))].append(j)
scaling = {}
for P, reps in sorted(sc.items()):
    rows = []
    for r, js in sorted(reps.items()):
        n = sum(int(j['count']) for j in js)
        t0 = min(float(j['t0']) for j in js); t1 = max(float(j['t1']) for j in js)
        per_proc = [3600 * int(j['count']) / float(j['wall_s']) for j in js]
        searches = sum(game_stats(games[j['tag']])[0]['searches'] for j in js)
        tsearch = sum(game_stats(games[j['tag']])[0]['t_decide'] for j in js)
        rows.append({'rep': r, 'games': n, 'span_s': round(t1 - t0, 1),
                     'aggregate_games_per_hour_span': round(3600 * n / (t1 - t0), 1),
                     # sum of per-process rates: removes the "wait for the slowest" tail of a block
                     'aggregate_games_per_hour_sum_rates': round(sum(per_proc), 1),
                     'per_process_games_per_hour_median': round(st.median(per_proc), 1),
                     'secs_per_search': round(tsearch / searches, 5)})
    scaling[P] = {'repeats': rows, 'aggregate': {k: med([x[k] for x in rows]) for k in rows[0] if k != 'rep'}}
summary['scaling_sims100'] = scaling
json.dump(summary, open(os.path.join(RAW, 'games_summary.json'), 'w'), indent=1)
for k, v in summary.items():
    if k == 'scaling_sims100':
        for P, s in v.items():
            print('P', P, {kk: vv['median'] for kk, vv in s['aggregate'].items()})
    else:
        a = v['aggregate']
        print(k, 'g/h', a['games_per_hour']['median'], a['games_per_hour']['min'], a['games_per_hour']['max'],
              'dec', a['decisions_per_game']['median'], 'multi', a['multi_option_per_game']['median'],
              'search', a['searches_per_game']['median'], 's/search', a['secs_per_search']['median'],
              'rune_t', a['rune_frac_of_game_time']['median'], 'enc', a['encode_frac_of_game']['median'])
