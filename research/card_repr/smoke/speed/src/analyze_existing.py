"""Existing data: per-game secs / decisions in smoke/data/out/*.games.jsonl, and the 45 training
runs in smoke/train/runs (wall-clock to the best validation checkpoint per head, still improving?).
Writes raw/existing_games.json and raw/existing_runs.json."""
import json, glob, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); SM = os.path.join(HERE, '..', '..')
def q(xs, p): xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))]
def summ(xs): return {'n': len(xs), 'median': round(st.median(xs), 4), 'mean': round(st.mean(xs), 4),
                      'p10': round(q(xs, .1), 4), 'p90': round(q(xs, .9), 4)}
rows = {}
for f in glob.glob(os.path.join(SM, 'data', 'out', '*.games.jsonl')):
    for line in open(f, encoding='utf-8'):
        g = json.loads(line)
        if g['term'].startswith('error') or g['term'] == 'stopped': continue
        kind = g['agent_a'] if g['agent_a'] == g['agent_b'] else 'mixed'
        r = rows.setdefault(kind, {'secs': [], 'n_seq': [], 'n_multi': [], 'turns': [], 'secs_per_multi': []})
        r['secs'].append(g['secs']); r['n_seq'].append(g['n_seq']); r['n_multi'].append(g['n_multi'])
        r['turns'].append(g['turns']); r['secs_per_multi'].append(g['secs'] / max(1, g['n_multi']))
out = {k: {m: summ(v) for m, v in r.items()} | {'total_secs': round(sum(r['secs']), 1)} for k, r in rows.items()}
json.dump(out, open(os.path.join(HERE, '..', 'raw', 'existing_games.json'), 'w'), indent=1)
for k, v in out.items(): print(k, v['secs']['n'], 'secs', v['secs'], 'n_seq', v['n_seq']['median'], 'n_multi', v['n_multi']['median'], 'total', v['total_secs'])

runs = []
for f in sorted(glob.glob(os.path.join(SM, 'train', 'runs', '*', 'metrics.json'))):
    m = json.load(open(f)); vc = m['val_curve']; steps = m['steps']; ts = m['train_secs']
    pol = {s: a + b for s, a, b, c in vc}; val = {s: c for s, a, b, c in vc}
    last = vc[-1][0]
    def at(d, s): return d[s]
    runs.append({'name': m['name'], 'threads': m['threads'], 'bs': m['bs'], 'steps': steps,
                 'train_decisions': m['train_decisions'], 'train_secs': round(ts, 1), 'total_secs': round(m['total_secs'], 1),
                 'secs_per_step_incl_eval': round(ts / steps, 4),
                 'best_policy_step': m['best_policy_step'], 'best_value_step': m['best_value_step'],
                 'wall_to_best_policy_s': round(ts * m['best_policy_step'] / steps, 1),
                 'wall_to_best_value_s': round(ts * m['best_value_step'] / steps, 1),
                 # policy val improvement over the last 1000 steps (positive = still improving)
                 'policy_val_drop_last1000': round(at(pol, last - 1000) - at(pol, last), 5),
                 'policy_val_drop_last500': round(at(pol, last - 500) - at(pol, last), 5),
                 'policy_best_is_last_or_second_last': m['best_policy_step'] >= last - 250,
                 'value_val_at_best': round(m['best_value_val'], 4), 'value_val_final': round(at(val, last), 4)})
agg = {
  'n_runs': len(runs),
  'train_secs': summ([r['train_secs'] for r in runs]),
  'secs_per_step_incl_eval': summ([r['secs_per_step_incl_eval'] for r in runs]),
  'best_policy_step': summ([r['best_policy_step'] for r in runs]),
  'best_value_step': summ([r['best_value_step'] for r in runs]),
  'wall_to_best_policy_s': summ([r['wall_to_best_policy_s'] for r in runs]),
  'wall_to_best_value_s': summ([r['wall_to_best_value_s'] for r in runs]),
  'policy_val_drop_last1000': summ([r['policy_val_drop_last1000'] for r in runs]),
  'policy_val_drop_last500': summ([r['policy_val_drop_last500'] for r in runs]),
  'frac_policy_best_in_last_500_steps': round(sum(r['policy_best_is_last_or_second_last'] for r in runs) / len(runs), 3),
  'value_overfit_final_minus_best': summ([r['value_val_final'] - r['value_val_at_best'] for r in runs]),
  'note': 'runs used 3 concurrent workers x 2 torch threads on Windows CPU (launch.py), bs 256, 3000 steps, eval every 250 steps on 3000 val decisions',
}
json.dump({'aggregate': agg, 'runs': runs}, open(os.path.join(HERE, '..', 'raw', 'existing_runs.json'), 'w'), indent=1)
print(json.dumps(agg, indent=1))
