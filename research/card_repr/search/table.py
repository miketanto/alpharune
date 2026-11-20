"""Prints the M1 table from out/*.json (eval_search.py results)."""
import glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = ['n0_greedy', 'n1_pool', 'n16_pool', 'n64_pool', 'n200_pool', 'n64_decklist', 'n64_perfect', 'n64_c15',
        'n64_c30', 'n1_vs_ismcts200', 'n64_vs_ismcts200']
print(f'{"run":18s} {"N":>4s} {"opp":>14s} {"games":>5s} {"win":>6s} {"wilson95":>15s} {"pairs 0/1/2":>12s} '
      f'{"changed":>8s} {"games/h":>8s} {"sims/s":>7s}')
for r in ROWS:
    p = os.path.join(HERE, 'out', r + '.json')
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    st = d['stats']
    ch = f'{st["changed"] / st["searches"]:.3f}' if st['searches'] else '-'
    sps = st['sims'] / d['secs'] if d['secs'] else 0
    opp = d.get('opponent', 'greedy net')
    print(f'{r:18s} {d["sims"]:4d} {opp[:14]:>14s} {d["games"]:5d} {d["win_rate"]:6.3f} {str(d["wilson95"]):>15s} '
          f'{"/".join(map(str, d["pairs_won_0_1_2"])):>12s} {ch:>8s} {d["games_per_hour"]:8.0f} {sps:7.0f}'
          + (f'  errors {d["errors"]}' if d['errors'] else ''))
