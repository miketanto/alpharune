"""Head-to-head between league snapshots (greedy, seat-balanced, RB_DECK selects the mirror deck).

  RB_DECK=azir python3 league/h2h.py az_main 300:200 300:250 --games 2000
"""
import argparse, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import match  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('run')
ap.add_argument('pairs', nargs='+')
ap.add_argument('--games', type=int, default=2000)
ap.add_argument('--procs', type=int, default=8)
a = ap.parse_args()
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs', a.run, 'league_%05d.pt')
print('deck', match.DECK, flush=True)
for pair in a.pairs:
    x, y = (int(v) for v in pair.split(':'))
    r = match.play_nets(R % x, R % y, a.games, 'B2', 'B2', procs=a.procs)
    p = (r['a_wins'] + 0.5 * r['draws']) / r['games']
    h = 1.96 * math.sqrt(p * (1 - p) / r['games'])
    print(f'@{x} vs @{y}: {p:.3f} +-{h:.3f}  (n={r["games"]}, {r["secs"]}s)', flush=True)
