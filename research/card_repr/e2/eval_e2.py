"""E2 head-to-heads (greedy, seat-balanced, common seeds, Rengar mirror) via league/match.play_nets.

  RB_DECK=rengar python3 e2/eval_e2.py --games 2000 --out e2/h2h.jsonl B500=runs/e2b_main/league_00500.pt:base ...
  pair syntax: NAME_A=PATH_A:NAME_B  with names defined by --net NAME=PATH (paths relative to selfplay/)
"""
import argparse, json, math, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SP = os.path.normpath(os.path.join(HERE, '..', 'selfplay'))
sys.path.insert(0, os.path.join(SP, 'league'))
import match  # noqa: E402


def wilson(w, g, z=1.96):
    p = w / g
    d = 1 + z * z / g
    c = p + z * z / (2 * g)
    h = z * math.sqrt(p * (1 - p) / g + z * z / (4 * g * g))
    return (c - h) / d, (c + h) / d


ap = argparse.ArgumentParser()
ap.add_argument('--net', action='append', default=[], help='NAME=path (relative to selfplay/)')
ap.add_argument('--games', type=int, default=2000)
ap.add_argument('--procs', type=int, default=14)
ap.add_argument('--cores', default='0-15')
ap.add_argument('--out', default=os.path.join(HERE, 'h2h.jsonl'))
ap.add_argument('pairs', nargs='+', help='A:B (names from --net)')
a = ap.parse_args()
nets = dict(x.split('=', 1) for x in a.net)
lo, hi = (int(v) for v in a.cores.split('-'))
for pair in a.pairs:
    x, y = pair.split(':')
    px, py = (os.path.join(SP, nets[k]) for k in (x, y))
    t0 = time.time()
    r = match.play_nets(px, py, a.games, 'B2', 'B2', procs=a.procs, cores=list(range(lo, hi + 1)))
    w = r['a_wins'] + 0.5 * r['draws']
    p = w / r['games']
    l, h = wilson(w, r['games'])
    rec = dict(a=x, b=y, a_path=nets[x], b_path=nets[y], **r, win_rate=round(p, 4), wilson95=[round(l, 4), round(h, 4)])
    open(a.out, 'a').write(json.dumps(rec) + '\n')
    print(f'{x} vs {y}: {p:.3f} [{l:.3f}, {h:.3f}]  n={r["games"]} errors={r["errors"]} ({time.time() - t0:.0f}s)',
          flush=True)
