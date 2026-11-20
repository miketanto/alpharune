"""Demo: register existing checkpoints + anchors, play a greedy round robin on CPU cores 12-15, rate.

  python3 demo.py run    [--games 100] [--deadline-min 75]     # plays missing pairs in priority order
  python3 demo.py report                                       # writes league/ratings_demo.txt

Pairs already in results_demo.jsonl are skipped (incremental). Priority: net-net and cheap anchors first,
then ismcts:sims=100 for every net, then ismcts:sims=200 (fewer games) for the top provisional nets.
Writes league/registry_demo.json, league/results_demo.jsonl, league/ratings_demo.txt.
"""
import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE  # noqa: E402
from match import play  # noqa: E402
from ratings import ResultStore, fit_bt, ratings_table, residuals, winrate_matrix  # noqa: E402
from registry import Registry  # noqa: E402

RUNS_DEMO = ['real_smoke_id', 'gate1_sanity_id_ask', 'gate2v2_sanity_B2_ask', 'gate3_B2_v3', 'gate4_B2', 'gate4_id']
REG = os.path.join(HERE, 'registry_demo.json')
RES = os.path.join(HERE, 'results_demo.jsonl')
OUT = os.path.join(HERE, 'ratings_demo.txt')


HIST = [('gate4_id', [50, 100]), ('gate4_B2', [50, 100])]   # --hist: intermediate snapshots (role historical)


def build_registry(hist=False):
    reg = Registry(REG)
    reg.add_anchors()
    for r in RUNS_DEMO:
        reg.add_run(r, role='main', which='latest')
    if hist:
        for r, ups in HIST:
            for u in ups:
                reg.add_checkpoint(f'runs/{r}/ckpt_{u:05d}.pt', role='historical')
    reg.save()
    return reg


def run(a):
    reg = build_registry(a.hist)
    print(reg.table(), flush=True)
    st = ResultStore(RES)
    nets = [e['id'] for e in reg.members(kind='net', loadable=True)]
    t_end = time.time() + a.deadline_min * 60
    plan = [(x, y, a.games) for x, y in itertools.combinations(nets, 2)]
    plan += [(x, 'anchor:random-player', a.games) for x in nets]
    plan += [(x, 'anchor:ismcts:sims=25', a.games) for x in nets]
    plan += [(x, 'anchor:ismcts:sims=100', a.games) for x in nets]
    for (x, y, n) in plan:
        _play(reg, st, x, y, n, t_end, a.procs)
    # ismcts:sims=200 for the top provisional nets
    names, W, N = st.matrix()
    f = fit_bt(names, W, N)
    top = [k for k in sorted(f['elo'], key=lambda k: -f['elo'][k]) if k in nets]
    top = top if a.all200 else top[:a.top200]
    for x in top:
        _play(reg, st, x, 'anchor:ismcts:sims=200', a.games200, t_end, a.procs)
    print('done', flush=True)


def _play(reg, st, x, y, n, t_end, procs):
    have = st.pair_games().get(tuple(sorted((x, y))), 0)
    if have >= n:
        return
    if time.time() > t_end:
        print(f'deadline: skip {x} vs {y}', flush=True)
        return
    rec = play(reg, x, y, n - have, procs=procs)
    st.append(rec)
    print(json.dumps(rec), flush=True)


def report(a):
    reg = Registry(REG)
    st = ResultStore(RES)
    names, W, N = st.matrix()
    f = fit_bt(names, W, N, prior_games=a.prior, boot=a.boot)
    extra = {}
    for e in reg.members():
        if e['kind'] == 'net':
            extra[e['id']] = f"{e['repr']}, schema v{e['schema']}, " + ','.join(k for k, v in e['opts'].items() if v)
    short = {e['id']: (e['spec'] if e['kind'] == 'anchor' else f"{e['run']}@{e['update']}")
             .replace('ismcts:sims=', 'ismcts').replace('_sanity', '').replace('random-player', 'random')
             .replace('gate2v2_B2_ask', 'g2_B2_ask').replace('gate1_id_ask', 'g1_id_ask').replace('gate', 'g')
             for e in reg.members()}
    order = sorted(f['elo'], key=lambda k: -f['elo'][k])
    recs = st.load()
    lines = ['League ratings demo (league/demo.py), ' + time.strftime('%Y-%m-%d %H:%M'),
             'Greedy (argmax) networks on CPU cores 12-15; Rengar mirror; one game per env slot.',
             f'Bradley-Terry / Elo, anchor random-player = 0, prior {a.prior} virtual draw per played pair, '
             f'{a.boot} parametric bootstrap resamples.',
             'Net-vs-net pairs alternate seats; net-vs-anchor games (verify_eval.py) have the network on seat 1;',
             'anchor-vs-anchor games (anchor_games.py, riftbound CLI) alternate seats, agents pay runes themselves.',
             f'{len(recs)} match records, {int(N.sum() / 2)} games.', '']
    skipped = [e for e in reg.members(kind='net') if not e['loadable']]
    for e in skipped:
        lines.append(f"SKIPPED {e['id']}: {e['why']}")
    lines += ['', ratings_table(f, extra), '', winrate_matrix(names, W, N, order, short), '',
              'Fit check: pairs that fit the single rating line worst (z = (obs - pred) / binomial sd)',
              residuals(f, names, W, N, 10), '', 'Match records:']
    for r in recs:
        lines.append(f"  {r['a']:30s} vs {r['b']:28s} {r['a_wins']:4d}-{r['b_wins']:4d} (draws {r['draws']}, "
                     f"err {r['errors']}, A seat1 {r['a_seat1_wins']}/{r['a_seat1_games']}) {r['mode']} "
                     f"{r['secs']}s {r.get('via', '')}")
    txt = '\n'.join(lines) + '\n'
    open(OUT, 'w').write(txt)
    json.dump(f, open(os.path.join(HERE, 'ratings_demo.json'), 'w'), indent=1)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'report', 'registry'])
    ap.add_argument('--games', type=int, default=100)
    ap.add_argument('--games200', type=int, default=60)
    ap.add_argument('--top200', type=int, default=3)
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--deadline-min', type=float, default=75)
    ap.add_argument('--prior', type=float, default=1.0)
    ap.add_argument('--boot', type=int, default=300)
    ap.add_argument('--hist', action='store_true', help='also rate intermediate snapshots (HIST)')
    ap.add_argument('--all200', action='store_true', help='ismcts:sims=200 for every net, not only the top ones')
    a = ap.parse_args()
    if a.cmd == 'run':
        run(a)
    elif a.cmd == 'registry':
        print(build_registry(a.hist).table())
    else:
        report(a)


if __name__ == '__main__':
    main()
