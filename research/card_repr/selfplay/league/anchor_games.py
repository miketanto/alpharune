"""Anchor-vs-anchor games (C++ agent vs C++ agent) through the engine CLI, so anchors are linked directly in the
results matrix and not only through the networks (league/README.txt).

rbenv cannot seat two C++ agents (seat 1 is always external), so these games run in the WSL build's `riftbound`
binary (~/riftbound/alpharune/build/riftbound, Rengar mirror, Duel). Seats are balanced: half of the games with
A as Player 1, half with A as Player 2 (two runs, different seeds). Pinned to cores 12-15 (taskset).
Caveat: in the binary the agents also answer rune-payment prompts themselves; in rbenv (net games) those are
auto-paid. Both are the same engine and agents otherwise.

  python3 anchor_games.py random-player ismcts:sims=25 100 [--results league/results_demo.jsonl] [--threads 4]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EVAL_CORES  # noqa: E402
from ratings import ResultStore  # noqa: E402

REPO_WSL = os.path.expanduser('~/riftbound/alpharune')
BIN = os.path.join(REPO_WSL, 'build', 'riftbound')
DECK = os.path.join(REPO_WSL, 'decks', os.environ.get('RB_DECK', 'rengar') + '.txt')   # RB_DECK: mirror deck (default rengar)


def run_cli(a1, a2, n, seed, threads, cores=EVAL_CORES):
    cmd = ['taskset', '-c', ','.join(map(str, cores)), BIN, '--deck1', DECK, '--deck2', DECK, '--agent1', a1,
           '--agent2', a2, '--games', str(n), '--threads', str(threads), '--seed', str(seed), '--web', 'off']
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_WSL, env=dict(os.environ, RIFTBOUND_ROOT='.'))
    out = p.stdout
    g = lambda k: int(re.search(rf'{k}:\s+(\d+)', out).group(1))
    return g('P1 wins'), g('P2 wins'), g('Draws')


def play_anchors(a, b, n, seed=880001, threads=4):
    t0 = time.time()
    h = n // 2
    w1, l1, d1 = run_cli(a, b, h, seed, threads)               # a = Player 1
    l2, w2, d2 = run_cli(b, a, n - h, seed + 50000, threads)   # a = Player 2
    return dict(a='anchor:' + a, b='anchor:' + b, games=n, a_wins=w1 + w2, b_wins=l1 + l2, draws=d1 + d2, errors=0,
                a_seat1_games=h, a_seat1_wins=w1, mode='agent', seed=seed, secs=round(time.time() - t0, 1),
                kind='anchor-anchor', via='riftbound CLI', t=time.strftime('%Y-%m-%dT%H:%M:%S'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a'); ap.add_argument('b'); ap.add_argument('n', type=int)
    ap.add_argument('--results', default=None)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--seed', type=int, default=880001)
    x = ap.parse_args()
    rec = play_anchors(x.a, x.b, x.n, x.seed, x.threads)
    ResultStore(x.results).append(rec)
    print(json.dumps(rec), flush=True)


if __name__ == '__main__':
    main()
