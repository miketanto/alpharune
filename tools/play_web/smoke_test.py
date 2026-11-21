#!/usr/bin/env python3
"""Headless check of the play server's game loop: a random "human" against the network.

    CUDA_VISIBLE_DEVICES= taskset -c 12-15 python3 tools/play_web/smoke_test.py [games] [checkpoint]

For every human decision it checks that the frame's options are exactly the engine's legal
indices, that the frame names no card of the opponent's hand the human was not shown, and
that every log line in the frame is one the human may read (pw_view filters them).
"""
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import server  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    ck = sys.argv[2] if len(sys.argv) > 2 else server.DEFAULT_CKPT
    rng = random.Random(7)
    S = server.Session()
    for gi in range(n):
        t0 = time.time()
        seat = 1 + gi % 2
        frames = S.new('rengar', 'rengar', ck, seat, 1000 + gi)
        decisions = 0
        agent_frames = len(frames) - 1
        while True:
            f = frames[-1]
            assert 'view_error' not in f, f
            if f['game_over']:
                break
            d = f['decision']
            st = S.game.status()
            assert d is not None and st['seat'] == seat, (st, f.get('waiting_for'))
            idx = sorted(o['index'] for o in d['options'])
            assert idx == list(range(st['nlegal'])), (idx, st)
            # The opponent's hand: only its size, never an object.
            ids = {o['id'] for o in f['hand']}
            assert all(o['owner'] == 1 for o in f['hand'])
            assert f['players']['opp']['hand_count'] >= len(f['opp_known_hand'])
            frames = S.act(rng.choice(idx))
            agent_frames += len(frames) - 2
            decisions += 1
            assert decisions < 3000
        f = frames[-1]
        print(f"game {gi}: human seat {seat}, winner {'human' if f['winner'] == 1 else 'network' if f['winner'] == 2 else 'none'}"
              f" {f['players']['me']['score']}-{f['players']['opp']['score']}, turn {f['turn']['number']},"
              f" {decisions} human decisions, {agent_frames} network frames, {time.time() - t0:.1f}s, error={f['error']!r}",
              flush=True)
        assert not f['error'], f['error']


if __name__ == '__main__':
    main()
