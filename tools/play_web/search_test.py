#!/usr/bin/env python3
"""Headless check of the AZ / search opponent and of saved games (a random human).

    CUDA_VISIBLE_DEVICES= taskset -c 12-15 python3 tools/play_web/search_test.py [sims]

Against search from both seats and against the network alone: full games; at every human decision of a search game the
displayed game and the search harness must encode the same observation (history included); every saved game must be
listed and replay to its end. Writes to a temporary directory.
"""
import os
import random
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import playweb  # noqa: E402
import replay  # noqa: E402
import server  # noqa: E402

tmp = tempfile.mkdtemp()
server.GAMES_DIR = tmp
sims = int(sys.argv[1]) if len(sys.argv) > 1 else 64
rng = random.Random(5)
bad = 0
for seat, deck, adeck, n in ((1, 'azir', 'azir', sims), (2, 'rengar', 'azir', sims), (1, 'azir', 'azir', 0)):
    S = server.Session()
    t0 = time.time()
    S.new(deck, adeck, server.DEFAULT_CKPT, seat, 424200 + seat, n)
    k, slow, same = 0, 0.0, 0
    while True:
        st = S.game.status()
        if st['done']:
            break
        if S.search:
            o, b = S.game.observe(), S.search.S._buf
            for key in ('glob', 'tok', 'tok_mask', 'act', 'act_mask', 'hist'):
                if not np.array_equal(o[key][0].astype(np.int32), b[key][0].astype(np.int32)):
                    bad += 1
                    print('DIFFERENT', key, 'at decision', len(S.rows))
            same += 1
        t = time.time()
        S.act(rng.randrange(min(st['nlegal'], 96)))
        slow = max(slow, time.time() - t)
        k += 1
    st = S.game.status()
    err = S.game.view(seat, 1 << 40)['error']
    print(f'you P{seat} {deck} vs {adeck} sims {n}: winner P{st["winner"]}, {k} human decisions, {len(S.rows)} in all, '
          f'{time.time() - t0:.0f}s, slowest answer {slow:.1f}s, encodings compared {same}, error {err!r}')
    if err:
        bad += 1
    if S.search:
        S.search.close()
C = replay.Catalog(tmp, server.DECK_DIR)
for e in C.entries():
    r = replay.Replay(C, e['id'], 1, True)
    f = r.goto(e['decisions'])
    ok = f['game_over'] and f['replay'].get('winner') == e['winner']
    bad += not ok
    print('replay', e['name'], e['decks'], 'winner', e['winner'], 'ok' if ok else 'BAD')
print('PASS' if not bad else f'{bad} PROBLEMS')
sys.exit(bool(bad))
