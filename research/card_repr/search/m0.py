"""M0: leak test and ms-per-simulation benchmark (SEARCH_PLAN.md).

  python3 m0.py leak  --games 40 --worlds 8          # real games of the network (greedy, both seats)
  python3 m0.py bench --games 16 --sims 64 --threads 1 --decisions 300   # dummy evaluator

leak: at every external decision (>= 2 options) of real games of the network, `worlds` sampled
worlds are built for the decider; encode(world, decider) must be byte-identical to
encode(truth, decider). Also reports how often a world redrew the opponent's hidden cards (so the
test is not vacuous) and WorldSampler::check problems.
bench: a searched game slot set with the dummy evaluator (uniform prior, value 0); C++ timers per
phase, wall time per simulation.
"""
import argparse, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rbsearch  # noqa: E402

CKPT = os.path.join(rbsearch.SP, 'runs', 'ov_main', 'league_02080.pt')


def leak(a):
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.4)
    S = rbsearch.Search(a.games, a.deck, a.seed, sims=0, leaves=1, threads=a.threads, seat_mode='none', deck_b=a.deck_b,
                        strict=not a.loose, deck_model=a.deck_model)
    net = rbsearch.load_net(a.ckpt, S.dims, dev)
    ev = rbsearch.Evaluator(net, dev, S.dims['A'])
    acc, first = None, ''
    t0 = time.time()
    steps = 0
    while not S.finished():
        B = S.collect()
        if B == 0:
            break
        acc, msg = S.leaktest(a.worlds, acc)
        first = first or msg
        rows = S.rows(B)
        logp, v = ev(rows, B)
        S.feed(logp, v)
        steps += 1
        if steps % 50 == 0:
            print(f'[{time.time() - t0:5.0f}s] step {steps} done {S.n_done()}/{S.n} {acc}', flush=True)
    res = S.results()
    out = dict(games=len(res), errors=sum(1 for r in res if r['error']), leak=acc, first_mismatch=first,
               secs=round(time.time() - t0, 1), worlds_per_decision=a.worlds, strict_vis=not a.loose,
               deck_model=a.deck_model)
    print(json.dumps(out), flush=True)
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)


def bench(a):
    S = rbsearch.Search(a.games, a.deck, a.seed, sims=a.sims, leaves=a.leaves, threads=a.threads,
                        perfect=a.perfect)
    A = S.dims['A']
    t0 = time.time()
    tc = 0.0
    while not S.finished():
        s0 = time.time()
        B = S.collect()
        tc += time.time() - s0
        if B == 0:
            break
        rows = S.rows(B)
        logp, v = rbsearch.dummy_eval(rows, B, A)
        S.feed(logp, v)
        st = S.stats()
        if st['searches'] >= a.decisions:
            break
    wall = time.time() - t0
    st = S.stats()
    sims = max(1, st['sims'] + st['fails'])
    ms = lambda k: round(st[k] / 1e6 / sims, 4)
    out = dict(threads=a.threads, sims_per_decision=a.sims, leaves=a.leaves, perfect=a.perfect,
               searched_decisions=st['searches'], sims=st['sims'], fails=st['fails'], terminals=st['terminals'],
               reexpand=st['reexpand'], mean_depth=round(st['depth_sum'] / sims, 2),
               replay_steps_per_sim=round(st['replay_steps'] / sims, 2),
               ms_per_sim_build=ms('ns_build'), ms_per_sim_edit=ms('ns_edit'), ms_per_sim_descend=ms('ns_descend'),
               ms_per_sim_encode=ms('ns_encode'), ms_per_sim_teardown=ms('ns_teardown'),
               ms_per_sim_cpp_total=round((st['ns_build'] + st['ns_edit'] + st['ns_descend'] + st['ns_encode']
                                           + st['ns_teardown']) / 1e6 / sims, 4),
               wall_ms_per_sim=round(1000 * wall / sims, 4), collect_ms_per_sim=round(1000 * tc / sims, 4),
               sims_per_sec_wall=round(sims / wall, 1), snap_ms_total=round(st['ns_snap'] / 1e6, 1))
    print(json.dumps(out), flush=True)
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['leak', 'bench'])
    ap.add_argument('--games', type=int, default=40)
    ap.add_argument('--worlds', type=int, default=8)
    ap.add_argument('--sims', type=int, default=64)
    ap.add_argument('--leaves', type=int, default=4)
    ap.add_argument('--threads', type=int, default=14)
    ap.add_argument('--decisions', type=int, default=300)
    ap.add_argument('--perfect', action='store_true')
    ap.add_argument('--seed', type=int, default=770001)
    ap.add_argument('--deck', default='rengar')
    ap.add_argument('--deck-b', default=None)
    ap.add_argument('--ckpt', default=CKPT)
    ap.add_argument('--loose', action='store_true', help='encoder without strict visibility (the training default)')
    ap.add_argument('--deck-model', default='pool', choices=['pool', 'decklist'])
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    leak(a) if a.mode == 'leak' else bench(a)


if __name__ == '__main__':
    main()
