"""M1: searched network vs the same network greedy (SEARCH_PLAN.md).

  python3 eval_search.py --sims 64 --games 400 --out out/n64.json
  python3 eval_search.py --sims 0  --games 400            # greedy vs greedy sanity row (seat_mode none)

Game slots: game i plays seed (seed + i // 2); the search plays seat 1 in even slots, seat 2 in odd
slots (seat-balanced, common seeds). Each slot plays exactly one game. The non-search seat plays the
network's argmax (project rule: evals are greedy). The searched seat plays the most visited root
option (ties: higher prior). With --sims 0 nobody searches; the row reports seat-1 wins.
"""
import argparse, json, math, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rbsearch  # noqa: E402

CKPT = os.path.join(rbsearch.SP, 'runs', 'ov_main', 'league_02080.pt')


def wilson(w, g, z=1.96):
    if g == 0:
        return [float('nan'), float('nan')]
    p = w / g
    d = 1 + z * z / g
    c = p + z * z / (2 * g)
    h = z * math.sqrt(p * (1 - p) / g + z * z / (4 * g * g))
    return [round((c - h) / d, 3), round((c + h) / d, 3)]


def summarize(res, sims):
    # sims 0: nobody searches; the "side" is designated as for a search (seat 1 in even slots, 2 in odd)
    res = [dict(r, search_seat=r['search_seat'] or (1 if r['game'] % 2 == 0 else 2)) for r in res]
    done = [r for r in res if r['done'] and not r['error']]
    errs = sum(1 for r in res if r['error'])
    wins = sum(1 for r in done if r['winner'] == r['search_seat'])
    draws = sum(1 for r in done if r['winner'] == 0)
    p1 = sum(1 for r in done if r['winner'] == 1)
    g = len(done)
    by_seat = {}
    for s in (1, 2):
        d = [r for r in done if r['search_seat'] == s]
        if d:
            w = sum(1 for r in d if r['winner'] == s)
            by_seat[s] = dict(games=len(d), wins=w, win_rate=round(w / len(d), 3))
    # paired seeds: both slots of a seed done -> the search won 0, 1 or 2 of the pair
    pairs = {}
    for r in done:
        pairs.setdefault(r['seed'], []).append(r)
    pc = [0, 0, 0]
    for v in pairs.values():
        if len(v) == 2:
            pc[sum(1 for r in v if r['winner'] == r['search_seat'])] += 1
    return dict(games=g, errors=errs, wins=wins, draws=draws, win_rate=round(wins / g, 4) if g else None,
                seat1_win_rate=round(p1 / g, 4) if g else None,
                wilson95=wilson(wins, g), by_search_seat=by_seat, pairs_won_0_1_2=pc,
                mean_turns=round(float(np.mean([r['turns'] for r in done])), 1) if done else None,
                searched_decisions_per_game=round(float(np.mean([r['n_search'] for r in done])), 1) if done else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sims', type=int, default=64)
    ap.add_argument('--games', type=int, default=400)
    ap.add_argument('--leaves', type=int, default=4)
    ap.add_argument('--cpuct', type=float, default=2.0)
    ap.add_argument('--fpu', type=float, default=0.0, help='FPU reduction below the node value')
    ap.add_argument('--vl', type=float, default=1.0)
    ap.add_argument('--threads', type=int, default=14)
    ap.add_argument('--perfect', action='store_true', help='diagnostics only: worlds = the true state')
    ap.add_argument('--loose', action='store_true', help='encoder without strict visibility')
    ap.add_argument('--deck-model', default='pool', choices=['pool', 'decklist'],
                    help='opponent unseen cards: legend pool (honest default) or public decklist')
    ap.add_argument('--opponent', default='', help='C++ agent on the non-search seat, e.g. ismcts:sims=200')
    ap.add_argument('--seed', type=int, default=990001)
    ap.add_argument('--deck', default='rengar')
    ap.add_argument('--ckpt', default=CKPT)
    ap.add_argument('--chunk', type=int, default=2048)
    ap.add_argument('--every', type=float, default=60.0, help='progress print interval (s)')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    if a.games % 2:
        raise SystemExit('--games must be even (seat-balanced pairs)')
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.4)
    torch.set_num_threads(2)
    S = rbsearch.Search(a.games, a.deck, a.seed, sims=a.sims, leaves=a.leaves, threads=a.threads, cpuct=a.cpuct,
                        fpu_red=a.fpu, vl=a.vl, perfect=a.perfect, seat_mode='balanced' if a.sims > 0 else 'none',
                        strict=not a.loose, deck_model=a.deck_model,
                        opponent=a.opponent or None)
    net = rbsearch.load_net(a.ckpt, S.dims, dev)
    ev = rbsearch.Evaluator(net, dev, S.dims['A'], a.chunk)
    t0 = time.time()
    last = t0
    t_c = t_r = t_n = t_f = 0.0
    rounds = rows_total = 0
    while True:
        s = time.time(); B = S.collect(); t_c += time.time() - s
        if B == 0:
            break
        s = time.time(); rows = S.rows(B); t_r += time.time() - s
        s = time.time(); logp, v = ev(rows, B); t_n += time.time() - s
        s = time.time(); S.feed(logp, v); t_f += time.time() - s
        rounds += 1
        rows_total += B
        if time.time() - last > a.every:
            last = time.time()
            sm = summarize(S.results(), a.sims)
            print(f'[{last - t0:6.0f}s] done {S.n_done()}/{S.n}  wins {sm["wins"]}/{sm["games"]}  '
                  f'rate {sm["win_rate"]}  CI {sm["wilson95"]}  batch {B}', flush=True)
    secs = time.time() - t0
    res = S.results()
    st = S.stats()
    out = dict(ckpt=a.ckpt, sims=a.sims, leaves=a.leaves, cpuct=a.cpuct, fpu=a.fpu, vl=a.vl, perfect=a.perfect, deck_model=a.deck_model, opponent=a.opponent or 'greedy net',
               strict_vis=not a.loose, seed=a.seed, deck=a.deck, **summarize(res, a.sims),
               secs=round(secs, 1), games_per_hour=round(len(res) / secs * 3600, 1), rounds=rounds,
               mean_batch=round(rows_total / max(1, rounds), 1),
               time_split=dict(collect=round(t_c, 1), rows=round(t_r, 1), net=round(t_n, 1), feed=round(t_f, 1)),
               stats=st)
    errs = [r['error'] for r in res if r['error']]
    if errs:
        out['first_errors'] = errs[:3]
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(dict(out, games_detail=res), open(a.out, 'w'), indent=1)
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
