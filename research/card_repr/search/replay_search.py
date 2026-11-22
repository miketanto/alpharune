"""Annotated replay of one searched game (search seat vs the same network greedy).

Step 1 (`play`): play deal SEED twice with the search (search on P1 in slot 0, on P2 in slot 1; as
eval_search.py), save each game's per-decision log (rbs_declog) to OUT/declog_seedSEED.json.
Step 2 (`viz`): replay the chosen slot's choices in the viz env (librbenv_viz) -- same engine, same
seed, same choice indices -- and write the HTML replay + annotations. Run in separate processes
(two different rbenv libraries).

  python3 replay_search.py play --seed 7001 --sims 200 --out out/replay
  python3 replay_search.py viz  --seed 7001 --search-seat 1 --out out/replay
  python3 ../selfplay/viz/transcript.py out/replay/game_seed7001.html
"""
import argparse, ctypes, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, '..', 'selfplay', 'runs', 'ov_main', 'league_02080.pt')


def play(a):
    import torch
    sys.path.insert(0, HERE)
    import rbsearch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.4)
    S = rbsearch.Search(2, a.deck, a.seed, sims=a.sims, leaves=4, threads=a.threads, seat_mode='balanced', strict=True,
                        deck_b=a.deck_b or a.deck)
    L = S.L
    L.rbs_declog.argtypes = [ctypes.c_void_p, ctypes.c_int, np.ctypeslib.ndpointer(np.float32, flags='C'), ctypes.c_int]
    L.rbs_declog.restype = ctypes.c_int
    ev = rbsearch.Evaluator(rbsearch.load_net(a.ckpt, S.dims, dev), dev, S.dims['A'])
    while True:
        B = S.collect()
        if B == 0:
            break
        S.feed(*ev(S.rows(B), B))
    res = S.results()
    out = []
    for i, r in enumerate(res):
        buf = np.zeros((4000, 18), np.float32)
        n = L.rbs_declog(S.h, i, buf, 4000)
        out.append(dict(r, declog=buf[:n].tolist()))
        print(f'slot {i}: search seat {r["search_seat"]}, winner P{r["winner"]}, turns {r["turns"]}, '
              f'{n} decisions, searched {r["n_search"]}', flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(seed=a.seed, sims=a.sims, deck=a.deck, deck_b=a.deck_b or a.deck, name=a.name or os.path.basename(a.ckpt), ckpt=a.ckpt, games=out), open(os.path.join(a.out, f'declog_seed{a.seed}.json'), 'w'))


def viz(a):
    sys.path.insert(0, os.path.join(HERE, '..', 'selfplay', 'viz'))
    from rbenv_viz import RBEnv
    D = json.load(open(os.path.join(a.out, f'declog_seed{a.seed}.json')))
    g = next(x for x in D['games'] if x['search_seat'] == a.search_seat)
    log = g['declog']
    N = D['sims']
    nm = a.name or os.path.basename(a.ckpt)
    lab = {a.search_seat: f'search N={N} ({nm})', 3 - a.search_seat: f'network greedy ({nm})'}
    env = RBEnv(1, a.deck, a.deck_b or a.deck, a.seed, autopay=True, opponent=None, viz_dir=os.path.abspath(a.out),
                opp_seat=0, labels=(lab[1], lab[2]))
    k = 0
    overrides = []
    while True:
        obs = env.observe()
        if obs['need'][0]:
            d = log[k]
            k += 1
            searched, ch, am = int(d[0]), int(d[1]), int(d[2])
            seat = int(obs['seat'][0])
            assert seat == int(d[11]), (k, seat, d[11])
            txt = lambda i: env.option_text(0, int(i))
            if searched:
                tot = max(1.0, d[7])
                lines = [f'[NET] value {d[10]:+.2f} (win ~{(d[10] + 1) * 50:.0f}%), SEARCH {int(tot)} visits; top by visits:']
                for t in range(3):
                    i, n = int(d[12 + 2 * t]), d[13 + 2 * t]
                    if i < 0:
                        continue
                    lines.append(f'[NET]   {"* " if i == ch else "  "}{100 * n / tot:5.1f}% : {txt(i)}')
                if ch != am:
                    lines.append(f'[NET]   (search override: network preferred "{txt(am)}" at {100 * d[4]:.1f}%, '
                                 f'search Q {d[9]:+.2f} vs chosen Q {d[8]:+.2f} with network prior {100 * d[3]:.1f}%)')
                    overrides.append(dict(decision=k, chosen=txt(ch), chosen_prior=round(d[3], 3), chosen_visits=int(d[5]),
                                          chosen_q=round(d[8], 2), net_pick=txt(am), net_prior=round(d[4], 3),
                                          net_visits=int(d[6]), net_q=round(d[9], 2), total=int(tot), value=round(d[10], 2)))
            else:
                lines = [f'[NET] value {d[10]:+.2f} (win ~{(d[10] + 1) * 50:.0f}%), greedy; top by probability:',
                         f'[NET]   * {100 * d[3]:5.1f}% : {txt(ch)}']
            env.note(0, '\n'.join(lines), float(d[10]))
            env.step(np.array([ch], np.int64))
        else:
            env.step(np.zeros(1, np.int64))
        res = env.results()
        if res:
            r = res[0]
            ok = r['winner'] == g['winner'] and k == len(log)
            print(dict(r, replay_matches_search=ok, decisions_replayed=k, decisions_logged=len(log)), flush=True)
            break
    env.close()
    json.dump(overrides, open(os.path.join(a.out, f'overrides_seed{a.seed}_seat{a.search_seat}.json'), 'w'), indent=1)
    for o in overrides:
        print(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['play', 'viz'])
    ap.add_argument('--seed', type=int, default=7001)
    ap.add_argument('--sims', type=int, default=200)
    ap.add_argument('--search-seat', type=int, default=1)
    ap.add_argument('--ckpt', default=CKPT)
    ap.add_argument('--out', default=os.path.join(HERE, 'out', 'replay'))
    ap.add_argument('--deck', default='rengar', help='seat-1 deck')
    ap.add_argument('--deck-b', default=None, help='seat-2 deck (default: --deck)')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--name', default='', help='network label in the replay')
    a = ap.parse_args()
    play(a) if a.mode == 'play' else viz(a)


if __name__ == '__main__':
    main()
