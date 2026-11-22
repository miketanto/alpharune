"""G2 evaluation: six-legend generalist (g2_main@1000) against G1@500 and the specialists, greedy, CPU.
  1. forgetting on the G1 decks: G2 vs G1 / S_R / S_A on Rengar and Azir (mirrors and the cross pairs);
  2. learning on the new decks: G2 vs G1 piloting the same deck (mirror), 2000 games, seat-balanced, same deals;
  3. strength on the new decks: G2 and G1 (network seat 1) vs ISMCTS-100 (seat 2), mirror, N games each.
Results: g2/eval/results.jsonl (one line per pair) and g2/eval/REPORT.txt.
"""
import json, multiprocessing as mp, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'g1'))
import eval_g1 as E  # noqa: E402  (play_pair, wilson, import paths)
from common import act_rows, import_rbenv, load_net  # noqa: E402

RUNS = '/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/selfplay/runs'
G2, G1 = f'{RUNS}/g2_main/league_01000.pt', f'{RUNS}/g1_main/league_00500.pt'
SR, SA = f'{RUNS}/e2a_main/league_00480.pt', f'{RUNS}/az_main/league_00300.pt'
NEW = ['diana', 'kaisa', 'yi_rebuild', 'kennen']
OUT = os.path.join(HERE, 'eval')
GAMES = int(os.environ.get('GAMES', 2000))
PROCS, ISM_GAMES, ISM = 14, int(os.environ.get('ISM_GAMES', 200)), os.environ.get('ISM', 'ismcts:sims=100')
DP = E.deck_path if hasattr(E, 'deck_path') else (lambda n: os.path.join(HERE, '..', '..', '..', 'decks', n + '.txt'))


def ism_worker(ck, deck, spec, n, seed, q):
    import torch
    torch.set_num_threads(1)
    RBEnv = import_rbenv()
    env = RBEnv(n, DP(deck), DP(deck), seed, autopay=True, opponent=spec)
    net = load_net(ck, env.dims)
    seen = set()
    while len(seen) < n:
        obs = env.observe()
        rows = np.nonzero(obs['need'])[0]
        ch = np.zeros(len(obs['need']), np.int64)
        if len(rows):
            ch[rows] = act_rows(net, obs, rows, 'cpu', True)
        env.step(ch)
        for r in env.results():
            if r['env'] not in seen:
                seen.add(r['env'])
                q.put((int(r['winner']), bool(r.get('error'))))
    q.put((None, None))


def play_ism(ck, deck, n, seed=990001):
    t0 = time.time()
    P = min(PROCS, n)
    sizes = [n // P + (1 if i < n % P else 0) for i in range(P)]
    ctx = mp.get_context('fork')
    q = ctx.Queue()
    ps = [ctx.Process(target=ism_worker, args=(ck, deck, ISM, sizes[i], seed + 100000 * i, q)) for i in range(P)]
    for p in ps:
        p.start()
    done = g = w = e = 0
    while done < P:
        win, err = q.get()
        if win is None:
            done += 1
        elif err:
            e += 1
        else:
            g += 1
            w += 1.0 if win == 1 else 0.5 if win == 0 else 0.0
    for p in ps:
        p.join()
    return dict(wr=round(w / g, 4) if g else None, ci=E.wilson(w, g), games=g, errors=e, secs=round(time.time() - t0, 1))


def main():
    os.makedirs(OUT, exist_ok=True)
    res_path = os.path.join(OUT, 'results.jsonl')
    done = {}
    if os.path.exists(res_path):
        for line in open(res_path):
            r = json.loads(line); done[r['name']] = r
    def rec(name, r):
        r = dict(name=name, **r); done[name] = r
        with open(res_path, 'a') as f:
            f.write(json.dumps(r) + '\n')
        print(f"{time.strftime('%H:%M:%S')} {name}: {r.get('wr')} {r.get('ci')} n={r.get('games')} err={r.get('errors')} ({r.get('secs')}s)", flush=True)
    jobs = []
    for nm, ckA, dA, ckB, dB in [
            ('G2 vs G1 | rengar mirror', G2, 'rengar', G1, 'rengar'),
            ('G2 vs G1 | azir mirror', G2, 'azir', G1, 'azir'),
            ('G2(rengar) vs G1(azir)', G2, 'rengar', G1, 'azir'),
            ('G2(azir) vs G1(rengar)', G2, 'azir', G1, 'rengar'),
            ('G2 vs S_R | rengar mirror', G2, 'rengar', SR, 'rengar'),
            ('G2 vs S_A | azir mirror', G2, 'azir', SA, 'azir')] + \
            [(f'G2 vs G1 | {d} mirror', G2, d, G1, d) for d in NEW]:
        jobs.append(('net', nm, (ckA, DP(dA), ckB, DP(dB))))
    for d in NEW:
        for tag, ck in (('G2', G2), ('G1', G1)):
            jobs.append(('ism', f'{tag} vs {ISM} | {d} mirror', (ck, d)))
    for kind, nm, args in jobs:
        if nm in done:
            continue
        if kind == 'net':
            rec(nm, E.play_pair(*args, GAMES, PROCS, 770001, None))
        else:
            rec(nm, play_ism(*args, ISM_GAMES))
    L = [f'G2 evaluation ({time.ctime()}), greedy. G2 = g2_main@1000 (Rengar, Azir, Diana, Kai\'Sa, Master Yi, Kennen), '
         f'G1 = g1_main@500, S_R = e2a_main@480, S_A = az_main@300.',
         'Net pairs: 2000 games, both seat orders, same deals; win rate of the first player. '
         f'ISMCTS: {ISM_GAMES} games, network always seat 1.', '']
    for kind, nm, _ in jobs:
        r = done.get(nm)
        if r:
            L.append(f'  {nm:42s} {r["wr"]:.3f} [{r["ci"][0]:.3f}, {r["ci"][1]:.3f}]  n={r["games"]} err={r["errors"]}')
    open(os.path.join(OUT, 'REPORT.txt'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
