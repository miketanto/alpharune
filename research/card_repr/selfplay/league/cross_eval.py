"""Cross-deck match: each agent plays its own deck (greedy, one game per env slot, both seat orders).

  python3 league/cross_eval.py runs/az_main/league_00300.pt azir runs/ov_main/league_02080.pt rengar --games 500
Reports agent A's win rate with A on seat 1 and on seat 2, and overall (Wilson 95% CI).
"""
import argparse, math, multiprocessing as mp, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import act_rows, import_rbenv, load_net  # noqa: E402


def wilson(w, n, z=1.96):
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((c - h) / d, 3), round((c + h) / d, 3)


def worker(ck1, ck2, d1, d2, n, seed, q):
    import torch
    torch.set_num_threads(1)
    RBEnv = import_rbenv()
    env = RBEnv(n, d1, d2, seed, autopay=True, opponent=None)
    nets = {1: load_net(ck1, env.dims, 'B2'), 2: load_net(ck2, env.dims, 'B2')}
    seen = set()
    while len(seen) < n:
        obs = env.observe()
        rows = np.nonzero(obs['need'])[0]
        ch = np.zeros(len(obs['need']), np.int64)
        for s in (1, 2):
            rr = rows[obs['seat'][rows] == s]
            if len(rr):
                ch[rr] = act_rows(nets[s], obs, rr, 'cpu', True)
        env.step(ch)
        for r in env.results():
            if r['env'] not in seen:
                seen.add(r['env'])
                q.put((int(r['winner']), bool(r.get('error'))))
    q.put(None)


def run(ck1, ck2, d1, d2, games, procs, seed):
    ctx = mp.get_context('fork')
    q = ctx.Queue()
    sizes = [games // procs + (1 if i < games % procs else 0) for i in range(procs)]
    ps = [ctx.Process(target=worker, args=(ck1, ck2, d1, d2, sizes[i], seed + 100000 * i, q)) for i in range(procs)]
    for p in ps:
        p.start()
    done, w1, n = 0, 0, 0
    while done < procs:
        r = q.get()
        if r is None:
            done += 1
            continue
        if r[1] or r[0] not in (1, 2):
            continue
        n += 1
        w1 += r[0] == 1
    for p in ps:
        p.join()
    return w1, n


ap = argparse.ArgumentParser()
ap.add_argument('ck_a'); ap.add_argument('deck_a'); ap.add_argument('ck_b'); ap.add_argument('deck_b')
ap.add_argument('--games', type=int, default=500)
ap.add_argument('--procs', type=int, default=8)
ap.add_argument('--seed', type=int, default=660001)
a = ap.parse_args()
w1, n1 = run(a.ck_a, a.ck_b, a.deck_a, a.deck_b, a.games, a.procs, a.seed)          # A on seat 1
w2b, n2 = run(a.ck_b, a.ck_a, a.deck_b, a.deck_a, a.games, a.procs, a.seed + 7)     # A on seat 2
w2 = n2 - w2b
print(f'A ({a.deck_a}) on seat 1: {w1}/{n1} = {w1 / n1:.3f} {wilson(w1, n1)}')
print(f'A ({a.deck_a}) on seat 2: {w2}/{n2} = {w2 / n2:.3f} {wilson(w2, n2)}')
print(f'A overall: {w1 + w2}/{n1 + n2} = {(w1 + w2) / (n1 + n2):.3f} {wilson(w1 + w2, n1 + n2)}')
