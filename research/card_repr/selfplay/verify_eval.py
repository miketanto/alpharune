"""Unbiased evaluation: exactly one game per env slot (the first game each slot plays),
so short games are not over-counted. Network = seat 1 (deck A), opponent in C++ = seat 2.

Games are split over --procs worker processes (one RBEnv each, distinct seeds). The
opponent's search is the bottleneck and runs one decision at a time inside each env
step, so more processes means more searches in parallel. Workers run the (small)
network on CPU with one thread, so they don't each hold a CUDA context.

Progress (games, wins, running win rate) is printed every --every games, and the
final JSON is printed as the last line.

usage: python3 verify_eval.py <ckpt|untrained> <opponent spec> <n_games>
                              [--repr id] [--procs 6] [--every 20] [--seed 990001]
                              [--deck rengar] [--variants a,b,c] [--out res.json]
"""
import os, sys, time, json, math, argparse
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def wilson(w, g, z=1.96):
    if g == 0:
        return [float('nan'), float('nan')]
    p = w / g
    d = 1 + z * z / g
    c = p + z * z / (2 * g)
    h = z * math.sqrt(p * (1 - p) / g + z * z / (4 * g * g))
    return [round((c - h) / d, 3), round((c + h) / d, 3)]


def worker(wid, ck, opp, n, rep, seed, q, deck='rengar', variants=None):
    import torch
    sys.path.insert(0, os.path.join(HERE, 'learner'))
    sys.path.insert(0, os.path.join(HERE, 'rbenv'))
    from model import Net, load_vocab
    from selfplay import act_rows
    from rbenv import RBEnv
    torch.set_num_threads(1)
    dev = 'cpu'
    env = RBEnv(n, deck, deck, seed, variant=variants, autopay=True, opponent=opp)
    sd = None
    if ck != 'untrained':
        sd = torch.load(ck, map_location=dev)
        sd = sd.get('net', sd.get('model', sd)) if isinstance(sd, dict) else sd
    ask = sd is not None and 'W_ask.weight' in sd   # --ask-head checkpoints (model.py)
    cnorm = sd is None or 'card_ln.weight' in sd   # pre-gate-2 checkpoints have no card LayerNorm
    net = Net(env.dims, load_vocab()[0], rep, ask_head=ask, card_norm=cnorm).to(dev)
    if sd is not None:
        net.load_state_dict(sd)
    net.eval()
    seen = set()
    while len(seen) < n:
        obs = env.observe()
        rows = np.nonzero(obs['need'])[0]
        ch = np.zeros(len(obs['need']), np.int64)
        if len(rows):
            a, _, _ = act_rows(net, obs, rows, dev)
            ch[rows] = a
        env.step(ch)
        for r in env.results():
            if r['env'] not in seen:
                seen.add(r['env'])
                q.put((wid, int(r['winner']), bool(r.get('error')), int(r['turns'])))
    q.put((wid, None, None, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt'); ap.add_argument('opp'); ap.add_argument('n', type=int)
    ap.add_argument('--repr', default='id')
    ap.add_argument('--procs', type=int, default=6)
    ap.add_argument('--every', type=int, default=20)
    ap.add_argument('--seed', type=int, default=990001)
    ap.add_argument('--deck', default='rengar', help='deck of BOTH seats (mirror)')
    ap.add_argument('--variants', default='', help='comma-separated variants to register')
    ap.add_argument('--out', default='', help='also write the final JSON here')
    a = ap.parse_args()
    P = max(1, min(a.procs, a.n))
    sizes = [a.n // P + (1 if i < a.n % P else 0) for i in range(P)]
    q = mp.get_context('fork').Queue()
    ps = [mp.get_context('fork').Process(target=worker,
                                         args=(i, a.ckpt, a.opp, sizes[i], a.repr, a.seed + 100000 * i, q, a.deck,
                                               [v for v in a.variants.split(',') if v] or None))
          for i in range(P)]
    t0 = time.time()
    for p in ps:
        p.start()
    done = wins = games = errs = 0
    turns = []
    while done < P:
        wid, winner, err, t = q.get()
        if winner is None:
            done += 1
            continue
        if err:
            errs += 1
            continue
        games += 1
        wins += winner == 1
        turns.append(t)
        if games % a.every == 0:
            print(f'[{time.time() - t0:6.0f}s] {games}/{a.n} games  wins {wins}  '
                  f'win rate {wins / games:.3f}  CI {wilson(wins, games)}', flush=True)
    for p in ps:
        p.join()
    res = {'ckpt': a.ckpt, 'repr': a.repr, 'deck': a.deck, 'opp': a.opp, 'games': games, 'errors': errs, 'wins': wins,
                      'win_rate': round(wins / games, 3) if games else None, 'wilson95': wilson(wins, games),
                      'mean_turns': round(float(np.mean(turns)), 1) if turns else None,
                      'procs': P, 'secs': round(time.time() - t0, 1)}
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
