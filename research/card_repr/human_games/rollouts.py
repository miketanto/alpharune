"""Counterfactuals in a rebuilt human game.
 1. At chosen network decisions: every distinct option (by words; the likeliest first, at most MAXO) played out N times by the
    same network on both seats, sampled; win rate of the network's seat. TRUE world: real hands, real deck orders.
    Then the chosen option and the best alternative again with fresh seeds (selection is not scored on its own sample).
 2. Turn 10 with other facedown cards: what G2's policy and value say before its attack.
python3 rollouts.py GAME.json idx[,idx...]
"""
import json, os, sys
from multiprocessing import Pool
PW = '/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/.claude/worktrees/play-web-ui-e914cf/tools/play_web'
sys.path.insert(0, PW)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PLAYWEB_TORCH_THREADS'] = '1'
HERE = os.path.dirname(os.path.abspath(__file__))
DECKS = os.path.join(PW, '..', '..', 'decks')
G = json.load(open(os.path.join(HERE, sys.argv[1])))
MAXO, N1, N2 = 6, 100, 400
_A = None


def net():
    global _A
    import playweb
    if _A is None:
        _A = playweb.NetAgent(os.path.join(playweb.SELFPLAY, 'runs', G['checkpoint']))
    return _A


def ev(o):
    import numpy as np, torch
    a = net()
    with torch.no_grad():
        lp, v = a.net(a._common.to_dev(o, np.array([0]), 'cpu'))
    return lp[0].exp().numpy(), float(v.reshape(-1)[0])


def replay(choices):
    import playweb
    g = playweb.Game(os.path.join(DECKS, G['p1']), os.path.join(DECKS, G['p2']), G['human'], G['seed'])
    for _, c in choices:
        g.step(c)
    return g


def options(g):
    st = g.status()
    return {o['index']: o['text'] for o in g.view(st['seat'], 1 << 40)['decision']['options']}


def roll(task):
    import numpy as np
    i, k, seeds = task
    w = 0
    for sd in seeds:
        rng = np.random.default_rng(sd)
        g = replay(G['choices'][:i])
        g.step(k)
        while True:
            st = g.status()
            if st['done'] or not st['need']:
                break
            p, _ = ev(g.observe())
            p = p[:st['nlegal']].astype(np.float64); p /= p.sum()
            g.step(int(rng.choice(len(p), p=p)))
        w += g.status()['winner'] == 3 - G['human']
        g.close()
    return i, k, w, len(seeds)


def wil(w, n):
    import math
    z, p = 1.96, w / n
    d = 1 + z * z / n
    c, h = (p + z * z / 2 / n) / d, z * math.sqrt(p * (1 - p) / n + z * z / 4 / n / n) / d
    return f'{p:.3f} [{c - h:.3f},{c + h:.3f}]'


def run(P, items, n, base):
    tasks = [(i, k, list(range(base + s, base + s + 10))) for i, k in items for s in range(0, n, 10)]
    res = {}
    for i, k, w, m in P.imap_unordered(roll, tasks):
        a = res.setdefault((i, k), [0, 0]); a[0] += w; a[1] += m
    return res


def main():
    import numpy as np
    idxs = [int(x) for x in sys.argv[2].split(',')]
    plan, info = [], {}
    for i in idxs:
        g = replay(G['choices'][:i])
        st = g.status()
        assert st['seat'] == 3 - G['human'] == G['choices'][i][0], (i, st)
        op = options(g)
        p, v = ev(g.observe())
        ch = G['choices'][i][1]
        ks, seen = [ch], {op[ch]}
        for k in np.argsort(-p[:st['nlegal']]):
            if op[int(k)] not in seen and len(ks) < MAXO:
                ks.append(int(k)); seen.add(op[int(k)])
        info[i] = (op, p, v, ch, g.view(1, 1 << 40)['turn']['number'])
        plan += [(i, k) for k in ks]
        g.close()
    with Pool(12) as P:
        R = run(P, plan, N1, 10_000_000)
        conf = []
        for i in idxs:
            ch = info[i][3]
            alts = [(R[x][0] / R[x][1], x[1]) for x in R if x[0] == i and x[1] != ch]
            if alts:
                conf += [(i, ch), (i, max(alts)[1])]
        F = run(P, conf, N2, 50_000_000)
    for i in idxs:
        op, p, v, ch, t = info[i]
        print(f'\n#{i} turn {t}  G2 value (its win chance) {(v + 1) / 2:.2f}')
        for k in sorted([x[1] for x in R if x[0] == i], key=lambda k: -R[(i, k)][0]):
            c = F.get((i, k))
            print(f"  {'CHOSEN ' if k == ch else '       '}{100 * p[k]:5.1f}%  N1 {wil(*R[(i, k)])}" + (f'  FRESH {wil(*c)}' if c else '') + f'   {op[k][:95]}')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
