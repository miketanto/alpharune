"""Gate 4 steps B2-B4: for each position, the original and its edits are played out N times each by the
playout net on both seats (GREEDY; the same seeds for every variant of a position = common random numbers:
same deck reshuffle and engine RNG), and the model responses (value at the root, policy over the root options)
of every net in --nets are recorded.

usage: python3 run.py positions.json --out results.jsonl --playout-ckpt B2.pt --playout-repr B2
                      --nets B2=ckpt,id=ckpt [--n 256] [--cap 24] [--shard 0/1]
Writes one JSON line per position (appends; positions already in --out are skipped).
"""
import argparse, json, os, sys, time
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import po as P  # noqa: E402


def select(positions, cap, rng):
    """Greedy cover: visit positions in random order, keep the edits whose cell has < cap positions so far."""
    order = rng.permutation(len(positions))
    cnt = Counter()
    sel = []
    for i in order:
        p = positions[i]
        keep = [e for e in p['edits'] if cnt[e['cell']] < cap]
        if not keep:
            continue
        for e in keep:
            cnt[e['cell']] += 1
        sel.append(dict(p, edits=keep))
    return sel, cnt


def tv(p, q, lp, lq):
    a = dict(zip(lp, p)); b = dict(zip(lq, q))
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('positions'); ap.add_argument('--out', required=True)
    ap.add_argument('--playout-ckpt', required=True); ap.add_argument('--playout-repr', default='B2')
    ap.add_argument('--nets', required=True, help='name=ckpt[:repr],...')
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--cap', type=int, default=24)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--seed', type=int, default=4242)
    ap.add_argument('--n-envs', type=int, default=512)
    ap.add_argument('--sample', action='store_true', help='playouts sample from the policy (default greedy)')
    ap.add_argument('--max-minutes', type=float, default=0)
    ap.add_argument('--screen', type=int, default=64)
    ap.add_argument('--screen-lo', type=float, default=0.08)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(int(os.environ.get('G4_THREADS', '2')))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cuda':   # coordinator 2026-11-19: cap each process's CUDA memory (WDDM oversubscription)
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get('G4_CUDA_FRAC', '0.2')))
    pos = json.load(open(a.positions))
    k, m = (int(x) for x in a.shard.split('/'))
    order = np.random.default_rng(a.seed).permutation(len(pos))
    sel = [pos[i] for i in order][k::m]
    cap = max(1, -(-a.cap // m))   # per shard
    print('shard', k, 'of', m, ':', len(sel), 'candidate positions; cap per cell per shard', cap, flush=True)
    done = set()
    cnt = Counter()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line); done.add((r['seed'], len(r['hist'])))
            if not r.get('skipped'):
                for v in r['variants'][1:]:
                    if v['ok']:
                        cnt[v['cell']] += 1
    pnet = P.load_net(a.playout_ckpt, a.playout_repr, dev)
    nets = {}
    for spec in a.nets.split(','):
        name, ck = spec.split('=')
        rep = name.split('_')[0] if ':' not in ck else ck.split(':')[1]
        ck = ck.split(':')[0]
        nets[name] = P.load_net(ck, rep, dev)
    t0 = time.time()
    seeds = [a.seed * 1000 + j for j in range(a.n)]
    for pi, p in enumerate(sel):
        if (p['seed'], len(p['hist'])) in done:
            continue
        if a.max_minutes and time.time() - t0 > 60 * a.max_minutes:
            print('time limit'); break
        keep = [e for e in p['edits'] if cnt[e['cell']] < cap]
        if not keep:
            continue
        t1 = time.time()
        po = P.Po(p['seed'], p['hist'], n_envs=a.n_envs)
        d0 = po.describe()
        dec = d0['decider']
        # screen: positions already decided under the playout policy carry no signal
        v0, _ = po.add_variant('none')
        sres = P.play(po, pnet, [(v0, s) for s in seeds[:a.screen]], dev, greedy=not a.sample)
        sw = np.mean([1.0 if sres[(v0, s)] == dec else 0.0 for s in seeds[:a.screen]])
        if not (a.screen_lo <= sw <= 1 - a.screen_lo):
            with open(a.out, 'a') as f:
                f.write(json.dumps(dict(seed=p['seed'], hist=p['hist'], turn=p['turn'], skipped=True, screen_win=sw)) + '\n')
            po.close()
            print('[%5.0fs] skip seed %d turn %d: screen win %.2f' % (time.time() - t0, p['seed'], p['turn'], sw), flush=True)
            continue
        po.close()
        po = P.Po(p['seed'], p['hist'], n_envs=a.n_envs)
        variants = [('orig', 'none', 1, '')] + [(e['cell'], e['edit'], e['orient'], e['note']) for e in keep]
        vidx, recs = [], []
        for cell, ed, orient, note in variants:
            v, err = po.add_variant(ed)
            recs.append(dict(cell=cell, edit=ed, orient=orient, note=note, ok=v is not None, err=err))
            vidx.append(v)
        # model responses at the root
        for r, v in zip(recs, vidx):
            if v is None:
                continue
            ro = po.root_obs(v)
            info = po.info(v)
            r['legal'] = info['legal'][:ro['nlegal']]
            # the edited object as the engine left it after resume + Cleanup (legality / edit-stuck check)
            parts = r['edit'].split()
            if len(parts) > 1 and parts[0] not in ('played', 'ss'):
                r['post'] = next((o for o in info['objs'] if o['id'] == int(parts[1])), None)
            elif parts[0] == 'played':
                r['post'] = info['players'][dec - 1]
            elif parts[0] == 'ss':
                r['post'] = info['bfs'][int(parts[1])]
            r['model'] = {}
            for name, net in nets.items():
                pr, val = P.net_eval(net, ro, np.array([0]), dev)
                r['model'][name] = dict(v=float(val[0]), p=pr[0, :len(r['legal'])].tolist())
        # playouts
        jobs = [(v, s) for v in vidx if v is not None for s in seeds]
        res = P.play(po, pnet, jobs, dev, greedy=not a.sample)
        for r, v in zip(recs, vidx):
            if v is None:
                continue
            w = [res[(v, s)] for s in seeds]
            r['wins'] = [1 if x == dec else (0 if x in (1, 2, 0) else -1) for x in w]
        po.close()
        for r in recs[1:]:
            if r['ok']:
                cnt[r['cell']] += 1
        out = dict(screen_win=sw, seed=p['seed'], hist=p['hist'], turn=p['turn'], tp=p['tp'], decider=dec, scores=p['scores'],
                   n=a.n, playout=a.playout_repr, mode='sample' if a.sample else 'greedy', variants=recs,
                   secs=round(time.time() - t1, 1))
        with open(a.out, 'a') as f:
            f.write(json.dumps(out) + '\n')
        o = recs[0]
        print('[%5.0fs] pos %d/%d seed %d turn %d: %d variants (%d failed), orig win %.3f, %.1fs' % (
            time.time() - t0, pi + 1, len(sel), p['seed'], p['turn'], len(recs), sum(not r['ok'] for r in recs),
            np.mean([x for x in o['wins'] if x >= 0]), time.time() - t1), flush=True)


if __name__ == '__main__':
    main()
