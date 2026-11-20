"""Mulligan check (2026-11-20): does the opening hand matter, and has the network learned to mulligan?

Mirror games of ONE checkpoint against itself (the network plays both seats, on CPU), one game per env slot
(unbiased, as verify_eval.py). Every other decision is GREEDY (eval rule). Arms differ ONLY in the mulligan:

  net   : the network's own mulligan choice (greedy argmax)
  keep  : both seats always keep
  sample: the mulligan is SAMPLED from the policy (as in training); everything else greedy

The same --seed gives the same deals in every arm (env slot i of worker w plays the same game seed), so
arms are paired per deal: with greedy play a game is a function of deal + choices, and a deal where the
network keeps plays out identically in `net` and `keep`.

Per mulligan decision we record: arm, deal id, seat, first player, the policy's P(keep), the chosen number
of cards set aside, the value estimate before the mulligan, the outcome.

  python3 mull_check.py CKPT --games 4000 --procs 8 --out results/   (writes <arm>.jsonl + prints a summary)
"""
import os, sys, time, json, argparse
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SP = os.path.dirname(HERE)
PK_MULLIGAN, PK_MAIN = 2, 3   # core/prompt.h PromptKind
MU_COUNT_COL = 44              # rbenv.cpp: action col 44 = number of cards set aside


def worker(wid, ck, arm, n, seed, q):
    import torch
    sys.path.insert(0, os.path.join(SP, 'learner'))
    sys.path.insert(0, os.path.join(SP, 'rbenv'))
    from model import Net, load_vocab
    from selfplay import to_dev
    from rbenv import RBEnv
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    env = RBEnv(n, 'rengar', 'rengar', seed, autopay=True, opponent=None)
    sd = torch.load(ck, map_location='cpu')
    sd = sd.get('net', sd.get('model', sd)) if isinstance(sd, dict) else sd
    net = Net(env.dims, load_vocab()[0], 'B2', ask_head='W_ask.weight' in sd, card_norm='card_ln.weight' in sd,
              ask_opt='W_ask_opt.weight' in sd, opt_cost='W_oc.weight' in sd, opt_refs='W_mg.weight' in sd)
    net.load_state_dict(sd)
    net.eval()
    seen = set()
    first = {}          # env -> seat of the first main-phase decision
    mulls = {}          # env -> list of records
    while len(seen) < n:
        obs = env.observe()
        rows = np.nonzero(obs['need'])[0]
        ch = np.zeros(len(obs['need']), np.int64)
        if len(rows):
            with torch.no_grad():
                logp, v = net(to_dev(obs, rows, 'cpu'))
            p = logp.exp().numpy()
            a = p.argmax(-1)
            for j, r in enumerate(rows):
                r = int(r)
                if r in seen:
                    continue
                pk = int(obs['pk'][r])
                seat = int(obs['seat'][r])
                if pk == PK_MAIN and r not in first:
                    first[r] = seat
                if pk == PK_MULLIGAN:
                    nl = int(obs['nlegal'][r])
                    cnt = obs['act'][r, :nl, MU_COUNT_COL].astype(int)
                    keep = np.nonzero(cnt == 0)[0]
                    k = int(keep[0]) if len(keep) else -1
                    pr = p[j, :nl]
                    if arm == 'keep' and k >= 0:
                        a[j] = k
                    elif arm == 'sample':
                        a[j] = int(np.random.default_rng(seed * 1000 + r * 7 + seat).choice(nl, p=pr / pr.sum()))
                    mulls.setdefault(r, []).append(dict(seat=seat, p_keep=float(pr[k]) if k >= 0 else None,
                                                        n_opts=nl, set_aside=int(cnt[a[j]]),
                                                        p_best_mull=float(np.delete(pr, k).max()) if k >= 0 and nl > 1 else 0.0,
                                                        v=float(v[j])))
            ch[rows] = a
        env.step(ch)
        for res in env.results():
            r = int(res['env'])
            if r in seen:
                continue
            seen.add(r)
            rec = dict(arm=arm, deal=f'{wid}:{r}', winner=int(res['winner']), turns=int(res['turns']),
                       error=bool(res.get('error')), first=first.get(r, 0), mull=mulls.get(r, []))
            q.put(rec)
    q.put(None)


def summarize(recs):
    out = {}
    for arm in sorted(set(r['arm'] for r in recs)):
        rs = [r for r in recs if r['arm'] == arm and not r['error'] and r['winner'] in (1, 2)]
        d = [m for r in rs for m in r['mull']]
        fp = np.mean([r['winner'] == r['first'] for r in rs]) if rs else float('nan')
        out[arm] = dict(games=len(rs), first_player_wr=round(float(fp), 3),
                        p1_wr=round(float(np.mean([r['winner'] == 1 for r in rs])), 3),
                        mull_rate=round(float(np.mean([m['set_aside'] > 0 for m in d])), 3) if d else None,
                        mean_p_keep=round(float(np.mean([m['p_keep'] for m in d if m['p_keep'] is not None])), 3) if d else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--games', type=int, default=4000)
    ap.add_argument('--procs', type=int, default=8)
    ap.add_argument('--seed', type=int, default=770001)
    ap.add_argument('--arms', default='net,keep,sample')
    ap.add_argument('--out', default=os.path.join(HERE, 'results'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    P = a.procs
    sizes = [a.games // P + (1 if i < a.games % P else 0) for i in range(P)]
    allrecs = []
    for arm in a.arms.split(','):
        t0 = time.time()
        ctx = mp.get_context('fork')
        q = ctx.Queue()
        ps = [ctx.Process(target=worker, args=(i, a.ckpt, arm, sizes[i], a.seed + 100000 * i, q)) for i in range(P)]
        for p in ps:
            p.start()
        done, recs = 0, []
        with open(os.path.join(a.out, f'{arm}.jsonl'), 'w') as f:
            while done < P:
                r = q.get()
                if r is None:
                    done += 1
                    continue
                recs.append(r)
                f.write(json.dumps(r) + '\n')
                if len(recs) % 500 == 0:
                    print(f'[{arm} {time.time() - t0:5.0f}s] {len(recs)}/{a.games}', flush=True)
        for p in ps:
            p.join()
        allrecs += recs
        print(arm, json.dumps(summarize(recs)[arm]), f'{time.time() - t0:.0f}s', flush=True)
    json.dump(summarize(allrecs), open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
