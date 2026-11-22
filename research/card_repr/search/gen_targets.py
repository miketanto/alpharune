"""E2 search-teacher generator (CEILING_PLAN.md E2; e2/REPORT.txt).

  python3 gen_targets.py --weights ../selfplay/runs/e2b_main/gen_weights.pt --out ../e2/buf_B --threads 4

Self-play with the CURRENT main weights (reloaded whenever --weights changes on disk; the trainer
rewrites it every --export-every updates). Both seats are the network and SAMPLE their moves (as the
training actors); each external decision (>= 2 options, not a rune payment) is searched with
probability --p-search by rbsearch's determinized PUCT (--sims, net prior + value, strict visibility
inside the search, legend-pool deck model). For a searched decision the record is:
  the live observation encoded with the TRAINING encoder (loose visibility, as rbenv: identical to
  what selfplay.py's actors see), the net prior P, root visits N and mean values Q per option, the net
  value v0 and the visit-weighted root value vs at the root, and the game outcome z (decider's view).
The move played at a searched decision is still sampled from the net prior, so the positions follow
the network's own sampled-play distribution.
Records are written in chunks (<out>/c_<seq>.npz, atomically) and only the newest --keep chunks are
kept (rolling buffer). <out>/gen_log.jsonl: throughput every --log-every seconds.
--holdout N: write the first N records to <out>/holdout.npz instead and exit (fixed evaluation set).
"""
import argparse, glob, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rbsearch  # noqa: E402

KEYS = ('glob', 'tok', 'tok_mask', 'act', 'act_mask', 'pk', 'psrc', 'nlegal', 'seat', 'P', 'N', 'Q', 'v0', 'vs', 'z',
        'wupd')


def pack(recs):
    c = {k: np.concatenate([r[k] for r in recs]) for k in recs[0]}
    return dict(glob=np.clip(c['glob'], -32768, 32767).astype(np.int16),
                tok=np.clip(c['tok'], -32768, 32767).astype(np.int16), tok_mask=c['tok_mask'],
                act=np.clip(c['act'], -32768, 32767).astype(np.int16), act_mask=c['act_mask'],
                pk=c['pk'].astype(np.int16), psrc=c['psrc'].astype(np.int16), nlegal=c['nlegal'].astype(np.int16),
                seat=c['seat'].astype(np.int8), P=c['P'].astype(np.float16), N=c['N'].astype(np.float16),
                Q=c['Q'].astype(np.float16), v0=c['v0'].astype(np.float32), vs=c['vs'].astype(np.float32),
                z=c['z'].astype(np.float32), wupd=c['wupd'].astype(np.int32))


def save_npz(path, d):
    tmp = path + '.tmp.npz'
    np.savez(tmp, **d)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True, help='checkpoint to play / search with; reloaded when it changes')
    ap.add_argument('--out', required=True)
    ap.add_argument('--games', type=int, default=96, help='concurrent game slots')
    ap.add_argument('--sims', type=int, default=64)
    ap.add_argument('--p-search', type=float, default=0.3)
    ap.add_argument('--leaves', type=int, default=4)
    ap.add_argument('--cpuct', type=float, default=2.0)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--deck', default='rengar')
    ap.add_argument('--seed', type=int, default=5_000_000)
    ap.add_argument('--chunk', type=int, default=512, help='records per chunk file')
    ap.add_argument('--keep', type=int, default=80, help='chunk files kept (rolling buffer)')
    ap.add_argument('--gpu-mem-frac', type=float, default=0.15)
    ap.add_argument('--max-minutes', type=float, default=0)
    ap.add_argument('--log-every', type=float, default=60)
    ap.add_argument('--holdout', type=int, default=0)
    a = ap.parse_args()
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cuda' and a.gpu_mem_frac:
        torch.cuda.set_per_process_memory_fraction(a.gpu_mem_frac)
    torch.set_num_threads(1)
    os.makedirs(a.out, exist_ok=True)
    S = rbsearch.Search(a.games, a.deck, a.seed, sims=a.sims, leaves=a.leaves, threads=a.threads, cpuct=a.cpuct,
                        seat_mode='gen', strict=True, p_search=a.p_search)

    def load():
        while True:
            try:
                ck = torch.load(a.weights, map_location='cpu', weights_only=False)
                break
            except Exception as e:   # being rewritten
                print('weights not readable yet:', repr(e), flush=True)
                time.sleep(5)
        upd = int(ck.get('update', -1)) if isinstance(ck, dict) else -1
        net = rbsearch.load_net(a.weights, S.dims, dev)
        return net, upd, os.stat(a.weights).st_mtime

    net, wupd, wmt = load()
    ev = rbsearch.Evaluator(net, dev, S.dims['A'])
    print(f'generator: weights {a.weights} @{wupd}, {a.games} games, sims {a.sims}, p_search {a.p_search}, '
          f'threads {a.threads}', flush=True)
    seq = max([int(os.path.basename(p)[2:9]) for p in glob.glob(os.path.join(a.out, 'c_*.npz'))] or [0])
    next_seed = a.seed + a.games
    pend, npend = [], 0
    t0 = tlog = time.time()
    tot_rec = tot_games = reloads = 0
    n_changed = n_rec_seen = 0
    lg = open(os.path.join(a.out, 'gen_log.jsonl'), 'a')
    while True:
        B = S.collect()
        if B:
            rows = S.rows(B)
            logp, v = ev(rows, B)
            S.feed(logp, v)
        res = S.results()
        for r in res:
            if not r['done']:
                continue
            rec = S.gen_take(r['game'])
            if rec is not None and not r['error']:
                rec['wupd'] = np.full(len(rec['z']), wupd, np.int32)
                n_changed += int((rec['N'].argmax(1) != rec['P'].argmax(1)).sum()); n_rec_seen += len(rec['z'])
                pend.append(rec); npend += len(rec['z'])
            tot_games += 1
            S.restart(r['game'], next_seed); next_seed += 1
        if a.holdout and npend >= a.holdout:
            d = pack(pend)
            d = {k: x[:a.holdout] for k, x in d.items()}
            save_npz(os.path.join(a.out, 'holdout.npz'), d)
            print(f'holdout: {a.holdout} records from {tot_games} games in {time.time() - t0:.0f}s', flush=True)
            return
        while not a.holdout and npend >= a.chunk:
            d = pack(pend)
            seq += 1
            save_npz(os.path.join(a.out, f'c_{seq:07d}.npz'), {k: x[:a.chunk] for k, x in d.items()})
            rest = {k: x[a.chunk:] for k, x in d.items()}
            pend = [rest] if len(rest['z']) else []
            npend = len(rest['z'])
            tot_rec += a.chunk
            old = sorted(glob.glob(os.path.join(a.out, 'c_*.npz')))
            for p in old[:-a.keep]:
                try:
                    os.remove(p)
                except OSError:
                    pass
        try:
            mt = os.stat(a.weights).st_mtime
        except OSError:
            mt = wmt
        if mt != wmt:
            time.sleep(1.0)   # let the writer finish
            n2, wupd, wmt = load()
            net.load_state_dict(n2.state_dict()); reloads += 1
            print(f'reloaded weights @{wupd}', flush=True)
        now = time.time()
        if now - tlog > a.log_every:
            # (rbs_stats sums over the live game slots only, so it is not used after restarts)
            row = dict(t=round(now - t0), wupd=wupd, reloads=reloads, records=tot_rec, games=tot_games,
                       rec_per_s=round(tot_rec / (now - t0), 1), sims_per_s_est=round(tot_rec * a.sims / (now - t0)),
                       argmax_changed=round(n_changed / max(n_rec_seen, 1), 4), batch=B, chunk_seq=seq)
            lg.write(json.dumps(row) + '\n'); lg.flush()
            print(json.dumps(row), flush=True)
            tlog = now
        if a.max_minutes and now - t0 > a.max_minutes * 60:
            break


if __name__ == '__main__':
    main()
