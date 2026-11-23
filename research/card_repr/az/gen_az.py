"""AlphaZero-style self-play generator (PLAYTEST_PLAN.md; az/README.txt).

  python3 gen_az.py --weights ../az/run/gen_weights.pt --out ../az/buf --threads 14

Both seats are the network + search (rbsearch.cpp, seat_mode gen + az): every external decision is
searched; with probability --p-full a full search (--sims, Dirichlet noise at the root) whose root
statistics are recorded as a training target, else a fast search (--sims-fast, not recorded). The move
played is the search's: visit-proportional for the first --temp-moves external decisions of a game, then
the most visited option. Observations carry the public action history (rbenv history.h).

Matchups (--matchups, ordered seat1:seat2 deck pairs, repeated entries = more slots): slot i plays pair
i % len. Records: observation (training encoder), P, N, Q per option, v0, vs, outcome z (decider's view),
seat, the decider's deck id (index into --matchups' deck names, deck_names.json) and the weights update.
Chunks <out>/c_<seq>.npz (newest --keep kept); weights reloaded when --weights changes.
"""
import argparse, glob, json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'search'))
import rbsearch  # noqa: E402

DEFAULT_MU = 'azir:azir,azir:rengar,rengar:azir,azir:diana,diana:azir,azir:azir'


def pack(recs):
    c = {k: np.concatenate([r[k] for r in recs]) for k in recs[0]}
    i16 = lambda x: np.clip(x, -32768, 32767).astype(np.int16)
    return dict(glob=i16(c['glob']), tok=i16(c['tok']), tok_mask=c['tok_mask'], act=i16(c['act']),
                act_mask=c['act_mask'], hist=i16(c['hist']), pk=c['pk'].astype(np.int16),
                psrc=c['psrc'].astype(np.int16), nlegal=c['nlegal'].astype(np.int16), seat=c['seat'].astype(np.int8),
                deck=c['deck'].astype(np.int8), P=c['P'].astype(np.float16), N=c['N'].astype(np.float16),
                Q=c['Q'].astype(np.float16), v0=c['v0'].astype(np.float32), vs=c['vs'].astype(np.float32),
                z=c['z'].astype(np.float32), wupd=c['wupd'].astype(np.int32))


def save_npz(path, d):
    # temp name must not match the trainer's c_*.npz glob (it would open the file and block the rename on /mnt/c)
    tmp = os.path.join(os.path.dirname(path), 'tmp_' + os.path.basename(path))
    np.savez(tmp, **d)
    for i in range(30):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(1.0)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--matchups', default=DEFAULT_MU)
    ap.add_argument('--games', type=int, default=168, help='concurrent game slots')
    ap.add_argument('--sims', type=int, default=64)
    ap.add_argument('--sims-fast', type=int, default=16)
    ap.add_argument('--p-full', type=float, default=0.5)
    ap.add_argument('--temp-moves', type=int, default=30)
    ap.add_argument('--dir-alpha', type=float, default=0.3)
    ap.add_argument('--dir-eps', type=float, default=0.25)
    ap.add_argument('--dir-tap', type=float, default=0.1, help="a rune tap's share of the root noise relative to a move's")
    ap.add_argument('--leaves', type=int, default=4)
    ap.add_argument('--cpuct', type=float, default=2.0)
    ap.add_argument('--threads', type=int, default=14)
    ap.add_argument('--runes', type=int, default=0, help='1: both seats may tap their runes (TAPS.md; needs RB_WSL_COPY = a float-engine copy)')
    ap.add_argument('--seed', type=int, default=9_000_000)
    ap.add_argument('--chunk', type=int, default=1024)
    ap.add_argument('--keep', type=int, default=400)
    ap.add_argument('--gpu-mem-frac', type=float, default=0.3)
    ap.add_argument('--max-minutes', type=float, default=0)
    ap.add_argument('--max-games', type=int, default=0)
    ap.add_argument('--log-every', type=float, default=60)
    a = ap.parse_args()
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cuda' and a.gpu_mem_frac:
        torch.cuda.set_per_process_memory_fraction(a.gpu_mem_frac)
    torch.set_num_threads(1)
    os.makedirs(a.out, exist_ok=True)
    mus = [tuple(x.split(':')) for x in a.matchups.split(',')]
    names = sorted({d for m in mus for d in m})
    json.dump(dict(decks=names, matchups=mus), open(os.path.join(a.out, 'deck_names.json'), 'w'))
    S = rbsearch.Search(a.games, [m[0] for m in mus], a.seed, deck_b=[m[1] for m in mus], sims=a.sims,
                        leaves=a.leaves, threads=a.threads, cpuct=a.cpuct, seat_mode='gen', strict=True, az=True,
                        az_temp_moves=a.temp_moves, sims_fast=a.sims_fast, dir_alpha=a.dir_alpha,
                        dir_eps=a.dir_eps, p_full=a.p_full, runes=a.runes, dir_tap=a.dir_tap)

    def load():
        while True:
            try:
                ck = torch.load(a.weights, map_location='cpu', weights_only=False)
                break
            except Exception as e:
                print('weights not readable yet:', repr(e), flush=True)
                time.sleep(5)
        upd = int(ck.get('update', -1)) if isinstance(ck, dict) else -1
        return rbsearch.load_net(a.weights, S.dims, dev), upd, os.stat(a.weights).st_mtime

    net, wupd, wmt = load()
    ev = rbsearch.Evaluator(net, dev, S.dims['A'])
    print(f'az generator: weights {a.weights} @{wupd}, {a.games} slots over {len(mus)} matchups {mus}, sims '
          f'{a.sims}/{a.sims_fast} p_full {a.p_full}, threads {a.threads}', flush=True)
    seq = max([int(os.path.basename(p)[2:9]) for p in glob.glob(os.path.join(a.out, 'c_*.npz'))] or [0])
    next_seed = a.seed + a.games + 1_000_000 * (seq // 1000)
    pend, npend = [], 0
    t0 = tlog = time.time()
    tot_rec = tot_games = reloads = errs = 0
    taps = np.zeros(4, np.int64)   # finished games: exhaust, recycle, decisions offering a tap, decisions
    wins = {}   # (seat-1 deck, seat-2 deck) -> [seat-1 wins, games]
    lg = open(os.path.join(a.out, 'gen_log.jsonl'), 'a')
    while True:
        B = S.collect()
        if B:
            rows = S.rows(B)
            logp, v = ev(rows, B)
            S.feed(logp, v)
        for r in S.results():
            if not r['done']:
                continue
            g = r['game']
            mu = mus[g % len(mus)]
            rec = S.gen_take(g)
            if r['error']:
                errs += 1
            else:
                w = wins.setdefault(f'{mu[0]}:{mu[1]}', [0, 0])
                w[0] += int(r['winner'] == 1); w[1] += 1
                if rec is not None:
                    rec['wupd'] = np.full(len(rec['z']), wupd, np.int32)
                    rec['deck'] = np.where(rec['seat'] == 1, names.index(mu[0]), names.index(mu[1])).astype(np.int8)
                    pend.append(rec); npend += len(rec['z'])
            tot_games += 1
            if a.runes:
                taps += S.taps(g)
            S.restart(g, next_seed); next_seed += 1
        while npend >= a.chunk:
            d = pack(pend)
            seq += 1
            save_npz(os.path.join(a.out, f'c_{seq:07d}.npz'), {k: x[:a.chunk] for k, x in d.items()})
            rest = {k: x[a.chunk:] for k, x in d.items()}
            pend = [rest] if len(rest['z']) else []
            npend = len(rest['z'])
            tot_rec += a.chunk
            for p in sorted(glob.glob(os.path.join(a.out, 'c_*.npz')))[:-a.keep]:
                try:
                    os.remove(p)
                except OSError:
                    pass
        try:
            mt = os.stat(a.weights).st_mtime
        except OSError:
            mt = wmt
        if mt != wmt:
            time.sleep(1.0)
            n2, wupd, wmt = load()
            net.load_state_dict(n2.state_dict()); reloads += 1
            print(f'reloaded weights @{wupd}', flush=True)
        now = time.time()
        if now - tlog > a.log_every:
            row = dict(t=round(now - t0), wupd=wupd, reloads=reloads, records=tot_rec, games=tot_games, errors=errs,
                       rec_per_s=round(tot_rec / (now - t0), 1), games_per_h=round(3600 * tot_games / (now - t0)),
                       batch=B, chunk_seq=seq, taps_per_game=[round(float(x) / max(1, tot_games), 2) for x in taps] if a.runes else None, seat1_wr={k: round(w[0] / w[1], 3) for k, w in wins.items() if w[1]})
            lg.write(json.dumps(row) + '\n'); lg.flush()
            print(json.dumps(row), flush=True)
            tlog = now
        if (a.max_minutes and now - t0 > a.max_minutes * 60) or (a.max_games and tot_games >= a.max_games):
            break


if __name__ == '__main__':
    main()
