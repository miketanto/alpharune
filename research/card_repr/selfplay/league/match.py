"""Network-vs-network and network-vs-anchor evaluation games (league/README.txt).

Network vs network (play_nets): rbenv with opponent=None (both seats external). Each env slot plays exactly ONE
game (its first; later games of that slot are finished with the first legal option and never counted), as
verify_eval.py does, so short games are not over-counted. Seats alternate by slot: in global slot k, network A
is seat 1 when k is even, seat 2 when odd, so with an even game count each network has each seat equally often.
Both networks run on CPU, one thread per process, greedy (argmax) by default (--sample draws from the policy).
Games are split over --procs processes, each pinned to cores 12-15 (common.EVAL_CORES).

Network vs anchor (play_anchor): reuses verify_eval.py (network = seat 1, the C++ agent = seat 2; that script's
convention, so anchor games are NOT seat-balanced), run as a subprocess under `taskset -c 12-15`. If
verify_eval.py cannot run (e.g. the learner files are mid-edit), the same loop runs natively (fallback=True).

Result record (one dict, appended to a results JSONL by ratings.ResultStore):
  {a, b, games, a_wins, b_wins, draws, errors, a_seat1_games, a_seat1_wins, mode, seed, secs, kind}
  a / b are registry ids; for anchors b = 'anchor:<spec>'.

CLI
  python3 match.py <registry id A> <registry id B | anchor:spec> <n_games> [--procs 4] [--sample] [--seed S]
          [--results league/results.jsonl]
"""
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EVAL_CORES, SP, act_rows, import_rbenv, load_net, pin_cores  # noqa: E402

DECK = 'rengar'


def _first_legal(mask_rows):
    return mask_rows.argmax(1)


def _worker_nets(wid, ck_a, rep_a, ck_b, rep_b, n, slot0, seed, q, greedy, variants, cores):
    pin_cores(cores)
    import torch
    torch.set_num_threads(1)
    RBEnv = import_rbenv()
    env = RBEnv(n, DECK, DECK, seed, variant=variants, autopay=True, opponent=None)
    na = load_net(ck_a, env.dims, rep_a)
    nb = load_net(ck_b, env.dims, rep_b)
    a_seat = np.array([1 if (slot0 + i) % 2 == 0 else 2 for i in range(n)])
    seen = set()
    while len(seen) < n:
        obs = env.observe()
        need = obs['need']
        rows = np.nonzero(need)[0]
        ch = np.zeros(len(need), np.int64)
        if len(rows):
            done = np.array([r in seen for r in rows])
            if done.any():
                ch[rows[done]] = _first_legal(obs['act_mask'][rows[done]])
            live = rows[~done]
            isa = obs['seat'][live] == a_seat[live]
            ra, rb = live[isa], live[~isa]
            if len(ra):
                ch[ra] = act_rows(na, obs, ra, 'cpu', greedy)
            if len(rb):
                ch[rb] = act_rows(nb, obs, rb, 'cpu', greedy)
        env.step(ch)
        for r in env.results():
            e = r['env']
            if e in seen:
                continue
            seen.add(e)
            q.put((wid, int(a_seat[e]), int(r['winner']), bool(r.get('error')), int(r.get('turns', 0))))
    q.put((wid, None, None, None, None))


def play_nets(ck_a, ck_b, n_games, rep_a=None, rep_b=None, procs=4, greedy=True, seed=770001, variants=None,
              cores=EVAL_CORES, verbose=False):
    """ck_a / ck_b: checkpoint paths. Returns the result record (a/b fields left to the caller)."""
    P = max(1, min(procs, n_games))
    sizes = [n_games // P + (1 if i < n_games % P else 0) for i in range(P)]
    # make each worker's first slot even so seat alternation is balanced overall
    starts = np.cumsum([0] + sizes[:-1]).tolist()
    ctx = mp.get_context('fork')
    q = ctx.Queue()
    ps = [ctx.Process(target=_worker_nets, args=(i, ck_a, rep_a, ck_b, rep_b, sizes[i], starts[i],
                                                 seed + 100000 * i, q, greedy, variants, cores))
          for i in range(P)]
    t0 = time.time()
    for p in ps:
        p.start()
    rec = dict(games=0, a_wins=0, b_wins=0, draws=0, errors=0, a_seat1_games=0, a_seat1_wins=0)
    done = 0
    turns = []
    while done < P:
        wid, aseat, win, err, t = q.get()
        if aseat is None:
            done += 1
            continue
        if err:
            rec['errors'] += 1
            continue
        rec['games'] += 1
        turns.append(t)
        rec['a_seat1_games'] += aseat == 1
        if win == 0:
            rec['draws'] += 1
        elif win == aseat:
            rec['a_wins'] += 1
            rec['a_seat1_wins'] += aseat == 1
        else:
            rec['b_wins'] += 1
    for p in ps:
        p.join()
    rec.update(mode='greedy' if greedy else 'sample', seed=seed, secs=round(time.time() - t0, 1), kind='net-net',
               mean_turns=round(float(np.mean(turns)), 1) if turns else None)
    if verbose:
        print(rec, flush=True)
    return rec


def _anchor_native(ck, rep, spec, n_games, procs, greedy, seed, cores):
    """Same loop as verify_eval.worker (network seat 1, C++ agent seat 2), without importing the learner."""
    def w(wid, n, s, q):
        pin_cores(cores)
        import torch
        torch.set_num_threads(1)
        RBEnv = import_rbenv()
        env = RBEnv(n, DECK, DECK, s, autopay=True, opponent=spec)
        net = load_net(ck, env.dims, rep)
        seen = set()
        while len(seen) < n:
            obs = env.observe()
            rows = np.nonzero(obs['need'])[0]
            ch = np.zeros(len(obs['need']), np.int64)
            if len(rows):
                ch[rows] = act_rows(net, obs, rows, 'cpu', greedy)
            env.step(ch)
            for r in env.results():
                if r['env'] not in seen:
                    seen.add(r['env'])
                    q.put((int(r['winner']), bool(r.get('error'))))
        q.put((None, None))
    P = max(1, min(procs, n_games))
    sizes = [n_games // P + (1 if i < n_games % P else 0) for i in range(P)]
    ctx = mp.get_context('fork')
    q = ctx.Queue()
    ps = [ctx.Process(target=w, args=(i, sizes[i], seed + 100000 * i, q)) for i in range(P)]
    for p in ps:
        p.start()
    done = games = wins = errs = draws = 0
    while done < P:
        win, err = q.get()
        if win is None:
            done += 1
        elif err:
            errs += 1
        else:
            games += 1
            wins += win == 1
            draws += win == 0
    for p in ps:
        p.join()
    return dict(games=games, wins=wins, draws=draws, errors=errs)


def play_anchor(ck, spec, n_games, rep=None, procs=4, greedy=True, seed=990001, cores=EVAL_CORES,
                force_native=False):
    """Network (seat 1) vs C++ agent `spec` (seat 2) through verify_eval.py. Returns a result record."""
    t0 = time.time()
    res = None
    if not force_native:
        cmd = ['taskset', '-c', ','.join(map(str, cores)), sys.executable, os.path.join(SP, 'verify_eval.py'), ck,
               spec, str(n_games), '--procs', str(procs), '--seed', str(seed), '--every', str(10 ** 9),
               '--repr', rep or 'id']
        if not greedy:
            cmd.append('--sample')
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=SP)
        try:
            res = json.loads(p.stdout.strip().splitlines()[-1])
            res['via'] = 'verify_eval.py'
        except Exception:
            sys.stderr.write(f'verify_eval.py failed ({p.returncode}); native fallback\n{p.stderr[-2000:]}\n')
    if res is None:
        res = _anchor_native(ck, rep, spec, n_games, procs, greedy, seed, cores)
        res['via'] = 'native'
    g, w = int(res['games']), int(res['wins'])
    d = int(res.get('draws', 0))   # verify_eval counts a draw as a non-win; it reports no draw count
    return dict(games=g, a_wins=w, b_wins=g - w - d, draws=d, errors=int(res.get('errors', 0)), a_seat1_games=g,
                a_seat1_wins=w, mode='greedy' if greedy else 'sample', seed=seed, secs=round(time.time() - t0, 1),
                kind='net-anchor', via=res['via'], mean_turns=res.get('mean_turns'))


def play(reg, a, b, n_games, procs=4, greedy=True, seed=None):
    """Registry-level entry point: a is a net id; b a net id or an anchor id. Returns the record with a/b set."""
    ea, eb = reg.get(a), reg.get(b)
    assert ea['kind'] == 'net' and ea['loadable'], a
    if eb['kind'] == 'anchor':
        rec = play_anchor(reg.abspath(a), eb['spec'], n_games, ea['repr'], procs, greedy,
                          seed if seed is not None else 990001)
    else:
        assert eb['loadable'], b
        rec = play_nets(reg.abspath(a), reg.abspath(b), n_games, ea['repr'], eb['repr'], procs, greedy,
                        seed if seed is not None else 770001)
    rec.update(a=a, b=b, t=time.strftime('%Y-%m-%dT%H:%M:%S'))
    return rec


def main():
    import argparse
    from registry import Registry
    from ratings import ResultStore
    ap = argparse.ArgumentParser()
    ap.add_argument('a'); ap.add_argument('b'); ap.add_argument('n', type=int)
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--registry', default=None)
    ap.add_argument('--results', default=None)
    a = ap.parse_args()
    reg = Registry(a.registry)
    rec = play(reg, a.a, a.b, a.n, a.procs, not a.sample, a.seed)
    ResultStore(a.results).append(rec)
    print(json.dumps(rec))


if __name__ == '__main__':
    main()
