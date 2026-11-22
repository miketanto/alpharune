"""G1 evaluation (GENERALIST_PLAN.md sections 4-5): pilot swap (P1) per deck, forgetting, midpoint slope.

  python3 g1/eval_g1.py                       # defaults below, 2000 games per pair (1000 per seat order), 8 procs
  python3 g1/eval_g1.py --games 16 --procs 2 --out g1/eval_smoke   # smoke

Players (checkpoint, deck it pilots):
  G      g1 main final    runs/g1_main/league_00500.pt         (--g)
  Gmid   g1 main midpoint runs/g1_main/league_00250.pt         (--gmid)
  S_R    Rengar specialist = E2 control arm final: runs/e2a_main/league_00500.pt (ov_main@2080 + 500 league updates,
         the same added budget as G); if that file is missing runs/e2a_main/latest.pt; if E2 arm A does not exist,
         FALLBACK runs/ov_main/league_02080.pt (said so in the report). (--s-rengar overrides)
  S_A    Azir specialist  runs/az_main/league_00300.pt          (--s-azir)
  OV     ov_main@2080     runs/ov_main/league_02080.pt          (forgetting reference on Rengar)
Reference opponents: S_R piloting Rengar, S_A piloting Azir.

Games: greedy (argmax) both sides, one game per env slot (its first), rbenv with both seats external. A pair
(X on deck dx) vs (Y on deck dy) is played in both seat orders with the SAME game seeds, games/2 each, so every
deal is played once from each side (seat- and deal-balanced). Draws count 1/2. Wilson 95% CIs per win rate.
A player against itself on the same deck is exactly 0.5 under this design (deterministic greedy play, the same deal
from both sides), so those cells are filled with 0.5 and not played.

P1 pilot swap on deck X:  d_X = mean over refs r of [ WR(G on X vs r) - WR(S_X on X vs r) ]; bar: d_X >= -0.03.
  Summary: mean of d_rengar and d_azir, and the worst deck. CI of a difference: independent-binomial normal
  approximation (the pairs share seeds, so it is conservative).
Forgetting: G on the Rengar mirror vs OV; G on the Azir mirror vs S_A (= the P1 Azir cell vs S_A).
Midpoint: the same G cells for Gmid (slope = final - midpoint).
Pair results are cached in <out>/pairs.jsonl (keyed by checkpoint paths, decks, games, seed): a rerun only plays
what is missing. Report: <out>/REPORT.txt and <out>/report.json.
"""
import argparse, hashlib, json, math, multiprocessing as mp, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SP = os.path.normpath(os.path.join(HERE, '..', 'selfplay'))
sys.path.insert(0, os.path.join(SP, 'league'))
os.environ.setdefault('LEAGUE_MODEL', 'live')
from common import act_rows, import_rbenv, load_net  # noqa: E402

RUNS = os.path.join(SP, 'runs')
BAR = -0.03


def wilson(w, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((c - h) / d, 4), round((c + h) / d, 4)


def worker(ck1, d1, ck2, d2, n, seed, cores, q):
    if cores:
        os.sched_setaffinity(0, cores)
    import torch
    torch.set_num_threads(1)
    RBEnv = import_rbenv()
    env = RBEnv(n, d1, d2, seed, autopay=True, opponent=None)
    nets = {1: load_net(ck1, env.dims), 2: load_net(ck2, env.dims)}
    seen = set()
    while len(seen) < n:
        obs = env.observe()
        rows = np.nonzero(obs['need'])[0]
        ch = np.zeros(len(obs['need']), np.int64)
        if len(rows):
            done = np.array([r in seen for r in rows])
            if done.any():                         # slots already counted: finish with the first legal option
                ch[rows[done]] = obs['act_mask'][rows[done]].argmax(1)
            live = rows[~done]
            for s in (1, 2):
                rr = live[obs['seat'][live] == s]
                if len(rr):
                    ch[rr] = act_rows(nets[s], obs, rr, 'cpu', True)
        env.step(ch)
        for r in env.results():
            if r['env'] not in seen:
                seen.add(r['env'])
                q.put((int(r['winner']), bool(r.get('error')), int(r.get('turns', 0))))
    q.put(None)


def play_order(ck1, d1, ck2, d2, games, procs, seed, cores):
    """seat 1 = (ck1, d1). Returns seat-1 score, games, draws, errors, mean turns."""
    ctx = mp.get_context('fork')
    q = ctx.Queue()
    P = max(1, min(procs, games))
    sizes = [games // P + (1 if i < games % P else 0) for i in range(P)]
    ps = [ctx.Process(target=worker, args=(ck1, d1, ck2, d2, sizes[i], seed + 100000 * i, cores, q))
          for i in range(P)]
    for p in ps:
        p.start()
    done = n = draws = errs = 0
    s1 = 0.0
    turns = []
    while done < P:
        r = q.get()
        if r is None:
            done += 1
            continue
        win, err, t = r
        if err:
            errs += 1
            continue
        n += 1
        turns.append(t)
        s1 += 1.0 if win == 1 else 0.5 if win == 0 else 0.0
        draws += win == 0
    for p in ps:
        p.join()
    return dict(score=s1, games=n, draws=draws, errors=errs, mean_turns=round(float(np.mean(turns)), 1) if turns else None)


def play_pair(ckA, dA, ckB, dB, games, procs, seed, cores):
    t0 = time.time()
    o1 = play_order(ckA, dA, ckB, dB, games // 2, procs, seed, cores)          # A on seat 1
    o2 = play_order(ckB, dB, ckA, dA, games - games // 2, procs, seed, cores)  # A on seat 2, same deals
    a1, a2 = o1['score'], o2['games'] - o2['score']
    n = o1['games'] + o2['games']
    return dict(a_score=a1 + a2, games=n, wr=round((a1 + a2) / n, 4) if n else float('nan'),
                ci=wilson(a1 + a2, n), wr_seat1=round(a1 / o1['games'], 4) if o1['games'] else None,
                wr_seat2=round(a2 / o2['games'], 4) if o2['games'] else None,
                draws=o1['draws'] + o2['draws'], errors=o1['errors'] + o2['errors'],
                mean_turns=[o1['mean_turns'], o2['mean_turns']], secs=round(time.time() - t0, 1))


class Cache:
    def __init__(self, path):
        self.path = path
        self.d = {}
        if os.path.exists(path):
            for line in open(path):
                try:
                    r = json.loads(line)
                    self.d[r['key']] = r
                except Exception:
                    pass

    @staticmethod
    def key(ckA, dA, ckB, dB, games, seed):
        def fid(p):
            st = os.stat(p)
            return f'{os.path.relpath(p, SP)}:{st.st_size}:{int(st.st_mtime)}'
        return hashlib.md5(json.dumps([fid(ckA), dA, fid(ckB), dB, games, seed]).encode()).hexdigest()

    def get(self, k):
        return self.d.get(k)

    def put(self, k, r):
        r['key'] = k
        self.d[k] = r
        with open(self.path, 'a') as f:
            f.write(json.dumps(r) + '\n')


def resolve_s_rengar(arg):
    if arg:
        return arg, 'given (--s-rengar)'
    for p, why in ((os.path.join(RUNS, 'e2a_main', 'league_00500.pt'), 'E2 control arm @500 (ov_main@2080 + 500)'),
                   (os.path.join(RUNS, 'e2a_main', 'latest.pt'), 'E2 control arm latest.pt')):
        if os.path.exists(p):
            return p, why
    return os.path.join(RUNS, 'ov_main', 'league_02080.pt'), ('FALLBACK: runs/e2a_main missing, using ov_main@2080 '
                                                             '(no added updates: the comparison is NOT budget-matched)')


def ck_update(p):
    import torch
    try:
        return torch.load(p, map_location='cpu', weights_only=False).get('update')
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--g', default=os.path.join(RUNS, 'g1_main', 'league_00500.pt'))
    ap.add_argument('--gmid', default=os.path.join(RUNS, 'g1_main', 'league_00250.pt'), help="'' = skip")
    ap.add_argument('--s-rengar', default='')
    ap.add_argument('--s-azir', default=os.path.join(RUNS, 'az_main', 'league_00300.pt'))
    ap.add_argument('--ov', default=os.path.join(RUNS, 'ov_main', 'league_02080.pt'))
    ap.add_argument('--games', type=int, default=2000, help='per pair (half per seat order)')
    ap.add_argument('--procs', type=int, default=8)
    ap.add_argument('--cores', default='', help='e.g. 8-15 (default: no pinning)')
    ap.add_argument('--seed', type=int, default=880001)
    ap.add_argument('--out', default=os.path.join(HERE, 'eval'))
    a = ap.parse_args()
    cores = None
    if a.cores:
        lo, hi = (int(x) for x in a.cores.split('-'))
        cores = set(range(lo, hi + 1))
    os.makedirs(a.out, exist_ok=True)
    cache = Cache(os.path.join(a.out, 'pairs.jsonl'))
    s_r, s_r_why = resolve_s_rengar(a.s_rengar)
    P = {'G': a.g, 'S_R': s_r, 'S_A': a.s_azir, 'OV': a.ov}
    if a.gmid:
        P['Gmid'] = a.gmid
    for k, p in P.items():
        assert os.path.exists(p), f'{k}: {p} missing'
    info = {k: dict(path=os.path.relpath(p, SP), update=ck_update(p)) for k, p in P.items()}
    lines = [f'G1 evaluation ({time.ctime()}), greedy, {a.games} games per pair ({a.games // 2} per seat order, '
             f'same deals), Wilson 95% CIs', '']
    for k, v in info.items():
        lines.append(f'  {k:5s} {v["path"]} (update {v["update"]})' + (f'  [{s_r_why}]' if k == 'S_R' else ''))
    lines.append('')
    print('\n'.join(lines), flush=True)

    res = {}

    def wr(x, dx, y, dy):
        """Win rate of player x piloting dx vs player y piloting dy."""
        if P[x] == P[y] and dx == dy:
            return dict(wr=0.5, ci=(0.5, 0.5), games=0, note='self, exactly 0.5 by design')
        k = cache.key(P[x], dx, P[y], dy, a.games, a.seed)
        r = cache.get(k)
        if r is None:
            kr = cache.key(P[y], dy, P[x], dx, a.games, a.seed)     # the same pair the other way round
            r2 = cache.get(kr)
            if r2 is not None:
                return dict(wr=round(1 - r2['wr'], 4), ci=(round(1 - r2['ci'][1], 4), round(1 - r2['ci'][0], 4)),
                            games=r2['games'], note='from the reversed pair')
            r = play_pair(P[x], dx, P[y], dy, a.games, a.procs, a.seed, cores)
            r.update(x=x, dx=dx, y=y, dy=dy)
            cache.put(k, r)
            print(f'  {x}({dx}) vs {y}({dy}): {r["wr"]:.3f} {tuple(r["ci"])} n={r["games"]} '
                  f'seat1 {r["wr_seat1"]} seat2 {r["wr_seat2"]} err={r["errors"]} ({r["secs"]}s)', flush=True)
        return r

    refs = [('S_R', 'rengar'), ('S_A', 'azir')]
    spec = {'rengar': 'S_R', 'azir': 'S_A'}
    out = {'players': info, 's_rengar_choice': s_r_why, 'games_per_pair': a.games, 'p1': {}, 'forgetting': {}}

    def se(c):
        return (c['ci'][1] - c['ci'][0]) / (2 * 1.96)

    def p1_block(gname):
        blk = {}
        for X in ('rengar', 'azir'):
            cells = []
            for ref, dref in refs:
                g = wr(gname, X, ref, dref)
                s = wr(spec[X], X, ref, dref)
                d = g['wr'] - s['wr']
                cells.append(dict(ref=f'{ref}({dref})', G=g['wr'], G_ci=list(g['ci']), S=s['wr'], S_ci=list(s['ci']),
                                  diff=round(d, 4), diff_se=round(math.hypot(se(g), se(s)), 4)))
            dm = sum(c['diff'] for c in cells) / len(cells)
            dse = math.sqrt(sum(c['diff_se'] ** 2 for c in cells)) / len(cells)
            blk[X] = dict(cells=cells, diff=round(dm, 4), ci=[round(dm - 1.96 * dse, 4), round(dm + 1.96 * dse, 4)],
                          pass_point=dm >= BAR, pass_ci_lo=dm - 1.96 * dse >= BAR)
        mean = (blk['rengar']['diff'] + blk['azir']['diff']) / 2
        worst = min(('rengar', 'azir'), key=lambda X: blk[X]['diff'])
        blk['mean_diff'] = round(mean, 4)
        blk['worst'] = dict(deck=worst, diff=blk[worst]['diff'])
        blk['pass'] = mean >= BAR
        return blk

    for gname in ['G'] + (['Gmid'] if 'Gmid' in P else []):
        out['p1'][gname] = p1_block(gname)
        fr = wr(gname, 'rengar', 'OV', 'rengar')
        fa = wr(gname, 'azir', 'S_A', 'azir')
        out['forgetting'][gname] = {'rengar_mirror_vs_ov_main@2080': dict(wr=fr['wr'], ci=list(fr['ci'])),
                                    'azir_mirror_vs_az_main@300': dict(wr=fa['wr'], ci=list(fa['ci']))}
    ctx = wr('S_R', 'rengar', 'OV', 'rengar')
    out['context_S_R_vs_OV'] = dict(wr=ctx['wr'], ci=list(ctx['ci']))

    L = lines
    for gname, blk in out['p1'].items():
        L.append(f'== P1 pilot swap, {gname} ==  (win rate of the pilot on deck X vs each reference; diff = '
                 f'{gname} - specialist; bar {BAR:+.2f})')
        for X in ('rengar', 'azir'):
            b = blk[X]
            L.append(f'  deck {X} (specialist {spec[X]}):')
            for c in b['cells']:
                L.append(f'    vs {c["ref"]:12s} {gname} {c["G"]:.3f} [{c["G_ci"][0]:.3f},{c["G_ci"][1]:.3f}]   '
                         f'{spec[X]} {c["S"]:.3f} [{c["S_ci"][0]:.3f},{c["S_ci"][1]:.3f}]   diff {c["diff"]:+.3f} '
                         f'(se {c["diff_se"]:.3f})')
            L.append(f'    pilot-swap diff d_{X} = {b["diff"]:+.3f}  95% CI [{b["ci"][0]:+.3f}, {b["ci"][1]:+.3f}]  '
                     f'-> {"PASS" if b["pass_point"] else "FAIL"} vs {BAR:+.2f}'
                     f'{" (CI lower bound also clears it)" if b["pass_ci_lo"] else ""}')
        L.append(f'  mean over decks {blk["mean_diff"]:+.3f} -> {"PASS" if blk["pass"] else "FAIL"}; worst deck '
                 f'{blk["worst"]["deck"]} {blk["worst"]["diff"]:+.3f}')
        f = out['forgetting'][gname]
        L.append(f'  forgetting: {gname} Rengar mirror vs ov_main@2080 '
                 f'{f["rengar_mirror_vs_ov_main@2080"]["wr"]:.3f} {tuple(f["rengar_mirror_vs_ov_main@2080"]["ci"])}; '
                 f'Azir mirror vs az_main@300 {f["azir_mirror_vs_az_main@300"]["wr"]:.3f} '
                 f'{tuple(f["azir_mirror_vs_az_main@300"]["ci"])}')
        L.append('')
    L.append(f'context: S_R Rengar mirror vs ov_main@2080 {ctx["wr"]:.3f} {tuple(ctx["ci"])} (what the specialist '
             f'gained over the init)')
    if 'Gmid' in out['p1']:
        g, m = out['p1']['G'], out['p1']['Gmid']
        L.append(f'slope midpoint -> final: mean pilot-swap diff {m["mean_diff"]:+.3f} -> {g["mean_diff"]:+.3f}; '
                 f'rengar {m["rengar"]["diff"]:+.3f} -> {g["rengar"]["diff"]:+.3f}; '
                 f'azir {m["azir"]["diff"]:+.3f} -> {g["azir"]["diff"]:+.3f}')
    txt = '\n'.join(L) + '\n'
    open(os.path.join(a.out, 'REPORT.txt'), 'w').write(txt)
    json.dump(out, open(os.path.join(a.out, 'report.json'), 'w'), indent=1)
    print(txt, flush=True)


if __name__ == '__main__':
    main()
