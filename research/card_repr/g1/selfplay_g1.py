"""G1 (GENERALIST_PLAN.md section 4): league PPO trainer on a MIX of matchups.

A copy of selfplay/learner/selfplay.py (2026-11-20, the E2 trainer) with one change of substance: the actor steps
several RBEnv handles, one per ORDERED matchup (seat-1 deck : seat-2 deck), instead of one mirror handle.
  --matchups rengar:rengar,rengar:azir,azir:rengar,azir:azir     (default: the 2x2 grid)
n_envs slots are split equally over the handles (n_envs must be a multiple of the number of matchups). Each step
the handles' observations are concatenated into ONE batch (same dims), so there is still one GPU forward for the
learner per step (+ one per league opponent), and the choices are split back per handle. A global slot index
g = offset[handle] + local index is what the league (assign/report), the learner seat and the trajectories see.
The learner seat alternates per game in every slot (as before), so with the four ordered matchups the learner plays
every (own deck, opponent deck, own seat) combination; slots are uniform over matchups ("uniform to start").
League opponents (self, OSFP history, PFSP pool, exploiter target) are networks, so they play whichever deck the
other seat of the slot holds. The league sampler itself is the shared, unmodified league/integration.py: ONE
OSFP/PFSP win-rate table over all matchups pooled (per-matchup PFSP would need a sampler per cell: not done).

Per-matchup logging: every finished game is attributed to the cell "<learner deck>_v_<opponent deck>"; log.csv has
per cell the learner win rate vs the pool (league opponents, not self-play) and its games, plus the learner's
decision count per cell; metrics.jsonl has the same with seat splits under 'matchups'.

Removed vs learner/selfplay.py (not used by the league recipe): --cf-data, the E2 search teacher / holdout, the
in-training eval vs C++ agents and probes (overnight.py runs with --eval-every 0), the non-league in-memory pool.
Shared modules are imported read-only: learner/model.py (Net), league/integration.py (LeagueOpponents), rbenv.
"""
import argparse, csv, json, os, random, sys, time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))                 # research/card_repr/g1
SP = os.path.normpath(os.path.join(HERE, '..', 'selfplay'))       # research/card_repr/selfplay
sys.path.insert(0, os.path.join(SP, 'learner'))
from model import Net, load_vocab  # noqa: E402

REPO = os.path.normpath(os.path.join(HERE, '..', '..', '..'))      # alpharune
OBS_KEYS = ('glob', 'tok', 'tok_mask', 'act', 'act_mask', 'pk', 'psrc')
CAT_KEYS = OBS_KEYS + ('need', 'seat')


def deck_path(name):
    p = os.path.join(REPO, 'decks', name + '.txt')
    return p if os.path.isfile(p) else name


def parse_matchups(s):
    out = []
    for m in s.split(','):
        a, b = m.strip().split(':')
        out.append((a, b))
    return out


class MultiEnv:
    """Several RBEnv handles (one per ordered matchup) behind the one-handle observe/step/results interface."""

    def __init__(self, n, matchups, seed):
        sys.path.insert(0, os.path.join(SP, 'rbenv'))
        from rbenv import RBEnv
        k = len(matchups)
        assert n % k == 0, f'n_envs {n} must be a multiple of {k} matchups'
        self.per = n // k
        self.matchups = matchups
        self.envs = [RBEnv(self.per, deck_path(a), deck_path(b), seed + 10**7 * i, autopay=True)
                     for i, (a, b) in enumerate(matchups)]
        self.dims = self.envs[0].dims
        self.n = n
        # deck held by each seat of each global slot
        self.deck1 = np.array([a for a, _ in matchups for _ in range(self.per)], dtype=object)
        self.deck2 = np.array([b for _, b in matchups for _ in range(self.per)], dtype=object)

    def observe(self):
        os_ = [e.observe() for e in self.envs]
        return {k: np.concatenate([o[k] for o in os_], 0) for k in CAT_KEYS}

    def step(self, ch):
        for i, e in enumerate(self.envs):
            e.step(ch[i * self.per:(i + 1) * self.per])

    def results(self):
        out = []
        for i, e in enumerate(self.envs):
            for r in e.results():
                r['env'] += i * self.per
                out.append(r)
        return out

    def deck_of(self, g, seat):
        return self.deck1[g] if seat == 1 else self.deck2[g]


def to_dev(obs, rows, dev):
    """Rows of an observe() dict -> torch on device, trimmed to the used T/A."""
    tm = obs['tok_mask'][rows]
    am = obs['act_mask'][rows]
    T = int(tm.any(0).nonzero()[0].max()) + 1 if tm.any() else 1
    A = int(am.any(0).nonzero()[0].max()) + 1 if am.any() else 1
    b = dict(glob=obs['glob'][rows], tok=obs['tok'][rows, :T], tok_mask=tm[:, :T],
             act=obs['act'][rows, :A], act_mask=am[:, :A], pk=obs['pk'][rows], psrc=obs['psrc'][rows])
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(dev, non_blocking=True) for k, v in b.items()}


@torch.no_grad()
def act_rows(net, obs, rows, dev, greedy=False):
    b = to_dev(obs, rows, dev)
    logp, v = net(b)
    a = logp.argmax(-1) if greedy else torch.multinomial(logp.exp().clamp(min=0), 1).squeeze(-1)
    lp = logp.gather(1, a[:, None]).squeeze(1)
    return a.cpu().numpy(), lp.cpu().numpy(), v.cpu().numpy()


class Ring:
    """Flat ring buffer of decisions (int16 obs + float stats)."""

    def __init__(self, cap, dims):
        G, T, Fd, A, K = (int(dims[k]) for k in ('G', 'T', 'F', 'A', 'K'))
        self.cap = cap
        self.glob = np.zeros((cap, G), np.int16)
        self.tok = np.zeros((cap, T, Fd), np.int16)   # int16: card keys reach ~980 with vocab v2 (int8 clipped them)
        self.tok_mask = np.zeros((cap, T), bool)
        self.act = np.zeros((cap, A, K), np.int16)
        self.act_mask = np.zeros((cap, A), bool)
        self.pk = np.zeros(cap, np.int16)
        self.psrc = np.zeros(cap, np.int16)
        self.a = np.zeros(cap, np.int64)
        self.logp = np.zeros(cap, np.float32)
        self.v = np.zeros(cap, np.float32)
        self.adv = np.zeros(cap, np.float32)
        self.ret = np.zeros(cap, np.float32)
        self.head = 0
        self.nbytes = sum(x.nbytes for x in (self.glob, self.tok, self.tok_mask, self.act, self.act_mask))

    def put(self, obs, rows, a, lp, v):
        n = len(rows)
        idx = (self.head + np.arange(n)) % self.cap
        for k in OBS_KEYS:
            src = obs[k][rows]
            dst = getattr(self, k)
            if src.dtype != bool:
                ii = np.iinfo(dst.dtype)
                src = np.clip(src, ii.min, ii.max)
            dst[idx] = src
        self.a[idx], self.logp[idx], self.v[idx] = a, lp, v
        ids = self.head + np.arange(n)
        self.head += n
        return ids

    def alive(self, gid):
        return gid > self.head - self.cap

    def batch(self, gids, dev):
        idx = gids % self.cap
        obs = {k: getattr(self, k) for k in OBS_KEYS}
        b = to_dev(obs, idx, dev)
        for k in ('glob', 'tok', 'act', 'pk', 'psrc'):
            b[k] = b[k].long()
        ex = {k: torch.from_numpy(getattr(self, k)[idx]).to(dev) for k in ('a', 'logp', 'v', 'adv', 'ret')}
        return b, ex


def gae(v, r, lam):
    """gamma=1, reward r only at the end."""
    n = len(v)
    nxt = np.append(v[1:], r)
    delta = nxt - v
    adv = np.zeros(n, np.float32)
    acc = 0.0
    for t in range(n - 1, -1, -1):
        acc = delta[t] + lam * acc
        adv[t] = acc
    return adv, adv + v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repr', default='B2', choices=['id', 'random', 'B2', 'B'])
    ap.add_argument('--matchups', default='rengar:rengar,rengar:azir,azir:rengar,azir:azir',
                    help='ordered seat1:seat2 deck pairs (deck names in alpharune/decks or paths), comma-separated')
    ap.add_argument('--init', default='', help='start from the weights of this checkpoint (fresh optimizer)')
    ap.add_argument('--ask-head', action='store_true')
    ap.add_argument('--ask-opt', action='store_true')
    ap.add_argument('--no-opt-cost', action='store_true')
    ap.add_argument('--no-card-norm', action='store_true')
    ap.add_argument('--n-envs', type=int, default=256)
    ap.add_argument('--updates', type=int, default=100)
    ap.add_argument('--samples-per-update', type=int, default=16384)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--minibatch', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--lam', type=float, default=0.95)
    ap.add_argument('--vf-coef', type=float, default=0.5)
    ap.add_argument('--ent-coef', type=float, default=0.01)
    ap.add_argument('--max-grad-norm', type=float, default=0.5)
    ap.add_argument('--eval-every', type=int, default=0, help='(ignored; kept for overnight.py command lines)')
    ap.add_argument('--ckpt-every', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--max-minutes', type=float, default=0)
    ap.add_argument('--gpu-mem-frac', type=float, default=0)
    ap.add_argument('--league', required=True, help='league config json (overnight_g1.py writes it)')
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.gpu_mem_frac and dev.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_frac)
    out = os.path.join(SP, 'runs', args.run_name)
    os.makedirs(out, exist_ok=True)
    t_start = time.time()

    MU = parse_matchups(args.matchups)
    env = MultiEnv(args.n_envs, MU, args.seed)
    dims = env.dims
    vocab, vocab_src = load_vocab()
    assert len(vocab) == int(dims['V']), f'vocab {len(vocab)} != dims V {dims["V"]} ({vocab_src})'
    mk = dict(ask_head=args.ask_head, card_norm=not args.no_card_norm, ask_opt=args.ask_opt,
              opt_cost=not args.no_opt_cost)
    net = Net(dims, vocab, args.repr, **mk).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    cfg = dict(vars(args), matchup_list=MU, dims={k: int(v) for k, v in dims.items()}, vocab_src=vocab_src,
               device=str(dev), params=sum(p.numel() for p in net.parameters()), missing_repr_keys=net.missing_keys,
               trainer='g1/selfplay_g1.py')
    json.dump(cfg, open(os.path.join(out, 'config.json'), 'w'), indent=1)
    print('config', json.dumps(cfg), flush=True)

    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=dev, weights_only=False)['model'])
        print('initialised from', args.init, flush=True)
    start_update = 0
    ck = {}
    if args.resume and os.path.exists(os.path.join(out, 'latest.pt')):
        ck = torch.load(os.path.join(out, 'latest.pt'), map_location=dev, weights_only=False)
        net.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        start_update = ck['update']
        print('resumed at update', start_update, flush=True)

    sys.path.insert(0, os.path.join(SP, 'league'))
    from integration import LeagueOpponents
    league = LeagueOpponents(args.league, dims, lambda e: Net(dims, vocab, e['repr'], **e['opts']), dev, out,
                             args.run_name, args.seed)
    if ck.get('league'):
        league.load_state_dict(ck['league'])
    league.begin_update(start_update + 1)
    print('league', json.dumps(league.log_row()), flush=True)

    cap = int(args.samples_per_update * 1.5 + args.n_envs * 300)
    ring = Ring(cap, dims)
    print(f'ring capacity {cap} decisions, {ring.nbytes / 2**20:.0f} MiB; matchups {MU}, {env.per} slots each',
          flush=True)

    N = args.n_envs
    learner_seat = np.where(np.arange(N) % 2 == 0, 1, 2)
    opp_mid = np.array([league.assign(e) or '' for e in range(N)], dtype=object)
    opp_self = opp_mid == ''
    traj = defaultdict(list)
    ready = []
    cells = sorted({f'{a}_v_{b}' for a, b in MU} | {f'{b}_v_{a}' for a, b in MU})

    logf = open(os.path.join(out, 'log.csv'), 'a', newline='')
    cols = (['update', 'wall_s', 'games', 'decisions', 'games_per_s', 'dec_per_s', 'mean_len', 'draws', 'errors',
             'lost_traj', 'wr_vs_pool', 'pool_games', 'pol_loss', 'v_loss', 'entropy', 'approx_kl', 'clipfrac',
             'explained_var']
            + [f'{p}_{c}' for c in cells for p in ('wr', 'n', 'dec')])
    w = csv.DictWriter(logf, fieldnames=cols)
    if logf.tell() == 0:
        w.writeheader()
    mj = open(os.path.join(out, 'metrics.jsonl'), 'a')

    for u in range(start_update, args.updates):
        if args.max_minutes and time.time() - t_start > args.max_minutes * 60:
            print('time limit reached', flush=True); break
        t0 = time.time()
        ep = defaultdict(float)
        mc = {c: defaultdict(float) for c in cells}   # per-matchup cell (learner deck _v_ opponent deck)
        net.eval()
        # ---------------------------------------------------------------- collect
        while len(ready) < args.samples_per_update:
            obs = env.observe()
            rows = np.nonzero(obs['need'])[0]
            ch = np.zeros(N, np.int64)
            if len(rows):
                seat = obs['seat'][rows]
                mine = (seat == learner_seat[rows]) | opp_self[rows]
                r_cur, r_pool = rows[mine], rows[~mine]
                if len(r_cur):
                    a, lp, v = act_rows(net, obs, r_cur, dev)
                    ch[r_cur] = a
                    ids = ring.put(obs, r_cur, a, lp, v)
                    for e, s, i in zip(r_cur.tolist(), obs['seat'][r_cur].tolist(), ids.tolist()):
                        traj[(e, s)].append(i)
                        mc[f'{env.deck_of(e, s)}_v_{env.deck_of(e, 3 - s)}']['dec'] += 1
                for mid in set(opp_mid[r_pool].tolist()):
                    rr = r_pool[opp_mid[r_pool] == mid]
                    a, _, _ = act_rows(league.net(mid), obs, rr, dev)
                    ch[rr] = a
                ep['decisions'] += len(r_cur)
            env.step(ch)
            for res in env.results():
                e = res['env']
                ep['games'] += 1
                ep['len'] += res.get('length', 0)
                win = res['winner']
                bad = bool(res.get('error'))
                ep['errors'] += bad
                ep['draws'] += (win == 0) and not bad
                ls = int(learner_seat[e])
                c = mc[f'{env.deck_of(e, ls)}_v_{env.deck_of(e, 3 - ls)}']
                c['games'] += 1
                if not opp_self[e] and not bad:
                    sc = 0.5 if win == 0 else float(win == ls)
                    ep['pool_games'] += 1; ep['pool_wins'] += sc
                    c['n'] += 1; c['w'] += sc
                    c[f'n_s{ls}'] += 1; c[f'w_s{ls}'] += sc
                for s in (1, 2):
                    ids = traj.pop((e, s), [])
                    if not ids or bad:
                        continue
                    if not ring.alive(ids[0]):
                        ep['lost_traj'] += 1
                        continue
                    r = 0.0 if win == 0 else (1.0 if win == s else -1.0)
                    ids = np.array(ids)
                    adv, ret = gae(ring.v[ids % cap], r, args.lam)
                    ring.adv[ids % cap] = adv
                    ring.ret[ids % cap] = ret
                    ready.extend(ids.tolist())
                league.report(e, None if bad else (0.5 if win == 0 else float(win == ls)))
                learner_seat[e] = 3 - ls
                opp_mid[e] = league.assign(e) or ''
                opp_self[e] = opp_mid[e] == ''
        t_collect = time.time() - t0
        rid = np.array(ready); ready = []
        ok = ring.alive(rid)
        ep['lost_rows'] = int((~ok).sum()); rid = rid[ok]

        # ---------------------------------------------------------------- PPO update (unchanged)
        net.train()
        adv_all = ring.adv[rid % cap]
        am, asd = float(adv_all.mean()), float(adv_all.std() + 1e-8)
        agg = defaultdict(float); nb = 0
        for _ in range(args.epochs):
            perm = rid[np.random.permutation(len(rid))]
            for i in range(0, len(perm), args.minibatch):
                g = perm[i:i + args.minibatch]
                if len(g) < 64:
                    continue
                b, ex = ring.batch(g, dev)
                logp, v = net(b)
                lp = logp.gather(1, ex['a'][:, None]).squeeze(1)
                ratio = (lp - ex['logp']).exp()
                adv = (ex['adv'] - am) / asd
                pl = -torch.min(ratio * adv, ratio.clamp(1 - args.clip, 1 + args.clip) * adv).mean()
                vl = F.mse_loss(v, ex['ret'])
                p = logp.exp()
                ent = -(p * logp.masked_fill(~b['act_mask'], 0)).sum(-1).mean()
                loss = pl + args.vf_coef * vl - args.ent_coef * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    agg['pol_loss'] += pl.item(); agg['v_loss'] += vl.item(); agg['entropy'] += ent.item()
                    agg['approx_kl'] += ((ratio - 1) - (lp - ex['logp'])).mean().item()
                    agg['clipfrac'] += ((ratio - 1).abs() > args.clip).float().mean().item()
                nb += 1
        for k in agg:
            agg[k] /= max(nb, 1)
        ret_all, v_all = ring.ret[rid % cap], ring.v[rid % cap]
        ev = 1 - np.var(ret_all - v_all) / (np.var(ret_all) + 1e-8)
        upd = u + 1

        # ---------------------------------------------------------------- league / ckpt
        league_act = league.end_update(upd, net, len(rid), lambda p: torch.save(
            {'model': net.state_dict(), 'update': upd, 'repr': args.repr, 'dims': dims}, p))
        if league_act['reset']:
            if league_act['init']:
                net.load_state_dict(torch.load(league_act['init'], map_location=dev, weights_only=False)['model'])
            else:
                net.load_state_dict(Net(dims, vocab, args.repr, **mk).to(dev).state_dict())
            opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
        league.begin_update(upd + 1)
        if upd % args.ckpt_every == 0 or upd == args.updates:
            torch.save({'model': net.state_dict(), 'update': upd, 'repr': args.repr, 'dims': dims},
                       os.path.join(out, f'ckpt_{upd:05d}.pt'))
            torch.save({'model': net.state_dict(), 'opt': opt.state_dict(), 'update': upd, 'pool': [],
                        'repr': args.repr, 'dims': dims, 'league': league.state_dict()},
                       os.path.join(out, 'latest.pt'))
        row = {'update': upd, 'wall_s': round(time.time() - t_start, 1), 'games': int(ep['games']),
               'decisions': int(ep['decisions']), 'games_per_s': round(ep['games'] / t_collect, 2),
               'dec_per_s': round(ep['decisions'] / t_collect, 1),
               'mean_len': round(ep['len'] / max(ep['games'], 1), 1), 'draws': int(ep['draws']),
               'errors': int(ep['errors']), 'lost_traj': int(ep['lost_traj'] + ep['lost_rows']),
               'wr_vs_pool': round(ep['pool_wins'] / ep['pool_games'], 3) if ep['pool_games'] else '',
               'pool_games': int(ep['pool_games']),
               **{k: round(agg[k], 5) for k in ('pol_loss', 'v_loss', 'entropy', 'approx_kl', 'clipfrac')},
               'explained_var': round(float(ev), 3)}
        mrow = {}
        for c in cells:
            d = mc[c]
            row[f'wr_{c}'] = round(d['w'] / d['n'], 3) if d['n'] else ''
            row[f'n_{c}'] = int(d['n'])
            row[f'dec_{c}'] = int(d['dec'])
            mrow[c] = dict(games=int(d['games']), pool_games=int(d['n']), pool_wins=d['w'],
                           seat1=[int(d['n_s1']), d['w_s1']], seat2=[int(d['n_s2']), d['w_s2']],
                           decisions=int(d['dec']))
        extra = {'league': {**league.log_row(), **league_act}, 'matchups': mrow}
        w.writerow(row); logf.flush()
        mj.write(json.dumps({**row, **extra, 't_collect': round(t_collect, 1),
                             't_learn': round(time.time() - t0 - t_collect, 1)}) + '\n'); mj.flush()
        cell_s = ' '.join(f'{c}={row[f"wr_{c}"]}/{row[f"n_{c}"]}' for c in cells)
        print(f'update={upd} games={row["games"]} gps={row["games_per_s"]} wr_vs_pool={row["wr_vs_pool"]} '
              f'| {cell_s} | kl={row["approx_kl"]} ent={row["entropy"]} ev={row["explained_var"]} '
              f'snapshot={league_act.get("snapshot")} reset={league_act.get("reset")}', flush=True)
    print('done', out, flush=True)


if __name__ == '__main__':
    main()
