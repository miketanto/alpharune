"""No-search self-play PPO trainer for Riftbound (Rengar mirror, reward +1 win / -1 loss at game end).

  python3 selfplay.py --repr id --n-envs 256 --updates 200 --run-name id_s0
  python3 selfplay.py --env fake --repr B2 --n-envs 64 --updates 3 --run-name fake_test

Env: rbenv.RBEnv (SELFPLAY.md "Python interface contract"), or FakeEnv (--env fake) with the same interface.
Actor: N envs, one batched GPU forward per step for the current policy (+ one for the pool opponent).
Learner seat alternates per game; the other seat is the current policy (p=--p-self) or a past checkpoint.
Only the current policy's own decisions are trained on (both seats when it plays itself).
Writes selfplay/runs/<run-name>/{config.json, log.csv, metrics.jsonl, ckpt_*.pt, latest.pt}.
"""
import argparse, csv, json, math, os, random, sys, time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import Net, load_vocab  # noqa: E402
import probes  # noqa: E402

SP = os.path.normpath(os.path.join(HERE, '..'))           # research/card_repr/selfplay
REPO = os.path.normpath(os.path.join(HERE, '..', '..', '..', '..'))
OBS_KEYS = ('glob', 'tok', 'tok_mask', 'act', 'act_mask', 'pk', 'psrc')


def make_env(kind, n, seed, deck, opponent=None, variant=None):
    if kind.startswith('fake'):
        from fakeenv import FakeEnv
        return FakeEnv(n, deck, deck, seed, variant=variant, opponent=opponent, V=len(load_vocab()[0]),
                       learnable=kind == 'fake-learnable')
    sys.path.insert(0, os.path.join(SP, 'rbenv'))
    from rbenv import RBEnv
    return RBEnv(n, deck, deck, seed, variant=variant, autopay=True, opponent=opponent)


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
    if greedy:
        a = logp.argmax(-1)
    else:
        a = torch.multinomial(logp.exp().clamp(min=0), 1).squeeze(-1)
    lp = logp.gather(1, a[:, None]).squeeze(1)
    return a.cpu().numpy(), lp.cpu().numpy(), v.cpu().numpy()


class Ring:
    """Flat ring buffer of decisions (int16 obs + float stats)."""

    def __init__(self, cap, dims):
        G, T, Fd, A, K = (int(dims[k]) for k in ('G', 'T', 'F', 'A', 'K'))
        self.cap = cap
        self.glob = np.zeros((cap, G), np.int16)
        self.tok = np.zeros((cap, T, Fd), np.int8)   # int8: values clipped to [-128, 127]
        self.tok_mask = np.zeros((cap, T), bool)
        self.act = np.zeros((cap, A, K), np.int8)
        self.act_mask = np.zeros((cap, A), bool)
        self.pk = np.zeros(cap, np.int16)
        self.psrc = np.zeros(cap, np.int16)
        self.a = np.zeros(cap, np.int64)
        self.logp = np.zeros(cap, np.float32)
        self.v = np.zeros(cap, np.float32)
        self.adv = np.zeros(cap, np.float32)
        self.ret = np.zeros(cap, np.float32)
        self.head = 0       # total rows ever written
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


def play_eval(net, env, n_games, dev, greedy):
    """Network is seat 1; the env plays seat 2 in C++. Returns (win rate, games, draws, errors)."""
    net.eval()
    env.results()
    wins = games = draws = errs = 0
    t0 = time.time()
    while games < n_games and time.time() - t0 < 600:
        obs = env.observe()
        rows = np.nonzero(obs['need'])[0]
        ch = np.zeros(len(obs['need']), np.int64)
        if len(rows):
            a, _, _ = act_rows(net, obs, rows, dev, greedy)
            ch[rows] = a
        env.step(ch)
        for r in env.results():
            if r.get('error'):
                errs += 1
                continue
            games += 1
            wins += r['winner'] == 1
            draws += r['winner'] == 0
    net.train()
    return (wins / games if games else float('nan')), games, draws, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repr', default='id', choices=['id', 'random', 'B2', 'B'])
    ap.add_argument('--env', default='real', choices=['real', 'fake', 'fake-learnable'])
    ap.add_argument('--deck', default=os.path.join(REPO, 'decks', 'rengar.txt'))
    ap.add_argument('--variants', default='', help='comma-separated variant names registered in the env (mixed-variant decks)')
    ap.add_argument('--init', default='', help='start from the weights of this checkpoint (fine-tuning; fresh optimizer)')
    ap.add_argument('--ask-head', action='store_true', help='direct asking-card -> option-score path (model.py)')
    ap.add_argument('--n-envs', type=int, default=256)
    ap.add_argument('--updates', type=int, default=100)
    ap.add_argument('--samples-per-update', type=int, default=16384, help='learner decisions from finished games')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--minibatch', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--lam', type=float, default=0.95)
    ap.add_argument('--vf-coef', type=float, default=0.5)
    ap.add_argument('--ent-coef', type=float, default=0.01)
    ap.add_argument('--max-grad-norm', type=float, default=0.5)
    ap.add_argument('--p-self', type=float, default=0.5, help='P(opponent seat = current policy) per game')
    ap.add_argument('--pool-every', type=int, default=5, help='add a snapshot to the opponent pool every N updates')
    ap.add_argument('--pool-size', type=int, default=20)
    ap.add_argument('--eval-every', type=int, default=10)
    ap.add_argument('--eval-games', type=int, default=200)
    ap.add_argument('--eval-envs', type=int, default=64)
    ap.add_argument('--eval-ismcts-games', type=int, default=0, help='games vs ismcts:sims=25 per eval (0 = off)')
    ap.add_argument('--eval-greedy', action='store_true')
    ap.add_argument('--ckpt-every', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--max-minutes', type=float, default=0, help='stop after this wall time (0 = no limit)')
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = os.path.join(SP, 'runs', args.run_name)
    os.makedirs(out, exist_ok=True)
    t_start = time.time()

    VARIANTS = [v for v in args.variants.split(',') if v] or None
    env = make_env(args.env, args.n_envs, args.seed, args.deck, variant=VARIANTS)
    dims = env.dims
    vocab, vocab_src = load_vocab()
    assert len(vocab) == int(dims['V']), f'vocab {len(vocab)} != dims V {dims["V"]} ({vocab_src})'
    net = Net(dims, vocab, args.repr, ask_head=args.ask_head).to(dev)
    pool_net = Net(dims, vocab, args.repr, ask_head=args.ask_head).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    nparam = sum(p.numel() for p in net.parameters())
    cfg = dict(vars(args), dims={k: int(v) for k, v in dims.items()}, vocab_src=vocab_src, device=str(dev),
               params=nparam, missing_repr_keys=net.missing_keys)
    json.dump(cfg, open(os.path.join(out, 'config.json'), 'w'), indent=1)
    print('config', json.dumps(cfg), flush=True)

    if args.init:
        ck0 = torch.load(args.init, map_location=dev)
        net.load_state_dict(ck0['model'])
        print('initialised from', args.init, flush=True)
    start_update = 0
    pool = []  # list of (update, cpu state dict)
    if args.resume and os.path.exists(os.path.join(out, 'latest.pt')):
        ck = torch.load(os.path.join(out, 'latest.pt'), map_location=dev)
        net.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        start_update = ck['update']; pool = ck.get('pool', [])
        print('resumed at update', start_update, flush=True)

    cap = int(args.samples_per_update * 1.5 + args.n_envs * 300)
    ring = Ring(cap, dims)
    print(f'ring capacity {cap} decisions, {ring.nbytes / 2**20:.0f} MiB', flush=True)

    N = args.n_envs
    rng = np.random.default_rng(args.seed + 17)
    learner_seat = np.where(np.arange(N) % 2 == 0, 1, 2)
    opp_self = rng.random(N) < args.p_self   # opponent seat is current policy
    pool_id = -1                             # snapshot loaded into pool_net
    traj = defaultdict(list)                 # (env, seat) -> ring ids
    ready = []                               # ring ids with adv/ret computed
    stats = defaultdict(float)
    eval_envs = {}

    logf = open(os.path.join(out, 'log.csv'), 'a', newline='')
    cols = ['update', 'wall_s', 'games', 'decisions', 'games_per_s', 'dec_per_s', 'mean_len', 'draws', 'errors',
            'lost_traj', 'wr_vs_pool', 'pool_games', 'pol_loss', 'v_loss', 'entropy', 'approx_kl', 'clipfrac',
            'explained_var', 'eval_random_wr', 'eval_ismcts_wr', 'pool_size']
    w = csv.DictWriter(logf, fieldnames=cols)
    if logf.tell() == 0:
        w.writeheader()
    mj = open(os.path.join(out, 'metrics.jsonl'), 'a')

    def choose_pool():
        nonlocal pool_id
        if pool:
            k = rng.integers(len(pool))
            pool_net.load_state_dict(pool[k][1]); pool_net.eval()
            pool_id = pool[k][0]

    choose_pool()
    for u in range(start_update, args.updates):
        if args.max_minutes and time.time() - t_start > args.max_minutes * 60:
            print('time limit reached', flush=True); break
        t0 = time.time()
        ep = defaultdict(float)
        net.eval()
        # ---------------------------------------------------------------- collect
        while len(ready) < args.samples_per_update:
            obs = env.observe()
            need = obs['need']
            rows = np.nonzero(need)[0]
            ch = np.zeros(N, np.int64)
            if len(rows):
                seat = obs['seat'][rows]
                mine = (seat == learner_seat[rows]) | opp_self[rows] | (not pool)
                r_cur, r_pool = rows[mine], rows[~mine]
                if len(r_cur):
                    a, lp, v = act_rows(net, obs, r_cur, dev)
                    ch[r_cur] = a
                    ids = ring.put(obs, r_cur, a, lp, v)
                    for e, s, i in zip(r_cur.tolist(), obs['seat'][r_cur].tolist(), ids.tolist()):
                        traj[(e, s)].append(i)
                if len(r_pool):
                    a, _, _ = act_rows(pool_net, obs, r_pool, dev)
                    ch[r_pool] = a
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
                if not opp_self[e] and pool and not bad:
                    ep['pool_games'] += 1
                    ep['pool_wins'] += win == learner_seat[e]
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
                learner_seat[e] = 3 - learner_seat[e]
                opp_self[e] = rng.random() < args.p_self
        t_collect = time.time() - t0
        # drop ready rows that were overwritten (should not happen with the ring size above)
        rid = np.array(ready); ready = []
        ok = ring.alive(rid)
        ep['lost_rows'] = int((~ok).sum()); rid = rid[ok]

        # ---------------------------------------------------------------- PPO update
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

        # ---------------------------------------------------------------- pool / ckpt / eval
        if upd % args.pool_every == 0:
            pool.append((upd, {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}))
            pool[:] = pool[-args.pool_size:]
        choose_pool()
        if upd % args.ckpt_every == 0 or upd == args.updates:
            torch.save({'model': net.state_dict(), 'update': upd, 'repr': args.repr, 'dims': dims},
                       os.path.join(out, f'ckpt_{upd:05d}.pt'))
            torch.save({'model': net.state_dict(), 'opt': opt.state_dict(), 'update': upd, 'pool': pool,
                        'repr': args.repr, 'dims': dims}, os.path.join(out, 'latest.pt'))
        row = {'update': upd, 'wall_s': round(time.time() - t_start, 1), 'games': int(ep['games']),
               'decisions': int(ep['decisions']), 'games_per_s': round(ep['games'] / t_collect, 2),
               'dec_per_s': round(ep['decisions'] / t_collect, 1),
               'mean_len': round(ep['len'] / max(ep['games'], 1), 1), 'draws': int(ep['draws']),
               'errors': int(ep['errors']), 'lost_traj': int(ep['lost_traj'] + ep['lost_rows']),
               'wr_vs_pool': round(ep['pool_wins'] / ep['pool_games'], 3) if ep['pool_games'] else '',
               'pool_games': int(ep['pool_games']),
               **{k: round(agg[k], 5) for k in ('pol_loss', 'v_loss', 'entropy', 'approx_kl', 'clipfrac')},
               'explained_var': round(float(ev), 3), 'eval_random_wr': '', 'eval_ismcts_wr': '',
               'pool_size': len(pool)}
        extra = {}
        if args.eval_every and (upd % args.eval_every == 0 or upd == args.updates):
            specs = [('random', 'random-player', args.eval_games)]
            if args.eval_ismcts_games:
                specs.append(('ismcts', 'ismcts:sims=25', args.eval_ismcts_games))
            for name, spec, ng in specs:
                # a fresh env per eval (fixed seeds), so no game started by an older policy is counted
                eenv = make_env(args.env, min(args.eval_envs, ng), 10**6 + args.seed, args.deck, opponent=spec, variant=VARIANTS)
                te = time.time()
                wr, ng_, dr, er = play_eval(net, eenv, ng, dev, args.eval_greedy)
                if hasattr(eenv, 'close'):
                    eenv.close()
                del eenv
                row[f'eval_{name}_wr'] = round(wr, 3)
                extra[f'eval_{name}'] = dict(wr=wr, games=ng_, draws=dr, errors=er, secs=round(time.time() - te, 1))
            extra['probes'] = probes.run_probes(net, dev)
        w.writerow(row); logf.flush()
        mj.write(json.dumps({**row, **extra, 't_collect': round(t_collect, 1),
                             't_learn': round(time.time() - t0 - t_collect, 1), 'pool_id': pool_id}) + '\n'); mj.flush()
        print(' '.join(f'{k}={v}' for k, v in row.items() if v != ''), extra if extra else '', flush=True)
    print('done', out, flush=True)


if __name__ == '__main__':
    main()
