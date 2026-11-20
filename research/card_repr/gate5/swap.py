"""Gate 5: card-swap counterfactuals (TESTPLAN gate 5). Extends the gate-4 harness (gate4/po.py) with the
rbenv edit `swap <obj> <cardkey>` (gate5/rbenv_gate5_ext.cpp: riftbound::becomeCard + Cleanup on resume).

  collect : positions from the cf_B2 policy's SAMPLED self-play (mirror of --deck), top-level main-phase
            decisions from turn 2, p per decision, <= per-game per game; each with a swap plan.
  run     : per position: screen (orig win in [lo, 1-lo] under the B2 policy), then the original and every swap
            are played out N times by the B2 policy AND N times by the id policy (both seats, SAMPLED, common
            seeds for every variant: paired differences); every net's root value / policy is recorded.
  respond : add more nets' root responses to a results file (variants rebuilt deterministically).

Swap plan per position (one target per kind, the same target for every pool -> paired pools):
  board: a unit on the board (either side; not facedown / token / attached) -> a unit card of the pool;
  hand : a card in the DECIDER's hand (the only hand the nets can see)    -> any card of the pool.
  pools: seen (Rengar main deck), unseen (Yi-only cards), synth (registered synthetic variants, never trained).
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'gate4'))
import po as P  # noqa: E402

SEEN_UNITS = ['noxus_hopeful', 'pit_rookie', 'inferna', 'grim_apothecary', 'kinkou_initiate', 'irresistible_faefolk',
              'first_mate', 'pakaa_cub', 'ferrous_forerunner', 'kai_sa_survivor', 'nidalee_cat_form', 'darius_trifarian',
              'pyke_dockside_butcher', 'rengar_trophy_hunter']
SEEN_OTHER = ['sabotage', 'punch_first', 'thrill_of_the_hunt', 'rampage']
UNSEEN_UNITS = ['lonely_poro', 'scuttle_crab', 'ruin_runner', 'master_yi_tempered', 'fiora_peerless']
UNSEEN_OTHER = ['charm', 'defy', 'discipline', 'en_garde', 'zhonya_s_hourglass', 'back_off', 'onslaught']
SYNTH_UNITS = ['synthetic:pit_rookie__might3', 'synthetic:pit_rookie__score1', 'synthetic:pit_rookie__score_alone',
               'synthetic:lonely_poro__draw2']
SYNTH_OTHER = ['synthetic:punch_first__amount4', 'synthetic:punch_first__kill', 'synthetic:punch_first__score1',
               'synthetic:sabotage__unit', 'synthetic:discipline__draw2', 'synthetic:defy__cost2']
VARIANTS = SYNTH_UNITS + SYNTH_OTHER
VSTR = ",".join(sorted(VARIANTS))
POOLS = {'seen': (SEEN_UNITS, SEEN_UNITS + SEEN_OTHER), 'unseen': (UNSEEN_UNITS, UNSEEN_UNITS + UNSEEN_OTHER),
         'synth': (SYNTH_UNITS, SYNTH_UNITS + SYNTH_OTHER)}
DECKS = {'rengar': 'rengar', 'yi': os.path.normpath(os.path.join(HERE, '..', '..', '..', 'decks', 'yi_rebuild.txt'))}
CK = {'B2': os.path.join(P.SP, 'runs', 'cf_B2', 'latest.pt'), 'id': os.path.join(P.SP, 'runs', 'cf_id', 'latest.pt')}


def plan(d, rng):
    dec = d['decider']
    objs = d['objs']
    board = [o for o in objs if o['unit'] and o['zone'] in ('base', 'bf') and not o['hidden'] and not o['token']]
    hand = [o for o in objs if o['zone'] == 'hand' and o['owner'] == dec and not o['key'].startswith('token')]
    E = []
    if board:
        t = board[rng.integers(len(board))]
        for pool, (units, _) in POOLS.items():
            c = [k for k in units if k != t['key']]
            k = c[rng.integers(len(c))]
            E.append(dict(kind='board', pool=pool, edit='swap %d %s' % (t['id'], k), new=k, old=t['key'],
                          side='own' if t['ctrl'] == dec else 'opp', obj=t['id'], zone=t['zone']))
    if hand:
        t = hand[rng.integers(len(hand))]
        for pool, (_, allk) in POOLS.items():
            c = [k for k in allk if k != t['key']]
            k = c[rng.integers(len(c))]
            E.append(dict(kind='hand', pool=pool, edit='swap %d %s' % (t['id'], k), new=k, old=t['key'],
                          side='own', obj=t['id'], zone='hand'))
    return E


def init_lib():
    from rbenv import RBEnv
    e = RBEnv(1, 'rengar', 'rengar', 1, variant=VSTR)   # registers the synthetic variants (one set per process)
    e.close() if hasattr(e, 'close') else None


def collect(a):
    import torch
    from rbenv import RBEnv
    from selfplay import act_rows
    torch.set_num_threads(2)
    dev = 'cpu'
    net = P.load_net(CK['B2'], 'B2', dev)
    rng = np.random.default_rng(a.seed)
    dk = DECKS[a.deck]
    env = RBEnv(a.n_envs, dk, dk, a.seed, variant=VSTR)
    cands, per, games = [], {}, 0
    t0 = time.time()
    while games < a.games:
        o = env.observe()
        rows = np.nonzero(o['need'])[0]
        for i in rows:
            if rng.random() < a.p and int(o['glob'][i][0]) >= 2 and P.env_resumable(env, i):
                s, h = P.env_hist(env, i)
                if per.get(s, 0) < a.per_game:
                    per[s] = per.get(s, 0) + 1
                    cands.append((s, h.tolist()))
        ch = np.zeros(a.n_envs, np.int64)
        if len(rows):
            ac, _, _ = act_rows(net, o, rows, dev, False)
            ch[rows] = ac
        env.step(ch)
        games += len(env.results())
    print('games', games, 'candidates', len(cands), '%.0fs' % (time.time() - t0), flush=True)
    out = []
    for s, h in cands:
        try:
            p = P.Po(s, h, n_envs=1, deck=dk, variant=VSTR)
        except RuntimeError as ex:
            print('replay failed', s, ex); continue
        d = p.describe()
        out.append(dict(seed=s, hist=h, deck=a.deck, turn=d['turn'], tp=d['turn_player'], decider=d['decider'],
                        scores=[x['score'] for x in d['players']], edits=plan(d, rng)))
        p.close()
    json.dump(out, open(a.out, 'w'))
    print('positions', len(out), 'edits', sum(len(x['edits']) for x in out))


def obj_name(info, oid):
    o = next((x for x in info['objs'] if x['id'] == oid), None)
    return o


def responses(po, recs, vidx, nets, dev):
    for r, v in zip(recs, vidx):
        if v is None:
            continue
        ro = po.root_obs(v)
        info = po.info(v)
        r['legal'] = info['legal'][:ro['nlegal']]
        # options that name each swap target (act[1] src / act[2] ability source -> token -> object)
        objs = {x.get("obj") for x in recs if x.get("obj") is not None}
        nl = len(r["legal"])
        refs = [{int(ro["tok_obj"][t]) for t in (ro["act"][0, a, 1], ro["act"][0, a, 2]) if 0 <= t < len(ro["tok_obj"])} for a in range(nl)]
        if r.get('obj') is not None:
            o = obj_name(info, r['obj'])
            r['post'] = o
            r['src_name'] = o['name'] if o else None
        r.setdefault('model', {})
        for name, net in nets.items():
            pr, val = P.net_eval(net, ro, np.array([0]), dev)
            r['model'][name] = dict(v=float(val[0]), p=pr[0, :len(r['legal'])].tolist())
            r["model"][name]["pobj"] = {str(ob): float(sum(pr[0, a] for a in range(nl) if ob in refs[a])) for ob in objs}


def build(p):
    po = P.Po(p['seed'], p['hist'], n_envs=p.get('_n_envs', 1), deck=DECKS[p['deck']], variant=VSTR)
    recs = [dict(kind='orig', pool='orig', edit='none', obj=None)] + [dict(e) for e in p['edits']]
    vidx = []
    for r in recs:
        v, err = po.add_variant(r['edit'])
        r['ok'] = v is not None; r['err'] = err
        vidx.append(v)
    return po, recs, vidx


def load_nets(spec, dev):
    nets = {}
    for s in spec.split(','):
        name, ck = s.split('=')
        rep = ck.split(':')[1] if ':' in ck else name.split('_')[0]
        nets[name] = P.load_net(ck.split(':')[0], rep, dev)
    return nets


def run(a):
    import torch
    torch.set_num_threads(int(os.environ.get('G5_THREADS', '2')))
    dev = 'cpu'
    pos = json.load(open(a.positions))
    k, m = (int(x) for x in a.shard.split('/'))
    sel = pos[k::m]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line); done.add((r['seed'], len(r['hist'])))
    pol = {'B2': P.load_net(CK['B2'], 'B2', dev), 'id': P.load_net(CK['id'], 'id', dev)}
    nets = dict(pol)
    t0 = time.time()
    seeds = [a.seed * 1000 + j for j in range(a.n)]
    for pi, p in enumerate(sel):
        if (p['seed'], len(p['hist'])) in done or not p['edits']:
            continue
        if a.max_minutes and time.time() - t0 > 60 * a.max_minutes:
            print('time limit', flush=True); break
        t1 = time.time()
        p = dict(p, _n_envs=a.n_envs)
        po, recs, vidx = build(p)
        dec = po.describe()['decider']
        v0 = vidx[0]
        sres = P.play(po, pol['B2'], [(v0, s) for s in seeds[:a.screen]], dev, greedy=False)
        sw = float(np.mean([1.0 if sres[(v0, s)] == dec else 0.0 for s in seeds[:a.screen]]))
        if not (a.screen_lo <= sw <= 1 - a.screen_lo):
            with open(a.out, 'a') as f:
                f.write(json.dumps(dict(seed=p['seed'], hist=p['hist'], deck=p['deck'], skipped=True, screen_win=sw)) + '\n')
            po.close()
            print('[%5.0fs] skip %s seed %d: screen %.2f' % (time.time() - t0, p['deck'], p['seed'], sw), flush=True)
            continue
        responses(po, recs, vidx, nets, dev)
        for pname, pnet in pol.items():
            jobs = [(v, s) for v in vidx if v is not None for s in seeds]
            res = P.play(po, pnet, jobs, dev, greedy=False)
            for r, v in zip(recs, vidx):
                if v is None:
                    continue
                r['wins_' + pname] = [1 if res[(v, s)] == dec else (0 if res[(v, s)] in (0, 1, 2) else -1) for s in seeds]
        po.close()
        out = dict(screen_win=sw, seed=p['seed'], hist=p['hist'], deck=p['deck'], turn=p['turn'], decider=dec,
                   scores=p['scores'], n=a.n, variants=recs, secs=round(time.time() - t1, 1))
        with open(a.out, 'a') as f:
            f.write(json.dumps(out) + '\n')
        o = recs[0]
        print('[%5.0fs] pos %d/%d %s seed %d turn %d: %d variants (%d failed) orig B2 %.3f id %.3f, %.1fs' % (
            time.time() - t0, pi + 1, len(sel), p['deck'], p['seed'], p['turn'], len(recs), sum(not r['ok'] for r in recs),
            np.mean(o['wins_B2']), np.mean(o['wins_id']), time.time() - t1), flush=True)


def respond(a):
    import torch
    torch.set_num_threads(int(os.environ.get('G5_THREADS', '2')))
    dev = 'cpu'
    nets = load_nets(a.nets, dev)
    rows = [json.loads(l) for l in open(a.inp)]
    with open(a.out, 'w') as f:
        for r in rows:
            if not r.get('skipped'):
                p = dict(seed=r['seed'], hist=r['hist'], deck=r['deck'],
                         edits=[{k: v for k, v in x.items() if k in ('kind', 'pool', 'edit', 'new', 'old', 'side', 'obj', 'zone')}
                                for x in r['variants'][1:]])
                po, recs, vidx = build(p)
                tmp = [dict(obj=x.get('obj')) for x in r['variants']]
                responses(po, tmp, vidx, nets, dev)
                for x, t, v in zip(r['variants'], tmp, vidx):
                    if v is None:
                        continue
                    assert t['legal'] == x['legal'], 'legal list changed on rebuild'
                    x['model'].update(t['model'])
                po.close()
            f.write(json.dumps(r) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['collect', 'run', 'respond'])
    ap.add_argument('--deck', default='rengar')
    ap.add_argument('--out'); ap.add_argument('--inp'); ap.add_argument('--positions')
    ap.add_argument('--games', type=int, default=150); ap.add_argument('--n-envs', type=int, default=64)
    ap.add_argument('--p', type=float, default=0.08); ap.add_argument('--per-game', type=int, default=2)
    ap.add_argument('--seed', type=int, default=9100001)
    ap.add_argument('--n', type=int, default=256); ap.add_argument('--shard', default='0/1')
    ap.add_argument('--screen', type=int, default=32); ap.add_argument('--screen-lo', type=float, default=0.08)
    ap.add_argument('--max-minutes', type=float, default=0)
    ap.add_argument('--nets', default='')
    a = ap.parse_args()
    {'collect': collect, 'run': run, 'respond': respond}[a.mode](a)


if __name__ == '__main__':
    main()
