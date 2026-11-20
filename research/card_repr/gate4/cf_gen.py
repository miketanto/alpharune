"""Counterfactual-pair TRAINING data for the value head (gate 4 follow-up, 2026-11-19).

Positions: the playout net's own SAMPLED self-play games (fresh game seeds, disjoint from every gate-4 test
position: those use 7770001.. and 8880001..). Edits: random draws from the TRAINING edit families applied to
arbitrary units / cards (not only the gate-4 key units):
  assault_give   kw_give <unit> A v (v in 1..3), any unit EXCEPT an enemy unit in base (= gate-4 null cell)
  assault_remove kw_remove <unit> A
  might          might <unit> d, d in +-1..3
  to_base        to_base <unit at a battlefield> (own or enemy)
  to_bf          to_bf <own unit in base> <slot>
  unhide         unhide <facedown card>
  played         played 0/1, only when a reader (Noxus Hopeful in hand, Darius in hand / on board) exists
                 (with no reader it is the gate-4 null cell)
  ss             Star Spring stamp of the turn player toggled
--shield-test generates ONLY the held-out family (Shield given to / removed from arbitrary units), as a test set.
HELD OUT from training (never generated without --shield-test): every Shield edit (kw_give / kw_remove ... S), and the null-edit cells.
Each variant (original + up to --k edits) is played out --n times with the SAMPLED policy on both seats, the
same seeds for every variant (common random numbers). A --screen-playout screen of the original skips the
edits of decided positions (the original is still stored as a level sample).

Output: <out>_NNN.npz chunks: root observations of every variant + meta (position id, family, edit, win, n,
paired diff vs the original, se of the diff).

usage: python3 cf_gen.py <ckpt> --repr B2 --out DIR/w0 --base-seed 5550001 --minutes 35
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import po as P  # noqa: E402
from rbenv import RBEnv  # noqa: E402

TEST_SEED_RANGES = [(7770001, 7779999), (8880001, 8889999)]
FAMILIES = ('assault_give', 'assault_remove', 'might', 'to_base', 'to_bf', 'unhide', 'played', 'ss')


def candidate_edits(d, rng):
    tp = d['turn_player']; opp = 3 - tp
    units = [o for o in d['objs'] if o['unit'] and o['zone'] in ('base', 'bf')]
    own = [o for o in units if o['ctrl'] == tp]
    hand = [o for o in d['objs'] if o['zone'] == 'hand' and o['owner'] == tp]
    C = {f: [] for f in FAMILIES}
    for o in units:
        if o['assault'] == 0 and not (o['ctrl'] == opp and o['zone'] == 'base'):
            C['assault_give'].append('kw_give %d A %d' % (o['id'], rng.integers(1, 4)))
        if o['assault'] > 0:
            C['assault_remove'].append('kw_remove %d A' % o['id'])
        for dd in (-3, -2, -1, 1, 2, 3):
            if o['base_might'] + dd >= 0:
                C['might'].append('might %d %d' % (o['id'], dd))
        if o['zone'] == 'bf':
            C['to_base'].append('to_base %d' % o['id'])
    slots = [b['slot'] for b in d['bfs']]
    for o in own:
        if o['zone'] == 'base' and slots:
            C['to_bf'].append('to_bf %d %d' % (o['id'], slots[rng.integers(len(slots))]))
    for b in d['bfs']:
        for fid in b['facedown']:
            C['unhide'].append('unhide %d' % fid)
    kh = {o['key'] for o in hand}; kb = {o['key'] for o in own}
    if 'noxus_hopeful' in kh or 'darius_trifarian' in kh | kb:
        cp = d['players'][tp - 1]['cards_played']
        C['played'].append('played %d' % (1 if cp == 0 else 0))
    for b in d['bfs']:
        if b['name'] == 'Star Spring':
            used = b.get('ss_used_p%d' % tp, 0)
            C['ss'].append('ss %d %d' % (b['slot'], 0 if used else 1))
    return C


def shield_edits(d, rng):
    """TEST-ONLY held-out family: Shield given (v 1..3, 'this turn') to / removed from arbitrary units."""
    units = [o for o in d['objs'] if o['unit'] and o['zone'] in ('base', 'bf')]
    C = {'shield_give': [], 'shield_remove': []}
    for o in units:
        if o['shield'] == 0:
            C['shield_give'].append('kw_give %d S %d' % (o['id'], rng.integers(1, 4)))
        else:
            C['shield_remove'].append('kw_remove %d S' % o['id'])
    return C


def sample_edits(C, k, rng):
    """Up to k distinct edits: families drawn uniformly among the non-empty ones, then an edit within it."""
    pools = {f: list(rng.permutation(v)) for f, v in C.items() if v}
    out = []
    while len(out) < k and pools:
        f = list(pools)[rng.integers(len(pools))]
        out.append((f, str(pools[f].pop())))
        if not pools[f]:
            del pools[f]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt'); ap.add_argument('--repr', default='B2')
    ap.add_argument('--out', required=True)
    ap.add_argument('--base-seed', type=int, required=True)
    ap.add_argument('--minutes', type=float, default=35)
    ap.add_argument('--n', type=int, default=48)
    ap.add_argument('--k', type=int, default=7)
    ap.add_argument('--screen', type=int, default=24)
    ap.add_argument('--screen-lo', type=float, default=0.05)
    ap.add_argument('--game-envs', type=int, default=32)
    ap.add_argument('--p', type=float, default=0.12)
    ap.add_argument('--per-game', type=int, default=3)
    ap.add_argument('--chunk', type=int, default=40)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--shield-test', action='store_true', help='generate ONLY the held-out Shield family (test set)')
    a = ap.parse_args()
    import torch
    from selfplay import act_rows
    torch.set_num_threads(a.threads)
    dev = 'cpu'
    for lo, hi in TEST_SEED_RANGES:
        assert not (lo <= a.base_seed + 20000 and a.base_seed <= hi), 'seed range overlaps the gate-4 test seeds'
    net = P.load_net(a.ckpt, a.repr, dev)
    rng = np.random.default_rng(a.base_seed)
    env = RBEnv(a.game_envs, 'rengar', 'rengar', a.base_seed)
    t0 = time.time()
    seeds = [a.base_seed * 10 + j for j in range(a.n)]
    buf = {k: [] for k in ('glob', 'tok', 'tok_mask', 'act', 'act_mask', 'pk', 'psrc')}
    meta = []
    chunk_id, npos, nvar, per = 0, 0, 0, {}

    def flush():
        nonlocal chunk_id, buf, meta
        if not meta:
            return
        arr = {k: np.concatenate(v) for k, v in buf.items()}
        arr['glob'] = arr['glob'].astype(np.int16)
        arr['tok'] = np.clip(arr['tok'], -32768, 32767).astype(np.int16)
        arr['act'] = np.clip(arr['act'], -32768, 32767).astype(np.int16)
        np.savez_compressed('%s_%03d.npz' % (a.out, chunk_id), meta=np.array(json.dumps(meta)), **arr)
        chunk_id += 1
        buf = {k: [] for k in buf}
        meta = []

    while time.time() - t0 < 60 * a.minutes:
        # ---- gather candidates from fresh self-play games (sampled)
        cands = []
        while len(cands) < a.chunk:
            o = env.observe()
            rows = np.nonzero(o['need'])[0]
            for i in rows:
                if rng.random() < a.p and int(o['glob'][i][0]) >= 2 and P.env_resumable(env, i):
                    s, h = P.env_hist(env, i)
                    assert not any(lo <= s <= hi for lo, hi in TEST_SEED_RANGES)
                    if per.get(s, 0) < a.per_game:
                        per[s] = per.get(s, 0) + 1
                        cands.append((s, h.tolist()))
            ch = np.zeros(a.game_envs, np.int64)
            if len(rows):
                ac, _, _ = act_rows(net, o, rows, dev, False)
                ch[rows] = ac
            env.step(ch)
            env.results()
        # ---- play out
        for s, h in cands:
            if time.time() - t0 > 60 * a.minutes:
                break
            try:
                po = P.Po(s, h, n_envs=a.screen)
            except RuntimeError:
                continue
            d = po.describe()
            dec = d['decider']
            v0, _ = po.add_variant('none')
            sres = P.play(po, net, [(v0, x) for x in seeds[:a.screen]], dev, greedy=False)
            sw = [1.0 if sres[(v0, x)] == dec else 0.0 for x in seeds[:a.screen] if sres[(v0, x)] >= 0]
            ro0 = po.root_obs(v0)
            po.close()
            pid = '%d_%d' % (s, len(h))
            if not sw:
                continue
            sw = float(np.mean(sw))
            if not (a.screen_lo <= sw <= 1 - a.screen_lo):
                for k in buf:
                    buf[k].append(ro0[k])
                meta.append(dict(pid=pid, fam='orig', edit='none', win=sw, n=a.screen, d=0.0, dse=0.0, screened=1, turn=d['turn']))
                npos += 1; nvar += 1
                continue
            E = sample_edits(shield_edits(d, rng) if a.shield_test else candidate_edits(d, rng), a.k, rng)
            po = P.Po(s, h, n_envs=(1 + len(E)) * a.n)
            vs = [('orig', 'none')] + E
            vidx = []
            for f, ed in vs:
                v, err = po.add_variant(ed)
                vidx.append(v)
            if vidx[0] is None:
                po.close(); continue
            jobs = [(v, x) for v in vidx if v is not None for x in seeds]
            res = P.play(po, net, jobs, dev, greedy=False)
            wo = np.array([1.0 if res[(vidx[0], x)] == dec else (0.0 if res[(vidx[0], x)] >= 0 else np.nan) for x in seeds])
            for (f, ed), v in zip(vs, vidx):
                if v is None:
                    continue
                we = np.array([1.0 if res[(v, x)] == dec else (0.0 if res[(v, x)] >= 0 else np.nan) for x in seeds])
                m = ~np.isnan(we) & ~np.isnan(wo)
                dd = we[m] - wo[m]
                ro = po.root_obs(v)
                if int(ro['seat'][0]) != int(ro0['seat'][0]):
                    continue
                for k in buf:
                    buf[k].append(ro[k])
                meta.append(dict(pid=pid, fam=f, edit=ed, win=float(np.nanmean(we)), n=int(m.sum()), d=float(dd.mean()),
                                 dse=float(dd.std(ddof=1) / np.sqrt(len(dd))) if len(dd) > 1 else 0.0, screened=0, turn=d['turn']))
                nvar += 1
            po.close()
            npos += 1
        flush()
        print('[%5.0fs] positions %d variants %d chunks %d' % (time.time() - t0, npos, nvar, chunk_id), flush=True)
    flush()
    print('done', npos, nvar, flush=True)


if __name__ == '__main__':
    main()
