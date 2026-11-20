"""Gate 4 step B1: position snapshots from the trained B2 policy's own self-play games, and the edit plan.

Self-play: the B2 net on both seats, SAMPLED (the training actors' policy; position diversity).
A candidate = a top-level main-phase decision (resumable), turn >= 2, kept with probability --p, at most
--per-game per game. Each candidate is replayed in a Po handle and the edit plan is built from the engine's
description of the position and the root observation (which units would start a combat as attacker).

usage: python3 collect.py <ckpt> --out positions.json [--games 400] [--p 0.12]
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import po as P  # noqa: E402
from rbenv import RBEnv  # noqa: E402

MAIN_PK = None  # filled from the first resumable decision


def attack_info(ro, d):
    """From a root observation: TP units that some legal Standard Move would send into a combat as attacker,
    and the enemy units at those destinations."""
    act, am, to = ro['act'][0], ro['act_mask'][0], ro['tok_obj']
    att, dst = set(), set()
    for a in np.nonzero(am)[0]:
        r = act[a]
        if r[47] == 1:
            for j in range(12, 16):
                if r[j] >= 0 and to[r[j]] >= 0:
                    att.add(int(to[r[j]]))
            if r[30] >= 1:
                dst.add(int(r[30]) - 1)
    tp = d['turn_player']
    opp = 3 - tp
    defs = {o['id'] for o in d['objs'] if o['unit'] and o['zone'] == 'bf' and o['ctrl'] == opp and o['loc'] in dst}
    return att, defs


def plan_edits(d, ro):
    tp = d['turn_player']; opp = 3 - tp
    objs = {o['id']: o for o in d['objs']}
    units = [o for o in d['objs'] if o['unit'] and o['zone'] in ('base', 'bf')]
    own = [o for o in units if o['ctrl'] == tp]
    oppu = [o for o in units if o['ctrl'] == opp]
    hand = [o for o in d['objs'] if o['zone'] == 'hand' and o['owner'] == tp]
    att, defs = attack_info(ro, d)
    E = []   # (cell, edit, orient, note)

    def top(lst):
        return max(lst, key=lambda o: (o['might'], -o['id'])) if lst else None
    attu = [objs[i] for i in att if i in objs]
    defu = [objs[i] for i in defs]
    a = top([o for o in attu if o['assault'] == 0])
    if a:
        E.append(('assault+:attacker', 'kw_give %d A 2' % a['id'], 1, a['key']))
        E.append(('shield+:attacker', 'kw_give %d S 2' % a['id'], 1, a['key']))
    dd = top([o for o in defu if o['assault'] == 0])
    if dd:
        E.append(('assault+:defender', 'kw_give %d A 2' % dd['id'], 1, dd['key']))
    dd = top([o for o in defu if o['shield'] == 0])
    if dd:
        E.append(('shield+:defender', 'kw_give %d S 2' % dd['id'], 1, dd['key']))
    # Inferna's Assault: removed this turn, when Inferna can attack now vs when it cannot
    inf = [o for o in units if o['key'] == 'inferna' and o['assault'] > 0]
    ia = [o for o in inf if o['id'] in att]
    ii = [o for o in inf if o['id'] not in att]
    if ia:
        E.append(('inferna_assault-:can_attack', 'kw_remove %d A' % ia[0]['id'], -1, 'inferna'))
    if ii:
        E.append(('inferna_assault-:cannot_attack', 'kw_remove %d A' % ii[0]['id'], -1,
                  'inferna own' if ii[0]['ctrl'] == tp else 'inferna opp'))
    # hidden card present vs absent (returned to its owner's hand)
    for b in d['bfs']:
        for fid in b['facedown']:
            o = objs.get(fid)
            if o:
                E.append(('hidden-:%s' % ('own' if o['owner'] == tp else 'opp'), 'unhide %d' % fid, -1, o['key']))
                break
    # Legion / Darius / null: cards played this turn >= 1 (on) vs 0 (off)
    cp = d['players'][tp - 1]['cards_played']
    edit, orient = ('played 1', 1) if cp == 0 else ('played 0', -1)
    keys_hand = {o['key'] for o in hand}
    keys_board = {o['key'] for o in own}
    if 'noxus_hopeful' in keys_hand:
        E.append(('legion:noxus_in_hand', edit, orient, 'noxus_hopeful'))
    if 'darius_trifarian' in keys_hand or 'darius_trifarian' in keys_board:
        E.append(('darius_second_card:%s' % ('board' if 'darius_trifarian' in keys_board else 'hand'), edit, orient, 'darius'))
    if 'noxus_hopeful' not in keys_hand and 'darius_trifarian' not in keys_hand | keys_board:
        E.append(('null:played_no_reader', edit, orient, ''))
    # Star Spring stamp of the turn player: unused (on) vs used
    for b in d['bfs']:
        if b['name'] == 'Star Spring':
            used = b.get('ss_used_p%d' % tp, 0)
            here = [o for o in own if o['zone'] == 'bf' and o['loc'] == b['slot']]
            unit_in_hand = any(o['unit'] for o in hand)
            rel = b['controller'] == tp and here and unit_in_hand
            E.append(('starspring_unused:%s' % ('relevant' if rel else 'irrelevant'),
                      'ss %d %d' % (b['slot'], 0 if used else 1), 1 if used else -1, ''))
    # Might +-2 on key units
    k = top([o for o in own if o['zone'] == 'bf'] or own)
    if k:
        E.append(('might+2:own_key', 'might %d 2' % k['id'], 1, k['key']))
        if k['base_might'] >= 3:
            E.append(('might-2:own_key', 'might %d -2' % k['id'], 1, k['key']))
    k = top([o for o in oppu if o['zone'] == 'bf'] or oppu)
    if k:
        E.append(('might+2:opp_key', 'might %d 2' % k['id'], 1, k['key']))
    # Context: the same unit at a battlefield vs in base
    atbf = [o for o in own if o['zone'] == 'bf']
    if atbf:
        u = top(atbf)
        E.append(('location:bf_to_base', 'to_base %d' % u['id'], 1, u['key']))
    # Kinkou Initiate in hand: other units' total Might crosses 5
    if 'kinkou_initiate' in keys_hand and own:
        tot = sum(o['might'] for o in own)
        lo = min(own, key=lambda o: o['might'])
        if 3 <= tot <= 4:
            E.append(('kinkou_condition:false_to_true', 'might %d %d' % (lo['id'], 5 - tot), 1, 'kinkou'))
        elif 5 <= tot <= 6 and lo['base_might'] >= tot - 4:
            E.append(('kinkou_condition:true_to_false', 'might %d %d' % (lo['id'], 4 - tot), 1, 'kinkou'))
    # Null controls
    nb = [o for o in oppu if o['zone'] == 'base' and o['assault'] == 0]
    if nb:
        E.append(('null:assault_opp_unit_in_base', 'kw_give %d A 2' % nb[0]['id'], 1, nb[0]['key']))
    ob = [o for o in own if o['zone'] == 'base' and o['exhausted'] and o['shield'] == 0 and o['id'] not in att]
    if ob:
        E.append(('null:shield_own_exhausted_base_unit', 'kw_give %d S 2' % ob[0]['id'], 1, ob[0]['key']))
    return E, sorted(att), sorted(defs)


RARE = ('inferna', 'hidden', 'legion', 'darius', 'kinkou', 'starspring_unused:relevant', 'assault+:attacker')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt'); ap.add_argument('--repr', default='B2')
    ap.add_argument('--out', required=True)
    ap.add_argument('--games', type=int, default=400)
    ap.add_argument('--n-envs', type=int, default=64)
    ap.add_argument('--p', type=float, default=0.12)
    ap.add_argument('--per-game', type=int, default=3)
    ap.add_argument('--seed', type=int, default=7770001)
    ap.add_argument('--min-turn', type=int, default=2)
    a = ap.parse_args()
    import torch
    from selfplay import act_rows
    torch.set_num_threads(2)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = P.load_net(a.ckpt, a.repr, dev)
    rng = np.random.default_rng(a.seed)
    env = RBEnv(a.n_envs, 'rengar', 'rengar', a.seed)
    cands, per, games = [], {}, 0
    gseed = {}
    t0 = time.time()
    while games < a.games:
        o = env.observe()
        rows = np.nonzero(o['need'])[0]
        for i in rows:
            if rng.random() < a.p and env_ok(env, i, o, a.min_turn):
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
            p = P.Po(s, h, n_envs=1)
        except RuntimeError as ex:
            print('replay failed', s, ex); continue
        d = p.describe()
        v, err = p.add_variant('none')
        if v is None:
            print('orig variant failed', err); continue
        ro = p.root_obs(v)
        E, att, defs = plan_edits(d, ro)
        out.append(dict(seed=s, hist=h, turn=d['turn'], tp=d['turn_player'], scores=[x['score'] for x in d['players']],
                        n_legal=len(d['legal']), attackers=att, defenders=defs,
                        edits=[dict(cell=c, edit=e, orient=r, note=n) for c, e, r, n in E]))
        p.close()
    json.dump(out, open(a.out, 'w'))
    from collections import Counter
    c = Counter(e['cell'] for x in out for e in x['edits'])
    print('positions', len(out))
    for k, n in sorted(c.items()):
        print('  %-45s %d' % (k, n))


def env_ok(env, i, o, min_turn):
    # glob[0] is the turn number (SCHEMA.txt g)
    return int(o['glob'][i][0]) >= min_turn and P.env_resumable(env, i)


if __name__ == '__main__':
    main()
