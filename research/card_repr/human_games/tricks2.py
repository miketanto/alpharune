"""1. equip: choosing which Equipment to attach, among UNATTACHED copies in base only (identity is the only difference that
   matters): probability by card, and whether the observation gives the cards different keys.
2. search: the network with search vs the same network greedy, Azir mirror, search on alternating seats. Attack
   opportunities (ready Might > defenders at a battlefield) taken, and attacks won, by the facedown card there; both seats.
python3 tricks2.py equip|search N [sims]
"""
import json, os, re, sys
from multiprocessing import Pool
import tricks as T   # sets the environment and paths
CKPT = T.CK if os.path.isabs(T.CK) else None


def net():
    import playweb
    if T._A is None:
        T._A = playweb.NetAgent(os.path.join(playweb.SELFPLAY, 'runs', T.CK))
    return T._A


def equip(seed):
    import playweb
    net()
    dk = os.path.join(T.DECKS, 'azir.txt')
    g = playweb.Game(dk, dk, 1, seed)
    out = []
    while True:
        st = g.status()
        if st['done'] or not st['need']:
            break
        o = g.observe()
        p, v = T.ev(o)
        vw = g.view(st['seat'], 1 << 40)
        d = vw['decision']
        if d['kind'] == 'target' and 'Equipment' in (d.get('question') or ''):
            gear = {x['id']: x for x in vw['board'] if x['loc'] == 'base' and x['ctrl'] == 1 and x.get('attached_to') is None}
            q = {}
            for x in d['options']:
                if len(x['objs']) == 1 and x['objs'][0] in gear:
                    q.setdefault(gear[x['objs'][0]]['name'], []).append(float(p[x['index']]))
            if len(q) >= 2:
                avg = {k: sum(x) / len(x) for k, x in q.items()}      # per copy
                s = sum(avg.values())
                avg = {k: x / s for k, x in avg.items()}
                out.append(avg)
        g.step(int(p.argmax()))
    t = o['tok'][0]
    keys = sorted({int(k) for k in t[(t[:, 0] == 1) & o['tok_mask'][0], 1]})
    g.close()
    return out, keys


def scan(lg, vw_by_seat, rec):
    pass


def search(a):
    import playweb
    i, seed, sims = a
    A = net()
    dk = os.path.join(T.DECKS, 'azir.txt')
    ss = 1 + i % 2
    S = playweb.SearchAgent(T.CK, dk, dk, seed, ss, sims)
    g = playweb.Game(dk, dk, 3 - ss, seed)
    opp, att = [], []
    open_c = None
    nlog = 0
    nrow = 0
    import numpy as np, torch

    def before(seat):
        """Attack opportunities of a main-phase decision of `seat`: per battlefield (dest, ready might, defenders, facedown)."""
        vw = g.view(seat, 1 << 40)
        d = vw['decision']
        if not d or d['kind'] != 'main phase action':
            return None, vw
        res = {}
        for b in vw['battlefields']:
            L = f"bf{b['slot']}"
            dfn = sum(u['might'] for u in vw['board'] if u['loc'] == L and u['ctrl'] == 2 and u.get('type') == 'Unit')
            n = sum(u['loc'] == L and u['ctrl'] == 2 and u.get('type') == 'Unit' for u in vw['board'])
            ks = [x['index'] for x in d['options'] if x['text'].startswith('move') and x.get('dest') == L]
            if not n or not ks:
                continue
            am = sum(u['might'] for u in vw['board'] if u['ctrl'] == 1 and u.get('type') == 'Unit' and not u['exhausted'] and u['loc'] != L)
            res[L] = (ks, am, dfn, sum(f['owner'] == 2 for f in b['facedown']), vw['players']['opp']['ready_runes'])
        return res, vw

    def apply(seat, k):
        nonlocal nlog, open_c
        res, _ = before(seat)
        if res:
            for L, (ks, am, dfn, fd, rd) in res.items():
                opp.append([int(seat == ss), int(am > dfn), int(fd > 0), int(k in ks), rd])
        g.step(k)
        lg = g.view(1, nlog)
        nlog = lg['log_next']
        for l in lg['log']:
            t = re.sub(r'^\[D:[0-9.]+\] ', '', l['msg'])
            mm = re.match(r'COMBAT: BF#(\d) attacker=P(\d)', t)
            if mm:
                bf, at = int(mm.group(1)), int(mm.group(2))
                ar, dr = (1, 2) if at == 1 else (2, 1)
                us = [u for u in lg['board'] if u['loc'] == f'bf{bf}' and u.get('type') == 'Unit']
                b = next(b for b in lg['battlefields'] if b['slot'] == bf)
                open_c = dict(search=int(at == ss), bf=bf, at=at, atk=sum(u['might'] for u in us if u['ctrl'] == ar),
                              dfn=sum(u['might'] for u in us if u['ctrl'] == dr), fd=sum(f['owner'] == dr for f in b['facedown']), win=0)
                att.append(open_c)
            if re.search(r'TURN \d+', t) or t.startswith('MOVE:'):
                open_c = None
            mm = re.match(r'CONQUER: P(\d) conquers BF#(\d)', t)
            if mm and open_c and int(mm.group(1)) == open_c['at'] and int(mm.group(2)) == open_c['bf']:
                open_c['win'] = 1
                open_c = None

    while True:
        S.advance()
        for d in S.log()[nrow:]:
            apply(int(d[11]), int(d[1])); nrow += 1
        st = g.status()
        if st['done']:
            break
        playweb.lib().rbs_set_strict_vis(1)
        a_, _ = A.choose(g.observe(), st['nlegal'])
        a_ = min(a_, S.A - 1)
        S.human(a_)
        apply(st['seat'], a_); nrow += 1
    w = g.status()['winner']
    S.close(); g.close()
    return dict(win=int(w == ss), opp=opp, att=att)


if __name__ == '__main__':
    import numpy as np
    mode, n = sys.argv[1], int(sys.argv[2])
    NP = int(os.environ.get('NP', '12'))
    if mode == 'equip':
        with Pool(NP) as P:
            R = P.map(equip, range(991000, 991000 + n))
        E = [e for r in R for e in r[0]]
        print(T.CK.split('/')[-1], 'choices among 2+ different unattached Equipments', len(E), '| distinct card keys seen in final positions', len(set(k for r in R for k in r[1])))
        tv = [0.5 * sum(abs(x - 1 / len(e)) for x in e.values()) for e in E]
        print('  distance from an even split between the cards: mean', np.mean(tv).round(3), 'median', np.median(tv).round(3), 'under 0.05:', np.mean([x < .05 for x in tv]).round(2))
        names = sorted({k for e in E for k in e})
        for nm in names:
            z = [e[nm] * len(e) for e in E if nm in e]
            print(f'    {nm:18s} preference x{np.mean(z):.2f} of even (n {len(z)})')
    else:
        sims = int(sys.argv[3])
        with Pool(NP) as P:
            R = []
            for i, r in enumerate(P.imap_unordered(search, [(i, 993000 + i // 2, sims) for i in range(n)])):
                R.append(r)
        json.dump(R, open('tricks_search.json', 'w'))
        print('games', len(R), 'search wins', sum(r['win'] for r in R))
        O = [x for r in R for x in r['opp']]
        At = [x for r in R for x in r['att']]
        for who, nm in ((1, 'SEARCH'), (0, 'greedy')):
            for fd in (0, 1):
                z = [x[3] for x in O if x[0] == who and x[1] and x[2] == fd]
                y = [x['win'] for x in At if x['search'] == who and x['atk'] > x['dfn'] and (x['fd'] > 0) == bool(fd)]
                print(f"  {nm} {'facedown there' if fd else 'no facedown   '}: took {np.mean(z) if z else float('nan'):.3f} of {len(z)} winning-looking attack chances; won {np.mean(y) if y else float('nan'):.3f} of {len(y)} winning-looking attacks")
            z = [x for x in At if x['search'] == who]
            print(f'  {nm} attacks with less Might than the defenders: {np.mean([x["atk"] < x["dfn"] for x in z]):.3f} of {len(z)}')
