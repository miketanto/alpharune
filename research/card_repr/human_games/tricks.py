"""Does the network respect tricks? G2-vs-G2 Azir mirrors (greedy), CPU processes.
 A. At main-phase decisions with an attack on offer (a Standard Move to a battlefield holding enemy units), the probability
    mass on attacks and the value, re-evaluated with ONE observation edit: opponent runes all exhausted / all ready,
    opponent hand 1 / 6, opponent facedown tokens removed.
 B. Attacks made: visible Might when the combat starts (attackers vs defenders), opponent ready runes and facedown there,
    and whether the attacker conquered. Ground truth of what open runes / facedown cards cost.
 C. Where each trick spell is cast (own main phase outside a combat / in a showdown).
python3 tricks.py OUT N_GAMES SEED
"""
import json, os, re, sys
from multiprocessing import Pool
PW = '/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/.claude/worktrees/play-web-ui-e914cf/tools/play_web'
sys.path.insert(0, PW)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PLAYWEB_TORCH_THREADS'] = '1'
DECKS = os.path.join(PW, '..', '..', 'decks')
CK = os.environ.get('CK', 'g2_main/latest.pt')
TRICKS = ('Discipline', 'En Garde', 'Back Off', 'Hidden Blade', 'Guards!', 'Deathgrip', 'Defy')
OR, OX, OH = slice(51, 57), slice(57, 63), 10   # glob: opponent ready runes / exhausted runes by Domain, opponent hand
_A = None
LOC0 = int(os.environ.get('LOC0', '1'))   # decision_export.cpp locOf: code of battlefield 0


def ev(o):
    import numpy as np, torch
    with torch.no_grad():
        lp, v = _A.net(_A._common.to_dev(o, np.array([0]), 'cpu'))
    return lp[0].exp().numpy(), float(v.reshape(-1)[0])


def edit(o, **kw):
    import numpy as np
    e = dict(o)
    e['glob'] = o['glob'].copy()
    g = e['glob'][0]
    if kw.get('runes') == 0:
        g[OX] += g[OR]; g[OR] = 0
    if kw.get('runes') == 1:
        g[OR] += g[OX]; g[OX] = 0
    if 'hand' in kw:
        g[OH] = kw['hand']
    if kw.get('nofd_bf') is not None or kw.get('nofd'):
        # a facedown card is seen twice: a 'hidden' card token with is_hidden (no location), and the battlefield token's
        # count of the opponent's facedown cards (f[9] -> row[11]; battlefield id is row[2]). Both are edited.
        e['tok'] = o['tok'].copy(); e['tok_mask'] = o['tok_mask'].copy()
        t = e['tok'][0]; mk = e['tok_mask'][0]
        bfs = (t[:, 0] == 2) & mk
        if kw.get('nofd_bf') is not None:
            bfs = bfs & (t[:, 2] == kw['nofd_bf'])
        n = int(t[bfs, 11].sum())
        t[bfs, 11] = 0
        hid = np.nonzero((t[:, 0] == 1) & (t[:, 1] == 1) & (t[:, 3] == 2) & (t[:, 13] == 1) & mk)[0]
        e['_nfd'] = (n, len(hid))
        mk[hid[:n]] = False
    return e


def game(seed):
    global _A
    import numpy as np, playweb
    if _A is None:
        _A = playweb.NetAgent(os.path.join(playweb.SELFPLAY, 'runs', CK))
    dk = os.path.join(DECKS, 'azir.txt')
    g = playweb.Game(dk, dk, 1, seed)
    A, B, C, EQ = [], [], [], []
    open_c = None
    nlog = 0
    while True:
        st = g.status()
        if st['done'] or not st['need']:
            break
        s = st['seat']
        o = g.observe()
        p, v = ev(o)
        a = int(p.argmax())
        vw = g.view(s, 1 << 40)
        d = vw['decision']
        op = {x['index']: x for x in d['options']}
        txt = (op.get(a) or {}).get('text') or ''
        if d['kind'] == 'target' and 'Equipment' in (d.get('question') or ''):
            # which Equipment to equip: probability by card NAME (copies summed), relative to a blind even split over options
            by, cnt = {}, {}
            for k, x in op.items():
                n_ = re.sub(r'^your | (token|at your base|in hand|at battlefield).*$|\(exhausted\)', '', x['text']).strip()
                by[n_] = by.get(n_, 0) + float(p[k]); cnt[n_] = cnt.get(n_, 0) + 1
            if len(by) >= 2:
                tot = sum(cnt.values())
                EQ.append([0.5 * sum(abs(by[q] - cnt[q] / tot) for q in by), len(by), max(by.values())])
        if d['kind'] == 'main phase action':
            eu = {f"bf{b['slot']}" for b in vw['battlefields'] if any(u['loc'] == f"bf{b['slot']}" and u['ctrl'] == 2 and u.get('type') == 'Unit' for u in vw['board'])}
            idx = [k for k, x in op.items() if x['text'].startswith('move') and x.get('dest') in eu]
            if idx:
                fd = sum(len([f for f in b['facedown'] if f['owner'] == 2]) for b in vw['battlefields'])
                rd = vw['players']['opp']['ready_runes']; tot = rd + vw['players']['opp']['exhausted_runes']
                rec = dict(p=float(p[idx].sum()), v=v, ready=rd, total=tot, hand=vw['players']['opp']['hand_count'], fd=fd)
                for nm, kw in (('r0', dict(runes=0)), ('r1', dict(runes=1)), ('h1', dict(hand=1)), ('h6', dict(hand=6)), ('nofd', dict(nofd=1))):
                    if nm == 'nofd' and not fd:
                        continue
                    if nm in ('r0', 'r1') and not tot:
                        continue
                    pe, vv = ev(edit(o, **kw))
                    rec[nm] = [float(pe[idx].sum()), vv]
                # per battlefield: this battlefield's own facedown card masked; visible Might of what could join the attack
                for b in vw['battlefields']:
                    L = f"bf{b['slot']}"
                    if L not in eu or not any(f['owner'] == 2 for f in b['facedown']):
                        continue
                    ib = [k for k in idx if op[k].get('dest') == L]
                    if not ib:
                        continue
                    ed = edit(o, nofd_bf=b['slot'])
                    pe, vv = ev(ed)
                    dm = sum(u['might'] for u in vw['board'] if u['loc'] == L and u['ctrl'] == 2 and u.get('type') == 'Unit')
                    am = sum(u['might'] for u in vw['board'] if u['ctrl'] == 1 and u.get('type') == 'Unit' and not u['exhausted'] and u['loc'] != L)
                    rec.setdefault('bf', []).append([float(p[ib].sum()), float(pe[ib].sum()), v, vv, am, dm, int(ed['_nfd'][0] == 1 and ed['_nfd'][1] >= 1)])
                A.append(rec)
        m = re.match(r'(Play\w*) (?:your )?(.+?)(?: in hand| face ?down|$)', txt)
        if m and any(t in txt for t in TRICKS):
            nm = next(t for t in TRICKS if t in txt)
            C.append((nm, 'main' if d['kind'] == 'main phase action' else d['kind'], bool(vw['turn']['showdown'] or any(b['combat'] for b in vw['battlefields'])), 'face down' in txt))
        g.step(a)
        # combats that began / conquers since
        lg = g.view(1, nlog)
        nlog = lg['log_next']
        for l in lg['log']:
            t = re.sub(r'^\[D:[0-9.]+\] ', '', l['msg'])
            mm = re.match(r'COMBAT: BF#(\d) attacker=P(\d)', t)
            if mm:
                bf, at = int(mm.group(1)), int(mm.group(2))
                # board right after this step (tricks of this step, if any, included: the first focus decision)
                at_rel, de_rel = (1, 2) if at == 1 else (2, 1)
                us = [u for u in lg['board'] if u['loc'] == f'bf{bf}' and u.get('type') == 'Unit']
                b = next(b for b in lg['battlefields'] if b['slot'] == bf)
                de = lg['players']['opp' if de_rel == 2 else 'me']
                open_c = dict(bf=bf, at=at, atk=sum(u['might'] for u in us if u['ctrl'] == at_rel), dfn=sum(u['might'] for u in us if u['ctrl'] == de_rel),
                              n_def=sum(u['ctrl'] == de_rel for u in us), ready=de['ready_runes'], hand=de['hand_count'],
                              fd=sum(f['owner'] == de_rel for f in b['facedown']), win=0)
                B.append(open_c)
            if re.search(r'TURN \d+', t) or t.startswith('MOVE:'):
                open_c = None
            mm = re.match(r'CONQUER: P(\d) conquers BF#(\d)', t)
            if mm and open_c and int(mm.group(1)) == open_c['at'] and int(mm.group(2)) == open_c['bf']:
                open_c['win'] = 1; open_c = None
    w = g.status()['winner']
    g.close()
    return dict(seed=seed, winner=w, A=A, B=B, C=C, EQ=EQ)


if __name__ == '__main__':
    out, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    with Pool(int(os.environ.get('NP', '12'))) as P:
        R = []
        for i, r in enumerate(P.imap_unordered(game, range(seed, seed + n))):
            R.append(r)
            if (i + 1) % 50 == 0:
                print(i + 1, flush=True)
    json.dump(R, open(out, 'w'))
    import numpy as np
    rng = np.random.default_rng(0)
    A = [a for r in R for a in r['A']]
    print('games', len(R), 'attack-offered decisions', len(A), 'mean P(attack)', np.mean([a['p'] for a in A]).round(3))

    def boot(f):
        """f(list of games) -> number; 95% interval over games."""
        x = [f([R[i] for i in rng.integers(0, len(R), len(R))]) for _ in range(300)]
        return f'{f(R):+.4f} [{np.nanpercentile(x, 2.5):+.4f},{np.nanpercentile(x, 97.5):+.4f}]'

    for nm, (x, y) in dict(runes_open_vs_tapped=('r1', 'r0'), hand6_vs_hand1=('h6', 'h1')).items():
        f = lambda G, k: np.mean([a[x][k] - a[y][k] for r in G for a in r['A'] if x in a and y in a])
        print(f'{nm:24s} d P(attack) {boot(lambda G: f(G, 0))}   d value(attacker, win units x2) {boot(lambda G: f(G, 1))}')
    f = lambda G, k: np.mean([(a['p'] if k == 0 else a['v']) - a['nofd'][k] for r in G for a in r['A'] if 'nofd' in a])
    print(f"{'facedown_vs_none':24s} d P(attack) {boot(lambda G: f(G, 0))}   d value {boot(lambda G: f(G, 1))}  n={sum('nofd' in a for a in A)}")
    X = [x for r in R for a in r['A'] for x in a.get('bf', [])]
    print('facedown at the attacked battlefield: decisions', len(X), 'tokens masked ok', np.mean([x[6] == 1 for x in X]).round(3))
    for lab, sel in (('all', lambda x: True), ('ready Might > defenders', lambda x: x[4] > x[5])):
        fx = lambda G, k: np.mean([x[k] - x[k + 1] for r in G for a in r['A'] for x in a.get('bf', []) if sel(x)])
        base = np.mean([x[1] for x in X if sel(x)])
        print(f'  {lab}: n {sum(map(sel, X))} P(attack there) without the card {base:.3f}; with it {boot(lambda G: fx(G, 0))}; value {boot(lambda G: fx(G, 2))}')
    B = [b for r in R for b in r['B']]
    print('attacks', len(B), 'conquered', np.mean([b['win'] for b in B]).round(3))
    for lab, sel in (('visibly winning atk>def', lambda b: b['atk'] > b['dfn']), ('even', lambda b: b['atk'] == b['dfn']), ('visibly losing atk<def', lambda b: b['atk'] < b['dfn'])):
        S = [b for b in B if sel(b)]
        print(f'  {lab}: n {len(S)} ({100 * len(S) / len(B):.0f}%) conquered {np.mean([b["win"] for b in S]):.3f}')
        for l2, s2 in (('def 0 ready, no facedown', lambda b: b['ready'] == 0 and not b['fd']), ('def 1-2 ready, no fd', lambda b: 1 <= b['ready'] <= 2 and not b['fd']),
                       ('def 3+ ready, no fd', lambda b: b['ready'] >= 3 and not b['fd']), ('facedown there', lambda b: b['fd'] > 0)):
            T = [b for b in S if s2(b)]
            if T:
                print(f'      {l2:26s} n {len(T):4d} conquered {np.mean([b["win"] for b in T]):.3f}')
    E = [e for r in R for e in r['EQ']]
    if E:
        print(f'equipment choices with 2+ different cards: {len(E)}; distance from a blind even split (0 blind, 1 decided) mean {np.mean([e[0] for e in E]):.3f}, share under 0.05: {np.mean([e[0] < .05 for e in E]):.2f}')
    import collections
    c = collections.Counter()
    for r in R:
        for nm, kind, sd, fd in r['C']:
            c[(nm, 'in combat/showdown' if sd else ('own main phase' if kind == 'main' else kind))] += 1
    for k, n_ in sorted(c.items()):
        print('  cast', k, n_)
