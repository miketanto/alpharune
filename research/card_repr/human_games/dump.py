"""Per-decision dump of a rebuilt game: both hands, G2's probabilities and value (decider's view)."""
import json, os, sys
import numpy as np
PW = '/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/.claude/worktrees/play-web-ui-e914cf/tools/play_web'
sys.path.insert(0, PW)
import playweb, torch
HERE = os.path.dirname(os.path.abspath(__file__))
G = json.load(open(os.path.join(HERE, sys.argv[1])))
DECKS = os.path.join(PW, '..', '..', 'decks')
g = playweb.Game(os.path.join(DECKS, G['p1']), os.path.join(DECKS, G['p2']), G['human'], G['seed'])
ag = playweb.NetAgent(os.path.join(playweb.SELFPLAY, 'runs', G['checkpoint']))


def probs(g):
    o = g.observe()
    with torch.no_grad():
        lp, v = ag.net(ag._common.to_dev(o, np.array([0]), 'cpu'))
    return lp[0].exp().numpy(), float(v.reshape(-1)[0])


def board(v):
    c = v['cards']
    nm = lambda o: o.get('name') or '?'
    s = []
    for b in v['battlefields']:
        L = 'bf%d' % b['slot']
        us = [f"{'P%d' % (v['me'] if o['ctrl'] == 1 else 3 - v['me'])}:{nm(o)}{o['might']}{'x' if o['exhausted'] else ''}" for o in v['board']
              if o['loc'] == L and o.get('type') == 'Unit']
        fd = [f"P{v['me'] if x['owner'] == 1 else 3 - v['me']}" for x in b['facedown']]
        s.append(f"{L}[ctl {b['controller']}] {us} facedown {fd}")
    base = {r: [f"{nm(o)}{o['might']}{'x' if o['exhausted'] else ''}" for o in v['board'] if o['loc'] == 'base' and o['ctrl'] == r and o.get('type') == 'Unit'] for r in (1, 2)}
    s.append(f"base P1 {base[1]}  base P2 {base[2]}")
    return s


i = 0
last = None
vals = []
for seat, ch in G['choices']:
    st = g.status()
    assert st['seat'] == seat, (i, st, seat)
    v1, v2 = g.view(1, 1 << 40), g.view(2, 1 << 40)
    dv = v1 if seat == 1 else v2
    t = v1['turn']['number']
    p, val = probs(g)
    vals.append((i, t, seat, val if seat == 1 else -val))
    if t != last:
        print(f"\n=== TURN {t} (P{v1['turn']['turn_player']}) score P1 {v1['players']['me']['score']} - P2 {v1['players']['opp']['score']}"
              f" | runes ready P1 {v1['players']['me']['ready_runes']} P2 {v1['players']['opp']['ready_runes']}")
        h1 = [o['name'] for o in v1['hand']]; h2 = [o['name'] for o in v2['hand']]
        print('  P1 hand', h1); print('  P2 hand', h2)
        for s in board(v1):
            print('  ' + s)
        last = t
    d = dv['decision']
    op = {o['index']: (o.get('text') or o.get('label')) for o in d['options']}
    order = np.argsort(-p[:st['nlegal']])
    top = ' | '.join(f"{100 * p[k]:.0f}% {op.get(int(k), '?')[:60]}" for k in order[:4] if p[k] > .02)
    r = f"{v1['players']['me']['ready_runes']}/{v1['players']['opp']['ready_runes']}"
    print(f"#{i:3d} P{seat} {'YOU' if seat == G['human'] else 'G2 '} [{d['kind']}] runes {r} v(P1) {val if seat == 1 else -val:+.2f} CHOSE ({100 * p[ch]:.0f}%) {op.get(ch, '?')[:75]}"
          + ('' if len(op) < 2 else f"   <<  {top}"))
    g.step(ch)
    i += 1
print(g.status())
json.dump(vals, open(os.path.join(HERE, sys.argv[1].replace('.json', '_vals.json')), 'w'))
