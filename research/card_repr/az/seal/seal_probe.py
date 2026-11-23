"""Why is Seal of Discord undervalued? Net-vs-net Kennen mirrors (greedy, floating engine) with per-decision
logging of every Seal option, and an optional treated seat that is forced to take Seal options.
  python3 seal_probe.py --games N --arm none|play|play_use --out out.json
arm none: both seats the plain net (diagnostics).  play: the treated seat plays Seal whenever it may.
play_use: also activates a ready Seal whenever it may (the Power then floats for the rest of the turn)."""
import argparse, json, os, sys, time
sys.path.insert(0, '/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/.claude/worktrees/play-web-ui-e914cf/tools/play_web')
import numpy as np
import playweb as pw

ap = argparse.ArgumentParser()
ap.add_argument('--games', type=int, default=20)
ap.add_argument('--arm', default='none')
ap.add_argument('--ckpt', default='/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/az/run_kennen/az_010292.pt')
ap.add_argument('--seed', type=int, default=50000)
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
ap.add_argument('--out', default='')
ap.add_argument('--verbose', action='store_true')
a = ap.parse_args()
DECK = os.path.join(pw.REPO_WSL, 'decks/kennen.txt')
if not os.path.exists(DECK):
    DECK = '/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/decks/kennen.txt'
net = pw.NetAgent(a.ckpt)

def is_seal(o):
    return 'Seal of Discord' in (o.get('text', '') + ' ' + o.get('label', ''))

res = []
t0 = time.time()
for gi in range(a.shard, a.games, a.nshards):
    T = 1 + gi % 2                      # treated seat alternates
    g = pw.Game(DECK, DECK, 1, a.seed + gi, manual_runes=True)
    st_ = dict(seat_legal=0, seal_play_offered=0, seal_play_taken=0, seal_play_p=[], seal_act_offered=0,
               seal_act_taken=0, seal_act_p=[], kinds={}, forced=0)
    per = {1: json.loads(json.dumps(st_)), 2: json.loads(json.dumps(st_))}
    n = 0
    while True:
        s = g.status()
        if s['done'] or n > 3000:
            break
        seat = s['seat']
        obs = g.observe()
        k, _ = net.choose(obs, s['nlegal'])
        d = g.view(seat).get('decision') or {}
        opts = d.get('options', [])
        P = per[seat]
        seal_opts = [o for o in opts if is_seal(o)]
        if seal_opts:
            probs = net.last_probs
            for o in seal_opts:
                kind = o.get('type', '?')
                P['kinds'][kind] = P['kinds'].get(kind, 0) + 1
                p = float(probs[o['index']]) if o['index'] < len(probs) else -1
                if kind in ('PlayCard', 'PlayActionCard', 'PlayReaction', 'HideCard'):
                    if kind != 'HideCard':
                        P['seal_play_offered'] += 1; P['seal_play_p'].append((g.view(seat)['turn']['number'], round(p, 4), k == o['index']))
                        if k == o['index']: P['seal_play_taken'] += 1
                elif 'Activate' in kind or 'Add' in kind:
                    P['seal_act_offered'] += 1; P['seal_act_p'].append(round(p, 4))
                    if k == o['index']: P['seal_act_taken'] += 1
                if a.verbose:
                    print(gi, 'seat', seat, 'turn', g.view(seat)['turn']['number'], kind, o.get('text'), 'p=%.3f' % p, 'chosen' if k == o['index'] else '', flush=True)
            if seat == T and a.arm in ('play', 'play_use'):
                pick = [o for o in seal_opts if o.get('type') in ('PlayCard', 'PlayReaction', 'PlayActionCard')]
                if a.arm == 'play_use' and not pick:
                    pick = [o for o in seal_opts if 'Activate' in o.get('type', '')]
                if pick and k != pick[0]['index']:
                    k = pick[0]['index']; P['forced'] += 1
        g.step(k)
        n += 1
    s = g.status()
    res.append(dict(game=gi, treated=T, winner=s['winner'], decisions=n, per=per))
    g.close()
    w = sum(r['winner'] == r['treated'] for r in res)
    print(f"[{time.time()-t0:.0f}s] game {gi} winner {s['winner']} treated {T} | treated wins {w}/{len(res)}", flush=True)
if a.out:
    json.dump(dict(arm=a.arm, ckpt=a.ckpt, games=res), open(a.out, 'w'))
