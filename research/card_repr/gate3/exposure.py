"""Step 5 (D3): exposure counts per (card, pk) and per (card, context) in the trained policy's own self-play games.
The policy plays both seats (sampled, as in training). Counts are scaled to per 10k games.
usage: python3 exposure.py <ckpt> --games 1000"""
import argparse, json, os
from collections import Counter, defaultdict
import numpy as np
import torch
from common import *
from rbenv import RBEnv, load_vocab
from selfplay import act_rows
from pathtest_keys import DECK_CARDS, BFS

ap = argparse.ArgumentParser()
ap.add_argument('ckpt'); ap.add_argument('--games', type=int, default=1000)
ap.add_argument('--min', type=int, default=500)
a = ap.parse_args()
voc = load_vocab()
env = RBEnv(128, 'rengar', 'rengar', 777000, autopay=True)
net = load_net(a.ckpt, env.dims, dev='cuda')
REFIX = [1, 2] + list(range(4, 18)) + list(range(18, 30, 2)) + [33]
asked = Counter()            # (card, pk) card is the asking object
offered = Counter()          # (card, pk) card referenced by some legal option
chosen = Counter()           # (card, pk) card referenced by the chosen option
ctx = Counter()              # (card, context) decisions of its controller while the card is in that context
ctx_act = Counter()          # (card, context) the card referenced by the chosen option in that context
games = 0; decisions = 0
while games < a.games:
    o = env.observe()
    rows = np.nonzero(o['need'])[0]
    ch = np.zeros(env.n_envs, np.int64)
    if len(rows):
        aa, _, _ = act_rows(net, o, rows, 'cuda'); ch[rows] = aa
    for r, c in zip(rows, ch[rows]):
        decisions += 1
        pk = pkn(int(o['pk'][r])); tok = o['tok'][r]; act = o['act'][r]; n = int(o['act_mask'][r].sum())
        g = o['glob'][r]
        ps = int(o['psrc'][r])
        if ps >= 0:
            asked[(voc[tok[ps, 1]], pk)] += 1
        off = set()
        for i in range(n):
            off |= {int(t) for t in act[i, REFIX] if t >= 0}
        for t in off:
            offered[(voc[tok[t, 1]], pk)] += 1
        chs = {int(t) for t in act[c, REFIX] if t >= 0}
        for t in chs:
            chosen[(voc[tok[t, 1]], pk)] += 1
        # contexts of own cards
        for t in np.nonzero(o['tok_mask'][r] & (tok[:, 0] == 1) & (tok[:, 2] == 1))[0]:
            key = voc[tok[t, 1]]
            cs = []
            loc, zone, comb = tok[t, 5], tok[t, 4], tok[t, 14]
            if loc == 0: cs.append('in_base')
            elif loc >= 1: cs.append('at_battlefield')
            if comb == 1: cs.append('attacking')
            elif comb == 2: cs.append('defending')
            if zone == 2 and key in ('noxus_hopeful', 'darius_trifarian'):   # Legion / "second card" condition
                cs.append('cond_true(played>=1)' if g[69] >= 1 else 'cond_false(played=0)')
            if key == 'inferna' and loc >= 1:
                cs.append('assault_active' if tok[t, 22] > 0 else 'assault_inactive')
            if tok[t, 23] > 0: cs.append('shield_active')
            for cc in cs:
                ctx[(key, cc)] += 1
                if t in chs: ctx_act[(key, cc)] += 1
        for t in np.nonzero(o['tok_mask'][r] & (tok[:, 0] == 2))[0]:
            key = voc[tok[t, 1]]
            if key == 'star_spring':
                ctx[(key, 'stamp_used_self' if tok[t, 15] else 'stamp_unused_self')] += 1
    env.step(ch)
    games += len(env.results())
scale = 10000 / games


def tab(C):
    return {f'{k[0]}|{k[1]}': round(v * scale) for k, v in sorted(C.items(), key=lambda kv: -kv[1])}


res = dict(games=games, decisions=decisions, per10k_games=dict(asked=tab(asked), offered=tab(offered), chosen=tab(chosen),
                                                                context=tab(ctx), context_chosen=tab(ctx_act)))
# rare list: in-scope cards x relevant pk / contexts under the minimum
rare = []
for card in DECK_CARDS:
    for pk in sorted({k[1] for k in chosen} | {k[1] for k in asked}):
        x = max(asked.get((card, pk), 0), chosen.get((card, pk), 0)) * scale
        if 0 < x < a.min:
            rare.append((card, pk, round(x)))
    for cc in ('in_base', 'at_battlefield', 'attacking', 'defending', 'cond_true(played>=1)', 'cond_false(played=0)',
               'assault_active', 'assault_inactive', 'shield_active'):
        x = ctx_act.get((card, cc), 0) * scale
        y = ctx.get((card, cc), 0) * scale
        if y > 0 and x < a.min:
            rare.append((card, 'ctx:' + cc, round(x), round(y)))
res['rare'] = rare
res['never'] = [c for c in DECK_CARDS if not any(k[0] == c for k in chosen) and not any(k[0] == c for k in asked)]
json.dump(res, open(os.path.join(HERE, 'exposure.json'), 'w'), indent=1)
print('games', games, 'decisions', decisions)
for k in ('asked', 'chosen'):
    print(k, list(res['per10k_games'][k].items())[:80])
print('rare', rare)
print('never', res['never'])
