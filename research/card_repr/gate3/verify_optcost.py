"""Checks the schema-v3 optional-cost fields (act 49..55) against the engine: at every Accelerate / AdditionalCost
prompt a random answer is taken; at the env's next external decision in the same turn, the deciding seat's own runes
on the board (ready, total; from glob 39..62) are compared with the answer's predicted [4] ready-after and
[5] total-after. Random play, autopay on."""
import json, os
import numpy as np
from collections import Counter
from common import *
from rbenv import RBEnv

env = RBEnv(32, 'rengar', 'rengar', 5150, autopay=True)
rng = np.random.default_rng(1)
pend = {}   # env -> (seat, turn, pk, pred_ready, pred_total, yes, extraE, extraP, rec)
st = Counter(); ex = []
games = 0
while games < 400:
    o = env.observe()
    rows = np.nonzero(o['need'])[0]
    ch = np.zeros(env.n_envs, np.int64)
    for r in rows:
        g = o['glob'][r]; seat = int(o['seat'][r])
        if r in pend:
            ps, turn, pk, pr, pt, yes = pend.pop(r)
            if g[0] == turn:
                if seat == ps:
                    rd, tot = g[39:45].sum(), g[39:51].sum()
                else:
                    rd, tot = g[51:57].sum(), g[51:63].sum()
                k = f'{pk}|{"yes" if yes else "no"}'
                st[k + '|n'] += 1
                st[k + '|ready_ok'] += int(rd == pr); st[k + '|total_ok'] += int(tot == pt)
                if (rd != pr or tot != pt) and len(ex) < 12:
                    ex.append(dict(pk=pk, yes=yes, pred=(int(pr), int(pt)), seen=(int(rd), int(tot))))
            else:
                st['next_decision_other_turn'] += 1
        n = int(o['act_mask'][r].sum())
        c = int(rng.integers(n)); ch[r] = c
        pk = pkn(int(o['pk'][r]))
        if pk in ('Accelerate', 'AdditionalCost'):
            q = o['act'][r, c, 49:56]
            pend[r] = (seat, int(g[0]), pk, int(q[4]), int(q[5]), int(o['act'][r, c, 39]) == 0)
    env.step(ch)
    for res in env.results():
        games += 1
        pend.pop(res['env'], None)
out = {}
for k in sorted({k.rsplit('|', 1)[0] for k in st if k.count('|') == 2}):
    n = st[k + '|n']
    out[k] = dict(n=n, ready_match=round(st[k + '|ready_ok'] / n, 4), total_match=round(st[k + '|total_ok'] / n, 4))
out['next_decision_other_turn'] = st['next_decision_other_turn']
out['examples'] = ex
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, 'verify_optcost.json'), 'w'), indent=1)
