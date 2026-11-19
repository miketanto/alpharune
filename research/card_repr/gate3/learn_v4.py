"""Gate 3 completion: learnability of the schema-v4 option fields (TriggerOrder, MoveJoin).
Synthetic supervised labels on real states (inventory_v4_samples.npz) that depend ONLY on the re-drawn quantity;
fresh untrained net, Adam, --steps steps; accuracy on held-out states.
  --mode trig_key   TriggerOrder: each option's ordered chain item (act 33) gets a distinct random card key;
                    the label is the option whose key ranks highest in a fixed random order of keys.
  --mode mj_cons    MoveJoin: the move's consequence (act 46..48, the same on every option) is re-drawn;
                    label = "move no more" (dec 39) if it would start a combat as attacker (47), else the
                    strongest unit offered (Mights re-drawn distinct, so the join options differ by content).
  --mode mj_might   MoveJoin: each offered unit's Might is re-drawn (distinct, 0..6); label = the strongest unit
                    if its Might >= 4, else "move no more".
  --v3              blank the schema-v4 fields as the v3 observation had them (TriggerOrder: 1 = -1, 33 = -1;
                    MoveJoin: 30 = -1, 46 = -1, 47 = 48 = 0, 56..60 = -1, 39 = 0): the "before" numbers.
usage: python3 learn_v4.py --mode trig_key [--v3] [--net-kw '{"ask_opt": true}']"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from common import *
from rbenv import load_vocab

ap = argparse.ArgumentParser()
ap.add_argument('--mode', required=True)
ap.add_argument('--v3', action='store_true')
ap.add_argument('--steps', type=int, default=300)
ap.add_argument('--bs', type=int, default=256)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--net-kw', default='{"ask_opt": true}')
ap.add_argument('--out', default='learn_v4.jsonl')
ap.add_argument('--samples', default='inventory_v4_samples.npz')
a = ap.parse_args()
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
from pathtest_keys import SC_CARD
pkname = 'TriggerOrder' if a.mode.startswith('trig') else 'MoveJoin'
pk = PK_NAMES.index(pkname)
z = np.load(os.path.join(HERE, a.samples))
D0 = {k: z[f'{pk}_{k}'] for k in OBS_KEYS}
DIMS = dict(G=72, T=96, F=24, A=96, K=int(D0['act'].shape[-1]))
net = load_net('untrained', DIMS, dev=dev, **json.loads(a.net_kw)); net.train()
RANK = {k: r for r, k in enumerate(rng.permutation(SC_CARD).tolist())}

ok = []
for s in range(len(D0['pk'])):
    n = D0['act_mask'][s].sum(); A = D0['act'][s, :n]
    if n < 2: continue
    if pkname == 'TriggerOrder' and (A[:, 33] < 0).any(): continue
    if pkname == 'MoveJoin' and ((A[:, 39] == 1).sum() != 1 or n < 2): continue
    ok.append(s)
ok = np.array(ok); rng.shuffle(ok)
ntest = max(16, len(ok) // 4)
test, train = ok[:ntest], ok[ntest:]
print(pkname, a.mode, 'v3' if a.v3 else 'v4', 'usable', len(ok), 'train', len(train), 'test', len(test), flush=True)


def make(idx):
    d = {k: D0[k][idx].copy() for k in OBS_KEYS}
    y = np.zeros(len(idx), np.int64)
    for j in range(len(idx)):
        n = d['act_mask'][j].sum(); A = d['act'][j]
        if a.mode == 'trig_key':
            keys = rng.choice(SC_CARD, n, replace=False)
            for i in range(n):
                d['tok'][j, A[i, 33], 1] = keys[i]
            y[j] = int(np.argmax([RANK[int(k)] for k in keys]))
        elif a.mode == 'mj_cons':
            dc, atk, con = rng.integers(0, 3), rng.integers(0, 2), rng.integers(0, 2)
            A[:n, 30] = np.where(A[:n, 30] < 1, 1, A[:n, 30])     # a battlefield (base has no consequences)
            A[:n, 46], A[:n, 47], A[:n, 48] = dc, atk, con
            stop = int(np.nonzero(A[:n, 39] == 1)[0][0])
            joins = [i for i in range(n) if i != stop]   # options differ only by content: the joining
            mt = rng.permutation(7)[:len(joins)] if len(joins) <= 7 else rng.integers(0, 7, len(joins))
            for i, m in zip(joins, mt):                   # unit is told apart by a re-drawn Might
                d['tok'][j, A[i, 8], 9] = m
            y[j] = stop if atk else joins[int(np.argmax(mt))]
        elif a.mode == 'mj_might':
            stop = int(np.nonzero(A[:n, 39] == 1)[0][0])
            joins = [i for i in range(n) if i != stop]
            mt = rng.choice(7, len(joins), replace=len(joins) > 7)
            for i, m in zip(joins, mt):
                d['tok'][j, A[i, 8], 9] = m
            best = joins[int(np.argmax(mt))]
            y[j] = best if mt.max() >= 4 else stop
        if a.v3:
            if pkname == 'TriggerOrder':
                A[:n, 1] = -1; A[:n, 33] = -1
            else:
                A[:n, 30] = -1; A[:n, 46] = -1; A[:n, 47] = 0; A[:n, 48] = 0; A[:n, 56:61] = -1; A[:n, 39] = 0
    return d, y


def tb(d):
    b = {k: torch.from_numpy(np.ascontiguousarray(d[k])).to(dev) for k in OBS_KEYS}
    for k in ('glob', 'tok', 'act', 'pk', 'psrc'):
        b[k] = b[k].long()
    return b


opt = torch.optim.Adam(net.parameters(), lr=a.lr)
curve = []
for step in range(a.steps + 1):
    if step % 50 == 0:
        net.eval()
        with torch.no_grad():
            accs = []
            for rep in range(4):
                d, y = make(test)
                lp, _ = net(tb(d))
                accs.append((lp.argmax(-1).cpu().numpy() == y).mean())
        curve.append((step, float(np.mean(accs))))
        print(step, round(float(np.mean(accs)), 3), flush=True)
        net.train()
    if step == a.steps: break
    d, y = make(rng.choice(train, a.bs))
    lp, _ = net(tb(d))
    loss = F.nll_loss(lp, torch.from_numpy(y).to(dev))
    opt.zero_grad(); loss.backward(); opt.step()
rec = dict(pk=pkname, mode=a.mode, obs='v3' if a.v3 else 'v4', net_kw=a.net_kw, n_states=len(ok), steps=a.steps,
           curve=curve, final_acc=curve[-1][1])
open(os.path.join(HERE, a.out), 'a').write(json.dumps(rec) + '\n')
