"""Step 3: learnability. Synthetic supervised labels on real states of one pk that depend ONLY on the asking card
(mode=card) or ONLY on one field (mode=field:<spec>). Each training row is a real state with the asking key (or the
field) re-drawn at random, so nothing else can predict the label. Fresh untrained net, Adam, --steps steps; accuracy
on held-out states.
usage: python3 learn.py --pk May --mode card [--net-kw '{"ask_opt": true}']"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from common import *
from rbenv import load_vocab

ap = argparse.ArgumentParser()
ap.add_argument('--pk', required=True)
ap.add_argument('--mode', default='card')
ap.add_argument('--steps', type=int, default=300)
ap.add_argument('--bs', type=int, default=256)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--net-kw', default='{}')
ap.add_argument('--out', default='learn_results.jsonl')
ap.add_argument('--tag', default='')
ap.add_argument('--samples', default='inventory_samples.npz')
a = ap.parse_args()
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
dev = 'cuda'
voc = load_vocab(); vi = {k: i for i, k in enumerate(voc)}
from pathtest_keys import SC_CARD, SC_BF
pk = PK_NAMES.index(a.pk)
z = np.load(os.path.join(HERE, a.samples))
D0 = {k: z[f'{pk}_{k}'] for k in OBS_KEYS}
DIMS = dict(G=72, T=96, F=24, A=96, K=int(D0['act'].shape[-1]))
net = load_net('untrained', DIMS, dev=dev, **json.loads(a.net_kw)); net.train()
S_CARD = set(rng.choice(SC_CARD, len(SC_CARD) // 2, replace=False).tolist())
S_BF = {SC_BF[0]}


def opt_score(d, s):
    """per-option content score used to define the labelled option (argmax or argmin)"""
    n = d['act_mask'][s].sum(); A = d['act'][s, :n]; T = d['tok'][s]
    if a.pk in ('May', 'Accelerate', 'AdditionalCost'):
        return -A[:, 39].astype(float)                           # yes (dec 0) high
    if a.pk == 'Location':
        return A[:, 30].astype(float)
    if a.pk == 'TriggerOrder':
        return A[:, 32].astype(float)
    sc = []
    for r in A:
        refs = [t for t in r[[8, 9, 10, 11, 4, 5, 6, 7, 33, 1]] if t >= 0]
        if not refs: sc.append(-1e6); continue
        t = T[refs[0]]
        if a.pk == 'ChooseObject':        # cards picked from a revealed hand: same side and Might, use the card
            sc.append(float(t[1])); continue
        sc.append(t[2] * 100 + t[9] * 1 + t[8] * 0.01)          # controller, Might, damage
    return np.array(sc, float)


# usable states: argmax and argmin unique and different
ok = []
for s in range(len(D0['pk'])):
    if D0['act_mask'][s].sum() < 2: continue
    if a.mode == 'card' and D0['psrc'][s] < 0 and a.pk != 'TriggerOrder': continue
    f = opt_score(D0, s)
    if (f == f.max()).sum() == 1 and (f == f.min()).sum() == 1 and f.max() != f.min() and f.max() > -1e5 and f.min() > -1e5:
        ok.append(s)
ok = np.array(ok); rng.shuffle(ok)
ntest = max(16, len(ok) // 4)
test, train = ok[:ntest], ok[ntest:]
print(a.pk, a.mode, 'usable states', len(ok), 'train', len(train), 'test', len(test), flush=True)


def make(idx):
    d = {k: D0[k][idx].copy() for k in OBS_KEYS}
    y = np.zeros(len(idx), np.int64)
    for j, s in enumerate(idx):
        f = opt_score(D0, s)
        if a.mode == 'card':
            if a.pk == 'TriggerOrder':   # no asking token: the "asking" cards are the pending chain items
                ch = np.nonzero(d['tok'][j, :, 0] == 3)[0]
                t = ch[0] if len(ch) else 0
            else:
                t = d['psrc'][j]
            sc = SC_BF if d['tok'][j, t, 0] == 2 else SC_CARD
            key = rng.choice(sc); d['tok'][j, t, 1] = key
            pos = key in (S_BF if d['tok'][j, t, 0] == 2 else S_CARD)
        else:
            kind, where, col, lo, hi, thr = a.mode.split(':')[1:]
            lo, hi, thr, col = int(lo), int(hi), int(thr), int(col)
            val = rng.integers(lo, hi + 1)
            if where == 'glob':
                d['glob'][j, col] = val
            elif where == 'optyes':     # the accepting option's own action field (schema v3 optional-cost fields)
                nn_ = d['act_mask'][j].sum()
                yi = [i for i in range(nn_) if d['act'][j, i, 39] == 0]
                d['act'][j, yi[0] if yi else 0, col] = val
            elif where == 'psrc':
                d['tok'][j, d['psrc'][j], col] = val
            pos = val >= thr
        y[j] = int(np.argmax(f)) if pos else int(np.argmin(f))
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
rec = dict(pk=a.pk, mode=a.mode, net_kw=a.net_kw, tag=a.tag, n_states=len(ok), steps=a.steps, curve=curve,
           final_acc=curve[-1][1])
open(os.path.join(HERE, a.out), 'a').write(json.dumps(rec) + '\n')
