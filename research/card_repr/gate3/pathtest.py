"""Step 2: path tests (a) asking-card sensitivity, (b) option-relevant field sensitivity, (c) schema-v2 fields,
on real per-pk states (inventory_samples.npz), for an untrained and a trained B2 --ask-head net.
usage: python3 pathtest.py <ckpt|untrained> <tag> [--net-kw JSON]"""
import argparse, json, os, sys, time
import numpy as np
import torch
from common import *
from rbenv import load_vocab
import model as M

ap = argparse.ArgumentParser()
ap.add_argument('ckpt'); ap.add_argument('tag')
ap.add_argument('--n', type=int, default=64)
ap.add_argument('--samples', default='inventory_samples.npz')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--net-kw', default='{}')
a = ap.parse_args()
torch.manual_seed(a.seed)
dev = 'cuda'
voc = load_vocab(); vi = {k: i for i, k in enumerate(voc)}
DECK_CARDS = ['rengar_pridestalker', 'rengar_trophy_hunter', 'noxus_hopeful', 'pit_rookie', 'sabotage', 'punch_first',
              'inferna', 'grim_apothecary', 'kinkou_initiate', 'irresistible_faefolk', 'first_mate', 'thrill_of_the_hunt',
              'rampage', 'pakaa_cub', 'ferrous_forerunner', 'kai_sa_survivor', 'nidalee_cat_form', 'darius_trifarian',
              'pyke_dockside_butcher', 'token:mech', 'token:sand_soldier']
BFS = ['emperor_s_dais', 'seat_of_power', 'star_spring']
SC_CARD = np.array([vi[k] for k in DECK_CARDS]); SC_BF = np.array([vi[k] for k in BFS])
z = np.load(a.samples)
DIMS = dict(G=72, T=96, F=24, A=96, K=int(z[z.files[0].split('_')[0] + '_act'].shape[-1]))
net = load_net(a.ckpt, DIMS, dev=dev, **json.loads(a.net_kw))
z = np.load(a.samples)
DIMS = dict(G=72, T=96, F=24, A=96, K=int(z[z.files[0].split('_')[0] + '_act'].shape[-1]))
pks = sorted({int(k.split('_')[0]) for k in z.files})
rng = np.random.default_rng(a.seed)

_logits = {}
net.pol.register_forward_hook(lambda m, i, o: _logits.__setitem__('l', o.squeeze(-1)))


def batch(d):
    b = {k: torch.from_numpy(np.ascontiguousarray(d[k])).to(dev) for k in OBS_KEYS}
    for k in ('glob', 'tok', 'act', 'pk', 'psrc'):
        b[k] = b[k].long()
    return b


@torch.no_grad()
def run(d):
    lp, v = net(batch(d))
    return lp.exp().cpu().numpy(), v.cpu().numpy(), _logits['l'].cpu().numpy()


def tv(p, q):
    return 0.5 * np.abs(p - q).sum(-1)


def sub(d, idx):
    return {k: d[k][idx].copy() for k in OBS_KEYS}


def edit_val(x):
    x = int(x)
    if x == -1: return 0
    if x in (0, 1): return 1 - x
    return x + 1


REFIX = [1, 2] + list(range(4, 18)) + list(range(18, 30, 2)) + [33]


def ref_tokens(row):
    """token indices an option references (src, abs, tg/ch/mv/mu, dm tokens, ci)"""
    return [int(t) for t in row[REFIX] if t >= 0]


res = {}
for pk in pks:
    d0 = {k: z[f'{pk}_{k}'] for k in OBS_KEYS}
    n = min(a.n, len(d0['pk']))
    d = sub(d0, np.arange(n))
    nl = d['act_mask'].sum(1)
    if (nl < 2).all():
        continue
    p0, v0, l0 = run(d)
    R = dict(n=int(n))
    # ---------------- (a) asking card
    ps = d['psrc']
    has = np.nonzero((ps >= 0) & (nl >= 2))[0]
    if len(has):
        tvs, ncs, dv = [], [], []
        for rep in range(6):
            d2 = sub(d, has)
            for j, s in enumerate(has):
                kind = d2['tok'][j, ps[s], 0]
                sc = SC_BF if kind == 2 else SC_CARD
                cur = d2['tok'][j, ps[s], 1]
                d2['tok'][j, ps[s], 1] = rng.choice(sc[sc != cur])
            p2, v2, l2 = run(d2)
            tvs.append(tv(p0[has], p2))
            dl = np.where(d['act_mask'][has], l2 - l0[has], np.nan)
            ncs.append(np.nanmax(dl, 1) - np.nanmin(dl, 1))   # non-constant part of the logit change
            dv.append(np.abs(v2 - v0[has]))
        tvs = np.concatenate(tvs); ncs = np.concatenate(ncs)
        # gradient of logit_i - logit_j wrt the asking card's embedding (injected delta at the psrc token)
        dd = sub(d, has)
        b = batch(dd)
        delta = torch.zeros(len(has), M.D, device=dev, requires_grad=True)
        orig = net.card_emb
        T = b['tok'].shape[1]
        hit = (torch.arange(T, device=dev)[None] == b['psrc'][:, None]).float()

        def patched(key, _o=orig):
            x = _o(key)
            if key.shape == b['tok'][..., 1].shape:
                x = x + hit[..., None] * delta[:, None, :]
            return x
        net.card_emb = patched
        with torch.enable_grad():
            net(b)
            L = _logits['l']
            am = b['act_mask']
            gn_diff, gn_one = [], []
            for j in range(len(has)):
                legal = torch.nonzero(am[j]).squeeze(-1).tolist()
                gs = []
                for i in legal[:6]:
                    g, = torch.autograd.grad(L[j, i], delta, retain_graph=True)
                    gs.append(g[j])
                G = torch.stack(gs)
                gn_one.append(G.norm(dim=1).mean().item())
                gn_diff.append((G[1:] - G[0]).norm(dim=1).mean().item())
        del net.card_emb
        R['ask'] = dict(n_states=int(len(has)), tv_mean=float(tvs.mean()), tv_median=float(np.median(tvs)),
                        tv_max=float(tvs.max()), frac_tv_gt_1e3=float((tvs > 1e-3).mean()),
                        frac_tv_gt_1e2=float((tvs > 1e-2).mean()),
                        logit_nonconst_mean=float(np.nanmean(ncs)), dvalue_mean=float(np.concatenate(dv).mean()),
                        grad_logit_norm=float(np.mean(gn_one)), grad_diff_norm=float(np.mean(gn_diff)),
                        grad_diff_ratio=float(np.mean(gn_diff) / max(np.mean(gn_one), 1e-12)))
    # ---------------- (b) option-relevant fields: edit one field of ONE option's referenced token / of the option
    tokf, actf = {}, {}
    sel = np.array([s for s in range(n) if nl[s] >= 2])
    for f in range(1, 24):
        d2 = sub(d, sel); ok = []
        for j, s in enumerate(sel):
            i = rng.integers(nl[s]); refs = ref_tokens(d['act'][s, i])
            if not refs:
                continue
            t = refs[0]
            if f == 1:
                kind = d2['tok'][j, t, 0]; sc = SC_BF if kind == 2 else SC_CARD
                cur = d2['tok'][j, t, 1]; d2['tok'][j, t, 1] = rng.choice(sc[sc != cur])
            else:
                d2['tok'][j, t, f] = edit_val(d2['tok'][j, t, f])
            ok.append(j)
        if not ok:
            break
        p2, v2, _ = run(d2)
        x = tv(p0[sel], p2)[ok]
        tokf['key' if f == 1 else f'f{f}'] = dict(tv_mean=float(x.mean()), frac_gt_1e3=float((x > 1e-3).mean()), n=len(ok))
    if tokf:
        R['ref_token_fields'] = tokf
    # psrc token's own live fields (asking card state)
    if len(has):
        pf = {}
        for f in range(2, 24):
            d2 = sub(d, has)
            for j, s in enumerate(has):
                d2['tok'][j, ps[s], f] = edit_val(d2['tok'][j, ps[s], f])
            p2, v2, _ = run(d2)
            x = tv(p0[has], p2)
            pf[f'f{f}'] = dict(tv_mean=float(x.mean()), frac_gt_1e3=float((x > 1e-3).mean()))
        R['ask_token_fields'] = pf
    for c in [0, 3, 19, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48] + ([49, 50, 51, 52, 53, 54, 55] if DIMS['K'] >= 56 else []):
        d2 = sub(d, sel); ok = []
        for j, s in enumerate(sel):
            i = rng.integers(nl[s])
            if c == 19:
                if d2['act'][j, i, 18] < 0: continue
                d2['act'][j, i, 19] += 1
            elif c == 32:
                if d2['act'][j, i, 31] <= 0: continue
                d2['act'][j, i, 32] += 1
            else:
                d2['act'][j, i, c] = edit_val(d2['act'][j, i, c])
            ok.append(j)
        if not ok:
            continue
        p2, _, _ = run(d2)
        x = tv(p0[sel], p2)[ok]
        actf[f'a{c}'] = dict(tv_mean=float(x.mean()), frac_gt_1e3=float((x > 1e-3).mean()), n=len(ok))
    R['act_fields'] = actf
    # glob
    gd = []
    for gi in range(72):
        d2 = sub(d, sel)
        d2['glob'][:, gi] = [edit_val(x) for x in d2['glob'][:, gi]]
        p2, v2, _ = run(d2)
        x = tv(p0[sel], p2)
        if x.max() < 1e-7 and np.abs(v2 - v0[sel]).max() < 1e-7:
            gd.append(gi)
    R['glob_dead'] = gd
    res[pkn(pk)] = R
    print(pkn(pk), json.dumps({k: R[k] for k in R if k in ('n', 'ask')}), flush=True)

# ---------------- (c) schema-v2 fields: value head and decisions, edited on all tokens of the kind
V2 = {'card': {17: 'play_ordinal', 18: 'hidden_this_turn', 19: 'eff_energy_cost', 20: 'eff_power_cost', 21: 'affordable',
               22: 'eff_assault', 23: 'eff_shield'},
      'bf': {15: 'starspring_self', 16: 'starspring_opp'},
      'chain': {16: 'addcost_paid', 17: 'src_play_ordinal', 18: 'trig_subject'},
      'pile': {2: 'owner', 3: 'zone', 4: 'count'}}
KN = {'card': 1, 'bf': 2, 'chain': 3, 'pile': 4}
C = {}
for pk in pks:
    d0 = {k: z[f'{pk}_{k}'] for k in OBS_KEYS}
    n = min(a.n, len(d0['pk'])); d = sub(d0, np.arange(n))
    nl = d['act_mask'].sum(1)
    p0, v0, _ = run(d)
    out = {}

    def meas(d2, name, rows):
        if not len(rows): return
        p2, v2, _ = run(d2)
        m = nl[rows] > 1
        out[name] = dict(n=int(len(rows)), dvalue=float(np.abs(v2 - v0)[rows].mean()),
                         tv=float(tv(p0, p2)[rows].mean()),
                         tv_multi=float(tv(p0, p2)[rows][m].mean()) if m.any() else None)
    for kn, fl in V2.items():
        for f, nm in fl.items():
            d2 = sub(d, np.arange(n)); rows = set()
            for s in range(n):
                for t in np.nonzero((d2['tok'][s, :, 0] == KN[kn]) & d2['tok_mask'][s])[0]:
                    x = d2['tok'][s, t, f]
                    if kn == 'card' and f in (19, 20, 21) and x == -1: continue   # n/a sentinel
                    d2['tok'][s, t, f] = edit_val(x); rows.add(s)
            meas(d2, f'{kn}.{nm}', np.array(sorted(rows), int))
    d2 = sub(d, np.arange(n)); rows = []
    for s in range(n):
        m = d2['tok'][s, :, 0] == 4
        if m.any(): d2['tok_mask'][s, m] = False; rows.append(s)
    meas(d2, 'pile.remove_all', np.array(rows, int))
    d2 = sub(d, np.arange(n)); d3 = sub(d, np.arange(n)); rows = []
    for s in range(n):
        m = (d2['tok'][s, :, 0] == 1) & (d2['tok'][s, :, 4] == 2) & (d2['tok'][s, :, 3] == 2) & d2['tok_mask'][s]
        if m.any():
            d2['tok_mask'][s, m] = False; rows.append(s)
            for t in np.nonzero(m)[0]:
                d3['tok'][s, t, 1] = rng.choice(SC_CARD[SC_CARD != d3['tok'][s, t, 1]])
    meas(d2, 'opp_hand.remove', np.array(rows, int)); meas(d3, 'opp_hand.swap_key', np.array(rows, int))
    C[pkn(pk)] = out
res['_schema_v2'] = C
json.dump(res, open(os.path.join(HERE, f'path_{a.tag}.json'), 'w'), indent=1)
print('wrote', f'path_{a.tag}.json')
