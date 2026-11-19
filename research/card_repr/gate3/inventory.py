"""Step 1: prompt-kind inventory over ~300 random Rengar-mirror games (autopay on) + per-pk state samples.
Writes gate3/inventory.json and gate3/samples.npz (up to --per-pk states per pk, random policy)."""
import argparse, json, os, sys, time
from collections import Counter, defaultdict
import numpy as np
from common import *
from rbenv import RBEnv, load_vocab

ap = argparse.ArgumentParser()
ap.add_argument('--games', type=int, default=300)
ap.add_argument('--per-pk', type=int, default=400)
ap.add_argument('--ckpt', default='')   # optional: policy-driven inventory
ap.add_argument('--out', default='inventory')
a = ap.parse_args()
vocab = load_vocab()
env = RBEnv(64, 'rengar', 'rengar', 4242, autopay=True)
D = env.dims
net = None
if a.ckpt:
    import torch
    net = load_net(a.ckpt, D, dev='cuda')
    from selfplay import act_rows
rng = np.random.default_rng(0)
cnt = Counter(); psrc_kind = defaultdict(Counter); psrc_key = defaultdict(Counter)
itype = defaultdict(Counter); fields = defaultdict(Counter); nleg = defaultdict(list)
vvals = defaultdict(Counter); seen = Counter()
samples = defaultdict(list)
FLD = dict(src=[1], abs=[2], ai=[3], tg=[4, 5, 6, 7], ch=[8, 9, 10, 11], mv=[12, 13, 14, 15], mu=[16, 17],
           dm=list(range(18, 30, 2)), ci=[33])
games = 0; t0 = time.time()
while games < a.games:
    o = env.observe()
    rows = np.nonzero(o['need'])[0]
    ch = np.zeros(env.n_envs, np.int64)
    if net is not None and len(rows):
        aa, _, _ = act_rows(net, o, rows, 'cuda'); ch[rows] = aa
    for r in rows:
        pk = int(o['pk'][r]); n = int(o['act_mask'][r].sum())
        if net is None:
            ch[r] = rng.integers(n)
        cnt[pk] += 1; nleg[pk].append(n)
        ps = int(o['psrc'][r])
        if ps >= 0:
            t = o['tok'][r, ps]
            psrc_kind[pk][KIND.get(int(t[0]), '?')] += 1
            psrc_key[pk][vocab[int(t[1])]] += 1
        else:
            psrc_kind[pk]['none'] += 1
        A = o['act'][r, :n]
        for row in A:
            itype[pk][INTENT[int(row[0])]] += 1
            for f, ix in FLD.items():
                if (row[ix] >= 0).any() if f != 'ai' else row[3] >= 0:
                    fields[pk][f] += 1
            if row[30] >= 0: fields[pk]['dst'] += 1
            if row[31] > 0: fields[pk]['v'] += 1; vvals[pk][int(row[32])] += 1
            if row[34] > 0: fields[pk]['nc'] += 1
            if row[35] >= 0: fields[pk]['bf'] += 1
            if row[36] > 0: fields[pk]['ps'] += 1
            for j, nm in ((37, 'alt'), (38, 'plc'), (39, 'dec')):
                if row[j] > 0: fields[pk][nm] += 1
            if row[40] >= 0: fields[pk]['first'] += 1
            if row[46] >= 0: fields[pk]['dst_ctrl'] += 1
            if row[47] > 0: fields[pk]['starts_combat'] += 1
            if row[48] > 0: fields[pk]['contest'] += 1
        seen[pk] += 1
        # reservoir sample
        if len(samples[pk]) < a.per_pk:
            samples[pk].append({k: o[k][r].copy() for k in OBS_KEYS})
        else:
            j = rng.integers(seen[pk])
            if j < a.per_pk:
                samples[pk][j] = {k: o[k][r].copy() for k in OBS_KEYS}
    env.step(ch)
    for res in env.results():
        games += 1
print('games', games, 'secs', round(time.time() - t0, 1), 'counters', env.counters())
tot = sum(cnt.values())
inv = {}
for pk in sorted(cnt, key=lambda k: -cnt[k]):
    inv[pkn(pk)] = dict(pk=pk, n=cnt[pk], per_game=round(cnt[pk] / games, 2), frac=round(cnt[pk] / tot, 4),
                        nlegal_mean=round(float(np.mean(nleg[pk])), 2), nlegal_max=int(np.max(nleg[pk])),
                        psrc_kind=dict(psrc_kind[pk]), psrc_key=dict(psrc_key[pk].most_common(12)),
                        intents=dict(itype[pk]), fields_per_option=dict(fields[pk]),
                        v_values=dict(vvals[pk].most_common(10)))
    print(f"{pkn(pk):14s} n={cnt[pk]:7d} /game={cnt[pk]/games:6.2f} nleg={np.mean(nleg[pk]):5.2f} psrc={dict(psrc_kind[pk])} "
          f"intents={dict(itype[pk])} fields={dict(fields[pk])}")
json.dump(dict(games=games, total=tot, pk=inv), open(os.path.join(HERE, a.out + '.json'), 'w'), indent=1)
out = {}
for pk, L in samples.items():
    for k in OBS_KEYS:
        out[f'{pk}_{k}'] = np.stack([s[k] for s in L])
np.savez_compressed(os.path.join(HERE, a.out + '_samples.npz'), **out)
