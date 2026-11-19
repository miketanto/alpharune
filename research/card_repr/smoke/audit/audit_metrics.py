"""Independent re-derivation of headline numbers + game-clustered bootstrap. Seed 12345 for bootstrap."""
import json, os, sys, math, random, collections
import torch
torch.set_num_threads(4)
H = os.path.dirname(os.path.abspath(__file__))
TR = os.path.join(H, '..', 'train')
P = torch.load(os.path.join(TR, 'packed', 'teacher.pt'))
meta = P['meta']; vocab = meta['vocab']; variants = meta['variants']
te = (P['split'] == 2).nonzero().squeeze(1).tolist()
na = P['na'].tolist(); ao = P['act_off'].tolist(); to = P['tok_off'].tolist(); nt = P['nt'].tolist()
pi = P['pi'].tolist(); game = P['game'].tolist(); var = P['variant'].tolist()
act_src = P['act'][:, 1].tolist(); act_abs = P['act'][:, 2].tolist()
tokkey = P['tok'][:, 1].tolist()
base_vid = variants.index('')
out = {}

def per_dec_top1(run):
    pr = torch.load(os.path.join(TR, 'runs', run, 'preds.pt'))
    assert pr['test_idx'].tolist() == te
    pv = pr['pv'].tolist()
    res = []; off = 0
    for i in te:
        n = na[i]; p = pv[off:off + n]; off += n
        m = max(p); am = p.index(m)
        res.append(1.0 if am == pi[i] else 0.0)
    assert off == len(pv)
    return res

# (6) headline E0 top1, all regime, test_all
r = json.load(open(os.path.join(TR, 'results.json')))
e0 = {}
for c in ('B', 'random'):
    vals = [sum(per_dec_top1(f'{c}__all__s{s}')) / len(te) for s in range(3)]
    rep = r['summary'][f'{c}__all']['E0']['test_all']['top1_pi']['per_seed']
    e0[c] = {'recomputed_per_seed': vals, 'reported_per_seed': rep}
out['E0_top1_recompute'] = e0
# uniform baseline
out['uniform_top1'] = sum(1.0 / na[i] for i in te) / len(te)

# E1 variant subsets: decisions in variant game where variant key is src/abs (card-source set)
PAIRS = {'primal_strength': 'punch_first'}
cards = json.load(open(os.path.join(H, '..', '..', 'cards.json')))
for s in cards['t2_minimal_pairs_synthetic']:
    PAIRS['synthetic:' + s['id']] = s['base']
def involved_src(i, k):
    for a in range(ao[i], ao[i] + na[i]):
        for x in (act_src[a], act_abs[a]):
            if 0 <= x < nt[i] and tokkey[to[i] + x] == k:
                return True
    return False
sel = collections.defaultdict(list)
for j, i in enumerate(te):
    if var[i] == base_vid: continue
    vk = variants[var[i]]
    if involved_src(i, vocab.index(vk)):
        sel[vk].append(j)
# teacher play rate -> majority baseline on card-play
tot = 0; maj = 0
pdec = {}
for vk, js in sel.items():
    k = vocab.index(vk); n_play = 0
    for j in js:
        i = te[j]; a = ao[i] + pi[i]
        x = act_src[a]; y = act_abs[a]
        if (0 <= x < nt[i] and tokkey[to[i] + x] == k) or (0 <= y < nt[i] and tokkey[to[i] + y] == k): n_play += 1
    pdec[vk] = (len(js), n_play / len(js))
    tot += len(js); maj += max(n_play, len(js) - n_play)
out['E1_card_source_decisions'] = {k: {'n': v[0], 'teacher_play_rate': v[1]} for k, v in pdec.items()}
out['E1_majority_baseline_pooled'] = maj / tot
out['E1_n_pooled'] = tot

# (4) game-clustered bootstrap of A - random pooled variant top1 (E1 'variant' set uses wider 'involved'; we use
# the card-source set and the full involved set both)
allv = sorted(j for js in sel.values() for j in js)
def top1s(cond):
    return [per_dec_top1(f'{cond}__base__s{s}') for s in range(3)]
T = {c: top1s(c) for c in ('A', 'random', 'B', 'stats')}
rng = random.Random(12345)
gidx = collections.defaultdict(list)
for j in allv: gidx[game[te[j]]].append(j)
glist = sorted(gidx)
def diff(c1, c2, js):
    return sum(sum(T[c1][s][j] - T[c2][s][j] for j in js) / len(js) for s in range(3)) / 3
boots = {}
for c1, c2 in (('A', 'random'), ('B', 'random'), ('stats', 'random')):
    pt = diff(c1, c2, allv)
    bs = []
    for _ in range(1000):
        gs = [glist[rng.randrange(len(glist))] for _ in glist]
        js = [j for g in gs for j in gidx[g]]
        bs.append(diff(c1, c2, js))
    bs.sort()
    seeds = [sum(T[c1][s][j] - T[c2][s][j] for j in allv) / len(allv) for s in range(3)]
    boots[f'{c1}-{c2}'] = {'point': pt, 'per_seed': seeds, 'game_boot_95': [bs[25], bs[974]], 'n_dec': len(allv), 'n_games': len(glist)}
out['E1_card_source_top1_diff_game_bootstrap'] = boots
json.dump(out, open(os.path.join(H, 'audit_metrics.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
