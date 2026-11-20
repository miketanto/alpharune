"""Counterfactual-pair supervision for the value head (gate 4 follow-up, 2026-11-19).

Data: research/card_repr/gate4/cf_gen.py chunks (root observations of an original position and its edited
variants, each with a playout win rate under the sampled policy, decider's perspective, common random numbers).

Loss on a batch of positions (every variant of each position in the batch):
  value term : mean over variants of (v - (2 * win - 1))^2
  pair term  : mean over edits of ((v_edit - v_orig) - 2 * (win_edit - win_orig))^2   (value units, [-1, 1])
selfplay.py adds  cf_coef * value + cf_pair_coef * pair  to each PPO minibatch loss (--cf-data ...).
A fixed fraction of the positions (hash of the position id) is a validation split, never trained on.
"""
import glob, json, os, zlib
import numpy as np
import torch

KEYS = ('glob', 'tok', 'tok_mask', 'act', 'act_mask', 'pk', 'psrc')


class CFData:
    def __init__(self, pattern, val_frac=0.1, seed=0):
        files = sorted(glob.glob(pattern))
        assert files, 'no CF data: ' + pattern
        arr = {k: [] for k in KEYS}
        meta = []
        for fn in files:
            z = np.load(fn)
            m = json.loads(str(z['meta']))
            for k in KEYS:
                arr[k].append(z[k])
            meta += m
        self.obs = {k: np.concatenate(v) for k, v in arr.items()}
        self.meta = meta
        self.win = np.array([m['win'] for m in meta], np.float32)
        self.d = np.array([m['d'] for m in meta], np.float32)
        by = {}
        for i, m in enumerate(meta):
            by.setdefault(m['pid'], []).append(i)
        self.groups = {'train': [], 'val': []}
        for pid, ix in by.items():
            o = [i for i in ix if meta[i]['fam'] == 'orig']
            if len(o) != 1:
                continue
            e = [i for i in ix if meta[i]['fam'] != 'orig']
            split = 'val' if (zlib.crc32(pid.encode()) % 1000) < 1000 * val_frac else 'train'
            self.groups[split].append((o[0], e))
        self.rng = np.random.default_rng(seed)
        fam = {}
        for m in meta:
            fam[m['fam']] = fam.get(m['fam'], 0) + 1
        self.summary = dict(files=len(files), variants=len(meta), positions=len(by),
                            train_pos=len(self.groups['train']), val_pos=len(self.groups['val']),
                            with_edits=sum(1 for g in self.groups['train'] + self.groups['val'] if g[1]), families=fam)

    def batch(self, n_pos, split='train', groups=None):
        G = self.groups[split]
        if groups is None:
            groups = [G[i] for i in self.rng.choice(len(G), min(n_pos, len(G)), replace=False)]
        idx, pairs = [], []
        for o, e in groups:
            io = len(idx); idx.append(o)
            for j in e:
                pairs.append((io, len(idx), self.d[j])); idx.append(j)
        return np.array(idx), pairs

    def tensors(self, idx, dev):
        from selfplay import to_dev
        b = to_dev(self.obs, idx, dev)
        for k in ('glob', 'tok', 'act', 'pk', 'psrc'):
            b[k] = b[k].long()
        return b

    def loss(self, net, n_pos, dev, split='train', groups=None):
        idx, pairs = self.batch(n_pos, split, groups)
        b = self.tensors(idx, dev)
        _, v = net(b)
        tgt = torch.from_numpy(2 * self.win[idx] - 1).to(dev)
        lv = ((v - tgt) ** 2).mean()
        if pairs:
            io = torch.tensor([p[0] for p in pairs], device=dev)
            ie = torch.tensor([p[1] for p in pairs], device=dev)
            dd = torch.tensor([2 * p[2] for p in pairs], device=dev, dtype=torch.float32)
            lp = ((v[ie] - v[io] - dd) ** 2).mean()
        else:
            lp = torch.zeros((), device=dev)
        return lv, lp

    @torch.no_grad()
    def evaluate(self, net, dev, split='val', chunk=32):
        """Value MSE, pair MSE, and r(dv, dwin) over the whole split."""
        G = self.groups[split]
        sv = sp = 0.0; nv = npr = 0
        xs, ys = [], []
        was = net.training
        net.eval()
        for i in range(0, len(G), chunk):
            gs = G[i:i + chunk]
            idx, pairs = self.batch(0, split, gs)
            b = self.tensors(idx, dev)
            _, v = net(b)
            v = v.float().cpu().numpy()
            sv += float(((v - (2 * self.win[idx] - 1)) ** 2).sum()); nv += len(idx)
            for io, ie, d in pairs:
                dv = (v[ie] - v[io]) / 2
                sp += float((2 * dv - 2 * d) ** 2); npr += 1
                xs.append(dv); ys.append(d)
        if was:
            net.train()
        r = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 and np.std(xs) > 0 else float('nan')
        return dict(v_mse=sv / max(nv, 1), pair_mse=sp / max(npr, 1), pair_r=r, n_pairs=npr,
                    dv_abs=float(np.mean(np.abs(xs))) if xs else float('nan'))
