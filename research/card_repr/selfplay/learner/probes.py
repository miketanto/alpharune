"""Card-check hook: fixed probe states with rule-defined right answers (cardexp/collect_probes.py).

A probe file is selfplay/learner/probes/<name>.npz holding one observation batch in the rbenv
layout (glob, tok, tok_mask, act, act_mask, pk) plus `good` (B,A) bool: the options that are
right by rule. run_probes reports, per file, the mean policy mass on the right options and the
fraction of states whose argmax is right; for conditional probes (npz with `play`, `cond`) also the
policy mass on / argmax rate of playing the card when the condition holds vs not. With no probe files it
returns {}. run_probes(net, dev, names=[...]) restricts to those probe files.
"""
import glob, os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE_DIR = os.path.join(HERE, 'probes')


@torch.no_grad()
def run_probes(net, dev, names=None):
    out = {}
    for path in sorted(glob.glob(os.path.join(PROBE_DIR, '*.npz'))):
        if names and os.path.splitext(os.path.basename(path))[0] not in names:
            continue
        d = np.load(path)
        if d['tok'].shape[-1] != net.F or d['act'].shape[-1] != net.K:   # recorded under another schema
            out[os.path.splitext(os.path.basename(path))[0]] = dict(
                skipped='probe recorded with F=%d K=%d, net has F=%d K=%d (schema v1 probe: re-collect with '
                        'cardexp/collect_probes.py)' % (d['tok'].shape[-1], d['act'].shape[-1], net.F, net.K))
            continue
        b = {k: torch.from_numpy(d[k]).to(dev) for k in ('glob', 'tok', 'tok_mask', 'act', 'act_mask', 'pk')}
        if 'psrc' in d:
            b['psrc'] = torch.from_numpy(d['psrc']).to(dev)
        good = torch.from_numpy(d['good']).to(dev)
        was = net.training
        net.eval()
        logp, v = net(b)
        net.train(was)
        p = logp.exp()
        r = dict(mass_right=float((p * good).sum(-1).mean()),
                 argmax_right=float(good.gather(1, p.argmax(-1, keepdim=True)).float().mean()),
                 value=float(v.mean()), n=int(len(p)))
        if 'play' in d and d['play'].any():   # conditional probes: P(play the card) by condition
            play = torch.from_numpy(d['play']).to(dev)
            pp = (p * play).sum(-1)
            top = play.gather(1, p.argmax(-1, keepdim=True)).squeeze(1).float()
            cond = torch.from_numpy(d['cond']).to(dev).bool()
            for nm, msk in (('true', cond), ('false', ~cond)):
                if msk.any():
                    r[f'play_mass_cond_{nm}'] = float(pp[msk].mean())
                    r[f'play_top_cond_{nm}'] = float(top[msk].mean())
                    r[f'n_cond_{nm}'] = int(msk.sum())
        out[os.path.splitext(os.path.basename(path))[0]] = r
    return out
