"""Card-check hook: fixed probe states with rule-defined right answers (to be filled later).

A probe file is selfplay/learner/probes/<name>.npz holding one observation batch in the rbenv
layout (glob, tok, tok_mask, act, act_mask, pk) plus `good` (B,A) bool: the options that are
right by rule. run_probes reports, per file, the mean policy mass on the right options and the
fraction of states whose argmax is right. With no probe files it returns {}.
"""
import glob, os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE_DIR = os.path.join(HERE, 'probes')


@torch.no_grad()
def run_probes(net, dev):
    out = {}
    for path in sorted(glob.glob(os.path.join(PROBE_DIR, '*.npz'))):
        d = np.load(path)
        b = {k: torch.from_numpy(d[k]).to(dev) for k in ('glob', 'tok', 'tok_mask', 'act', 'act_mask', 'pk')}
        if 'psrc' in d:
            b['psrc'] = torch.from_numpy(d['psrc']).to(dev)
        good = torch.from_numpy(d['good']).to(dev)
        was = net.training
        net.eval()
        logp, v = net(b)
        net.train(was)
        p = logp.exp()
        out[os.path.splitext(os.path.basename(path))[0]] = dict(
            mass_right=float((p * good).sum(-1).mean()),
            argmax_right=float(good.gather(1, p.argmax(-1, keepdim=True)).float().mean()),
            value=float(v.mean()), n=int(len(p)))
    return out
