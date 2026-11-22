"""Sanity checks of generator records: python3 check_buf.py <dir or holdout.npz> [ckpt] [max_chunks]"""
import glob, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'search'))
import rbsearch  # noqa: E402

src = sys.argv[1]
ck = sys.argv[2] if len(sys.argv) > 2 else os.path.join(rbsearch.SP, 'runs', 'ov_main', 'league_02080.pt')
fs = [src] if src.endswith('.npz') else sorted(glob.glob(os.path.join(src, 'c_*.npz')))[:int(sys.argv[3]) if len(sys.argv) > 3 else 8]
d = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in np.load(fs[0]).files}
n = len(d['z'])
P = d['P'].astype(np.float32); N = d['N'].astype(np.float32); am = d['act_mask']
print('records', n, 'P sum', P.sum(1).min(), P.sum(1).max(), 'N sum', np.unique(N.sum(1))[:6])
print('act_mask count == min(nlegal, A):', (am.sum(1) == np.minimum(d['nlegal'], am.shape[1])).mean())
print('visits outside mask', float((N * ~am).sum()), 'z values', np.unique(d['z'], return_counts=True))
pi = N / N.sum(1, keepdims=True)
H = lambda p: -(p * np.log(np.clip(p, 1e-9, 1))).sum(1)
print('argmax(N) != argmax(P):', (pi.argmax(1) != P.argmax(1)).mean())
print('H(P) %.3f  H(pi) %.3f  KL(pi||P) %.3f' % (H(P).mean(), H(pi).mean(),
      (pi * (np.log(np.clip(pi, 1e-9, 1)) - np.log(np.clip(P, 1e-9, 1)))).sum(1).mean()))
dims = rbsearch.init()
net = rbsearch.load_net(ck, dims, 'cuda')
ev = rbsearch.Evaluator(net, 'cuda', dims['A'])
rows = dict(glob=d['glob'].astype(np.int32), tok=d['tok'].astype(np.int32), tok_mask=d['tok_mask'],
            act=d['act'].astype(np.int32), act_mask=am, pk=d['pk'].astype(np.int32), psrc=d['psrc'].astype(np.int32))
lp, v = ev(rows, n)
p2 = np.exp(lp)
if ck.endswith('league_02080.pt') or 'wupd' not in d or (d['wupd'] == d['wupd'][0]).all():
    print('net(stored obs) vs stored P: max abs %.4f, mean L1 %.4f; |v - v0| max %.4f mean %.4f' % (
        np.abs(p2 - P).max(), np.abs(p2 - P).sum(1).mean(), np.abs(v - d['v0']).max(), np.abs(v - d['v0']).mean()))
print('corr(vs, z) %.3f  corr(v0, z) %.3f' % (np.corrcoef(d['vs'], d['z'])[0, 1], np.corrcoef(d['v0'], d['z'])[0, 1]))
