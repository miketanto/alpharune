"""Buffer diagnostics: is root Q in the decider's view, and how good are the value targets (z, vs, v0)?"""
import glob
import numpy as np

fs = sorted(glob.glob('buf/c_*.npz'))[-4:]
for f in fs:
    z = np.load(f)
    N = z['N'].astype(np.float32); Q = z['Q'].astype(np.float32)
    vs, zz, v0 = z['vs'], z['z'], z['v0']
    m = (N > 0).sum(1) >= 2
    aN = N.argmax(1); aQ = np.where(N > 0, Q, -9).argmax(1)
    print(f[-12:], 'n', len(N), 'multi', m.mean().round(3), 'argmaxN==argmaxQ', (aN == aQ)[m].mean().round(3),
          'corr(vs,z)', np.corrcoef(vs, zz)[0, 1].round(3), 'corr(v0,z)', np.corrcoef(v0, zz)[0, 1].round(3),
          'mse vs', ((vs - zz) ** 2).mean().round(3), 'mse v0', ((v0 - zz) ** 2).mean().round(3),
          'maxN', N.max(1).mean().round(1), 'visited', (N > 0).sum(1).mean().round(2),
          'sumN', N.sum(1).mean().round(1))
