"""Search policy targets from generator records (shared by learner/selfplay.py and the E2 checks).

visits: pi'(a) = N(a) / sum N  (AlphaZero).
cq:     completed-Q improved policy (Danihelka et al. 2022, "Policy improvement by planning with Gumbel",
        Eq. 11 / App. D): pi'(a) = softmax(log P(a) + sigma(completedQ(a))) over the legal options,
        completedQ = Q(a) for visited options, v_mix for the others,
        sigma(q) = (c_visit + max_b N(b)) * c_scale * (q + 1) / 2   (values in [-1, 1] mapped to [0, 1]),
        v_mix = (v0 + sumN / sum_{visited} P * sum_{visited} P(b) Q(b)) / (1 + sumN).
"""
import numpy as np


def policy_target(P, N, Q, v0, mask, kind='visits', c_visit=50.0, c_scale=0.1):
    P = P.astype(np.float32); N = N.astype(np.float32); Q = Q.astype(np.float32)
    if kind == 'visits':
        t = N * mask
        s = t.sum(1, keepdims=True)
        return np.where(s > 0, t / np.maximum(s, 1e-9), P * mask)
    vis = (N > 0) & mask
    sn = N.sum(1)
    pv = (P * vis).sum(1)
    pq = (P * np.where(vis, Q, 0)).sum(1)
    vmix = (v0 + sn / np.maximum(pv, 1e-9) * pq) / (1 + sn)
    cq = np.where(vis, Q, vmix[:, None])
    sig = (c_visit + N.max(1, keepdims=True)) * c_scale * (cq + 1) / 2
    lg = np.where(mask, np.log(np.clip(P, 1e-8, 1)) + sig, -1e9)
    lg -= lg.max(1, keepdims=True)
    e = np.exp(lg) * mask
    return e / e.sum(1, keepdims=True)
