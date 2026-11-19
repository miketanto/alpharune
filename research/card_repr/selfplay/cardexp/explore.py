import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'rbenv'))
from rbenv import RBEnv, load_vocab
from make_exp_decks import ALL_VARIANTS
voc = load_vocab(); V = {k: i for i, k in enumerate(voc)}
ctrl = V['synthetic:punch_first__ctrl_scores']
env = RBEnv(8, 'rengar__exp_train', 'rengar__exp_train', 5, variant=ALL_VARIANTS)
rng = np.random.default_rng(0)
after = {}
shown = 0
for step in range(4000):
    o = env.observe()
    for e in np.nonzero(o['need'])[0]:
        if after.get(e):
            ps = o['psrc'][e]; m = o['act_mask'][e]
            print('pk', o['pk'][e], 'psrc', ps, 'psrc key', voc[o['tok'][e, ps, 1]] if ps >= 0 else None,
                  'acts', [(int(a[0]), int(a[4]), int(a[8]), (voc[o['tok'][e, a[4], 1]], int(o['tok'][e, a[4], 2]), int(o['tok'][e, a[4], 4])) if a[4] >= 0 else None) for a in o['act'][e][m]][:8])
            after[e] -= 1; shown += 1
    c = np.zeros(8, np.int32)
    for e in np.nonzero(o['need'])[0]:
        m = o['act_mask'][e]
        cand = [i for i in np.nonzero(m)[0] if o['act'][e, i, 0] in (0, 8, 5) and o['act'][e, i, 1] >= 0 and o['tok'][e, o['act'][e, i, 1], 1] == ctrl]
        if cand and rng.random() < 0.5:
            c[e] = cand[0]; after[e] = 3
        else:
            c[e] = rng.choice(np.nonzero(m)[0])
    env.step(c); env.results()
    if shown > 25: break
