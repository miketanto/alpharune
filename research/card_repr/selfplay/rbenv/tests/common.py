import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from rbenv import RBEnv

def random_choices(o, rng):
    m = o['act_mask']
    r = rng.random(m.shape) * m
    c = r.argmax(1).astype(np.int32)
    c[~o['need']] = 0
    return c

def play(env, n_games, rng, record=None, max_steps=10**7):
    res = []
    steps = 0
    while len(res) < n_games and steps < max_steps:
        o = env.observe()
        c = random_choices(o, rng)
        if record is not None:
            record.append((o, c))
        env.step(c)
        steps += 1
        res += env.results()
    return res, steps
