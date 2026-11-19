# Test 2: same seed + same choices -> identical observations and winners.
from common import *
def run():
    env = RBEnv(8, 'rengar', 'rengar', seed=77)
    rec = []
    res, _ = play(env, 24, np.random.default_rng(5), record=rec)
    return rec, res
a, ra = run(); b, rb = run()
assert len(a) == len(b), (len(a), len(b))
for (oa, ca), (ob, cb) in zip(a, b):
    for k in oa: assert np.array_equal(oa[k], ob[k]), k
    assert np.array_equal(ca, cb)
key = lambda r: [(x['env'], x['game_seed'], x['winner'], x['length']) for x in r]
assert key(ra) == key(rb)
print('determinism ok:', len(a), 'steps,', len(ra), 'games identical')
