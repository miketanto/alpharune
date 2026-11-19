# Test 1: random legal actions, 200 games of the Rengar mirror, no errors.
import time, collections
from common import *
env = RBEnv(32, 'rengar', 'rengar', seed=1000)
print('dims', env.dims)
rng = np.random.default_rng(0)
t = time.time()
res, steps = play(env, 200, rng)
res = res[:200]
err = [r for r in res if r['error']]
print('games', len(res), 'errors', len(err), 'winners', collections.Counter(r['winner'] for r in res),
      'mean ext', np.mean([r['n_ext'] for r in res]), 'mean len', np.mean([r['length'] for r in res]),
      'mean turns', np.mean([r['turns'] for r in res]), 'secs %.1f' % (time.time() - t))
print('counters', env.counters())
for r in err[:5]: print(r)
assert not err
