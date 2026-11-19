# Test 5: opponent mode. Seat 2 played in C++ ('random-player', optionally
# 'ismcts:sims=N' as argv[1]); only seat 1 decisions reach Python.
import collections, sys, time
from common import *
spec = sys.argv[1] if len(sys.argv) > 1 else 'random-player'
ng = int(sys.argv[2]) if len(sys.argv) > 2 else 100
env = RBEnv(16, 'rengar', 'rengar', seed=5000, opponent=spec)
rng = np.random.default_rng(1)
res, steps, seats = [], 0, collections.Counter()
t = time.time()
while len(res) < ng:
    o = env.observe()
    seats.update(o['seat'][o['need']].tolist())
    env.step(random_choices(o, rng)); res += env.results()
err = [r for r in res if r['error']]
print(spec, 'games', len(res), 'errors', len(err), 'seats seen', dict(seats),
      'winners', collections.Counter(r['winner'] for r in res), 'secs %.1f' % (time.time() - t))
assert not err and set(seats) == {1}
