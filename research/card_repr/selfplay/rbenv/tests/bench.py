# Test 4: throughput with random actions. argv: n_envs n_procs seconds
import multiprocessing as mp, sys, time
from common import *
def worker(args):
    n, seed, secs = args
    env = RBEnv(n, 'rengar', 'rengar', seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(20): env.step(random_choices(env.observe(), rng)); env.results()   # warm-up
    g = d = 0; t = time.time()
    while time.time() - t < secs:
        o = env.observe(); d += int(o['need'].sum())
        env.step(random_choices(o, rng)); g += len(env.results())
    return g, d, time.time() - t
if __name__ == '__main__':
    n, p, secs = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    with mp.get_context('fork').Pool(p) as pool:
        out = pool.map(worker, [(n, 10**6 * (i + 1), secs) for i in range(p)])
    g = sum(x[0] for x in out); d = sum(x[1] for x in out); s = max(x[2] for x in out)
    print('n_envs %d procs %d: %.0f games/s, %.0f decisions/s (%.0f games/h)' % (n, p, g / s, d / s, 3600 * g / s))
