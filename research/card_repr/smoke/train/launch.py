"""Run every (condition, regime, seed) with a fixed number of steps, 3 workers x 2 threads
(6 torch threads in total). Jobs are ordered seed-major so that, if time runs out, whole
seeds are complete for every condition."""
import subprocess, sys, os, time, threading, json

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
WORKERS, THREADS = 3, 2
CONDS = ['id', 'random', 'stats', 'A', 'B', 'B2', 'B2A']
jobs = []
for seed in (0, 1, 2):
    for regime in ('base', 'all'):
        for c in CONDS + (['Bshuf'] if regime == 'base' else []):
            jobs.append((c, regime, seed))
lock = threading.Lock()
status = open(os.path.join(HERE, 'launch_log.txt'), 'a')


def worker(wid):
    while True:
        with lock:
            if not jobs:
                return
            c, r, s = jobs.pop(0)
        name = f'{c}__{r}__s{s}'
        if os.path.exists(os.path.join(HERE, 'runs', name, 'metrics.json')):
            continue
        t = time.time()
        with lock:
            status.write(f'{time.strftime("%H:%M:%S")} start {name} w{wid}\n'); status.flush()
        p = subprocess.run([sys.executable, 'train.py', '--cond', c, '--regime', r, '--seed', str(s),
                            '--steps', str(STEPS), '--threads', str(THREADS)], cwd=HERE,
                           stdout=subprocess.DEVNULL, stderr=open(os.path.join(HERE, 'runs', f'{name}.stderr'), 'w'))
        with lock:
            status.write(f'{time.strftime("%H:%M:%S")} end {name} rc={p.returncode} {time.time()-t:.0f}s\n'); status.flush()


os.makedirs(os.path.join(HERE, 'runs'), exist_ok=True)
ts = [threading.Thread(target=worker, args=(i,)) for i in range(WORKERS)]
for t in ts:
    t.start()
for t in ts:
    t.join()
status.write('ALL DONE\n'); status.flush()
