"""Data prep cost: prep.parse_file (the JSON-lines -> int16 packing of smoke/train/prep.py) on a
sample of teacher decision files, single process and Pool(NPOOL), affinity CPUs 8-15. Nothing is written
except raw/prep.json. Reports decisions/s; the full prep also runs pack() + torch.save (~once)."""
import os, sys, time, json, glob, statistics as st
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'train'))
sys.path.insert(0, HERE)
from bench_train import set_affinity
import prep
import torch
NPOOL = 4   # memory: each spawned worker imports torch (~150 MB)

def n_dec(r): return len(r[-1]['pi'])

if __name__ == '__main__':
    print(set_affinity(list(range(8, 16))))
    files = sorted(glob.glob(os.path.join(prep.DATA, 'out', 'base_teacher__s*.decisions.jsonl.gz')))[:16]
    out = {'files': [os.path.basename(f) for f in files], 'serial': [], 'pool': []}
    for rep in range(3):
        t = time.time(); r = prep.parse_file(files[rep]); dt = time.time() - t
        out['serial'].append({'file': os.path.basename(files[rep]), 'decisions': n_dec(r), 'secs': round(dt, 2),
                              'dec_per_s': round(n_dec(r) / dt, 1)})
        print(out['serial'][-1], flush=True)
    for rep in range(3):
        t = time.time()
        with Pool(NPOOL) as pool:
            rs = pool.map(prep.parse_file, files, chunksize=1)
        dt = time.time() - t; n = sum(n_dec(r) for r in rs)
        out['pool'].append({'decisions': n, 'secs': round(dt, 2), 'dec_per_s': round(n / dt, 1)})
        print(out['pool'][-1], flush=True)
    out['serial_dec_per_s_median'] = st.median(x['dec_per_s'] for x in out['serial'])
    out['pool_dec_per_s_median'] = st.median(x['dec_per_s'] for x in out['pool'])
    # subset for the training benchmarks (memory: the full packed/teacher.pt is ~0.5 GB on disk)
    t = time.time()
    rs.sort(key=lambda r: r[0])
    P = prep.pack([r[1:] for r in rs])
    gids = P.pop('gid')
    games = sorted(set(gids)); gi = {g: i for i, g in enumerate(games)}
    # split by game: every 10th game -> val (1), the rest train (0)
    P['split'] = torch.tensor([1 if gi[g] % 10 == 7 else 0 for g in gids], dtype=torch.long)
    P['variant'] = torch.zeros(len(gids), dtype=torch.long)
    pks = sorted(set(P['pk']))
    P['pk'] = torch.tensor([pks.index(x) for x in P['pk']], dtype=torch.long)
    P['meta'] = {'vocab': prep.VOCAB, 'variants': [''], 'pk_names': pks, 'files': out['files']}
    torch.save(P, os.path.join(HERE, '..', 'data', 'subset_teacher.pt'))
    out['pack_and_save_secs'] = round(time.time() - t, 2)
    out['subset_decisions'] = len(gids)
    print('subset', len(gids), 'decisions, pack+save', out['pack_and_save_secs'], 's')
    json.dump(out, open(os.path.join(HERE, '..', 'raw', 'prep.json'), 'w'), indent=1)
