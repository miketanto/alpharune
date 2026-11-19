"""Faithful micro-benchmark of smoke/train/train.py: the same Data.batch, Net, losses, AdamW,
LambdaLR schedule, grad clipping and loss sum, on the real packed/teacher.pt (cond A, regime all).
Only the number of steps differs, and there is no periodic validation pass.

  python bench_train.py --device cpu  --threads 1,2,4,8 --bs 64,256,1024 --d 64,128 --out ../raw/train_cpu.json
  python bench_train.py --device cuda --bs 64,256,1024,4096 --d 64,128 --out ../raw/train_gpu.json

Per (d, threads, bs) and repeat: run >= --min-steps steps and >= --min-secs seconds after 3 warm-up
steps, timing separately
  t_data    Data.batch(idx) on CPU (the index gather / padding, as in train.py),
  t_h2d     moving the batch to the device (cuda only),
  t_comp    forward + backward + clip + optimizer step (cuda: synchronised).
--infer also measures forward-only throughput (eval, no_grad) at batch 1 / 32 / 256 on pre-built
batches (positions/s), the number relevant to network-guided search.
--resident (cuda) also measures compute on batches already on the GPU (feeding cost removed).
CPU affinity is set to logical CPUs 8-15 (psutil if importable, else Win32 SetProcessAffinityMask).
"""
import argparse, json, os, sys, time, statistics as st, platform

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.normpath(os.path.join(HERE, '..', '..', 'train'))
sys.path.insert(0, TRAIN)


def set_affinity(cpus):
    try:
        import psutil
        psutil.Process().cpu_affinity(cpus)
        return 'psutil', psutil.Process().cpu_affinity()
    except ImportError:
        if os.name == 'nt':
            import ctypes
            mask = sum(1 << c for c in cpus)
            k = ctypes.windll.kernel32
            ok = k.SetProcessAffinityMask(k.GetCurrentProcess(), ctypes.c_size_t(mask))
            return 'win32', hex(mask) if ok else 'FAILED'
        os.sched_setaffinity(0, cpus)
        return 'sched', sorted(os.sched_getaffinity(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', default='8')
    ap.add_argument('--bs', default='256')
    ap.add_argument('--d', default='64')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--min-steps', type=int, default=8)
    ap.add_argument('--min-secs', type=float, default=6.0)
    ap.add_argument('--infer', action='store_true')
    ap.add_argument('--resident', action='store_true')
    ap.add_argument('--out', required=True)
    ap.add_argument('--data', default=os.path.join(HERE, '..', 'data', 'subset_teacher.pt'),
                    help='packed decisions in prep.py format (default: the ~60k-decision subset bench_prep.py writes)')
    a = ap.parse_args()
    aff = set_affinity(list(range(8, 16)))
    import torch
    import train as T
    dev = torch.device(a.device)
    t0 = time.time()
    try:
        P = torch.load(a.data, weights_only=False)
        RP = torch.load(os.path.join(T.PK, 'reprs.pt'), weights_only=False)
    except TypeError:
        P = torch.load(a.data); RP = torch.load(os.path.join(T.PK, 'reprs.pt'))
    t_load = time.time() - t0
    meta = P['meta']; vocab = meta['vocab']; special = RP['special']
    special_idx = torch.tensor([special.index(k) + 1 if k in special else 0 for k in vocab] + [0])
    tr_idx = (P['split'] == 0).nonzero().squeeze(1)
    table = torch.cat([RP['tables']['A'], torch.zeros(1, RP['tables']['A'].shape[1])])
    n_pk = len(meta['pk_names'])
    data = T.Data(P)
    info = {'device': a.device, 'torch': torch.__version__, 'python': sys.version.split()[0],
            'affinity': [aff[0], str(aff[1])], 'load_data_secs': round(t_load, 2),
            'n_decisions_packed': int(len(P['pi'])), 'n_train_decisions': int(len(tr_idx)),
            'tokens_per_decision_mean': round(float(P['nt'].float().mean()), 2),
            'actions_per_decision_mean': round(float(P['na'].float().mean()), 2)}
    if a.device == 'cuda':
        info['gpu'] = torch.cuda.get_device_name(0)
    print(info, flush=True)
    results, infer = [], []
    gen = torch.Generator().manual_seed(1)

    def to_dev(b):
        return {k: v.to(dev, non_blocking=False) for k, v in b.items()}

    import contextlib
    # train.py's Net.forward builds one tensor without a device (torch.zeros(len(g), 1) for the
    # global token's padding flag), so on cuda the model calls run under torch.device(dev) as the
    # default device; on cpu this is a no-op. Data.batch stays on the CPU, as in train.py.
    def ondev():
        return torch.device(dev) if a.device != 'cpu' else contextlib.nullcontext()

    def sync():
        if a.device == 'cuda':
            torch.cuda.synchronize()

    for d in [int(x) for x in a.d.split(',')]:
        T.D = d
        threads_list = [int(x) for x in a.threads.split(',')]
        for th in threads_list:
            torch.set_num_threads(th)
            for bs in [int(x) for x in a.bs.split(',')]:
                for rep in range(a.reps):
                    torch.manual_seed(rep)
                    net = T.Net(table, special_idx, False, n_pk).to(dev)
                    nparam = sum(p.numel() for n_, p in net.named_parameters() if n_ != 'R')
                    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
                    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200))
                    td = th2 = tc = 0.0; n = 0; tstart = None; resident = []
                    i = 0
                    while True:
                        if i == 3:
                            td = th2 = tc = 0.0; n = 0; tstart = time.perf_counter()
                        s0 = time.perf_counter()
                        idx = tr_idx[torch.randint(len(tr_idx), (bs,), generator=gen)]
                        b = data.batch(idx)
                        s1 = time.perf_counter()
                        if a.device != 'cpu':
                            b = to_dev(b); sync()
                        s2 = time.perf_counter()
                        if a.resident and len(resident) < 4:
                            resident.append(b)
                        with ondev():
                            lv_, lq_, lval = T.losses(net, b)
                        loss = lv_ + lq_ + lval
                        opt.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                        opt.step(); sched.step()
                        _ = [lv_.item(), lq_.item(), lval.item()]   # train.py reads the losses every step
                        sync()
                        s3 = time.perf_counter()
                        td += s1 - s0; th2 += s2 - s1; tc += s3 - s2; n += i >= 3
                        i += 1
                        if i > 3 and n >= a.min_steps and time.perf_counter() - tstart >= a.min_secs:
                            break
                    wall = time.perf_counter() - tstart
                    r = {'d': d, 'threads': th, 'bs': bs, 'rep': rep, 'params': nparam, 'steps': n,
                         'wall_s': round(wall, 3), 'steps_per_s': round(n / wall, 3),
                         'samples_per_s': round(n * bs / wall, 1), 'data_s_per_step': round(td / n, 5),
                         'h2d_s_per_step': round(th2 / n, 5), 'compute_s_per_step': round(tc / n, 5),
                         'data_frac': round(td / (td + th2 + tc), 3)}
                    if a.resident and resident:
                        sync(); r0 = time.perf_counter(); m = 0
                        while m < max(20, n) and time.perf_counter() - r0 < a.min_secs:
                            bb = resident[m % len(resident)]
                            with ondev():
                                lv_, lq_, lval = T.losses(net, bb)
                            loss = lv_ + lq_ + lval
                            opt.zero_grad(); loss.backward()
                            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                            opt.step(); sched.step(); _ = loss.item(); m += 1
                        sync(); rw = time.perf_counter() - r0
                        r['resident_steps_per_s'] = round(m / rw, 2)
                        r['resident_samples_per_s'] = round(m * bs / rw, 1)
                    results.append(r)
                    print(r, flush=True)
            # inference throughput for this d / threads
            if a.infer:
                torch.manual_seed(0)
                net = T.Net(table, special_idx, False, n_pk).to(dev).eval()
                for ib in (1, 32, 256):
                    batches = [data.batch(tr_idx[torch.randint(len(tr_idx), (ib,), generator=gen)]) for _ in range(16)]
                    for rep in range(a.reps):
                        with torch.no_grad():
                            for bb in batches[:2]:
                                with ondev():
                                    net(to_dev(bb) if a.device != 'cpu' else bb)
                            sync(); r0 = time.perf_counter(); m = 0; tf = 0.0
                            while m < 10 or time.perf_counter() - r0 < 3.0:
                                bb = batches[m % len(batches)]
                                f0 = time.perf_counter()
                                bb = to_dev(bb) if a.device != 'cpu' else bb
                                with ondev():
                                    net(bb)
                                sync(); tf += time.perf_counter() - f0; m += 1
                            w = time.perf_counter() - r0
                        x = {'d': d, 'threads': th, 'infer_batch': ib, 'rep': rep, 'calls': m,
                             'calls_per_s': round(m / w, 1), 'positions_per_s': round(m * ib / w, 1),
                             'ms_per_call': round(1000 * w / m, 3),
                             'note': 'forward only, prebuilt CPU batches, incl. host->device copy on cuda'}
                        infer.append(x); print(x, flush=True)

    def agg(rows, keys, val):
        groups = {}
        for r in rows:
            groups.setdefault(tuple(r[k] for k in keys), []).append(r)
        out = []
        for k, rs in groups.items():
            o = dict(zip(keys, k))
            for v in val:
                xs = [r[v] for r in rs if v in r]
                if xs:
                    o[v + '_median'] = round(st.median(xs), 4)
                    o[v + '_min'] = min(xs); o[v + '_max'] = max(xs)
            out.append(o)
        return out
    summary = agg(results, ['d', 'threads', 'bs'], ['steps_per_s', 'samples_per_s', 'data_s_per_step',
                                                   'h2d_s_per_step', 'compute_s_per_step', 'data_frac',
                                                   'resident_steps_per_s', 'resident_samples_per_s'])
    isum = agg(infer, ['d', 'threads', 'infer_batch'], ['positions_per_s', 'ms_per_call'])
    json.dump({'info': info, 'host': platform.node(), 'summary': summary, 'infer_summary': isum,
               'raw': results, 'infer_raw': infer, 'time': time.strftime('%Y-%m-%d %H:%M:%S')},
              open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
