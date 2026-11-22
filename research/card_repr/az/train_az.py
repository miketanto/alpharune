"""AlphaZero-style trainer (az/README.txt): supervised on the generator's search records.

  python3 train_az.py --init ../selfplay/runs/g2_main/league_01000.pt --buf buf --run run

Loss per record: cross-entropy to the search's visit distribution pi = N / sum N (policy) +
--vf * (v - z)^2 (value; z = the game outcome from the decider's view). Records come from the rolling
chunk files the generator writes (gen_az.py); the newest --window records are kept in memory and sampled
uniformly. Replay ratio: training stops when (records trained on) / (records ingested) reaches --reuse
and waits for new data. Every --export-every steps the weights are written to <run>/gen_weights.pt
(the generator reloads them); every --ckpt-every steps a checkpoint <run>/az_<step>.pt ({'net', 'update',
'repr'}, loadable by league/common.load_net and the eval scripts). Before a new chunk is trained on, its
loss is measured ("fresh": data the net has not seen yet). Log: <run>/train_log.jsonl.
"""
import argparse, collections, glob, json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SP = os.path.normpath(os.path.join(HERE, '..', 'selfplay'))
sys.path.insert(0, os.path.join(SP, 'learner'))
sys.path.insert(0, os.path.join(SP, 'league'))
sys.path.insert(0, os.path.join(SP, 'rbenv'))

KEYS = ('glob', 'tok', 'tok_mask', 'act', 'act_mask', 'hist', 'pk', 'psrc', 'N', 'z', 'deck', 'seat', 'wupd', 'vs')


def batch_of(chunks, picks, dev):
    import torch
    parts = collections.defaultdict(list)
    for ci, rows in picks:
        c = chunks[ci]
        for k in KEYS:
            parts[k].append(c[k][rows])
    d = {k: np.concatenate(v) for k, v in parts.items()}
    tm, am = d['tok_mask'], d['act_mask']
    T = int(tm.any(0).nonzero()[0].max()) + 1
    A = int(am.any(0).nonzero()[0].max()) + 1
    b = dict(glob=d['glob'], tok=d['tok'][:, :T], tok_mask=tm[:, :T], act=d['act'][:, :A], act_mask=am[:, :A],
             pk=d['pk'], psrc=d['psrc'], hist=d['hist'])
    b = {k: torch.from_numpy(np.ascontiguousarray(v)).to(dev, non_blocking=True) for k, v in b.items()}
    for k in ('glob', 'tok', 'act', 'pk', 'psrc', 'hist'):
        b[k] = b[k].long()
    N = torch.from_numpy(d['N'][:, :A].astype(np.float32)).to(dev)
    z = torch.from_numpy(d['z'].astype(np.float32)).to(dev)
    return b, N, z, d


def losses(net, b, N, z):
    import torch
    logp, v = net(b)
    pi = N / N.sum(1, keepdim=True).clamp(min=1.0)
    lp = logp.clamp(min=-30.0)
    ce = -(pi * lp).sum(1)
    h_pi = -(pi * torch.log(pi.clamp(min=1e-12))).sum(1)
    vl = (v - z) ** 2
    ent = -(logp.exp() * lp).sum(1)
    agree = (logp.argmax(1) == N.argmax(1)).float()
    return ce, h_pi, vl, ent, agree, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', required=True)
    ap.add_argument('--buf', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--vf', type=float, default=1.0)
    ap.add_argument('--reuse', type=float, default=4.0)
    ap.add_argument('--window', type=int, default=200_000)
    ap.add_argument('--min-records', type=int, default=30_000)
    ap.add_argument('--export-every', type=int, default=100)
    ap.add_argument('--ckpt-every', type=int, default=1000)
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--max-hours', type=float, default=0)
    ap.add_argument('--max-steps', type=int, default=0)
    ap.add_argument('--gpu-mem-frac', type=float, default=0.45)
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    import torch
    from common import load_net
    from rbenv import RBEnv
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cuda' and a.gpu_mem_frac:
        torch.cuda.set_per_process_memory_fraction(a.gpu_mem_frac)
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.run, exist_ok=True)
    dims = RBEnv(1, 'rengar', 'rengar', 1).dims
    start = a.init
    state = {}
    cks = sorted(glob.glob(os.path.join(a.run, 'az_*.pt')))
    if cks:   # resume
        start = cks[-1]
    b2 = os.environ.get('RB_B2', 'v1')   # this run's B2 vector file (overrides the init checkpoint's)
    net = load_net(start, dims, rep='B2', device=dev, b2=b2)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    step = 0
    if start != a.init:
        ck = torch.load(start, map_location='cpu', weights_only=False)
        step = int(ck.get('update', 0))
        if 'opt' in ck:
            opt.load_state_dict(ck['opt'])
        state = ck.get('state', {})
    json.dump(dict(vars(a), dims=dims, start=start, start_step=step), open(os.path.join(a.run, 'config.json'), 'w'),
              indent=1)

    def save(path, extra=None):
        d = {'net': net.state_dict(), 'update': step, 'repr': 'B2', 'init': a.init, 'b2': b2}
        if extra:
            d.update(extra)
        tmp = path + '.tmp'
        torch.save(d, tmp)
        os.replace(tmp, path)

    save(os.path.join(a.run, 'gen_weights.pt'))
    chunks, names, seen = collections.deque(), collections.deque(), set(state.get('seen', []))
    ingested = int(state.get('ingested', 0))
    trained = int(state.get('trained', 0))
    lg = open(os.path.join(a.run, 'train_log.jsonl'), 'a')
    acc = collections.defaultdict(float)
    nacc = 0
    fresh = collections.deque(maxlen=20)
    t0 = tlast = time.time()
    print(f'az trainer: start {start} @{step}, device {dev}, batch {a.batch}, lr {a.lr}, reuse {a.reuse}', flush=True)
    while True:
        new = [p for p in sorted(glob.glob(os.path.join(a.buf, 'c_*.npz'))) if os.path.basename(p) not in seen]
        for p in new:
            try:
                z = np.load(p)
                c = {k: z[k] for k in KEYS}
            except Exception:
                continue   # being written / removed
            seen.add(os.path.basename(p))
            n = len(c['z'])
            if ingested >= a.min_records:   # fresh loss: before training on it
                with torch.no_grad():
                    net.eval()
                    rows = np.arange(min(n, 512))
                    b, N, zz, _ = batch_of([c], [(0, rows)], dev)
                    ce, hpi, vl, ent, ag, _ = losses(net, b, N, zz)
                    fresh.append((float((ce - hpi).mean()), float(vl.mean()), float(ag.mean())))
                    net.train()
            chunks.append(c); names.append(os.path.basename(p))
            ingested += n
            while sum(len(x['z']) for x in chunks) - len(chunks[0]['z']) >= a.window:
                chunks.popleft(); names.popleft()
        total = sum(len(x['z']) for x in chunks)
        if ingested < a.min_records or trained >= a.reuse * ingested:
            time.sleep(2.0)
        else:
            sizes = np.array([len(x['z']) for x in chunks])
            ci = rng.choice(len(chunks), size=a.batch, p=sizes / sizes.sum())
            picks = [(int(i), rng.integers(0, sizes[i], size=int((ci == i).sum()))) for i in np.unique(ci)]
            b, N, z, d = batch_of(list(chunks), picks, dev)
            ce, hpi, vl, ent, ag, v = losses(net, b, N, z)
            loss = ce.mean() + a.vf * vl.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            step += 1
            trained += a.batch
            for k, x in (('kl', (ce - hpi).mean()), ('ce', ce.mean()), ('v_mse', vl.mean()), ('ent', ent.mean()),
                         ('agree', ag.mean()), ('gn', gn)):
                acc[k] += float(x)
            azir = d['deck'] == 0
            if azir.any():
                acc['kl_azir'] += float((ce - hpi)[torch.from_numpy(azir).to(dev)].mean()); acc['n_azir'] += 1
            nacc += 1
            if step % a.export_every == 0:
                save(os.path.join(a.run, 'gen_weights.pt'))
            if step % a.ckpt_every == 0:
                save(os.path.join(a.run, f'az_{step:06d}.pt'),
                     dict(opt=opt.state_dict(), state=dict(seen=sorted(seen), ingested=ingested, trained=trained)))
            if step % a.log_every == 0:
                now = time.time()
                row = dict(step=step, t=round(now - t0), records=total, ingested=ingested, trained=trained,
                           steps_per_s=round(a.log_every / (now - tlast), 2),
                           **{k: round(acc[k] / nacc, 4) for k in ('kl', 'ce', 'v_mse', 'ent', 'agree', 'gn')},
                           kl_azir=round(acc['kl_azir'] / max(acc['n_azir'], 1), 4))
                if fresh:
                    f = np.array(fresh)
                    row.update(fresh_kl=round(float(f[:, 0].mean()), 4), fresh_v_mse=round(float(f[:, 1].mean()), 4),
                               fresh_agree=round(float(f[:, 2].mean()), 4))
                lg.write(json.dumps(row) + '\n'); lg.flush()
                print(json.dumps(row), flush=True)
                acc.clear(); nacc = 0; tlast = now
        if (a.max_hours and time.time() - t0 > a.max_hours * 3600) or (a.max_steps and step >= a.max_steps):
            save(os.path.join(a.run, f'az_{step:06d}.pt'),
                 dict(opt=opt.state_dict(), state=dict(seen=sorted(seen), ingested=ingested, trained=trained)))
            break


if __name__ == '__main__':
    main()
