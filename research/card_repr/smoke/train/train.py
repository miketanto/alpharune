"""Train the minimal decision network (NETWORK.md) for one (condition, regime, seed).

  python train.py --cond B2 --regime base --seed 0 --steps 6000 --threads 2

regime 'base': train on base_teacher train games only (every variant card unseen: E1).
regime 'all' : train on base + all variant teacher train games (E2, E0).
Writes runs/<cond>__<regime>__s<seed>/{log.txt, metrics.json, preds.pt}.
"""
import argparse, json, math, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PK = os.path.join(HERE, 'packed')
TAU_Q = 0.05          # fixed a priori: q-softmax temperature for the q-policy target
D = 64


# --------------------------------------------------------------------------- data
class Data:
    def __init__(self, P):
        self.P = P

    def batch(self, idx):
        P = self.P
        B = idx.numel()
        nt, na = P['nt'][idx], P['na'][idx]
        T, A = int(nt.max()), int(na.max())
        ar_t = torch.arange(T)
        ti = P['tok_off'][idx, None] + ar_t
        mt = ar_t[None, :] < nt[:, None]
        tok = P['tok'][ti.clamp(max=P['tok'].shape[0] - 1)].long() * mt[..., None]
        ar_a = torch.arange(A)
        ai = P['act_off'][idx, None] + ar_a
        ma = ar_a[None, :] < na[:, None]
        aic = ai.clamp(max=P['act'].shape[0] - 1)
        act = P['act'][aic].long()
        act = torch.where(ma[..., None], act, torch.full_like(act, -1))
        nr = P['nref'][aic] * ma
        R = max(1, int(nr.max()))
        ar_r = torch.arange(R)
        ri = P['ref_off'][aic][..., None] + ar_r
        mr = ar_r[None, None, :] < nr[..., None]
        ref = P['ref'][ri.clamp(max=max(0, P['ref'].shape[0] - 1))].long() if P['ref'].numel() else torch.zeros(B, A, R, dtype=torch.long)
        ref = ref * mr
        vis = P['vis'][aic] * ma
        q = P['q'][aic] * ma
        return dict(tok=tok, mt=mt, act=act, ma=ma, ref=ref, mr=mr, g=P['g'][idx].long(),
                    pk=P['pk'][idx], vis=vis, q=q, pi=P['pi'][idx], z=P['z'][idx],
                    searched=P['searched'][idx], aidx=ai)


# --------------------------------------------------------------------------- model
def onehot_clip(x, lo, hi):
    return F.one_hot((x.clamp(lo, hi) - lo), hi - lo + 1).float()


class Net(nn.Module):
    def __init__(self, table, special_idx, trainable_table, n_pk, unk_row=None):
        super().__init__()
        if trainable_table:
            self.R = nn.Parameter(table.clone())
        else:
            self.register_buffer('R', table.clone())
        self.register_buffer('special_idx', special_idx)
        self.unk_row = unk_row
        self.W_card = nn.Linear(table.shape[1], D, bias=False)
        self.E_special = nn.Embedding(int(special_idx.max()) + 1, D)
        # situational state: per-kind, per-field, per-value embedding (= a linear map of the
        # one-hot encoding, computed as a lookup) + per-kind linear on the numeric copies
        self.E_state = nn.Embedding(3 * 15 * 12, D)
        self.W_num = nn.Linear(15 + 2, 3 * D)
        self.register_buffer('field_off', torch.arange(15) * 12)
        self.E_kind = nn.Embedding(5, D)  # 0 pad 1 c 2 b 3 k 4 global
        self.n_pk = n_pk
        self.W_glob = nn.Linear(72 + 11 + 37 + n_pk, D)
        layer = nn.TransformerEncoderLayer(D, 4, 2 * D, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(D)
        self.E_akind = nn.Embedding(23, D)
        self.E_dst = nn.Embedding(8, D)
        self.W_v = nn.Linear(17 + 6 + 6 + 5 + 5 + 2, D)
        self.val = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 1))
        self.pol_v = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 1))
        self.pol_q = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 1))

    def forward(self, b, key_override=None):
        tok = b['tok']
        kind = tok[..., 0]
        key = tok[..., 1] if key_override is None else key_override
        fields = tok[..., 2:17]
        x_card = self.W_card(self.R)[key]
        x_spec = self.E_special(self.special_idx[key])
        kc = kind.clamp(1, 3) - 1
        sidx = kc[..., None] * 180 + self.field_off + (fields.clamp(-1, 10) + 1)
        st_cat = self.E_state(sidx).sum(-2)
        fe = torch.cat([fields.float().clamp(-20, 40) / 5.0,
                        tok[..., 17:18].float(), torch.log1p(tok[..., 18:19].float().clamp(min=0))], -1)
        st = self.W_num(fe).view(*kind.shape, 3, D)
        st = torch.gather(st, 2, kc[..., None, None].expand(*kind.shape, 1, D)).squeeze(2) + st_cat
        x = x_card + x_spec + st + self.E_kind(kind)
        g = b['g'].float()
        gf = torch.cat([torch.sign(g) * torch.log1p(g.abs()), onehot_clip(b['g'][:, 1], 0, 10),
                        onehot_clip(b['g'][:, 68], 0, 36), F.one_hot(b['pk'], self.n_pk).float()], -1)
        xg = self.W_glob(gf) + self.E_kind.weight[4]
        h = torch.cat([xg[:, None, :], x], 1)
        pad = torch.cat([torch.zeros(len(g), 1, dtype=torch.bool), ~b['mt']], 1)
        h = self.ln(self.enc(h, src_key_padding_mask=pad))
        hg, ht = h[:, 0], h[:, 1:]
        value = torch.tanh(self.val(hg)).squeeze(-1)
        act = b['act']
        Bn, A, _ = act.shape
        T = ht.shape[1]

        def tok_h(ix):  # ix [B, A] token indices (-1 none) -> [B, A, D]
            ok = (ix >= 0) & (ix < T)
            hh = torch.gather(ht, 1, ix.clamp(0, T - 1)[..., None].expand(Bn, A, D))
            return hh * ok[..., None]

        h_src = tok_h(act[..., 1]) + tok_h(act[..., 2])
        ref = b['ref']
        R = ref.shape[2]
        hr = torch.gather(ht, 1, ref.clamp(0, T - 1).view(Bn, A * R)[..., None].expand(Bn, A * R, D)).view(Bn, A, R, D)
        mr = b['mr'].float()
        h_ref = (hr * mr[..., None]).sum(2) / mr.sum(2).clamp(min=1)[..., None]
        sc = torch.cat([onehot_clip(act[..., 4], -1, 15), onehot_clip(act[..., 5], -1, 4),
                        onehot_clip(act[..., 6], 0, 5), act[..., 7:11].clamp(0, 1).float(),
                        act[..., 12:13].clamp(0, 1).float(), onehot_clip(act[..., 11], -1, 3),
                        torch.log1p(act[..., 13:14].clamp(min=0).float()),
                        torch.log1p(mr.sum(2, keepdim=True))], -1)
        ha = (self.E_akind(act[..., 0].clamp(0, 22)) + h_src + h_ref + self.E_dst((act[..., 3] + 1).clamp(0, 7))
              + self.W_v(sc) + hg[:, None, :])
        neg = torch.finfo(ha.dtype).min / 4
        lv = self.pol_v(ha).squeeze(-1).masked_fill(~b['ma'], neg)
        lq = self.pol_q(ha).squeeze(-1).masked_fill(~b['ma'], neg)
        return value, F.log_softmax(lv, -1), F.log_softmax(lq, -1)


def q_target(q, ma):
    neg = torch.finfo(q.dtype).min / 4
    return F.softmax((q / TAU_Q).masked_fill(~ma, neg), -1)


def losses(net, b, key_override=None):
    v, lpv, lpq = net(b, key_override)
    ma = b['ma']
    lv_ = -(b['vis'] * lpv.masked_fill(~ma, 0)).sum(-1).mean()
    s = b['searched'].float()
    tq = q_target(b['q'], ma)
    lq_ = (-(tq * lpq.masked_fill(~ma, 0)).sum(-1) * s).sum() / s.sum().clamp(min=1)
    lval = ((v - b['z']) ** 2).mean()
    return lv_, lq_, lval


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', required=True)
    ap.add_argument('--regime', required=True, choices=['base', 'all'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    name = f'{args.cond}__{args.regime}__s{args.seed}{args.tag}'
    out = os.path.join(HERE, 'runs', name)
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, 'log.txt'), 'w')

    def log(*a):
        s = ' '.join(str(x) for x in a)
        print(s, flush=True)
        logf.write(s + '\n'); logf.flush()

    t0 = time.time()
    P = torch.load(os.path.join(PK, 'teacher.pt'))
    RP = torch.load(os.path.join(PK, 'reprs.pt'))
    SW = torch.load(os.path.join(PK, 'swap.pt'))
    meta = P['meta']
    vocab = meta['vocab']
    variants = meta['variants']
    base_vid = variants.index('')
    special = RP['special']
    special_idx = torch.tensor([special.index(k) + 1 if k in special else 0 for k in vocab] + [0])
    split, var = P['split'], P['variant']
    if args.regime == 'base':
        tr_mask = (split == 0) & (var == base_vid)
    else:
        tr_mask = split == 0
    tr_idx = tr_mask.nonzero().squeeze(1)
    # keys seen in training tokens
    tok_dec = torch.repeat_interleave(torch.arange(len(P['nt'])), P['nt'])
    seen = torch.zeros(len(vocab) + 1, dtype=torch.bool)
    seen[P['tok'][tr_mask[tok_dec], 1].long().unique()] = True
    unseen_cards = [vocab[i] for i in range(len(vocab)) if not seen[i] and special_idx[i] == 0 and vocab[i] != '<pad>']
    log('run', name, 'train decisions', len(tr_idx), 'unseen card keys in training:', unseen_cards)
    if args.regime == 'base':
        for v in variants:
            if v:
                assert not seen[vocab.index(v)], ('variant leaked into base training', v)

    cond = args.cond
    unk = len(vocab)  # extra row for UNK (only the id condition uses it)
    if cond == 'id':
        table = torch.cat([RP['tables']['id_init'], torch.randn(1, RP['tables']['id_init'].shape[1],
                                                               generator=torch.Generator().manual_seed(99))])
        trainable = True
    else:
        table = torch.cat([RP['tables'][cond], torch.zeros(1, RP['tables'][cond].shape[1])])
        trainable = False
    n_pk = len(meta['pk_names'])
    net = Net(table, special_idx, trainable, n_pk)
    nparam = sum(p.numel() for n_, p in net.named_parameters() if n_ != 'R')
    log('params (excluding the card table)', nparam, 'cond', cond, 'trainable table', trainable)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    warm = 200
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    data = Data(P)
    gen = torch.Generator().manual_seed(1000 + args.seed)
    # validation from the regime's own games only (base regime: base games, so no held-out card
    # influences model selection)
    val_mask = (split == 1) & ((var == base_vid) if args.regime == 'base' else torch.ones_like(split, dtype=torch.bool))
    val_idx = val_mask.nonzero().squeeze(1)
    val_idx = val_idx[torch.randperm(len(val_idx), generator=torch.Generator().manual_seed(5))[:3000]]
    is_card = (special_idx == 0)
    is_card[vocab.index('<pad>')] = False

    def key_map(b):
        """id condition: unseen keys -> UNK at evaluation."""
        if cond != 'id':
            return None
        k = b['tok'][..., 1]
        return torch.where(seen[k] | ~is_card[k], k, torch.full_like(k, unk))

    def evaluate(ix, bs=1024):
        net.eval()
        tot = torch.zeros(3); n = 0
        with torch.no_grad():
            for i in range(0, len(ix), bs):
                b = data.batch(ix[i:i + bs])
                l = losses(net, b, key_map(b))
                tot += torch.tensor([x.item() for x in l]) * len(b['pi']); n += len(b['pi'])
        net.train()
        return (tot / n).tolist()

    import copy
    best = {'policy': (float('inf'), 0, None), 'value': (float('inf'), 0, None)}
    val_curve = []
    run_loss = torch.zeros(3); t1 = time.time()
    for step in range(args.steps):
        idx = tr_idx[torch.randint(len(tr_idx), (args.bs,), generator=gen)]
        b = data.batch(idx)
        ko = None
        if cond == 'id':  # card-key dropout to UNK (2%) so the UNK row is trained
            k = b['tok'][..., 1]
            drop = (torch.rand(k.shape, generator=gen) < 0.02) & is_card[k]
            ko = torch.where(drop, torch.full_like(k, unk), k)
        lv_, lq_, lval = losses(net, b, ko)
        loss = lv_ + lq_ + lval
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        run_loss += torch.tensor([lv_.item(), lq_.item(), lval.item()])
        if (step + 1) % 250 == 0:
            ev = evaluate(val_idx)
            val_curve.append([step + 1] + ev)
            # per-head checkpoint selection on validation loss (same rule for every condition)
            if ev[0] + ev[1] < best['policy'][0]:
                best['policy'] = (ev[0] + ev[1], step + 1, copy.deepcopy(net.state_dict()))
            if ev[2] < best['value'][0]:
                best['value'] = (ev[2], step + 1, copy.deepcopy(net.state_dict()))
            msg = f'step {step+1} train(ce_vis,ce_q,mse_v) {[round(x/250,4) for x in run_loss.tolist()]} val {[round(x,4) for x in ev]} {time.time()-t1:.0f}s'
            run_loss.zero_()
            log(msg)
    train_secs = time.time() - t1

    # ---------------------------------------------------------------- predictions
    final_val = evaluate(val_idx)
    net.eval()
    vnet = copy.deepcopy(net)
    vnet.load_state_dict(best['value'][2])
    net.load_state_dict(best['policy'][2])
    vnet.eval(); net.eval()
    log('best policy step', best['policy'][1], 'best value step', best['value'][1])
    te_idx = (split == 2).nonzero().squeeze(1)
    pv_all, pq_all, val_all = [], [], []
    with torch.no_grad():
        for i in range(0, len(te_idx), 1024):
            b = data.batch(te_idx[i:i + 1024])
            _, lpv, lpq = net(b, key_map(b))
            v = vnet(b, key_map(b))[0]
            ma = b['ma']
            pv_all.append(lpv.exp()[ma]); pq_all.append(lpq.exp()[ma]); val_all.append(v)
        sw = {}
        for ver in ('orig', 'swap'):
            Dv = Data(SW[ver])
            ix = torch.arange(len(SW[ver]['pi']))
            b = Dv.batch(ix)
            _, lpv, lpq = net(b, key_map(b))
            v = vnet(b, key_map(b))[0]
            ma = b['ma']
            sw[ver] = dict(pv=lpv.exp()[ma], pq=lpq.exp()[ma], value=v)
    torch.save({'test_idx': te_idx, 'pv': torch.cat(pv_all), 'pq': torch.cat(pq_all),
                'value': torch.cat(val_all), 'swap': sw}, os.path.join(out, 'preds.pt'))
    torch.save({'policy': best['policy'][2], 'value': best['value'][2]}, os.path.join(out, 'model.pt'))
    json.dump({'name': name, 'cond': cond, 'regime': args.regime, 'seed': args.seed, 'steps': args.steps,
               'bs': args.bs, 'lr': args.lr, 'tau_q': TAU_Q, 'params_excl_table': nparam,
               'train_decisions': len(tr_idx), 'unseen_card_keys': unseen_cards,
               'final_val': final_val, 'val_curve': val_curve,
               'best_policy_step': best['policy'][1], 'best_policy_val': best['policy'][0],
               'best_value_step': best['value'][1], 'best_value_val': best['value'][0],
               'n_val': len(val_idx), 'train_secs': train_secs, 'total_secs': time.time() - t0,
               'threads': args.threads}, open(os.path.join(out, 'metrics.json'), 'w'), indent=1)
    log('done', name, f'train {train_secs:.0f}s total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
