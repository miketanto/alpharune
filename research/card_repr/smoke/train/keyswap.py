"""Supplementary to E1: how differently does a trained network treat an unseen variant from
its base card? On the E1 decisions (variant-deck test games where the variant is involved),
re-run the network with every variant token's key replaced by the base card's key and
measure the change in output: mean |dP(card actions)|, mean KL(policy_variant || policy_base)
for both heads, and mean |d value|. Writes keyswap.json. Uses runs/*__<regime>__s*/model.pt.
"""
import json, os, glob, sys
import torch
torch.set_num_threads(int(os.environ.get('THREADS', '2')))
import train as T
from aggregate import PAIRS, structure, involved, ci

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    P = torch.load(os.path.join(HERE, 'packed', 'teacher.pt'))
    RP = torch.load(os.path.join(HERE, 'packed', 'reprs.pt'))
    meta = P['meta']; vocab = meta['vocab']; variants = meta['variants']
    special = RP['special']
    special_idx = torch.tensor([special.index(k) + 1 if k in special else 0 for k in vocab] + [0])
    S = structure(P)
    te = S['idx']
    data = T.Data(P)
    sets = {}
    for vk, bk in PAIRS.items():
        sel = (S['variant'] == variants.index(vk)) & involved(S, vocab.index(vk))
        sets[vk] = (te[sel], vocab.index(vk), vocab.index(bk))
    out = {}
    for mf in sorted(glob.glob(os.path.join(HERE, 'runs', '*', 'metrics.json'))):
        M = json.load(open(mf))
        cond = M['cond']
        sd = torch.load(os.path.join(os.path.dirname(mf), 'model.pt'))
        if cond == 'id':
            table = torch.zeros_like(sd['policy']['R'])
        else:
            table = torch.cat([RP['tables'][cond], torch.zeros(1, 256)])
        net = T.Net(table, special_idx, cond == 'id', len(meta['pk_names']))
        net.load_state_dict(sd['policy']); net.eval()
        vnet = T.Net(table, special_idx, cond == 'id', len(meta['pk_names']))
        vnet.load_state_dict(sd['value']); vnet.eval()
        unk = len(vocab)
        seen = set(M['unseen_card_keys'])
        res = {}
        with torch.no_grad():
            for vk, (ix, kv, kb) in sets.items():
                b = data.batch(ix)
                k = b['tok'][..., 1]
                if cond == 'id':  # unseen -> UNK, exactly as in evaluation
                    unseen_ix = torch.tensor([vocab.index(x) for x in M['unseen_card_keys']] or [-5])
                    k1 = torch.where(torch.isin(k, unseen_ix), torch.full_like(k, unk), k)
                    k2 = torch.where(k == kv, torch.full_like(k, kb), k)
                    k2 = torch.where(torch.isin(k2, unseen_ix), torch.full_like(k2, unk), k2)
                else:
                    k1 = k
                    k2 = torch.where(k == kv, torch.full_like(k, kb), k)
                _, lpv1, lpq1 = net(b, k1); _, lpv2, lpq2 = net(b, k2)
                v1 = vnet(b, k1)[0]; v2 = vnet(b, k2)[0]
                ma = b['ma']
                src = torch.gather(k, 1, b['act'][..., 1].clamp(min=0)) * (b['act'][..., 1] >= 0)
                isk = (src == kv) & ma
                def kl(a, c):
                    return ((a.exp() * (a - c)).masked_fill(~ma, 0)).sum(1)
                hasc = isk.any(1)
                res[vk] = {'n': len(ix), 'n_card_src': int(hasc.sum()),
                           'abs_dP_card_v': ((lpv1.exp() * isk).sum(1) - (lpv2.exp() * isk).sum(1)).abs()[hasc].mean().item() if hasc.any() else None,
                           'abs_dP_card_q': ((lpq1.exp() * isk).sum(1) - (lpq2.exp() * isk).sum(1)).abs()[hasc].mean().item() if hasc.any() else None,
                           'kl_v': kl(lpv1, lpv2).mean().item(), 'kl_q': kl(lpq1, lpq2).mean().item(),
                           'abs_dvalue': (v1 - v2).abs().mean().item()}
        out[M['name']] = {'cond': cond, 'regime': M['regime'], 'seed': M['seed'], 'pairs': res}
        print(M['name'], 'done', flush=True)
    summ = {}
    groups = {}
    for n, r in out.items():
        groups.setdefault(f"{r['cond']}__{r['regime']}", []).append(r)
    for g, rs in groups.items():
        summ[g] = {vk: {m: ci([r['pairs'][vk][m] for r in rs]) for m in ('abs_dP_card_v', 'abs_dP_card_q', 'kl_v', 'kl_q', 'abs_dvalue')}
                   for vk in PAIRS}
        summ[g]['pooled_mean_abs_dP_card_q'] = ci([sum(r['pairs'][vk]['abs_dP_card_q'] * r['pairs'][vk]['n_card_src'] for vk in PAIRS)
                                                  / sum(r['pairs'][vk]['n_card_src'] for vk in PAIRS) for r in rs])
        summ[g]['pooled_mean_kl_q'] = ci([sum(r['pairs'][vk]['kl_q'] * r['pairs'][vk]['n'] for vk in PAIRS)
                                         / sum(r['pairs'][vk]['n'] for vk in PAIRS) for r in rs])
        summ[g]['pooled_mean_abs_dvalue'] = ci([sum(r['pairs'][vk]['abs_dvalue'] * r['pairs'][vk]['n'] for vk in PAIRS)
                                               / sum(r['pairs'][vk]['n'] for vk in PAIRS) for r in rs])
    json.dump({'summary': summ, 'per_run': out}, open(os.path.join(HERE, 'keyswap.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
