"""Aggregate runs/* into results.json (+ CSV tables). E0, E1, E2, E3 as in the task.

Per-run metrics are computed from preds.pt; across seeds: mean and 95% t-interval
(t_{0.975, n-1} * sd / sqrt(n)). Paired differences between conditions use the same seed
(same init for every non-id condition and the same data order), CI over seeds.
"""
import json, math, os, glob, collections
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PK = os.path.join(HERE, 'packed')
TAU_Q = 0.05
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
PAIRS = {  # variant key -> base key
    'primal_strength': 'punch_first',
    'synthetic:punch_first__amount4': 'punch_first', 'synthetic:punch_first__energy3': 'punch_first',
    'synthetic:punch_first__power0': 'punch_first', 'synthetic:punch_first__reaction': 'punch_first',
    'synthetic:discipline__draw2': 'discipline', 'synthetic:discipline__enemy': 'discipline',
    'synthetic:lonely_poro__draw2': 'lonely_poro', 'synthetic:defy__cost2': 'defy',
    'synthetic:sabotage__unit': 'sabotage', 'synthetic:pit_rookie__might3': 'pit_rookie'}


def ci(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    n = len(xs)
    if n == 0:
        return None
    m = sum(xs) / n
    if n == 1:
        return {'mean': m, 'ci95': None, 'n_seeds': 1, 'per_seed': xs}
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return {'mean': m, 'ci95': T975.get(n - 1, 1.96) * sd / math.sqrt(n), 'n_seeds': n, 'per_seed': xs}


def pearson(a, b):
    n = len(a)
    if n < 3:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a); vb = sum((y - mb) ** 2 for y in b)
    if va == 0 or vb == 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


# ------------------------------------------------------------------ test-set structure
def dense(P, idx, flat, off_key, n_key, fill):
    n = P[n_key][idx]
    A = int(n.max())
    ar = torch.arange(A)
    ix = P[off_key][idx, None] + ar
    m = ar[None] < n[:, None]
    out = flat[ix.clamp(max=flat.shape[0] - 1)]
    if out.dim() == 2:
        out = torch.where(m, out, torch.full_like(out, fill))
    else:
        out = torch.where(m[..., None], out, torch.full_like(out, fill))
    return out, m


def structure(P):
    te = (P['split'] == 2).nonzero().squeeze(1)
    S = {'idx': te}
    S['vis'], S['ma'] = dense(P, te, P['vis'], 'act_off', 'na', 0.0)
    S['q'], _ = dense(P, te, P['q'], 'act_off', 'na', 0.0)
    act, _ = dense(P, te, P['act'].long(), 'act_off', 'na', -1)
    tok, mt = dense(P, te, P['tok'].long(), 'tok_off', 'nt', 0)
    key = tok[..., 1]; T = key.shape[1]
    def gk(ix):
        ok = (ix >= 0) & (ix < T)
        return torch.where(ok, torch.gather(key, 1, ix.clamp(0, T - 1)), torch.full_like(ix, -1))
    S['src_key'] = gk(act[..., 1])
    S['abs_key'] = gk(act[..., 2])
    # refs: per action
    na = P['na'][te]
    aoff = P['act_off'][te]
    A = act.shape[1]
    aidx = (aoff[:, None] + torch.arange(A)).clamp(max=P['nref'].shape[0] - 1)
    nr = P['nref'][aidx] * S['ma']
    R = max(1, int(nr.max()))
    ri = P['ref_off'][aidx][..., None] + torch.arange(R)
    mr = torch.arange(R)[None, None] < nr[..., None]
    ref = P['ref'][ri.clamp(max=P['ref'].shape[0] - 1)].long()
    ref_key = torch.where(mr, torch.gather(key, 1, ref.clamp(0, T - 1).view(len(te), -1)).view(len(te), A, R),
                          torch.full_like(ref, -1))
    S['ref_key'] = ref_key
    S['psrc_key'] = torch.where(tok[..., 17] == 1, key, torch.full_like(key, -1))
    S['pi'] = P['pi'][te]; S['z'] = P['z'][te]; S['searched'] = P['searched'][te]
    S['variant'] = P['variant'][te]; S['pk'] = P['pk'][te]; S['na'] = na
    return S


def involved(S, k):
    return ((S['src_key'] == k).any(1) | (S['abs_key'] == k).any(1) | (S['ref_key'] == k).flatten(1).any(1)
            | (S['psrc_key'] == k).any(1))


def kind_baseline(P, train_mask):
    """choice rate of each (prompt kind, action kind) on the training decisions."""
    tr = train_mask.nonzero().squeeze(1)
    act, ma = dense(P, tr, P['act'][:, 0].long(), 'act_off', 'na', -1)
    pk = P['pk'][tr][:, None].expand_as(act)
    chosen = torch.zeros_like(act, dtype=torch.bool)
    chosen[torch.arange(len(tr)), P['pi'][tr]] = True
    npk = int(P['pk'].max()) + 1
    k = (pk * 32 + act.clamp(min=0))[ma]
    off = torch.bincount(k, minlength=npk * 32).float()
    ch = torch.bincount(k[chosen[ma]], minlength=npk * 32).float()
    rate = (ch + 0.5) / (off + 1.0)
    zmean = P['z'][tr].mean().item()
    return rate, zmean


def decision_metrics(S, pv, pq, value, sel):
    """pv, pq: dense [N, A] probabilities; value [N]; sel: bool mask of decisions."""
    if sel.sum() == 0:
        return None
    ma = S['ma'][sel]; pv = pv[sel]; pq = pq[sel]; pi = S['pi'][sel]
    q = S['q'][sel]; vis = S['vis'][sel]; srch = S['searched'][sel]
    eps = 1e-9
    r = torch.arange(len(pi))
    nll = -(pv[r, pi] + eps).log()
    top1 = (pv.masked_fill(~ma, -1).argmax(1) == pi).float()
    top1_q_pi = (pq.masked_fill(~ma, -1).argmax(1) == pi).float()
    qm = q.masked_fill(~ma, -1e9)
    qbest = qm == qm.max(1, keepdim=True).values
    informative = srch & (qbest & ma).sum(1).lt(ma.sum(1))
    qb_hit = qbest[r, pq.masked_fill(~ma, -1).argmax(1)].float()
    kl = (vis * ((vis + eps).log() - (pv + eps).log())).masked_fill(~ma, 0).sum(1)
    pvq = (pq.masked_fill(~ma, -1).argmax(1))
    brier = (((value[sel] + 1) / 2 - (S['z'][sel] + 1) / 2) ** 2)
    out = {'n': int(sel.sum()), 'nll_pi': nll.mean().item(), 'top1_pi': top1.mean().item(),
           'qhead_top1_pi': top1_q_pi.mean().item(),
           'qhead_qbest': qb_hit[informative].mean().item() if informative.any() else None,
           'n_q_informative': int(informative.sum()),
           'kl_vis': kl.mean().item(), 'value_brier': brier.mean().item()}
    return out


def baseline_metrics(S, rate, zmean, sel):
    ma = S['ma'][sel]; pi = S['pi'][sel]; na = S['na'][sel].float(); q = S['q'][sel]; srch = S['searched'][sel]
    r = torch.arange(len(pi))
    qm = q.masked_fill(~ma, -1e9)
    qbest = (qm == qm.max(1, keepdim=True).values) & ma
    informative = srch & qbest.sum(1).lt(ma.sum(1))
    uni = {'nll_pi': na.log().mean().item(), 'top1_pi': (1 / na).mean().item(),
           'qbest': (qbest.sum(1).float() / na)[informative].mean().item() if informative.any() else None}
    # kind-frequency
    act_kind = S['act_kind'][sel]
    pk = S['pk'][sel][:, None].expand_as(act_kind)
    rr = rate[(pk * 32 + act_kind.clamp(min=0))].masked_fill(~ma, 0)
    p = rr / rr.sum(1, keepdim=True)
    mx = rr.masked_fill(~ma, -1)
    tie = (mx == mx.max(1, keepdim=True).values) & ma
    top1 = (tie[r, pi].float() / tie.sum(1).float())
    kind = {'nll_pi': -(p[r, pi] + 1e-9).log().mean().item(), 'top1_pi': top1.mean().item(),
            'qbest': ((tie & qbest).sum(1).float() / tie.sum(1).float())[informative].mean().item() if informative.any() else None}
    y = (S['z'][sel] + 1) / 2
    const = {'value_brier': (((zmean + 1) / 2 - y) ** 2).mean().item(), 'value_brier_half': ((0.5 - y) ** 2).mean().item()}
    return {'uniform': uni, 'kind_freq': kind, 'value_const': const, 'n': int(sel.sum())}


def card_metrics(S, pv, pq, sel, k):
    """Decisions in sel where card k is the source of >= 1 legal action."""
    is_k = ((S['src_key'] == k) | (S['abs_key'] == k)) & S['ma']
    s2 = sel & is_k.any(1)
    if s2.sum() == 0:
        return None
    isk = is_k[s2]; pv = pv[s2]; pq = pq[s2]; pi = S['pi'][s2]; ma = S['ma'][s2]
    vis = S['vis'][s2]; q = S['q'][s2]
    r = torch.arange(len(pi))
    t_play = isk[r, pi]
    m_play_v = isk[r, pv.masked_fill(~ma, -1).argmax(1)]
    m_play_q = isk[r, pq.masked_fill(~ma, -1).argmax(1)]
    tq = torch.softmax((q / TAU_Q).masked_fill(~ma, -1e9), 1)
    mass_v = (pv * isk).sum(1); mass_q = (pq * isk).sum(1)
    t_vis = (vis * isk).sum(1); t_qs = (tq * isk).sum(1)
    qm = q.masked_fill(~ma, -1e9)
    qbest = (qm == qm.max(1, keepdim=True).values) & ma
    inf = qbest.sum(1).lt(ma.sum(1))
    t_qplay = (qbest & isk).any(1) & ~(qbest & ~isk).any(1)   # every q-best option is a card action
    t_qnot = ~(qbest & isk).any(1)                              # no q-best option is a card action
    clear = inf & (t_qplay | t_qnot)
    qagree = (m_play_q == t_qplay)[clear].float().mean().item() if clear.any() else None
    return {'n': int(s2.sum()), 'n_qclear': int(clear.sum()), 'play_agree_qhead_vs_qbest': qagree,
            'teacher_qplay_rate': t_qplay[clear].float().mean().item() if clear.any() else None,
            'teacher_play_rate': t_play.float().mean().item(),
            'play_agree_vhead': (t_play == m_play_v).float().mean().item(),
            'play_agree_qhead': (t_play == m_play_q).float().mean().item(),
            'nll_pi': -(pv[r, pi] + 1e-9).log().mean().item(),
            'mass_v': mass_v.mean().item(), 'teacher_vis_mass': t_vis.mean().item(),
            'mass_q': mass_q.mean().item(), 'teacher_qsoft_mass': t_qs.mean().item(),
            'mae_mass_v': (mass_v - t_vis).abs().mean().item(), 'mae_mass_q': (mass_q - t_qs).abs().mean().item()}


def to_dense(S, flat):
    out = torch.zeros(S['ma'].shape)
    out[S['ma']] = flat
    return out


def swap_metrics(SWP, preds_sw, meta, sims):
    """Per variant: network change vs teacher change between orig and swap versions."""
    res = {}
    for ver in ('orig', 'swap'):
        D = SWP[ver]
        n = len(D['pi'])
        dn = torch.zeros(n, int(D['na'].max()))
        m = torch.arange(dn.shape[1])[None] < D['na'][:, None]
        dn_v = dn.clone(); dn_v[m] = preds_sw[ver]['pv']
        dn_q = dn.clone(); dn_q[m] = preds_sw[ver]['pq']
        res[ver] = (dn_v, dn_q, preds_sw[ver]['value'])
    by_var = collections.defaultdict(list)
    for i, r in enumerate(meta):
        if r['sims'] != sims:
            continue
        pv_o = res['orig'][0][i, r['orig']['card_acts']].sum().item()
        pv_s = res['swap'][0][i, r['swap']['card_acts']].sum().item()
        pq_o = res['orig'][1][i, r['orig']['card_acts']].sum().item()
        pq_s = res['swap'][1][i, r['swap']['card_acts']].sum().item()
        v_o = res['orig'][2][i].item(); v_s = res['swap'][2][i].item()
        dvis = [b - a for a, b in zip(r['orig']['vis_mass_seeds'], r['swap']['vis_mass_seeds'])]
        drv = [b - a for a, b in zip(r['orig']['rv_seeds'], r['swap']['rv_seeds'])]
        def sig(d):
            m = sum(d) / len(d); sd = math.sqrt(sum((x - m) ** 2 for x in d) / max(1, len(d) - 1))
            return m, abs(m) > 2 * sd / math.sqrt(len(d)) and m != 0
        dvm, dvs = sig(dvis); drm, drs = sig(drv)
        legal_same = len(r['orig']['card_acts']) == len(r['swap']['card_acts']) and r['orig']['n_act'] == r['swap']['n_act']
        by_var[r['variant']].append(dict(net_dp=pv_s - pv_o, net_dpq=pq_s - pq_o, net_dv=v_s - v_o,
                                         t_dp=dvm, t_dp_sig=dvs, t_dv=drm, t_dv_sig=drs, legal_same=legal_same,
                                         net_p_orig=pv_o))
    out = {}
    for v, rows in by_var.items():
        o = {'n': len(rows), 'n_legal_same': sum(r['legal_same'] for r in rows)}
        for sub, rr in (('all', rows), ('legal_same', [r for r in rows if r['legal_same']])):
            sp = [r for r in rr if r['t_dp_sig']]
            sv = [r for r in rr if r['t_dv_sig']]
            o[sub] = {
                'corr_dpolicy': pearson([r['net_dp'] for r in rr], [r['t_dp'] for r in rr]),
                'corr_dvalue': pearson([r['net_dv'] for r in rr], [r['t_dv'] for r in rr]),
                'n_teacher_dpolicy_sig': len(sp),
                'sign_agree_dpolicy_on_sig': (sum((r['net_dp'] > 0) == (r['t_dp'] > 0) for r in sp) / len(sp)) if sp else None,
                'n_teacher_dvalue_sig': len(sv),
                'sign_agree_dvalue_on_sig': (sum((r['net_dv'] > 0) == (r['t_dv'] > 0) for r in sv) / len(sv)) if sv else None,
                'mean_net_dpolicy': sum(r['net_dp'] for r in rr) / len(rr) if rr else None,
                'mean_teacher_dpolicy': sum(r['t_dp'] for r in rr) / len(rr) if rr else None,
                'mean_abs_net_dpolicy': sum(abs(r['net_dp']) for r in rr) / len(rr) if rr else None,
                'mean_net_dvalue': sum(r['net_dv'] for r in rr) / len(rr) if rr else None,
                'mean_teacher_dvalue': sum(r['t_dv'] for r in rr) / len(rr) if rr else None,
            }
        out[v] = o
    return out


def main():
    P = torch.load(os.path.join(PK, 'teacher.pt'))
    SWP = torch.load(os.path.join(PK, 'swap.pt'))
    meta = P['meta']; vocab = meta['vocab']; variants = meta['variants']
    S = structure(P)
    kinds, _ = dense(P, S['idx'], P['act'][:, 0].long(), 'act_off', 'na', -1)
    S['act_kind'] = kinds
    base_vid = variants.index('')
    is_base_game = S['variant'] == base_vid
    runs = sorted(glob.glob(os.path.join(HERE, os.environ.get('RUNS_DIR', 'runs'), '*', 'metrics.json')))
    per_run = {}
    # baselines (computed per regime's training set)
    rate_all, z_all = kind_baseline(P, P['split'] == 0)
    rate_base, z_base = kind_baseline(P, (P['split'] == 0) & (P['variant'] == base_vid))
    subsets = {'test_all': torch.ones(len(S['pi']), dtype=torch.bool), 'test_base_games': is_base_game,
               'test_variant_games': ~is_base_game}
    e1_sets = {}
    for vk, bk in PAIRS.items():
        vid = variants.index(vk)
        e1_sets[vk] = {'variant_card_in_variant_games': (S['variant'] == vid) & involved(S, vocab.index(vk)),
                       'base_card_in_base_games': is_base_game & involved(S, vocab.index(bk))}
    baselines = {
        'all_regime': {n: baseline_metrics(S, rate_all, z_all, m) for n, m in subsets.items()},
        'base_regime_E1': {vk: {n: baseline_metrics(S, rate_base, z_base, m) for n, m in d.items()} for vk, d in e1_sets.items()},
    }
    for mf in runs:
        M = json.load(open(mf))
        d = os.path.dirname(mf)
        pr = torch.load(os.path.join(d, 'preds.pt'))
        pv = to_dense(S, pr['pv']); pq = to_dense(S, pr['pq']); val = pr['value']
        R = {'meta': M, 'E0': {n: decision_metrics(S, pv, pq, val, m) for n, m in subsets.items()}, 'E1': {}}
        for vk, bk in PAIRS.items():
            sets = e1_sets[vk]
            R['E1'][vk] = {
                'variant': decision_metrics(S, pv, pq, val, sets['variant_card_in_variant_games']),
                'base': decision_metrics(S, pv, pq, val, sets['base_card_in_base_games']),
                'variant_card': card_metrics(S, pv, pq, S['variant'] == variants.index(vk), vocab.index(vk)),
                'base_card': card_metrics(S, pv, pq, is_base_game, vocab.index(bk))}
        def pool(side, m, w='n'):
            num = den = 0.0
            for vk in PAIRS:
                x = R['E1'][vk][side]
                if x and x.get(m) is not None:
                    num += x[m] * x[w]; den += x[w]
            return num / den if den else None
        R['pooled'] = {'variant_nll': pool('variant', 'nll_pi'), 'variant_top1': pool('variant', 'top1_pi'),
                       'base_nll': pool('base', 'nll_pi'), 'base_top1': pool('base', 'top1_pi'),
                       'variant_card_nll': pool('variant_card', 'nll_pi'),
                       'variant_card_play_agree': pool('variant_card', 'play_agree_vhead'),
                       'variant_card_play_agree_q': pool('variant_card', 'play_agree_qhead'),
                       'variant_qhead_top1': pool('variant', 'qhead_top1_pi'),
                       'variant_qbest': pool('variant', 'qhead_qbest', 'n_q_informative'),
                       'variant_card_qbest_agree': pool('variant_card', 'play_agree_qhead_vs_qbest', 'n_qclear'),
                       'base_card_qbest_agree': pool('base_card', 'play_agree_qhead_vs_qbest', 'n_qclear'),
                       'base_qbest': pool('base', 'qhead_qbest', 'n_q_informative'),
                       'variant_brier': pool('variant', 'value_brier'),
                       'base_qhead_top1': pool('base', 'qhead_top1_pi'),
                       'base_card_nll': pool('base_card', 'nll_pi'),
                       'base_card_play_agree': pool('base_card', 'play_agree_vhead')}
        R['E2'] = {'sims100': swap_metrics(SWP, pr['swap'], SWP['meta'], 100),
                   'sims400': swap_metrics(SWP, pr['swap'], SWP['meta'], 400)}
        per_run[M['name']] = R
    # ---------------------------------------------------------------- across seeds
    groups = collections.defaultdict(list)
    for n, R in per_run.items():
        groups[(R['meta']['cond'], R['meta']['regime'])].append(R)

    def collect(rs, path):
        vals = []
        for R in rs:
            x = R
            for p in path:
                x = x.get(p) if isinstance(x, dict) else None
                if x is None:
                    break
            vals.append(x)
        return ci(vals)

    def by_seed(rs, path):
        out = {}
        for R in rs:
            x = R
            for p in path:
                x = x.get(p) if isinstance(x, dict) else None
                if x is None:
                    break
            out[R['meta']['seed']] = x
        return out

    summary = {}
    for (c, rg), rs in sorted(groups.items()):
        s = {'n_seeds': len(rs), 'seeds': sorted(R['meta']['seed'] for R in rs), 'E0': {}, 'E1': {}, 'E2': {}}
        for sub in subsets:
            s['E0'][sub] = {m: collect(rs, ['E0', sub, m]) for m in
                            ('nll_pi', 'top1_pi', 'qhead_top1_pi', 'qhead_qbest', 'kl_vis', 'value_brier')}
        for vk in PAIRS:
            e = {}
            for side in ('variant', 'base'):
                e[side] = {m: collect(rs, ['E1', vk, side, m]) for m in ('nll_pi', 'top1_pi', 'qhead_top1_pi', 'qhead_qbest', 'value_brier')}
                e[side]['n'] = rs[0]['E1'][vk][side]['n'] if rs[0]['E1'][vk][side] else 0
            for side in ('variant_card', 'base_card'):
                e[side] = {m: collect(rs, ['E1', vk, side, m]) for m in
                           ('play_agree_vhead', 'play_agree_qhead', 'play_agree_qhead_vs_qbest', 'nll_pi', 'mass_v', 'mass_q', 'mae_mass_v', 'mae_mass_q')}
                cm = rs[0]['E1'][vk][side]
                e[side]['n'] = cm['n'] if cm else 0
                e[side]['teacher_play_rate'] = cm['teacher_play_rate'] if cm else None
                e[side]['teacher_vis_mass'] = cm['teacher_vis_mass'] if cm else None
            s['E1'][vk] = e
        for sims in ('sims100', 'sims400'):
            s['E2'][sims] = {}
            for vk in PAIRS:
                if vk == 'primal_strength' or vk == 'synthetic:defy__cost2':
                    pass
                if not rs[0]['E2'][sims].get(vk):
                    continue
                o = {'n': rs[0]['E2'][sims][vk]['n'], 'n_legal_same': rs[0]['E2'][sims][vk]['n_legal_same']}
                for sub in ('all', 'legal_same'):
                    o[sub] = {m: collect(rs, ['E2', sims, vk, sub, m]) for m in
                              ('corr_dpolicy', 'corr_dvalue', 'sign_agree_dpolicy_on_sig', 'sign_agree_dvalue_on_sig',
                               'mean_net_dpolicy', 'mean_abs_net_dpolicy', 'mean_net_dvalue')}
                    o[sub]['mean_teacher_dpolicy'] = rs[0]['E2'][sims][vk][sub]['mean_teacher_dpolicy']
                    o[sub]['mean_teacher_dvalue'] = rs[0]['E2'][sims][vk][sub]['mean_teacher_dvalue']
                    o[sub]['n_teacher_dpolicy_sig'] = rs[0]['E2'][sims][vk][sub]['n_teacher_dpolicy_sig']
                    o[sub]['n_teacher_dvalue_sig'] = rs[0]['E2'][sims][vk][sub]['n_teacher_dvalue_sig']
                s['E2'][sims][vk] = o
        s['pooled'] = {m: collect(rs, ['pooled', m]) for m in rs[0]['pooled']}
        summary[f'{c}__{rg}'] = s

    # ---------------------------------------------------------------- paired differences (E1, E3)
    paired = {}
    def paired_diff(c1, c2, rg, path):
        a = by_seed(groups.get((c1, rg), []), path); b = by_seed(groups.get((c2, rg), []), path)
        common = [k for k in a if k in b and a[k] is not None and b[k] is not None]
        return ci([a[k] - b[k] for k in common])
    conds = sorted({c for c, _ in groups})
    for rg in ('base', 'all'):
        for c in conds:
            for ref in ('random', 'id', 'stats', 'Bshuf'):
                if c == ref or (c, rg) not in groups or (ref, rg) not in groups:
                    continue
                d = {}
                for vk in PAIRS:
                    d[vk] = {m: paired_diff(c, ref, rg, ['E1', vk, 'variant', m]) for m in ('nll_pi', 'top1_pi')}
                    d[vk]['card_play_agree_vhead'] = paired_diff(c, ref, rg, ['E1', vk, 'variant_card', 'play_agree_vhead'])
                    d[vk]['card_nll_pi'] = paired_diff(c, ref, rg, ['E1', vk, 'variant_card', 'nll_pi'])
                # pooled over the 11 pairs: decision-weighted mean of variant-card nll per seed
                d['pooled_variant_card_nll'] = paired_diff(c, ref, rg, ['pooled', 'variant_card_nll'])
                d['pooled_variant_nll'] = paired_diff(c, ref, rg, ['pooled', 'variant_nll'])
                d['pooled_variant_top1'] = paired_diff(c, ref, rg, ['pooled', 'variant_top1'])
                d['pooled_card_play_agree'] = paired_diff(c, ref, rg, ['pooled', 'variant_card_play_agree'])
                d['pooled_card_play_agree_q'] = paired_diff(c, ref, rg, ['pooled', 'variant_card_play_agree_q'])
                d['pooled_variant_qhead_top1'] = paired_diff(c, ref, rg, ['pooled', 'variant_qhead_top1'])
                d['pooled_variant_qbest'] = paired_diff(c, ref, rg, ['pooled', 'variant_qbest'])
                d['pooled_card_qbest_agree'] = paired_diff(c, ref, rg, ['pooled', 'variant_card_qbest_agree'])
                d['pooled_variant_brier'] = paired_diff(c, ref, rg, ['pooled', 'variant_brier'])
                paired[f'{c}-{ref}__{rg}'] = d
    results = {'notes': {'ci': '95% t-interval over seeds (n=3 unless stated)',
                         'tau_q': TAU_Q, 'pairs': PAIRS},
               'baselines': baselines, 'summary': summary, 'paired_diffs': paired,
               'per_run': per_run}
    json.dump(results, open(os.path.join(HERE, os.environ.get('RESULTS_OUT', 'results.json')), 'w'), indent=1)
    return results


def add_pooled(per_run):
    pass




def fmt(x):
    if not x:
        return ''
    if x.get('ci95') is None:
        return f"{x['mean']:.4f}"
    return f"{x['mean']:.4f} +- {x['ci95']:.4f}"


def write_tables(r):
    import csv
    S = r['summary']
    with open(os.path.join(HERE, 'table_E0.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'subset', 'n_seeds', 'nll_pi', 'top1_pi', 'qhead_top1_pi', 'qhead_qbest', 'kl_vis', 'value_brier'])
        for k, s in S.items():
            for sub, m in s['E0'].items():
                w.writerow([k, sub, s['n_seeds']] + [fmt(m[x]) for x in ('nll_pi', 'top1_pi', 'qhead_top1_pi', 'qhead_qbest', 'kl_vis', 'value_brier')])
        for sub, b in r['baselines']['all_regime'].items():
            w.writerow(['baseline_uniform', sub, '', f"{b['uniform']['nll_pi']:.4f}", f"{b['uniform']['top1_pi']:.4f}", '', f"{b['uniform']['qbest']:.4f}", '', ''])
            w.writerow(['baseline_kind_freq', sub, '', f"{b['kind_freq']['nll_pi']:.4f}", f"{b['kind_freq']['top1_pi']:.4f}", '', f"{b['kind_freq']['qbest']:.4f}", '', f"{b['value_const']['value_brier']:.4f}"])
    with open(os.path.join(HERE, 'table_E1.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'pair', 'side', 'n', 'nll_pi', 'top1_pi', 'qhead_top1_pi', 'qhead_qbest', 'value_brier',
                    'card_n', 'card_teacher_play_rate', 'card_play_agree_v', 'card_play_agree_q', 'card_play_agree_q_vs_qbest', 'card_nll_pi'])
        for k, s in S.items():
            for vk, e in s['E1'].items():
                for side, cs in (('variant', 'variant_card'), ('base', 'base_card')):
                    d, c = e[side], e[cs]
                    w.writerow([k, vk, side, d['n']] + [fmt(d.get(x)) for x in ('nll_pi', 'top1_pi', 'qhead_top1_pi', 'qhead_qbest', 'value_brier')]
                               + [c['n'], c['teacher_play_rate']] + [fmt(c.get(x)) for x in ('play_agree_vhead', 'play_agree_qhead', 'play_agree_qhead_vs_qbest', 'nll_pi')])
        for vk, d in r['baselines']['base_regime_E1'].items():
            for side, key in (('variant', 'variant_card_in_variant_games'), ('base', 'base_card_in_base_games')):
                b = d[key]
                w.writerow(['baseline_uniform', vk, side, b['n'], f"{b['uniform']['nll_pi']:.4f}", f"{b['uniform']['top1_pi']:.4f}", '', f"{b['uniform']['qbest']:.4f}" if b['uniform']['qbest'] is not None else '', ''])
                w.writerow(['baseline_kind_freq', vk, side, b['n'], f"{b['kind_freq']['nll_pi']:.4f}", f"{b['kind_freq']['top1_pi']:.4f}", '', f"{b['kind_freq']['qbest']:.4f}" if b['kind_freq']['qbest'] is not None else '', f"{b['value_const']['value_brier']:.4f}"])
    with open(os.path.join(HERE, 'table_E1_pooled.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        keys = None
        for k, s in S.items():
            if keys is None:
                keys = list(s['pooled'].keys())
                w.writerow(['run', 'n_seeds'] + keys)
            w.writerow([k, s['n_seeds']] + [fmt(s['pooled'].get(x)) for x in keys])
    with open(os.path.join(HERE, 'table_paired.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['comparison', 'metric', 'diff_mean +- ci95 (over seeds)'])
        for k, d in r['paired_diffs'].items():
            for m, v in d.items():
                if m.startswith('pooled'):
                    w.writerow([k, m, fmt(v)])
    with open(os.path.join(HERE, 'table_E2.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'sims', 'variant', 'subset', 'n', 'corr_dpolicy', 'corr_dvalue', 'sign_agree_dpolicy_on_sig', 'n_teacher_dpolicy_sig',
                    'sign_agree_dvalue_on_sig', 'n_teacher_dvalue_sig', 'mean_net_dpolicy', 'mean_teacher_dpolicy', 'mean_abs_net_dpolicy', 'mean_net_dvalue', 'mean_teacher_dvalue'])
        for k, s in S.items():
            for sims, e in s['E2'].items():
                for vk, o in e.items():
                    for sub in ('all', 'legal_same'):
                        x = o[sub]
                        w.writerow([k, sims, vk, sub, o['n'] if sub == 'all' else o['n_legal_same'], fmt(x['corr_dpolicy']), fmt(x['corr_dvalue']),
                                    fmt(x['sign_agree_dpolicy_on_sig']), x['n_teacher_dpolicy_sig'], fmt(x['sign_agree_dvalue_on_sig']), x['n_teacher_dvalue_sig'],
                                    fmt(x['mean_net_dpolicy']), x['mean_teacher_dpolicy'], fmt(x['mean_abs_net_dpolicy']), fmt(x['mean_net_dvalue']), x['mean_teacher_dvalue']])


if __name__ == '__main__':
    write_tables(main())
