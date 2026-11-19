"""Checks asked for by the coordinator (CARD_EXPERIMENT.md, "Checks").
1. decode a few oppctrl_tgt / ctrl_tgt probe states (is `good` = enemy units?)
2. sensitivity: does the target choice depend on WHICH card asks? Replace the asking token's card key
   (psrc) in the oppctrl states by other keys and compare the policy's mass on enemy units."""
import os, sys, json, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'learner'))
from model import Net, load_vocab
voc = load_vocab()[0]; V = {k: i for i, k in enumerate(voc)}
D = dict(G=72, T=64, F=17, A=96, K=46, V=len(voc))
P = os.path.join(HERE, '..', 'learner', 'probes')
ZONE = {2: 'hand', 3: 'base', 8: 'bf', 9: 'chain'}


def decode(name, n=3):
    d = np.load(os.path.join(P, name + '.npz'))
    for b in range(n):
        ps = d['psrc'][b]; t = d['tok'][b]
        print(f'{name} state {b}: pk {d["pk"][b]} asking token {ps} = {voc[t[ps, 1]]} (zone {ZONE.get(t[ps, 4], t[ps, 4])}, ctrl {t[ps, 2]})')
        for a in np.nonzero(d['act_mask'][b])[0]:
            r = d['act'][b, a, 8]
            print(f'   option {a}: type {d["act"][b, a, 0]} chooses token {r} = {voc[t[r, 1]]} controller {t[r, 2]} (1 self, 2 opp) zone {ZONE.get(t[r, 4], t[r, 4])} might {t[r, 9]}  good={bool(d["good"][b, a])}')


def sensitivity(run, rep):
    sd = torch.load(os.path.join(HERE, '..', 'runs', run, 'latest.pt'), map_location='cpu')['model']
    net = Net(D, voc, rep, ask_head='W_ask.weight' in sd, card_norm='card_ln.weight' in sd, ask_opt='W_ask_opt.weight' in sd, opt_cost='W_oc.weight' in sd, opt_refs='W_mg.weight' in sd)
    net.load_state_dict(sd)
    net.eval()
    d = np.load(os.path.join(P, 'oppctrl_tgt.npz'))
    out = {}
    for key in ['synthetic:punch_first__opp_of_ctrl_scores', 'synthetic:punch_first__ctrl_scores', 'punch_first',
                'hidden', 'noxus_hopeful', '<no psrc marker>']:
        tok = d['tok'].copy(); psrc = d['psrc'].copy()
        if key == '<no psrc marker>':
            psrc[:] = -1
        else:
            tok[np.arange(len(tok)), psrc, 1] = V[key]
        b = dict(glob=d['glob'], tok=tok, tok_mask=d['tok_mask'], act=d['act'], act_mask=d['act_mask'], pk=d['pk'], psrc=psrc)
        b = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in b.items()}
        with torch.no_grad():
            p = net(b)[0].exp().numpy()
        out[key] = round(float((p * d['good']).sum(1).mean()), 4)
    return out


if __name__ == '__main__':
    decode('oppctrl_tgt'); decode('ctrl_tgt', 1)
    res = {}
    for run in sys.argv[1:]:
        rep = 'id' if 'fix_' in run else run.split('cx_')[1].split('_s')[0]
        res[run] = sensitivity(run, rep)
    print(json.dumps(res, indent=1))
