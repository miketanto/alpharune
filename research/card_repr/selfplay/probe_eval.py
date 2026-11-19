"""Card-check (probe) evaluation of a checkpoint: sibling of verify_eval.py.

usage: python3 probe_eval.py <ckpt|untrained> --repr B2 [--probes ctrl_tgt,oppctrl_tgt] [--out res.json]
Reports per probe (learner/probes/<name>.npz, cardexp/collect_probes.py): mass_right (mean policy
probability on the rule-right options), argmax_right (top choice right), and for conditional
probes the policy's probability of playing the card when its condition holds vs not.
Evaluation plays GREEDY by default (user decision, 2026-11-19): each probe's headline "score" is argmax_right
(and play_top for conditional probes); --sample makes it the sampled policy's expectation, mass_right
(play_mass). Both metrics are always reported; the mode is recorded in the JSON ("mode").
"""
import argparse, json, os, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'learner'))
from model import Net, load_vocab
import probes

DIMS = dict(G=72, T=96, F=24, A=96, K=49)  # schema v2; K raised to 56 / 61 for opt_cost / opt_refs nets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt'); ap.add_argument('--repr', default='id')
    ap.add_argument('--probes', default=''); ap.add_argument('--ask-head', action='store_true'); ap.add_argument('--out', default='')
    ap.add_argument('--sample', action='store_true', help='headline score = sampled policy (mass); default greedy')
    ap.add_argument('--ask-opt', action='store_true', help='untrained net only: checkpoints are detected from their keys')
    a = ap.parse_args()
    voc = load_vocab()[0]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    sd = torch.load(a.ckpt, map_location=dev)['model'] if a.ckpt != 'untrained' else None
    cnorm = sd is None or 'card_ln.weight' in sd   # pre-gate-2 checkpoints have no card LayerNorm
    # Heads are detected from the checkpoint's keys (gate 3): --ask-head / --ask-opt nets, schema-v3 W_oc and
    # schema-v4 W_ci / W_mg. The flags apply to an untrained net only.
    ask = 'W_ask.weight' in sd if sd is not None else a.ask_head
    aopt = 'W_ask_opt.weight' in sd if sd is not None else a.ask_opt
    ocost = sd is not None and 'W_oc.weight' in sd
    orefs = sd is not None and 'W_mg.weight' in sd
    K = 61 if orefs else (56 if ocost else 49)
    net = Net(dict(DIMS, K=K, V=len(voc)), voc, a.repr, ask_head=ask, card_norm=cnorm, ask_opt=aopt,
              opt_cost=ocost, opt_refs=orefs).to(dev)
    if sd is not None:
        net.load_state_dict(sd)
    r = probes.run_probes(net, dev, [p for p in a.probes.split(',') if p] or None)
    for v in r.values():
        if isinstance(v, dict) and 'argmax_right' in v:
            v['score'] = v['mass_right'] if a.sample else v['argmax_right']
    res = dict(ckpt=a.ckpt, mode='sample' if a.sample else 'greedy', repr=a.repr, probes=r)
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps(res))


if __name__ == '__main__':
    main()
