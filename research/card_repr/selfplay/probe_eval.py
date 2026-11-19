"""Card-check (probe) evaluation of a checkpoint: sibling of verify_eval.py.

usage: python3 probe_eval.py <ckpt|untrained> --repr B2 [--probes ctrl_tgt,oppctrl_tgt] [--out res.json]
Reports per probe (learner/probes/<name>.npz, cardexp/collect_probes.py): mass_right (mean policy
probability on the rule-right options), argmax_right (top choice right), and for conditional
probes the policy's probability of playing the card when its condition holds vs not.
"""
import argparse, json, os, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'learner'))
from model import Net, load_vocab
import probes

DIMS = dict(G=72, T=96, F=24, A=96, K=49)  # schema v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt'); ap.add_argument('--repr', default='id')
    ap.add_argument('--probes', default=''); ap.add_argument('--ask-head', action='store_true'); ap.add_argument('--out', default='')
    a = ap.parse_args()
    voc = load_vocab()[0]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Net(dict(DIMS, V=len(voc)), voc, a.repr, ask_head=a.ask_head).to(dev)
    if a.ckpt != 'untrained':
        net.load_state_dict(torch.load(a.ckpt, map_location=dev)['model'])
    r = probes.run_probes(net, dev, [p for p in a.probes.split(',') if p] or None)
    res = dict(ckpt=a.ckpt, repr=a.repr, probes=r)
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps(res))


if __name__ == '__main__':
    main()
