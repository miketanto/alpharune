"""Adds a net's model responses (root value + root policy) to run.py output after the fact: each position is
replayed, its variants rebuilt with the same edits (deterministic), and the net evaluated at each root.
No playouts. Writes a new JSONL.

usage: python3 add_model.py in.jsonl out.jsonl --net id=<ckpt>
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import po as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inp'); ap.add_argument('out'); ap.add_argument('--net', required=True)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(2)
    name, ck = a.net.split('=')
    net = P.load_net(ck, name.split('_')[0], 'cpu')
    n = 0
    with open(a.out, 'w') as f:
        for line in open(a.inp):
            r = json.loads(line)
            if not r.get('skipped'):
                po = P.Po(r['seed'], r['hist'], n_envs=1)
                for v in r['variants']:
                    if not v['ok']:
                        continue
                    vi, err = po.add_variant(v['edit'])
                    assert vi is not None, err
                    ro = po.root_obs(vi)
                    legal = po.info(vi)['legal'][:ro['nlegal']]
                    assert legal == v['legal'], 'variant rebuilt differently'
                    pr, val = P.net_eval(net, ro, np.array([0]), 'cpu')
                    v['model'][name] = dict(v=float(val[0]), p=pr[0, :len(legal)].tolist())
                po.close()
                n += 1
            f.write(json.dumps(r) + '\n')
    print('positions', n)


if __name__ == '__main__':
    main()
