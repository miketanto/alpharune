"""Network-vs-network variant of play_viz.py: --ckpt plays P1, --ckpt2 plays P2, both in Python
(no C++ opponent). Writes the same HTML replay (librbenv_viz) with each network's top-3 options and
value annotated at every decision it makes.

  python3 play_viz_nn.py --seed 7001 --ckpt A.pt --ckpt2 B.pt --name1 main --name2 exploiter --greedy --out DIR
"""
import argparse, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import Net, load_vocab  # noqa: E402
from rbenv_viz import RBEnv  # noqa: E402
from play_viz import to_dev  # noqa: E402


def load_net(ck, dims):
    sd = torch.load(ck, map_location='cpu')
    sd = sd.get('net', sd.get('model', sd)) if isinstance(sd, dict) else sd
    net = Net(dims, load_vocab()[0], 'B2', ask_head='W_ask.weight' in sd, card_norm='card_ln.weight' in sd,
              ask_opt='W_ask_opt.weight' in sd, opt_cost='W_oc.weight' in sd, opt_refs='W_mg.weight' in sd)
    net.load_state_dict(sd)
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--ckpt2', required=True)
    ap.add_argument('--name1', default='P1 net')
    ap.add_argument('--name2', default='P2 net')
    ap.add_argument('--greedy', action='store_true')
    ap.add_argument('--deck', default='rengar', help='P1 deck (both seats unless --deck2)')
    ap.add_argument('--deck2', default='', help='P2 deck')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(1)
    os.makedirs(a.out, exist_ok=True)
    mode = 'greedy' if a.greedy else 'sampled'
    labels = ('%s (%s)' % (a.name1, mode), '%s (%s)' % (a.name2, mode))
    env = RBEnv(1, a.deck, a.deck2 or a.deck, a.seed, autopay=True, opponent=None,
                viz_dir=os.path.abspath(a.out), opp_seat=0, labels=labels)
    assert env.schema_version() == 4, env.schema_version()
    nets = {1: load_net(a.ckpt, env.dims), 2: load_net(a.ckpt2, env.dims)}
    names = {1: a.name1, 2: a.name2}
    vals = {1: [], 2: []}
    t0 = time.time()
    while True:
        obs = env.observe()
        if obs['need'][0]:
            seat = int(obs['seat'][0])
            with torch.no_grad():
                logp, v = nets[seat](to_dev(obs, np.array([0]), 'cpu'))
            p = logp[0].exp().numpy()
            nl = min(int(obs['nlegal'][0]), env.dims['A'])
            choice = int(p[:nl].argmax()) if a.greedy else int(torch.multinomial(logp[0].exp().clamp(min=0), 1))
            val = float(v[0])
            vals[seat].append(val)
            order = np.argsort(-p[:nl], kind='stable')
            # "[NET]" prefix and layout as play_viz.py, so transcript.py parses both seats' decisions.
            lines = ['[NET] value %+.2f (win ~%.0f%%), %d options; top by probability: (P%d %s)'
                     % (val, (val + 1) * 50, nl, seat, names[seat])]
            shown = list(order[:3])
            if choice not in shown:
                shown.append(choice)
            for i in shown:
                lines.append('[NET]   %s%5.1f%% : %s' % ('* ' if i == choice else '  ', 100 * p[i],
                                                       env.option_text(0, int(i))))
            env.note(0, '\n'.join(lines), val)
            ch = np.array([choice], np.int64)
        else:
            ch = np.zeros(1, np.int64)
        env.step(ch)
        res = env.results()
        if res:
            r = res[0]
            r['names'] = names
            r['secs'] = round(time.time() - t0, 1)
            r['value_first_last'] = {s: [round(x[0], 2), round(x[-1], 2)] if x else None for s, x in vals.items()}
            print(r, flush=True)
            break
    env.close()


if __name__ == '__main__':
    main()
