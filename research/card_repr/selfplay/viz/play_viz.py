"""Play one annotated game: network (checkpoint) vs an opponent spec run in C++, and write
an HTML replay (engine ReplayWriter, trace log, board, decisions) via librbenv_viz.so.

Every network decision gets annotation lines in the replay's trace panel (top-3 options by
probability, the chosen option, the value estimate) and its value in the win-probability
slot; every ISMCTS decision gets its root value and top-3 options by visits.

Schema v4 (2026-11-19): viz/model.py = working-tree learner/model.py, viz/rbenv_viz.py = rbenv.py
(+ librbenv_viz4.so). Net options are detected from the checkpoint as in gate4/po.py load_net.
Default network: runs/gate4_B2/latest.pt. --greedy = top choice (gate-4 eval); otherwise sampled.
The v2 tool (old network, old replays) is viz/v2/.

  python3 play_viz.py --seed 7001 --net-seat 1 [--opp ismcts:sims=100] [--greedy]
"""
import argparse, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import Net, load_vocab  # noqa: E402  (viz copy of learner/model.py)
from rbenv_viz import RBEnv  # noqa: E402



def to_dev(obs, rows, dev):
    """Verbatim from learner/selfplay.py: rows of an observe() dict -> torch, trimmed to used T/A."""
    tm = obs['tok_mask'][rows]
    am = obs['act_mask'][rows]
    T = int(tm.any(0).nonzero()[0].max()) + 1 if tm.any() else 1
    A = int(am.any(0).nonzero()[0].max()) + 1 if am.any() else 1
    b = dict(glob=obs['glob'][rows], tok=obs['tok'][rows, :T], tok_mask=tm[:, :T],
             act=obs['act'][rows, :A], act_mask=am[:, :A], pk=obs['pk'][rows], psrc=obs['psrc'][rows])
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(dev, non_blocking=True) for k, v in b.items()}


CKPT = os.path.join(HERE, '..', 'runs', 'gate4_B2', 'latest.pt')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--net-seat', type=int, default=1, choices=(1, 2))
    ap.add_argument('--opp', default='ismcts:sims=100')
    ap.add_argument('--ckpt', default=CKPT)
    ap.add_argument('--greedy', action='store_true')
    ap.add_argument('--deck', default='rengar', help='mirror deck')
    ap.add_argument('--out', default=os.path.join(HERE, 'replays_v4'))
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(1)
    os.makedirs(a.out, exist_ok=True)
    net_label = 'network(gate4 B2, %s)' % ('greedy' if a.greedy else 'sampled')
    labels = (net_label, a.opp) if a.net_seat == 1 else (a.opp, net_label)
    env = RBEnv(1, a.deck, a.deck, a.seed, autopay=True, opponent=a.opp,
                viz_dir=os.path.abspath(a.out), opp_seat=3 - a.net_seat, labels=labels)
    assert env.schema_version() == 4, env.schema_version()
    sd = torch.load(a.ckpt, map_location='cpu')
    sd = sd.get('net', sd.get('model', sd)) if isinstance(sd, dict) else sd
    net = Net(env.dims, load_vocab()[0], 'B2', ask_head='W_ask.weight' in sd, card_norm='card_ln.weight' in sd,
              ask_opt='W_ask_opt.weight' in sd, opt_cost='W_oc.weight' in sd, opt_refs='W_mg.weight' in sd)
    net.load_state_dict(sd)
    net.eval()
    t0 = time.time()
    n_dec = 0
    vals = []
    while True:
        obs = env.observe()
        if obs['need'][0]:
            assert obs['seat'][0] == a.net_seat, obs['seat']
            with torch.no_grad():
                logp, v = net(to_dev(obs, np.array([0]), 'cpu'))
            p = logp[0].exp().numpy()
            nl = min(int(obs['nlegal'][0]), env.dims['A'])
            choice = int(p[:nl].argmax()) if a.greedy else int(torch.multinomial(logp[0].exp().clamp(min=0), 1))
            val = float(v[0])
            vals.append(val)
            order = np.argsort(-p[:nl], kind='stable')
            lines = ['[NET] value %+.2f (win ~%.0f%%), %d options; top by probability:' % (val, (val + 1) * 50, nl)]
            shown = list(order[:3])
            if choice not in shown:
                shown.append(choice)
            for i in shown:
                lines.append('[NET]   %s%5.1f%% : %s' % ('* ' if i == choice else '  ', 100 * p[i],
                                                         env.option_text(0, int(i))))
            if choice not in order[:3]:
                lines.append('[NET]   (chosen option was outside the top 3: sampled)')
            env.note(0, '\n'.join(lines), val)
            ch = np.array([choice], np.int64)
            n_dec += 1
        else:
            ch = np.zeros(1, np.int64)
        env.step(ch)
        res = env.results()
        if res:
            r = res[0]
            r['net_seat'] = a.net_seat
            r['net_won'] = int(r['winner'] == a.net_seat)
            r['secs'] = round(time.time() - t0, 1)
            r['net_decisions'] = n_dec
            r['net_value_first_last'] = [round(vals[0], 2), round(vals[-1], 2)] if vals else None
            print(r, flush=True)
            break
    env.close()


if __name__ == '__main__':
    main()
