"""Print the network's mulligan decisions for a few deals: every option with its probability, and the
value estimate before the mulligan. Uses the viz env (option text). Stops each deal after both mulligans.

  python3 show_mulligans.py CKPT --seeds 7001-7005
"""
import os, sys, argparse, tempfile
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
VIZ = os.path.join(os.path.dirname(HERE), 'viz')
sys.path.insert(0, VIZ)
from play_viz_nn import load_net  # noqa: E402
from play_viz import to_dev  # noqa: E402
from rbenv_viz import RBEnv  # noqa: E402

PK_MULLIGAN = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--seeds', default='7001-7005')
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split('-'))
    torch.set_num_threads(1)
    tmp = tempfile.mkdtemp()
    for seed in range(lo, hi + 1):
        env = RBEnv(1, 'rengar', 'rengar', seed, autopay=True, opponent=None, viz_dir=tmp, opp_seat=0,
                    labels=('P1', 'P2'))
        net = load_net(a.ckpt, env.dims)
        done = 0
        while done < 2:
            obs = env.observe()
            if not obs['need'][0]:
                env.step(np.zeros(1, np.int64))
                continue
            with torch.no_grad():
                logp, v = net(to_dev(obs, np.array([0]), 'cpu'))
            p = logp[0].exp().numpy()
            nl = int(obs['nlegal'][0])
            if int(obs['pk'][0]) == PK_MULLIGAN:
                done += 1
                order = np.argsort(-p[:nl], kind='stable')
                # the hand = the cards named across all options (with copies: max count of a card in one option)
                import re
                from collections import Counter
                hand = Counter()
                for i in range(nl):
                    c = Counter(n.strip() for n in re.findall(r'([^#]+?) #\d+', env.option_text(0, i).replace('Mull:', '')))
                    for k, n in c.items():
                        hand[k] = max(hand[k], n)
                print(f'--- deal {seed}, P{int(obs["seat"][0])}   value before mulligan {float(v[0]):+.2f}   hand: '
                      + ', '.join(k if n == 1 else f'{k} x{n}' for k, n in sorted(hand.items())))
                for i in order[:4]:
                    print(f'   {100 * p[i]:5.1f}%  {env.option_text(0, int(i))}')
            env.step(np.array([int(p[:nl].argmax())], np.int64))
        env.close()


if __name__ == '__main__':
    main()
