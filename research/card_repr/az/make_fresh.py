"""A fresh (random-initialised) network with the architecture of G2 / the AZ runs, saved as an init checkpoint.

  RB_B2=v2 python3 az/make_fresh.py az/init_fresh.pt [seed]
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('rbenv', 'league', 'learner'):
    sys.path.insert(0, os.path.join(HERE, '..', 'selfplay', d))
import torch
from common import import_model, state_dict_of, detect_opts
from rbenv import RBEnv
m = import_model()
ref = torch.load(os.path.join(HERE, 'run_b2', 'az_009944.pt'), map_location='cpu', weights_only=False)
opts = detect_opts(state_dict_of(ref))
torch.manual_seed(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
dims = RBEnv(1, 'rengar', 'rengar', 1).dims
net = m.Net(dims, m.load_vocab()[0], 'B2', b2=os.environ.get('RB_B2', 'v2'), hist=True, **opts)
torch.save({'net': net.state_dict(), 'update': 0, 'repr': 'B2', 'b2': net.b2, 'init': 'fresh'}, sys.argv[1])
print('fresh net', opts, 'params', sum(p.numel() for p in net.parameters()), 'tap_bias', net.tap_bias.tolist(), '->', sys.argv[1])
