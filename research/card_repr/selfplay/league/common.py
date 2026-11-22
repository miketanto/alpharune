"""Shared helpers for the league package: paths, checkpoint inspection, network loading, acting.

Network loading mirrors verify_eval.py (the "loaders"): model options are detected from the state-dict keys
  ask_head  <- 'W_ask.weight'      card_norm <- 'card_ln.weight'   ask_opt <- 'W_ask_opt.weight'
  opt_cost  <- 'W_oc.weight'       opt_refs  <- 'W_mg.weight'
and the schema version from the stored dims (F 17 -> v1; K 49 -> v2, 56 -> v3, 61 -> v4; rbenv/README.txt).

Which model.py is used: learner/model.py is being edited by another agent, so by default the league code uses
league/model_snapshot.py (a verbatim copy of learner/model.py taken 2026-11-19 19:14, md5 0011b2e1...). Set
LEAGUE_MODEL=live to import learner/model.py instead (what a trainer integration would do).
to_dev / act_rows are copied from learner/selfplay.py (same semantics) so league code does not import the trainer.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SP = os.path.normpath(os.path.join(HERE, '..'))                 # research/card_repr/selfplay
RUNS = os.path.join(SP, 'runs')
LEARNER = os.path.join(SP, 'learner')
RBENV = os.path.join(SP, 'rbenv')
EVAL_CORES = (12, 13, 14, 15)                                   # CPU cores for evaluation games

ANCHORS = ('random-player', 'ismcts:sims=25', 'ismcts:sims=100', 'ismcts:sims=200')


def pin_cores(cores=EVAL_CORES):
    try:
        os.sched_setaffinity(0, set(cores))
    except (AttributeError, OSError):
        pass


def import_model():
    if os.environ.get('LEAGUE_MODEL', 'live') == 'live':
        sys.path.insert(0, LEARNER)
        import model as m
    else:
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        import model_snapshot as m
    return m


def import_rbenv():
    if RBENV not in sys.path:
        sys.path.insert(0, RBENV)
    from rbenv import RBEnv
    return RBEnv


def state_dict_of(ck):
    if isinstance(ck, dict):
        return ck.get('net', ck.get('model', ck))
    return ck


def detect_opts(sd):
    """Model options as verify_eval.py detects them from a state dict (None sd = untrained default net)."""
    return dict(ask_head=sd is not None and 'W_ask.weight' in sd,
                card_norm=sd is None or 'card_ln.weight' in sd,
                ask_opt=sd is not None and 'W_ask_opt.weight' in sd,
                opt_cost=sd is None or 'W_oc.weight' in sd,
                opt_refs=sd is None or 'W_mg.weight' in sd)


def schema_from_dims(dims):
    if not dims:
        return None
    if int(dims.get('F', 0)) == 17 or int(dims.get('T', 0)) == 64:
        return 1
    K = int(dims.get('K', 0))
    return {49: 2, 56: 3, 61: 4}.get(K, None)


def schema_from_sd(sd):
    """Fallback when a checkpoint has no dims: the per-kind numeric map W_num has F-2 inputs."""
    w = sd.get('W_num.weight') if sd is not None else None
    if w is None:
        return None
    nf = int(w.shape[1])
    if nf == 15:
        return 1
    if 'W_mg.weight' in sd:
        return 4
    if 'W_oc.weight' in sd:
        return 3
    return 2


def inspect_checkpoint(path):
    """-> dict(repr, update, dims, schema, opts) without building a network."""
    import torch
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = state_dict_of(ck)
    dims = ck.get('dims') if isinstance(ck, dict) else None
    rep = ck.get('repr') if isinstance(ck, dict) else None
    if rep is None:
        rep = 'id' if 'E_card.weight' in sd else None
    schema = schema_from_dims(dims) or schema_from_sd(sd)
    return dict(repr=rep, update=ck.get('update') if isinstance(ck, dict) else None,
                dims={k: int(v) for k, v in dims.items()} if dims else None, schema=schema, opts=detect_opts(sd))


def load_net(path, dims, rep=None, device='cpu', b2=None):
    """Build model.Net with options detected from the checkpoint and load it (eval mode)."""
    import torch
    m = import_model()
    sd = None
    if path != 'untrained':
        ck = torch.load(path, map_location=device, weights_only=False)
        sd = state_dict_of(ck)
        if rep is None and isinstance(ck, dict):
            rep = ck.get('repr')
        if b2 is None:   # the B2 vector file the checkpoint was trained with (absent = v1)
            b2 = ck.get('b2', 'v1') if isinstance(ck, dict) else 'v1'
    rep = rep or 'id'
    net = m.Net(dims, m.load_vocab()[0], rep, b2=b2, **detect_opts(sd)).to(device)
    if sd is not None:
        net.load_state_dict(sd)
    net.eval()
    return net


def to_dev(obs, rows, dev):
    """Copy of learner/selfplay.py to_dev: rows of an observe() dict -> torch, trimmed to the used T/A."""
    import torch
    tm = obs['tok_mask'][rows]
    am = obs['act_mask'][rows]
    T = int(tm.any(0).nonzero()[0].max()) + 1 if tm.any() else 1
    A = int(am.any(0).nonzero()[0].max()) + 1 if am.any() else 1
    b = dict(glob=obs['glob'][rows], tok=obs['tok'][rows, :T], tok_mask=tm[:, :T],
             act=obs['act'][rows, :A], act_mask=am[:, :A], pk=obs['pk'][rows], psrc=obs['psrc'][rows])
    if 'hist' in obs:   # rbenv history.h (nets without a history block ignore it)
        b['hist'] = obs['hist'][rows]
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(dev) for k, v in b.items()}


def act_rows(net, obs, rows, dev='cpu', greedy=True):
    """Copy of learner/selfplay.py act_rows (greedy = argmax, else sample)."""
    import torch
    with torch.no_grad():
        logp, v = net(to_dev(obs, rows, dev))
        a = logp.argmax(-1) if greedy else torch.multinomial(logp.exp().clamp(min=0), 1).squeeze(-1)
    return a.cpu().numpy()
