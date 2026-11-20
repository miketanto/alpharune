"""Trainer-side glue: who the learner plays, per game, from a league config (league/configs/*.json).

Used by the (unapplied) patch league/selfplay_integration.patch to learner/selfplay.py. Everything the trainer
needs is here, so the patch only adds calls.

  lo = LeagueOpponents(cfg_path, dims, make_net, device, run_dir, run_name, seed)
  lo.begin_update(update)           # sample n_slots opponents for this update, load their nets (cached)
  mid = lo.assign(env)              # per new game: None = self-play (the current policy), else a member id
  lo.net(mid)                       # the frozen nn.Module of that member (eval mode, on device; cached)
  lo.report(env, learner_score)     # per finished game: 1 win, 0 loss, 0.5 draw (for the learner);
                                    # None = errored game (only forgets the assignment)
  act = lo.end_update(update, net, n_learner_samples, save_fn)
        # snapshots / OSFP learning periods / exploiter resets. save_fn(path) must torch.save the learner there.
        # returns dict(snapshot=id|None, reset=bool, init=path) -- on reset the TRAINER reloads `init` (or
        # re-initialises) and builds a fresh optimizer.
  lo.state_dict() / lo.load_state_dict(d)   # put it into latest.pt as 'league'
  lo.log_row() -> dict                      # opponent probabilities and win rates, for metrics.jsonl

Per game vs per batch: config "resample": "game" draws an opponent for every new game among the n_slots nets
loaded for this update (slots are re-drawn from the sampler each update, so a sampler's distribution is followed
over updates while at most n_slots + 1 forward passes run per env step); "update" draws ONE opponent per update
(n_slots 1) -- today's behaviour.
Anchors (C++ agents) are league members for RATINGS only: an RBEnv runs one C++ opponent spec per handle, so
playing them in training would need a second env (not done; README "What's left").
Self-play games (slot -1) train on both seats, as today; games vs a slot train on the learner's seat only.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SP, state_dict_of  # noqa: E402
from exploiter import ExploiterConfig, RoleController  # noqa: E402
from opponents import OSFP, SELF, OSFPMix, RoleSampler, make_sampler  # noqa: E402
from registry import Registry  # noqa: E402


class LeagueOpponents:
    def __init__(self, cfg_path, dims, make_net, device, run_dir, run_name, seed=0, cfg=None):
        self.cfg = cfg if cfg is not None else json.load(open(cfg_path))
        self.dims, self.make_net, self.device = dims, make_net, device
        self.run_dir, self.run_name = run_dir, run_name
        rp = self.cfg.get('registry', 'league/registry.json')
        self.reg = Registry(rp if os.path.isabs(rp) else os.path.join(SP, rp))
        self.sampler = make_sampler(self.cfg['opponents'])
        rr = dict(self.cfg['role_rules'])
        self.role = rr['role']
        self.ctrl = RoleController(ExploiterConfig.for_role(rr.pop('role'), **rr), seed)
        self.n_slots = int(self.cfg.get('n_slots', 4))
        self.per_game = self.cfg.get('resample', 'game') == 'game'
        self.rng = np.random.default_rng(seed + 991)
        self.slots = []                  # member ids loaded this update
        self.nets = {}                   # member id -> nn.Module (cache)
        self.env_opp = {}                # env -> member id or SELF for the game in progress
        self.since_reset = {}            # games / wins per opponent since the last reset (exploiter rules)
        self.target_id = None
        self._seed_members()

    # ------------------------------------------------------------------ membership
    def _add(self, mid, origin=None, current=False):
        if isinstance(self.sampler, RoleSampler):
            e = self.reg.m.get(mid)
            self.sampler.add(mid, role='main' if current else 'historical',
                             origin=origin or (e['role'] if e else 'historical'), current=current)
        elif isinstance(self.sampler, OSFPMix):
            self.sampler.add(mid, origin=origin or 'league')
        else:
            self.sampler.add(mid)

    def _seed_members(self):
        sm = self.cfg.get('seed_members', [])
        if sm == 'all-loadable-nets':
            sm = [e['id'] for e in self.reg.members(kind='net', loadable=True)]
        for mid in sm:
            self._add(mid)
        for mid in self.cfg.get('seed_history', []):     # OSFPMix: the learner's own history H (origin main)
            self._add(mid, origin='main')
        tc = self.cfg.get('target_ckpt')
        if self.role == 'main-exploiter' and tc:
            self.target_id = self.reg.add_checkpoint(tc, role='main', name='target:' + os.path.dirname(tc))
            self.reg.save()
            self._add(self.target_id, current=True)

    # ------------------------------------------------------------------ nets
    def _load(self, mid):
        if mid not in self.nets:
            import torch
            sd = state_dict_of(torch.load(self.reg.abspath(mid), map_location='cpu', weights_only=False))
            net = self.make_net(self.reg.get(mid)).to(self.device)
            net.load_state_dict(sd)
            net.eval()
            if len(self.nets) > 4 * self.n_slots:       # simple bound on the cache
                self.nets.pop(next(iter(self.nets)))
            self.nets[mid] = net
        return self.nets[mid]

    def net(self, mid):
        return self._load(mid)

    # ------------------------------------------------------------------ per update / per game
    def sync(self):
        """Follow registry changes made by OTHER processes (overnight.py runs one trainer at a time):
        drop pool members that were pruned (registry entry gone / not loadable / file missing) and add new members
        whose role is in cfg['sync_roles'] (e.g. main-exploiter snapshots joining the main agent's PFSP pool)."""
        try:
            reg = Registry(self.reg.path)
        except Exception:
            return
        self.reg = reg
        for k in list(self.sampler.pool):
            e = reg.m.get(k)
            if e is None or not e.get('loadable') or (e['kind'] == 'net' and not os.path.exists(reg.abspath(k))):
                self.sampler.remove(k)
                if isinstance(self.sampler, OSFP):
                    self.sampler.G.pop(k, None); self.sampler.C.pop(k, None)
                self.nets.pop(k, None)
        roles = self.cfg.get('sync_roles', [])
        for e in reg.members(role=roles, kind='net', loadable=True) if roles else []:
            if e['id'] not in self.sampler.pool and os.path.exists(reg.abspath(e['id'])):
                self._add(e['id'], origin=e['role'])

    def begin_update(self, update):
        if self.cfg.get('sync_roles') is not None:
            self.sync()
        pr = self.sampler.probs()
        opp = [k for k in pr if k != SELF and pr[k] > 0]
        if not opp:
            self.slots = []
            return
        n = self.n_slots if self.per_game else 1
        w = np.array([pr[k] for k in opp])
        self.slots = [str(k) for k in self.rng.choice(opp, size=min(n, len(opp)), replace=False, p=w / w.sum())]
        self.slot_p = {k: pr[k] for k in self.slots}
        for k in self.slots:
            self._load(k)

    def assign(self, env):
        """New game in `env`: returns None (self-play) or a member id. Follows the sampler's SELF probability and,
        among this update's slots, the sampler's relative weights. A game keeps its opponent across updates."""
        pr = self.sampler.probs()
        p_self = pr.get(SELF, 0.0)
        if not self.slots or self.rng.random() < p_self:
            self.env_opp[env] = SELF
            return None
        w = np.array([self.slot_p[k] for k in self.slots])
        s = int(self.rng.choice(len(self.slots), p=w / w.sum()))
        self.env_opp[env] = self.slots[s]
        return self.slots[s]

    def report(self, env, learner_score):
        k = self.env_opp.pop(env, None)
        if k is None or learner_score is None:
            return
        self.sampler.update(k, learner_score)
        g = self.since_reset.setdefault(k, [0, 0.0])
        g[0] += 1
        g[1] += learner_score

    def end_update(self, update, net, n_samples, save_fn):
        out = dict(snapshot=None, reset=False, init=self.ctrl.cfg.init or None, refreshed=False)
        snap = False
        osfp_based = isinstance(self.sampler, (OSFP, OSFPMix))
        osfp_add = False
        if osfp_based:
            if self.sampler.tick(n_samples):
                osfp_add = snap = self.sampler.end_of_period()
        out['osfp_add'] = osfp_add
        if self.role == 'main':
            snap = self.ctrl.after_update(update)['snapshot'] or snap
        else:
            targets = self._targets()
            wr = {k: self.since_reset[k][1] / self.since_reset[k][0] for k in targets if k in self.since_reset}
            gm = {k: self.since_reset[k][0] for k in targets if k in self.since_reset}
            if len(wr) < len(targets):
                wr.update({k: 0.0 for k in targets if k not in wr}); gm.update({k: 0 for k in targets if k not in gm})
            act = self.ctrl.after_update(update, wr, gm)
            snap = act['snapshot']
            out.update(reason=act['reason'], wr_since_reset={k: round(v, 3) for k, v in wr.items()},
                       games_since_reset=gm)
            out['reset'] = act['reset']
            if act['refresh_target'] and self.target_id:
                self.nets.pop(self.target_id, None)     # reload the main run's latest.pt next time
                out['refreshed'] = True
        if snap:
            path = os.path.join(self.run_dir, f'league_{update:05d}.pt')
            save_fn(path)
            mid = self.reg.add_checkpoint(path, role=self.role, run=self.run_name,
                                          name=f'{self.run_name}@{update}', parent=None)
            self.reg.save()
            # a main agent plays its own snapshots; with OSFP only the snapshots OSFP admits join the history H
            # (periodic snapshot_every snapshots are then registered for ratings only)
            if self.role == 'main' and (osfp_add or not osfp_based):
                self._add(mid, origin='main')
            out['snapshot'] = mid
        if out['reset']:
            self.ctrl.did_reset(update)
            self.since_reset = {}
        return out

    def _targets(self):
        if self.role == 'main-exploiter':
            return [self.target_id] if self.target_id else [k for k in self.sampler.pool]
        return list(self.sampler.pool)

    # ------------------------------------------------------------------ persistence / logging
    def state_dict(self):
        return dict(sampler=self.sampler.state_dict(), ctrl=self.ctrl.state_dict(), since_reset=self.since_reset,
                    target_id=self.target_id)

    def load_state_dict(self, d):
        self.sampler.load_state_dict(d['sampler'])
        self.ctrl.load_state_dict(d['ctrl'])
        self.since_reset = d['since_reset']
        self.target_id = d['target_id']

    def log_row(self):
        pr = self.sampler.probs()
        return dict(league_probs={k: round(v, 4) for k, v in pr.items()}, league_wr=self.sampler.win_rates(),
                    league_slots=list(self.slots))


def _selftest(n_envs=16, steps=400):
    """Drive the real env on CPU with two frozen demo checkpoints as league members and an untrained learner,
    using exactly the call pattern of the patch. Checks assignment, reporting and an OSFP period end."""
    import torch
    from common import import_rbenv, load_net, act_rows, detect_opts, import_model, pin_cores
    pin_cores()
    torch.set_num_threads(1)
    RBEnv = import_rbenv()
    env = RBEnv(n_envs, 'rengar', 'rengar', 4242, autopay=True)
    m = import_model()
    vocab = m.load_vocab()[0]

    def make_net(entry):
        return m.Net(env.dims, vocab, entry['repr'], **entry['opts'])

    import tempfile
    tmpd = tempfile.mkdtemp(prefix='league_selftest_')
    reg_path = os.path.join(tmpd, 'registry_selftest.json')
    reg = Registry(reg_path)
    a = reg.add_checkpoint('runs/gate4_id/ckpt_00050.pt', role='historical')
    b = reg.add_checkpoint('runs/gate4_id/ckpt_00100.pt', role='historical')
    reg.save()
    cfg = dict(role_rules=dict(role='main', snapshot_every=None),
               opponents=dict(kind='osfp', p_self=0.3, optimism=1.0, xi=0.55, max_periods=0, period_samples=500),
               n_slots=2, resample='game', registry=reg_path, seed_members=[a, b])
    tmp = tmpd
    lo = LeagueOpponents(None, env.dims, make_net, 'cpu', tmp, 'selftest', cfg=cfg)
    learner = load_net(os.path.join(SP, 'runs/gate4_id/ckpt_00150.pt'), env.dims, 'id')
    lo.begin_update(1)
    learner_seat = np.where(np.arange(n_envs) % 2 == 0, 1, 2)
    slot = np.array([lo.assign(e) or '' for e in range(n_envs)], dtype=object)
    games = 0
    samples = 0
    for _ in range(steps):
        obs = env.observe()
        rows = np.nonzero(obs['need'])[0]
        ch = np.zeros(n_envs, np.int64)
        if len(rows):
            mine = (obs['seat'][rows] == learner_seat[rows]) | (slot[rows] == '')
            rc = rows[mine]
            if len(rc):
                ch[rc] = act_rows(learner, obs, rc, 'cpu', False)
                samples += len(rc)
            for s in set(slot[rows[~mine]].tolist()):
                rs = rows[~mine][slot[rows[~mine]] == s]
                ch[rs] = act_rows(lo.net(s), obs, rs, 'cpu', False)
        env.step(ch)
        for r in env.results():
            e = r['env']
            w = r['winner']
            lo.report(e, 0.5 if w == 0 else float(w == learner_seat[e]))
            games += 1
            learner_seat[e] = 3 - learner_seat[e]
            slot[e] = lo.assign(e) or ''
    act = lo.end_update(1, learner, samples, lambda p: torch.save({'model': learner.state_dict(), 'update': 1,
                                                                    'repr': 'id', 'dims': env.dims}, p))
    print('games', games, 'learner samples', samples)
    print(json.dumps(lo.log_row()))
    print('end_update ->', act)
    print('pool now', lo.sampler.pool)


if __name__ == '__main__':
    _selftest()
