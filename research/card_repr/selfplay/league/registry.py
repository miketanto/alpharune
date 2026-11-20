"""Checkpoint registry for league training (league/README.txt).

One JSON file (default league/registry.json) holding every league member:
  id           unique name ('gate4_id@150', 'anchor:random-player', ...)
  kind         'net' | 'anchor'
  path         checkpoint path relative to selfplay/ (nets), None for anchors
  spec         agent spec for anchors ('random-player', 'ismcts:sims=100'), None for nets
  repr         card representation (id / B2 / B / random) from the checkpoint or its run's config.json
  opts         model options detected as the loaders do (common.detect_opts)
  schema       rbenv schema version the checkpoint was trained on (1..4; common.schema_from_dims)
  loadable     whether it loads on the current env (schema >= 2) + 'why' if not
  parent       id of the member it was initialised / reset from (None for a fresh start)
  role         'main' | 'main-exploiter' | 'league-exploiter' | 'anchor' | 'historical'
  run          run directory name (selfplay/runs/<run>) or None
  update       learner update count at save time
  created      file mtime (ISO) for checkpoints, registration time for anchors
  frozen       True for anything the trainer must not update (all registered snapshots and anchors)

API
  reg = Registry(path=None)                  # loads or creates
  reg.add_anchors()                          # random-player, ismcts:sims=25/100/200 (role 'anchor')
  reg.add_checkpoint(path, role='main', parent=None, name=None, run=None) -> id
  reg.add_run(run_dir, role='main', which='latest'|'all'|[updates])      -> [ids]
  reg.get(id), reg.members(role=None, kind=None, loadable=None), reg.save()
CLI
  python3 registry.py add-run gate4_id --role main ; python3 registry.py list
"""
import datetime
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ANCHORS, HERE, RUNS, SP, inspect_checkpoint  # noqa: E402

ROLES = ('main', 'main-exploiter', 'league-exploiter', 'anchor', 'historical')
CURRENT_SCHEMA_MIN = 2   # rbenv README: v2+ checkpoints load on the v4 env; v1 (T 64, F 17) do not


def _iso(t):
    return datetime.datetime.fromtimestamp(t).isoformat(timespec='seconds')


class Registry:
    def __init__(self, path=None):
        self.path = path or os.path.join(HERE, 'registry.json')
        self.m = {}
        if os.path.exists(self.path):
            self.m = {e['id']: e for e in json.load(open(self.path))['members']}

    # ------------------------------------------------------------------ write
    def save(self):
        tmp = self.path + '.tmp'
        json.dump({'version': 1, 'members': list(self.m.values())}, open(tmp, 'w'), indent=1)
        os.replace(tmp, self.path)

    def add_anchors(self, specs=ANCHORS):
        out = []
        for s in specs:
            i = 'anchor:' + s
            if i not in self.m:
                self.m[i] = dict(id=i, kind='anchor', path=None, spec=s, repr=None, opts=None, schema=None,
                                 loadable=True, why=None, parent=None, role='anchor', run=None, update=None,
                                 created=_iso(time.time()), frozen=True)
            out.append(i)
        return out

    def add_checkpoint(self, path, role='main', parent=None, name=None, run=None, update=None):
        assert role in ROLES, role
        ap = path if os.path.isabs(path) else os.path.join(SP, path)
        info = inspect_checkpoint(ap)
        rel = os.path.relpath(ap, SP).replace(os.sep, '/')
        if run is None:
            mm = re.match(r'runs/([^/]+)/', rel)
            run = mm.group(1) if mm else None
        rep = info['repr']
        cfgp = os.path.join(os.path.dirname(ap), 'config.json')
        if rep is None and os.path.exists(cfgp):
            rep = json.load(open(cfgp)).get('repr')
        if parent is None and os.path.exists(cfgp):
            init = json.load(open(cfgp)).get('init') or ''
            parent = ('file:' + init) if init else None
        upd = update if update is not None else info['update']
        i = name or f'{run or os.path.basename(ap)}@{upd}'
        ok = info['schema'] is not None and info['schema'] >= CURRENT_SCHEMA_MIN
        why = None if ok else f"schema v{info['schema']} (dims {info['dims']}): input layout differs from the current env"
        self.m[i] = dict(id=i, kind='net', path=rel, spec=None, repr=rep, opts=info['opts'], schema=info['schema'],
                         dims=info['dims'], loadable=ok, why=why, parent=parent, role=role, run=run, update=upd,
                         created=_iso(os.path.getmtime(ap)), frozen=True)
        return i

    def add_run(self, run, role='main', which='latest'):
        d = run if os.path.isabs(run) else os.path.join(RUNS, run)
        name = os.path.basename(d.rstrip('/'))
        if which == 'latest':
            return [self.add_checkpoint(os.path.join(d, 'latest.pt'), role=role, run=name)]
        cks = sorted(f for f in os.listdir(d) if re.match(r'ckpt_\d+\.pt$', f))
        if which != 'all':
            keep = set(int(u) for u in which)
            cks = [f for f in cks if int(f[5:10]) in keep]
        ids, prev = [], None
        for f in cks:
            i = self.add_checkpoint(os.path.join(d, f), role=role if f == cks[-1] else 'historical', run=name,
                                    parent=prev)
            ids.append(i)
            prev = i
        return ids

    def set_role(self, i, role):
        assert role in ROLES
        self.m[i]['role'] = role

    # ------------------------------------------------------------------ read
    def get(self, i):
        return self.m[i]

    def members(self, role=None, kind=None, loadable=None):
        out = list(self.m.values())
        if role is not None:
            rs = (role,) if isinstance(role, str) else tuple(role)
            out = [e for e in out if e['role'] in rs]
        if kind is not None:
            out = [e for e in out if e['kind'] == kind]
        if loadable is not None:
            out = [e for e in out if e['loadable'] == loadable]
        return out

    def abspath(self, i):
        e = self.m[i]
        return None if e['path'] is None else os.path.join(SP, e['path'])

    def table(self):
        rows = [f"{'id':34s} {'kind':6s} {'role':16s} {'repr':5s} {'schema':6s} {'upd':>5s} load  opts"]
        for e in self.m.values():
            o = e['opts'] or {}
            os_ = ','.join(k for k, v in o.items() if v) if o else (e['spec'] or '')
            rows.append(f"{e['id']:34s} {e['kind']:6s} {e['role']:16s} {str(e['repr']):5s} {str(e['schema']):6s} "
                        f"{str(e['update']):>5s} {'yes ' if e['loadable'] else 'NO  '} {os_}")
        return '\n'.join(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['list', 'add-run', 'add-ckpt', 'anchors'])
    ap.add_argument('target', nargs='?')
    ap.add_argument('--role', default='main')
    ap.add_argument('--which', default='latest')
    ap.add_argument('--registry', default=None)
    a = ap.parse_args()
    reg = Registry(a.registry)
    if a.cmd == 'add-run':
        print(reg.add_run(a.target, a.role, a.which if a.which in ('latest', 'all') else a.which.split(',')))
    elif a.cmd == 'add-ckpt':
        print(reg.add_checkpoint(a.target, a.role))
    elif a.cmd == 'anchors':
        print(reg.add_anchors())
    reg.save()
    print(reg.table())


if __name__ == '__main__':
    main()
