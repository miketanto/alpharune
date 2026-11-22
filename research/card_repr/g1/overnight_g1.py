"""G1 league driver: selfplay/league/overnight.py (the Rengar / Azir / E2 recipe) on the mixed 2x2 matchup grid.

  python3 g1/overnight_g1.py run --deadline 06:00 --tag g1 --out league/g1 --stop-main-at 500
  python3 g1/overnight_g1.py summary --tag g1 --out league/g1

What is the same as overnight.py (and E2): one GPU process at a time; MAIN agent runs/<tag>_main (OSFPMix: 0.8 OSFP
over its own history seeded with the init + 0.2 PFSP (1-p)^2 over cf_id@230, gate4_B2@150, gate4_id@150 and the
main-exploiter snapshots), periodic snapshot every --snapshot-every (50); MAIN EXPLOITER runs/<tag>_exp vs a frozen
copy of the main's latest.pt, snapshot at >= 70% over >= 200 games or after 60 updates, reset to the init.
Phases --main-updates (40) main / --exp-updates (20) exploiter; --stop-main-at ends the run.
Init (env RB_LEAGUE_INIT, default runs/ov_main/league_02080.pt), registered as RB_LEAGUE_BASE (ov_main@2080).

What differs:
  * the trainer is g1/selfplay_g1.py (several env handles, one per ordered matchup, --matchups), for BOTH roles;
    the exploiter plays the same mixed grid. No --deck / RB_DECK.
  * no rating cycles (E2 ran --no-rate --no-ismcts; match.py's rating games are single-deck, so they would be the
    wrong games here). Evaluation after training: g1/eval_g1.py.
  * after every phase the per-matchup win rate vs the pool (league opponents) over that phase is written to log.txt
    and matchups.csv; `summary` writes SUMMARY.txt with the per-matchup table over the run.
  * --extra-members: more fixed nets for the main agent's PFSP pool (e.g. az_main@300=runs/az_main/league_00300.pt);
    default none, as in E2.
  * --cpu (smoke tests): CUDA_VISIBLE_DEVICES='' for the trainer; --nice N runs it under nice.
Outputs in <out>: log.txt, throughput.csv, matchups.csv, registry.json, state.json, SUMMARY.txt, train_*.log.
"""
import argparse
import csv
import datetime
import glob
import json
import os
import signal
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))                 # research/card_repr/g1
SP = os.path.normpath(os.path.join(HERE, '..', 'selfplay'))
sys.path.insert(0, os.path.join(SP, 'league'))
from common import RUNS  # noqa: E402

PY = sys.executable
TRAINER = os.path.join(HERE, 'selfplay_g1.py')
INIT = os.environ.get('RB_LEAGUE_INIT') or os.path.join(RUNS, 'ov_main', 'league_02080.pt')
BASE, BASE_ID = os.environ.get('RB_LEAGUE_BASE', 'ov_main@2080'), 'cf_id@230'
LINK = {'gate4_B2@150': 'runs/gate4_B2/latest.pt', 'gate4_id@150': 'runs/gate4_id/latest.pt'}
DEFAULT_MU = 'rengar:rengar,rengar:azir,azir:rengar,azir:azir'


class Log:
    def __init__(self, path):
        self.path = path

    def __call__(self, *a):
        s = time.strftime('%Y-%m-%d %H:%M:%S ') + ' '.join(str(x) for x in a)
        print(s, flush=True)
        with open(self.path, 'a') as f:
            f.write(s + '\n')


def parse_deadline(hhmm):
    now = datetime.datetime.now()
    h, m = (int(x) for x in hhmm.split(':'))
    d = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if d <= now:
        d += datetime.timedelta(days=1)
    return d.timestamp()


def mem_available_mb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) // 1024
    return -1


def ensure_ram(log, need_mb=1024, wait_s=600):
    t0 = time.time()
    while True:
        m = mem_available_mb()
        if m >= need_mb or m < 0:
            return m
        log(f'RAM low: MemAvailable {m} MB < {need_mb}; waiting')
        if time.time() - t0 > wait_s:
            log(f'WARN RAM still {m} MB after {wait_s}s; continuing anyway')
            return m
        time.sleep(30)


def disk_free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 2**30


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def append_csv(path, cols, row):
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerow(row)


def jload(path, default):
    try:
        return json.load(open(path))
    except Exception:
        return default


def jsave(path, obj):
    tmp = path + '.tmp'
    json.dump(obj, open(tmp, 'w'), indent=1)
    os.replace(tmp, path)


def tail(path, n):
    try:
        return ''.join(open(path).readlines()[-n:])
    except Exception:
        return ''


# ------------------------------------------------------------------------------------------------ setup
def paths(a):
    out = a.out if os.path.isabs(a.out) else os.path.join(SP, a.out)
    os.makedirs(out, exist_ok=True)
    return dict(out=out, reg=os.path.join(out, 'registry.json'), log=os.path.join(out, 'log.txt'),
                state=os.path.join(out, 'state.json'), thr=os.path.join(out, 'throughput.csv'),
                mu=os.path.join(out, 'matchups.csv'), target=os.path.join(out, 'exp_target.pt'),
                main_cfg=os.path.join(out, 'main.json'), exp_cfg=os.path.join(out, 'exp.json'),
                main_run=f'{a.tag}_main', exp_run=f'{a.tag}_exp', summary=os.path.join(out, 'SUMMARY.txt'))


def setup(a, P, log):
    from registry import Registry
    reg = Registry(P['reg'])
    reg.add_anchors()
    reg.add_checkpoint(INIT, role='historical', name=BASE)
    reg.add_checkpoint(os.path.join(RUNS, 'cf_id', 'latest.pt'), role='historical', name=BASE_ID)
    for k, p in LINK.items():
        reg.add_checkpoint(p, role='historical', name=k)
    extra = []
    for item in [x for x in a.extra_members.split(',') if x]:
        name, p = item.split('=')
        reg.add_checkpoint(p, role='historical', name=name)
        extra.append(name)
    reg.save()
    main_cfg = {
        '_doc': 'G1 main agent (overnight.py recipe): 0.8 OSFP over its own history + 0.2 PFSP (1-p)^2 over the '
                'league (exploiter snapshots synced from the registry + fixed nets); mixed matchups (g1/selfplay_g1.py).',
        'role_rules': {'role': 'main', 'snapshot_every': a.snapshot_every},
        'opponents': {'kind': 'osfp_mix', 'p_osfp': a.p_osfp,
                      'osfp': {'p_self': 0.6, 'optimism': 1.0, 'xi': 0.55, 'max_periods': 6,
                               'period_samples': a.osfp_period, 'f': None},
                      'pfsp': {'weighting': 'squared'}},
        'n_slots': a.n_slots, 'resample': 'game', 'registry': P['reg'],
        'seed_members': [BASE_ID] + list(LINK) + extra, 'seed_history': [BASE], 'sync_roles': ['main-exploiter']}
    exp_cfg = {
        '_doc': 'G1 main exploiter: only vs the frozen main copy exp_target.pt, on the same mixed matchups; resets to '
                'the init.',
        'role_rules': {'role': 'main-exploiter', 'init': INIT, 'win_threshold': 0.7, 'min_games': 200,
                       'max_updates': a.exp_max_updates, 'reset_prob': 1.0, 'target_refresh': 10, 'target': 'main'},
        'opponents': {'kind': 'role', 'role': 'main-exploiter'},
        'target_ckpt': P['target'], 'n_slots': 1, 'resample': 'game', 'registry': P['reg'], 'seed_members': []}
    jsave(P['main_cfg'], main_cfg)
    jsave(P['exp_cfg'], exp_cfg)


# ------------------------------------------------------------------------------------------------ training
def run_latest_update(run):
    p = os.path.join(RUNS, run, 'latest.pt')
    if not os.path.exists(p):
        return 0
    import torch
    try:
        return int(torch.load(p, map_location='cpu', weights_only=False)['update'])
    except Exception:
        return -1


def cells_of(rows):
    return sorted({k[3:] for r in rows for k in r if k.startswith('wr_') and '_v_' in k})


def phase_matchups(a, P, log, which, run, cur, new):
    """Per-matchup learner win rate vs the pool over the updates cur+1..new (from the trainer's log.csv)."""
    rows = [r for r in read_csv(os.path.join(RUNS, run, 'log.csv'))
            if r.get('update', '').isdigit() and cur < int(r['update']) <= new]
    if not rows:
        return
    parts = []
    for c in cells_of(rows):
        n = sum(int(r.get(f'n_{c}') or 0) for r in rows)
        wsum = sum(float(r[f'wr_{c}']) * int(r[f'n_{c}']) for r in rows if r.get(f'wr_{c}') not in ('', None))
        dec = sum(int(r.get(f'dec_{c}') or 0) for r in rows)
        wr = round(wsum / n, 3) if n else ''
        append_csv(P['mu'], ['time', 'phase', 'run', 'from', 'to', 'cell', 'pool_games', 'wr_vs_pool', 'decisions'],
                   dict(time=time.strftime('%Y-%m-%dT%H:%M:%S'), phase=which, run=run, **{'from': cur}, to=new,
                        cell=c, pool_games=n, wr_vs_pool=wr, decisions=dec))
        parts.append(f'{c} {wr} (n={n}, dec={dec})')
    log(f'  per-matchup wr vs pool, {which} {cur}->{new}: ' + '; '.join(parts))


def train_phase(a, P, log, which, n_upd, max_min, seed):
    run = P['main_run'] if which == 'main' else P['exp_run']
    cfg = P['main_cfg'] if which == 'main' else P['exp_cfg']
    cur = run_latest_update(run)
    if cur < 0:
        log(f'ERROR {run}/latest.pt unreadable; moving it aside')
        os.replace(os.path.join(RUNS, run, 'latest.pt'), os.path.join(RUNS, run, f'latest.bad.{int(time.time())}.pt'))
        cur = 0
    target = cur + n_upd
    cmd = (['nice', '-n', str(a.nice)] if a.nice else []) + [
        'taskset', '-c', a.train_cores, PY, TRAINER,
        '--repr', 'B2', '--ask-head', '--ask-opt', '--n-envs', str(a.n_envs), '--updates', str(target),
        '--samples-per-update', str(a.samples_per_update), '--ckpt-every', '1000000',
        '--run-name', run, '--league', cfg, '--gpu-mem-frac', str(a.gpu_mem_frac), '--seed', str(seed),
        '--threads', '2', '--max-minutes', f'{max_min:.2f}', '--resume', '--matchups', a.matchups]
    if cur == 0:
        cmd += ['--init', INIT]
    if a.train_extra:
        cmd += a.train_extra.split()
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', LEAGUE_MODEL='live')
    if a.cpu:
        env['CUDA_VISIBLE_DEVICES'] = ''
    lp = os.path.join(P['out'], f'train_{which}.log')
    log(f'phase {which}: {run} updates {cur}->{target} (max {max_min:.1f} min)')
    t0 = time.time()
    with open(lp, 'a') as f:
        f.write(f'\n===== {time.ctime()} {" ".join(cmd)}\n'); f.flush()
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=SP, env=env, start_new_session=True)
        STATE['train'] = p
        try:
            rc = p.wait(timeout=max_min * 60 + 600)
        except subprocess.TimeoutExpired:
            log(f'ERROR {which} phase overran; killing'); kill(p); rc = -9
        STATE['train'] = None
    secs = time.time() - t0
    new = run_latest_update(run)
    rows = [r for r in read_csv(os.path.join(RUNS, run, 'log.csv'))
            if r.get('update', '').isdigit() and cur < int(r['update']) <= max(new, cur)]
    gps = sorted(float(r['games_per_s']) for r in rows if r.get('games_per_s'))
    med = gps[len(gps) // 2] if gps else float('nan')
    upd_s = secs / max(new - cur, 1)
    append_csv(P['thr'], ['time', 'phase', 'run', 'from', 'to', 'rc', 'secs', 'secs_per_update', 'games_per_s_med',
                          'games_per_s_min', 'mem_avail_mb'],
               dict(time=time.strftime('%Y-%m-%dT%H:%M:%S'), phase=which, run=run, **{'from': cur}, to=new, rc=rc,
                    secs=round(secs), secs_per_update=round(upd_s, 1), games_per_s_med=round(med, 1),
                    games_per_s_min=round(gps[0], 1) if gps else '', mem_avail_mb=mem_available_mb()))
    log(f'phase {which} done rc={rc} updates {cur}->{new} in {secs:.0f}s ({upd_s:.1f} s/update), '
        f'games/s median {med:.1f}' + (f' min {gps[0]:.1f}' if gps else ''))
    try:
        phase_matchups(a, P, log, which, run, cur, new)
    except Exception as e:
        log(f'WARN per-matchup summary failed: {e!r}')
    if rc != 0:
        log(f'ERROR {which} phase exit code {rc}; tail of {lp}:\n' + tail(lp, 15))
    for f in glob.glob(os.path.join(RUNS, run, 'ckpt_*.pt')):
        os.remove(f)
    return rc, new


def copy_target(P, log):
    import torch
    ck = torch.load(os.path.join(RUNS, P['main_run'], 'latest.pt'), map_location='cpu', weights_only=False)
    torch.save({'model': ck['model'], 'update': ck['update'], 'repr': ck['repr'], 'dims': ck['dims']},
               P['target'] + '.tmp')
    os.replace(P['target'] + '.tmp', P['target'])
    log(f'exploiter target = main @{ck["update"]}')


def prune(a, P, log):
    """Keep at most --max-ckpts snapshot files: exploiter snapshots go first (newest 2 kept); main snapshots at
    multiples of --snapshot-every are never pruned (they are the evaluation points)."""
    from registry import Registry
    reg = Registry(P['reg'])
    live = [e for e in reg.members(kind='net', loadable=True) if e['run'] in (P['main_run'], P['exp_run'])
            and os.path.exists(reg.abspath(e['id']))]
    mains = sorted([e for e in live if e['run'] == P['main_run']], key=lambda e: e['update'])
    exps = sorted([e for e in live if e['run'] == P['exp_run']], key=lambda e: e['update'])
    prot = {e['id'] for e in mains[-3:]} | {e['id'] for e in exps[-2:]}
    prot |= {e['id'] for e in mains if e['update'] % a.snapshot_every == 0}
    victims = [e for e in exps if e['id'] not in prot] + [e for e in mains if e['id'] not in prot]
    n, gone = len(live), []
    while n > a.max_ckpts and victims:
        e = victims.pop(0)
        try:
            os.remove(reg.abspath(e['id']))
        except OSError:
            pass
        e.update(loadable=False, why='pruned (overnight_g1.py disk cap)', pruned=True)
        gone.append(e['id']); n -= 1
    if gone:
        reg.save()
        log(f'pruned {len(gone)} checkpoints: {gone}; {n} left')


# ------------------------------------------------------------------------------------------------ summary
def summary(a, P=None, log=None):
    P = P or paths(a)
    lines = [f'G1 league summary ({time.ctime()})', f'out: {P["out"]}', '']
    for which, run in (('main', P['main_run']), ('exp', P['exp_run'])):
        rows = [r for r in read_csv(os.path.join(RUNS, run, 'log.csv')) if r.get('update', '').isdigit()]
        if not rows:
            continue
        cells = cells_of(rows)
        lines.append(f'{which} ({run}): {rows[-1]["update"]} updates. Learner win rate vs the pool (league opponents, '
                     f'sampled play) per matchup cell "<learner deck>_v_<opponent deck>", per block of 50 updates:')
        lines.append(f"{'updates':>11s} " + ' '.join(f'{c:>22s}' for c in cells) + f" {'all':>8s}")
        for lo in range(0, int(rows[-1]['update']), 50):
            blk = [r for r in rows if lo < int(r['update']) <= lo + 50]
            parts = []
            for c in cells:
                n = sum(int(r.get(f'n_{c}') or 0) for r in blk)
                ws = sum(float(r[f'wr_{c}']) * int(r[f'n_{c}']) for r in blk if r.get(f'wr_{c}') not in ('', None))
                parts.append(f'{(ws / n if n else float("nan")):.3f} (n={n:5d})')
            pg = sum(int(r.get('pool_games') or 0) for r in blk)
            pw = sum(float(r['wr_vs_pool']) * int(r['pool_games']) for r in blk if r.get('wr_vs_pool'))
            lines.append(f'{lo + 1:>5d}-{lo + 50:<5d} ' + ' '.join(f'{p:>22s}' for p in parts) +
                         f' {(pw / pg if pg else float("nan")):8.3f}')
        lines.append('')
    thr = read_csv(P['thr'])
    if thr:
        lines.append('Throughput per phase: ' + ', '.join(f"{r['phase'][0]}{r['to']}:{r['secs_per_update']}s/u,"
                                                          f"{r['games_per_s_med']}g/s" for r in thr[-12:]))
    lines.append('Evaluation: bash g1/eval_g1.sh (pilot swap, forgetting; see g1/README.txt).')
    txt = '\n'.join(lines) + '\n'
    open(P['summary'], 'w').write(txt)
    (log or print)(('SUMMARY written:\n' if log else '') + txt)


# ------------------------------------------------------------------------------------------------ driver
STATE = {'train': None, 'stop': False}


def kill(p):
    if p is None or p.poll() is not None:
        return
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=30)
    except Exception:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass


def run(a):
    P = paths(a)
    log = Log(P['log'])
    deadline = parse_deadline(a.deadline)
    log(f'=== overnight_g1.py start; deadline {time.ctime(deadline)}; init {INIT} ({BASE}); args {vars(a)}')

    def on_sig(sig, frm):
        log(f'signal {sig}: stopping')
        STATE['stop'] = True
        kill(STATE['train'])
    signal.signal(signal.SIGTERM, on_sig)
    signal.signal(signal.SIGINT, on_sig)

    setup(a, P, log)
    st = jload(P['state'], {'phase_no': 0, 'fails': 0})
    while not STATE['stop'] and time.time() < deadline:
        try:
            ensure_ram(log, a.min_ram_mb)
            if disk_free_gb(SP) < a.min_disk_gb:
                log(f'WARN disk free {disk_free_gb(SP):.1f} GB < {a.min_disk_gb}; pruning harder')
                a.max_ckpts = max(8, a.max_ckpts // 2)
            prune(a, P, log)
            st['phase_no'] += 1
            which = 'main' if st['phase_no'] % 2 == 1 else 'exp'
            if which == 'exp':
                if run_latest_update(P['main_run']) <= 0:
                    log('no main latest.pt yet; skipping exploiter phase')
                    jsave(P['state'], st); continue
                copy_target(P, log)
            n_upd = a.main_updates if which == 'main' else a.exp_updates
            left_main = a.stop_main_at - run_latest_update(P['main_run'])
            if left_main <= 0:
                log(f'main reached {a.stop_main_at} updates: stopping')
                break
            if which == 'main':
                n_upd = min(n_upd, left_main)
            left = (deadline - time.time()) / 60
            if left < 1.0:
                break
            rc, _ = train_phase(a, P, log, which, n_upd, min(left, a.phase_max_min), seed=1000 + st['phase_no'])
            st['fails'] = st['fails'] + 1 if rc != 0 else 0
            jsave(P['state'], st)
            if st['fails'] >= 3:
                log(f'ERROR {st["fails"]} failed phases in a row; sleeping 120 s before retrying')
                time.sleep(120)
            if st['fails'] >= 6:
                log('ERROR 6 failed phases in a row: giving up'); break
        except Exception:
            log('ERROR in driver loop (continuing):\n' + traceback.format_exc())
            time.sleep(30)
    kill(STATE['train'])
    jsave(P['state'], st)
    try:
        summary(a, P, log)
    except Exception:
        log('ERROR writing SUMMARY:\n' + traceback.format_exc())
    log('=== overnight_g1.py done')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'summary'])
    ap.add_argument('--deadline', default='08:00', help='HH:MM local time (next occurrence); a safety stop')
    ap.add_argument('--out', default='league/g1')
    ap.add_argument('--tag', default='g1', help='run names runs/<tag>_main, runs/<tag>_exp')
    ap.add_argument('--matchups', default=DEFAULT_MU, help='ordered seat1:seat2 deck pairs')
    ap.add_argument('--main-updates', type=int, default=40)
    ap.add_argument('--exp-updates', type=int, default=20)
    ap.add_argument('--stop-main-at', type=int, default=500)
    ap.add_argument('--snapshot-every', type=int, default=50)
    ap.add_argument('--exp-max-updates', type=int, default=60)
    ap.add_argument('--p-osfp', type=float, default=0.8)
    ap.add_argument('--osfp-period', type=int, default=163840)
    ap.add_argument('--n-slots', type=int, default=1)
    ap.add_argument('--n-envs', type=int, default=256)
    ap.add_argument('--samples-per-update', type=int, default=16384)
    ap.add_argument('--gpu-mem-frac', type=float, default=0.4)
    ap.add_argument('--train-cores', default='0-11')
    ap.add_argument('--phase-max-min', type=float, default=30)
    ap.add_argument('--max-ckpts', type=int, default=30)
    ap.add_argument('--min-ram-mb', type=int, default=1024)
    ap.add_argument('--min-disk-gb', type=float, default=5)
    ap.add_argument('--extra-members', default='', help='name=path,... fixed nets added to the main PFSP pool')
    ap.add_argument('--train-extra', default='', help='extra selfplay_g1.py options for both roles')
    ap.add_argument('--cpu', action='store_true', help='(smoke tests) trainer on CPU')
    ap.add_argument('--nice', type=int, default=0)
    a = ap.parse_args()
    if a.cmd == 'run':
        run(a)
    else:
        summary(a)


if __name__ == '__main__':
    main()
