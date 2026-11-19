"""Builds ../results.json (measured rates + projections) from raw/*.json, raw/jobs.csv and the load logs.
Every projection states its formula and assumptions in the 'assumptions' field."""
import json, os, csv, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); SP = os.path.join(HERE, '..'); RAW = os.path.join(SP, 'raw')
J = lambda f: json.load(open(os.path.join(RAW, f)))
G = J('games_summary.json'); EX = J('existing_games.json'); RUNS = J('existing_runs.json')
C64 = J('train_cpu_wsl_d64.json'); C128 = J('train_cpu_wsl_d128.json'); GPU = J('train_gpu_wsl.json')
WIN = J('train_cpu_windows_torch20.json'); PREP = J('prep.json')


def m(cfg, k):
    return G[cfg]['aggregate'][k]['median']


def row(js, d, th, bs):
    for r in js['summary']:
        if (r['d'], r['threads'], r['bs']) == (d, th, bs):
            return r
    raise KeyError((d, th, bs))


def inf(js, d, th, b):
    for r in js['infer_summary']:
        if (r['d'], r['threads'], r['infer_batch']) == (d, th, b):
            return r


# ---------------------------------------------------------------- load during the runs
jobs = list(csv.DictReader(open(os.path.join(RAW, 'jobs.csv'))))
load = [list(map(float, r)) for r in list(csv.reader(open(os.path.join(RAW, 'wsl_load.csv'))))[1:] if len(r) == 18]


def load_between(t0, t1):
    rs = [r for r in load if t0 <= r[0] <= t1]
    if not rs:
        return None
    return {'samples': len(rs), 'loadavg1_mean': round(st.mean(r[1] for r in rs), 2),
            'cpu0_7_busy_pct_mean': round(st.mean(st.mean(r[2:10]) for r in rs), 1),
            'cpu8_15_busy_pct_mean': round(st.mean(st.mean(r[10:18]) for r in rs), 1)}


phases = {}
for name, pre in [('sims_sweep+autopay+random', ('sw_', 'ap_')), ('scaling_P1', ('sc_P1_',)), ('scaling_P2', ('sc_P2_',)),
                  ('scaling_P4', ('sc_P4_',)), ('scaling_P8', ('sc_P8_',))]:
    js = [j for j in jobs if j['tag'].startswith(pre)]
    phases[name] = load_between(min(float(j['t0']) for j in js), max(float(j['t1']) for j in js))

# ---------------------------------------------------------------- generation rates
sims = {}
for s in (25, 50, 100, 200, 400):
    c = f'sw_t{s}'
    sims[s] = {'games_per_hour_per_process': G[c]['aggregate']['games_per_hour'],
               'secs_per_game_median': m(c, 'secs_per_game'), 'ms_per_search_median': round(1000 * m(c, 'secs_per_search'), 2),
               'ms_per_simulation': round(1000 * m(c, 'secs_per_search') / s, 3),
               'decisions_per_game': m(c, 'decisions_per_game'), 'multi_option_decisions_per_game': m(c, 'multi_option_per_game'),
               'searches_per_game': m(c, 'searches_per_game'), 'games_per_repeat': G[c]['repeats'][0]['games']}
sc = G['scaling_sims100']
scaling = {P: {'aggregate_games_per_hour_sum_of_process_rates': v['aggregate']['aggregate_games_per_hour_sum_rates'],
               'aggregate_games_per_hour_block_span': v['aggregate']['aggregate_games_per_hour_span'],
               'ms_per_search': {k: round(1000 * x, 2) if isinstance(x, float) else x for k, x in v['aggregate']['secs_per_search'].items()}}
           for P, v in sc.items()}
ms_search = {int(P): v['aggregate']['secs_per_search']['median'] for P, v in sc.items()}
rand = G['sw_rand']['aggregate']
gen = {
    'random_player': {'games_per_s_per_process': rand['games_per_s'], 'games_per_hour_per_process': rand['games_per_hour'],
                      'decisions_per_game': m('sw_rand', 'decisions_per_game'), 'multi_option_per_game': m('sw_rand', 'multi_option_per_game'),
                      'export_share_of_game_time': m('sw_rand', 'encode_frac_of_game'),
                      'note': 'includes writing the decision export (observation encoding = ~3/4 of the time)'},
    'teacher_by_sims_one_process_per_physical_core': sims,
    'scaling_sims100': scaling,
    'scaling_note': 'P=1,2,4: one process per physical core (WSL cpus 8 / 8,10 / 8,10,12,14). P=8: cpus 8-15 = the same 4 '
                    'physical cores with both SMT threads busy. ms_per_search is the cleanest per-process speed (games differ in length).',
    'smt_gain_P8_vs_P4': round((8 / ms_search[8]) / (4 / ms_search[4]), 3),
    'export_overhead_teacher_sims100': m('sw_t100', 'encode_frac_of_game'),
    'decision_export_bytes_per_game_sims100_uncompressed': m('sw_t100', 'decisions_bytes_per_game'),
}
# ---------------------------------------------------------------- auto-pay
base, ap = G['sw_t100']['aggregate'], G['ap_t100']['aggregate']
rune_t = base['rune_frac_of_decide_time']['median'] * base['decide_frac_of_game']['median']
gen['autopay_sims100'] = {
    'measured': {'games_per_hour_per_process_search_all': base['games_per_hour'], 'games_per_hour_per_process_autopay': ap['games_per_hour'],
                 'speedup_games_per_hour': round(ap['games_per_hour']['median'] / base['games_per_hour']['median'], 2),
                 'secs_per_game': [m('sw_t100', 'secs_per_game'), m('ap_t100', 'secs_per_game')],
                 'searches_per_game': [m('sw_t100', 'searches_per_game'), m('ap_t100', 'searches_per_game')],
                 'turns_per_game': [m('sw_t100', 'turns_per_game'), m('ap_t100', 'turns_per_game')],
                 'ms_per_search': [round(1000 * m('sw_t100', 'secs_per_search'), 1), round(1000 * m('ap_t100', 'secs_per_search'), 1)],
                 'note': 'same 24 seeds (3 x 8), one process on cpu 14; auto-pay answers rune prompts with legal[0] (the engine order, '
                         'act0 of autopay_check.py) without calling the agent; games then diverge, and these were ~10% shorter.'},
    'derived_from_per_decision_timings': {
        'rune_share_of_multi_option_decisions': base['rune_frac_of_multi']['median'],
        'rune_share_of_search_time': base['rune_frac_of_decide_time']['median'],
        'rune_share_of_game_time': round(rune_t, 4),
        'speedup_if_game_unchanged': round(1 / (1 - rune_t), 2)},
}
# ---------------------------------------------------------------- cross-check with smoke/data
t = EX['ismcts:sims=100']
gen['crosscheck_existing_smoke_data'] = {
    'teacher_sims100_games': t['secs']['n'], 'secs_per_game_median': t['secs']['median'], 'secs_per_game_mean': t['secs']['mean'],
    'multi_option_per_game_mean': t['n_multi']['mean'], 'ms_per_search_implied': round(1000 * t['secs']['mean'] / t['n_multi']['mean'], 1),
    'total_cpu_seconds': t['total_secs'], 'random_secs_per_game_median': EX['random-player']['secs']['median'],
    'this_run_ms_per_search_P1_P4_P8': [round(1000 * ms_search[p], 1) for p in (1, 4, 8)],
    'note': 'existing data was played with run_play.sh (default P=10 processes, no pinning); its per-search time sits between '
            'this run\'s single-process and fully loaded numbers.'}

# ---------------------------------------------------------------- training
cpu = {'torch_2.7.1_linux_wsl': {f'd{r["d"]}_t{r["threads"]}_bs{r["bs"]}': {k: r[k] for k in ('steps_per_s_median', 'steps_per_s_min', 'steps_per_s_max', 'samples_per_s_median', 'data_frac_median', 'data_s_per_step_median', 'compute_s_per_step_median')}
                                   for r in C64['summary'] + C128['summary']},
       'torch_2.0.0_windows': {f'd{r["d"]}_t{r["threads"]}_bs{r["bs"]}': {k: r[k] for k in ('steps_per_s_median', 'steps_per_s_min', 'steps_per_s_max', 'samples_per_s_median', 'data_frac_median')}
                               for r in WIN['summary']},
       'infer_d64': {f't{r["threads"]}_b{r["infer_batch"]}': {'positions_per_s': r['positions_per_s_median'], 'ms_per_call': r['ms_per_call_median']} for r in C64['infer_summary']},
       'not_measured': 'd=128 at bs 1024 (killed twice by the memory guard when Windows available RAM fell below 900 MB)'}
gpu = {'device': GPU['info']['gpu'], 'torch': GPU['info']['torch'] + ' (WSL python3, already installed)',
       'train': {f'd{r["d"]}_bs{r["bs"]}': {k: r[k] for k in ('steps_per_s_median', 'steps_per_s_min', 'steps_per_s_max', 'samples_per_s_median', 'data_s_per_step_median', 'h2d_s_per_step_median', 'compute_s_per_step_median', 'data_frac_median', 'resident_samples_per_s_median')}
                 for r in GPU['summary']},
       'infer': {f'd{r["d"]}_b{r["infer_batch"]}': {'positions_per_s': r['positions_per_s_median'], 'ms_per_call': r['ms_per_call_median']} for r in GPU['infer_summary']},
       'feeding': 'CPU-side Data.batch + host->device copy are 5-9% of a step at every batch size; with batches already on the GPU '
                  '(resident) samples/s rises 15-25%. Feeding is not the bottleneck; at bs<=256 the step is kernel-launch bound.',
       'train_py_change_needed': 'train.py does not run on cuda unchanged: Net.forward builds torch.zeros(len(g), 1) on the CPU. The benchmark runs the model '
                                 'under torch.device("cuda") as the default device; a real run needs device=g.device on that line plus .to(device) for model and batches.'}
training = {'data_subset': {'decisions': PREP['subset_decisions'], 'source': '16 base_teacher shard files (smoke/data/out), packed by prep.parse_file',
                            'tokens_per_decision_mean': C64['info']['tokens_per_decision_mean'], 'actions_per_decision_mean': C64['info']['actions_per_decision_mean']},
            'prep_decisions_per_s': {'one_process': PREP['serial_dec_per_s_median'], 'pool_4': PREP['pool_dec_per_s_median']},
            'cpu': cpu, 'gpu': gpu, 'existing_runs': RUNS['aggregate']}

# ---------------------------------------------------------------- projections
S_GAME = t['n_multi']['mean']          # searches per teacher game (existing 3200 games; every multi-option decision is searched)
AP_SPEED = gen['autopay_sims100']['measured']['speedup_games_per_hour']
AP_DERIVED = gen['autopay_sims100']['derived_from_per_decision_timings']['speedup_if_game_unchanged']
ms8 = ms_search[8]                     # per-process ms/search with all 16 hardware threads busy (machine fully loaded)
ms4 = ms_search[4]


def gen_hours(n_games, procs, ms, speed=1.0):
    return n_games * S_GAME * ms / 0.997 / procs / 3600 / speed   # 0.997: search share of game time


def h(x):
    return round(x, 3)


r64_cpu = row(C64, 64, 8, 256)['samples_per_s_median']
r64_win = row(WIN, 64, 8, 256)['samples_per_s_median']
r64_gpu = row(GPU, 64, 8, 256)['samples_per_s_median']
r64_gpu1k = row(GPU, 64, 8, 1024)['samples_per_s_median']
EPOCHS = 2.3   # existing runs: 3000 x 256 samples over 336,592 train decisions = 2.28 epochs; best policy checkpoint at ~2750 steps
VAL_OVH = 1.05


def train_hours(n_dec, sps):
    return n_dec * EPOCHS / sps * VAL_OVH / 3600


prep_h = 1e6 / PREP['pool_dec_per_s_median'] / 3600
proj = {
    'teacher_10k_games_sims100': {
        '8_procs_on_4_physical_cores_SMT (as measured, P=8)': h(gen_hours(1e4, 8, ms8)),
        '8_procs_on_8_physical_cores (no SMT sharing; ms/search of P=4)': h(gen_hours(1e4, 8, ms4)),
        '14_procs (7 physical cores x 2 SMT; P=8 per-process rate)': h(gen_hours(1e4, 14, ms8)),
        '8_procs_SMT_with_autopay_measured': h(gen_hours(1e4, 8, ms8, AP_SPEED)),
        '14_procs_with_autopay_measured': h(gen_hours(1e4, 14, ms8, AP_SPEED)),
        '14_procs_with_autopay_derived': h(gen_hours(1e4, 14, ms8, AP_DERIVED))},
    'imitation_training_1M_decisions (2.3 epochs, bs256, +5% validation)': {
        'prep_pool4': h(prep_h),
        'cpu_wsl_torch2.7_8threads_d64': h(train_hours(1e6, r64_cpu)),
        'cpu_windows_torch2.0_8threads_d64': h(train_hours(1e6, r64_win)),
        'cpu_existing_launch_setup (2 threads, 3 runs in parallel; 0.291 s/step incl eval)': h(1e6 * EPOCHS / 256 * RUNS['aggregate']['secs_per_step_incl_eval']['median'] / 3600),
        'gpu_rtx3060_d64_bs256': h(train_hours(1e6, r64_gpu)),
        'gpu_rtx3060_d64_bs1024': h(train_hours(1e6, r64_gpu1k)),
        'cpu_wsl_8threads_d128': h(train_hours(1e6, row(C128, 128, 8, 256)['samples_per_s_median'])),
        'gpu_d128_bs256': h(train_hours(1e6, row(GPU, 128, 8, 256)['samples_per_s_median']))},
}
dec_it = 1000 * S_GAME
it = {}
for label, procs, ms in (('8_procs_SMT', 8, ms8), ('14_procs', 14, ms8)):
    for apl, sp in (('search_all', 1.0), ('autopay', AP_SPEED)):
        g_h = gen_hours(1000, procs, ms, sp)
        it[f'{label}_{apl}'] = {'generate_1k_games_h': h(g_h),
                                'train_cpu_h (1 epoch over the newest 1M-decision window)': h(1e6 / r64_cpu * VAL_OVH / 3600),
                                'train_gpu_h (same, bs256)': h(1e6 / r64_gpu * VAL_OVH / 3600),
                                'total_with_gpu_training_h': h(g_h + 1e6 / r64_gpu * VAL_OVH / 3600),
                                'total_with_cpu_training_h': h(g_h + 1e6 / r64_cpu * VAL_OVH / 3600)}
inf1 = inf(C64, 64, 1, 1)['ms_per_call_median']
gpu256 = inf(GPU, 64, 8, 256)['positions_per_s_median']
proj['self_play_iteration_1k_games_heuristic_leaf_ismcts_sims100'] = it
proj['self_play_iteration_note'] = (
    f'1k games = {dec_it:.0f} searched decisions. Search as today (heuristic leaf, no network in the loop). Training: one pass over a '
    '1M-decision replay window at bs 256 (CPU training runs on the cores that would otherwise play games).')
nn_evals = S_GAME * 100
proj['network_guided_search_cost_per_game_sims100'] = {
    'nn_evals_per_game': round(nn_evals),
    'cpu_1thread_batch1_secs': round(nn_evals * inf1 / 1000, 1),
    'gpu_batch1_secs': round(nn_evals * inf(GPU, 64, 8, 1)['ms_per_call_median'] / 1000, 1),
    'gpu_batched256_secs_of_gpu_time': round(nn_evals / gpu256, 2),
    'engine_search_secs_today_P8': round(S_GAME * ms8, 1),
    'note': 'one evaluation per simulation. Unbatched, the network adds 2-6 ms per simulation on top of ~0.5-0.65 ms of engine work, '
            'i.e. 4-12x slower games; GPU only helps if leaves from ~256 concurrent searches are batched. IsMctsSearch has no '
            'network evaluator or batching today, and the engine repo has no torch dependency.'}
assumptions = [
    f'Searches per teacher game = {S_GAME:.1f} (mean n_multi of the 3,200 existing sims=100 games; every multi-option decision is searched).',
    'Generation time = searches x ms/search / 0.997 (search is 99.7% of a teacher game) / processes; linear in games.',
    f'"8 cores" = 8 processes on 4 physical cores with SMT (measured P=8: {1000*ms8:.1f} ms/search). The "8 physical cores" row uses '
    f'the P=4 rate ({1000*ms4:.1f} ms/search), which needs the other 8 hardware threads idle.',
    f'"14 cores" = 14 processes at the P=8 per-process rate (all hardware threads busy, as during these measurements, when the other job '
    'filled cpus 0-7); assumes no memory-bandwidth limit beyond what P=8 already saw (each process is ~20 MB RSS).',
    f'Auto-pay speed-up: measured {AP_SPEED}x on 24 games (includes shorter games); derived {AP_DERIVED}x from rune share of time if games kept their length.',
    f'Training: {EPOCHS} epochs (as the existing runs) at the measured samples/s of the {PREP["subset_decisions"]}-decision base_teacher subset '
    f'(mean {C64["info"]["tokens_per_decision_mean"]} tokens); +5% for validation. Variant games may have longer decisions.',
    'Training memory: a 1M-decision packed set is ~1.1 GB in RAM (packed/teacher.pt is 0.52 GB for ~0.48M decisions); with ~1.5 GB '
    'available during these runs that does not fit next to the current jobs.',
    'GPU numbers from WSL python3 torch 2.7.1+cu126 (installed); train.py needs a small device fix to run there.']
res = {'machine': {'cpu': 'AMD Ryzen 7 5800X, 8 cores / 16 threads; WSL cpus 2k,2k+1 are SMT siblings, so cpus 8-15 = physical cores 4-7',
                   'ram_gb': 16, 'gpu': 'RTX 3060 12 GB'},
       'contention': {'other_job': '8 game processes pinned to WSL cpus 0-7 at ~100% for the whole session (autopay_match)',
                      'load_by_phase': phases,
                      'note': 'WSL vCPUs are not pinned to host CPUs; Hyper-V schedules them. With the other job, all 16 hardware threads '
                              'were busy during P=8 and partly during the sweep. Windows available RAM was 0.8-1.8 GB throughout.'},
       'generation': gen, 'training': training, 'projections_hours': proj, 'assumptions': assumptions}
json.dump(res, open(os.path.join(SP, 'results.json'), 'w'), indent=1)
print(json.dumps(proj, indent=1))
print(json.dumps(phases, indent=1))
print(gen['autopay_sims100'], gen['smt_gain_P8_vs_P4'], gen['crosscheck_existing_smoke_data'])
