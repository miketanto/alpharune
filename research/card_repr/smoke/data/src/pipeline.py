"""Orchestration for the decision dataset (research/card_repr/smoke/data).

Subcommands (run from anywhere; paths are fixed below):
  games      write the game lists (lists/*.txt) and jobs/play_jobs.txt
  verifyjobs write jobs/verify_<group>.txt (sample of games to replay)
  swapjobs   write jobs/swap_<variant>.txt from test-split base teacher games
  manifest   build manifest.json, card_games.json, counts.json from out/
Seeds: every game has a unique seed; split = seed % 10 (0-6 train, 7 val, 8-9 test).
Selection seed for swap candidates and verify samples: SEL_SEED below.
"""
import json, os, sys, random, collections, glob, gzip

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(DATA, '..', '..', '..', '..'))
sys.path.insert(0, HERE)
from make_decks import VARIANTS, DECKS, short  # noqa

SEL_SEED = 20261118
SIMS = int(os.environ.get('DX_SIMS', '100'))
TEACHER = f'ismcts:sims={SIMS}'
N_BASE = int(os.environ.get('DX_N_BASE', '1000'))
N_VAR = int(os.environ.get('DX_N_VAR', '200'))
N_RAND_BASE = int(os.environ.get('DX_N_RAND_BASE', '300'))
N_RAND_VAR = int(os.environ.get('DX_N_RAND_VAR', '40'))
GAMES_PER_SHARD = 20
RANDOM = 'random-player'
BASE_DECKS = ['decks/rengar.txt', 'decks/yi_rebuild.txt']


def split_of(seed):
    r = seed % 10
    return 'train' if r < 7 else ('val' if r == 7 else 'test')


def variant_decks():
    out = {}
    for v, base, vname in VARIANTS:
        for dk, path in DECKS.items():
            rel = f'research/card_repr/smoke/data/decks/{dk}__{short(v)}.txt'
            if os.path.exists(os.path.join(ROOT, rel)):
                out.setdefault(v, []).append((path, rel))
    return out


def matchups_base():
    r, y = BASE_DECKS
    return [(r, r), (y, y), (r, y), (y, r)]


def matchups_variant(pairs):
    # the variant deck against each base deck, both seats
    m = []
    for base_path, vpath in pairs:
        for opp in BASE_DECKS:
            m.append((vpath, opp))
            m.append((opp, vpath))
    return m


def cmd_games():
    lists = os.path.join(DATA, 'lists'); os.makedirs(lists, exist_ok=True)
    jobs = os.path.join(DATA, 'jobs'); os.makedirs(jobs, exist_ok=True)
    groups = []  # (name, variant, lines)
    def gen(prefix, seed0, n, mus, agent, swapcand):
        lines = []
        for i in range(n):
            seed = seed0 + i
            a, b = mus[i % len(mus)]
            sc = 1 if (swapcand and split_of(seed) == 'test') else 0
            lines.append(f'{prefix}{seed} {a} {b} {agent} {agent} {seed} {sc}')
        return lines
    groups.append(('base_teacher', '', gen('BT', 1, N_BASE, matchups_base(), TEACHER, True)))
    groups.append(('base_random', '', gen('BR', 50001, N_RAND_BASE, matchups_base(), RANDOM, False)))
    vd = variant_decks()
    for vi, (v, base, vname) in enumerate(VARIANTS):
        mus = matchups_variant(vd[v])
        groups.append((f'var_teacher__{short(v)}', v, gen(f'VT{vi:02d}_', 100000 * (vi + 1), N_VAR, mus, TEACHER, False)))
        groups.append((f'var_random__{short(v)}', v, gen(f'VR{vi:02d}_', 100000 * (vi + 1) + 50001, N_RAND_VAR, mus, RANDOM, False)))
    with open(os.path.join(jobs, 'play_jobs.txt'), 'w', newline='\n') as jf:
        for name, v, lines in groups:
            open(os.path.join(lists, name + '.txt'), 'w', newline='\n').write('\n'.join(lines) + '\n')
            # ismcts shards are small (cost), random-player shards one per group
            per = GAMES_PER_SHARD if 'teacher' in name else 10 ** 6
            ns = max(1, -(-len(lines) // per))
            for s in range(ns):
                jf.write(f'{name} {v or "-"} {s} {ns}\n')
    # interleave long (teacher) jobs first for balance
    jl = open(os.path.join(jobs, 'play_jobs.txt')).read().split('\n')
    jl = [x for x in jl if x]
    jl.sort(key=lambda x: (0 if 'teacher' in x else 1))
    open(os.path.join(jobs, 'play_jobs.txt'), 'w', newline='\n').write('\n'.join(jl) + '\n')
    print('groups', len(groups), 'jobs', len(jl), 'games', sum(len(g[2]) for g in groups))


def read_games(out):
    games = []
    for f in sorted(glob.glob(os.path.join(out, '*.games.jsonl'))):
        for line in open(f):
            if line.strip():
                g = json.loads(line)
                g['_file'] = os.path.basename(f)
                games.append(g)
    return games


def group_of(fname):
    return fname.split('.')[0].rsplit('__s', 1)[0]


def cmd_verifyjobs(n_per_group=6):
    out = os.path.join(DATA, 'out')
    rng = random.Random(SEL_SEED)
    by = collections.defaultdict(list)
    for g in read_games(out):
        by[(group_of(g['_file']), g['variant'] or '-')].append(g)
    jobs = os.path.join(DATA, 'jobs')
    lst = []
    for (grp, v), gs in sorted(by.items()):
        gs = sorted(gs, key=lambda g: g['gid'])
        pick = rng.sample(gs, min(n_per_group, len(gs)))
        path = os.path.join(jobs, f'verify__{grp}.txt')
        with open(path, 'w', newline='\n') as f:
            for g in pick:
                ch = g['choices']
                checks = g['checks']
                f.write(' '.join([g['gid'], g['deck_a'], g['deck_b'], str(g['seed']), str(g['winner']),
                                  str(len(ch))] + [str(c) for c in ch] + [str(len(checks))] +
                                 [f'{c[0]} {c[1]} {c[2]}' for c in checks]) + '\n')
        lst.append(f'{grp} {v}')
    open(os.path.join(jobs, 'verify_jobs.txt'), 'w', newline='\n').write('\n'.join(lst) + '\n')
    print('verify groups', len(lst))


def cmd_swapjobs(target=360):
    """Pick swap states from test-split base teacher games, at most 2 per game
    and base card, spread over base cards; one job file per variant."""
    out = os.path.join(DATA, 'out')
    rng = random.Random(SEL_SEED + 1)
    games = [g for g in read_games(out) if group_of(g['_file']) == 'base_teacher' and split_of(g['seed']) == 'test']
    games.sort(key=lambda g: g['gid'])
    cands = collections.defaultdict(list)   # base -> [(game, cand)]
    for g in games:
        per = collections.defaultdict(list)
        for c in g['swap_cands']:
            per[c['base']].append(c)
        for base, cs in per.items():
            for c in rng.sample(cs, min(2, len(cs))):
                cands[base].append((g, c))
    var_of = collections.defaultdict(list)
    for v, base, vname in VARIANTS:
        var_of[base].append(v)
    # bases with a main-phase playable variant test; defy is reaction-only
    bases = [b for b in sorted(cands) if b in var_of]
    per_base = max(1, target // max(1, len(bases)))
    picked = []
    for b in bases:
        cs = cands[b]
        # the real pair (punch_first -> primal_strength) and 4 synthetic variants share punch_first: twice the states
        nb = 2 * per_base if b == 'punch_first' else per_base
        picked += [(b, x) for x in rng.sample(cs, min(nb, len(cs)))]
    # top up from the larger pools if some base had fewer
    rest = [(b, x) for b in bases for x in cands[b] if (b, x) not in picked]
    rng.shuffle(rest)
    while len(picked) < target and rest:
        picked.append(rest.pop())
    jobs = os.path.join(DATA, 'jobs')
    nj = collections.Counter()
    files = {}
    for k, (b, (g, c)) in enumerate(sorted(picked, key=lambda t: (t[0], t[1][0]['gid'], t[1][1]['seq']))):
        ch = g['choices'][:c['seq']]
        for v in var_of[b]:
            if v not in files:
                files[v] = open(os.path.join(jobs, f'swap__{short(v)}.txt'), 'w', newline='\n')
            f = files[v]
            f.write(' '.join([f'S{k:04d}', g['gid'], g['deck_a'], g['deck_b'], str(g['seed']), str(c['seq']), c['fp'], b,
                              str(len(ch))] + [str(x) for x in ch]) + '\n')
            nj[v] += 1
    for f in files.values():
        f.close()
    avail = {b: len(cands[b]) for b in cands}
    print('states', len(picked), 'per base', collections.Counter(b for b, _ in picked), 'available', avail)
    print('jobs per variant', dict(nj))
    json.dump({'states': len(picked), 'per_base': collections.Counter(b for b, _ in picked),
               'available_candidates': avail, 'jobs_per_variant': nj, 'sel_seed': SEL_SEED + 1},
              open(os.path.join(DATA, 'swap_selection.json'), 'w'), indent=1)


def cmd_manifest():
    out = os.path.join(DATA, 'out')
    games = read_games(out)
    manifest, card_games = [], collections.defaultdict(list)
    counts = collections.defaultdict(lambda: collections.Counter())
    dec_per_card = collections.defaultdict(lambda: collections.Counter())
    for g in games:
        grp = group_of(g['_file'])
        sp = split_of(g['seed'])
        kind = 'teacher' if 'teacher' in grp else 'random'
        cards = sorted(set(g['deck_cards_a']) | set(g['deck_cards_b']) | set(g['obs_keys']))
        manifest.append({k: g[k] for k in ('gid', 'deck_a', 'deck_b', 'agent_a', 'agent_b', 'seed', 'engine_seed',
                                           'variant', 'winner', 'term', 'turns', 'scores', 'n_seq', 'n_rec',
                                           'valid_a', 'valid_b')} |
                        {'group': grp, 'set': kind, 'split': sp, 'decisions_file': g['_file'].replace('.games.', '.decisions.'),
                         'deck_cards_a': g['deck_cards_a'], 'deck_cards_b': g['deck_cards_b'], 'obs_keys': g['obs_keys']})
        for c in cards:
            card_games[c].append(g['gid'])
        key = (kind, g['variant'] or 'base')
        counts[key]['games_' + sp] += 1
        counts[key]['decisions_' + sp] += g['n_rec']
        if g['term'].startswith('error'):
            counts[key]['errors'] += 1
        for c, n in g['obs_keys'].items():
            dec_per_card[kind][c] += n
    json.dump({'split_rule': 'seed % 10: 0-6 train, 7 val, 8-9 test', 'teacher': TEACHER, 'games': manifest},
              open(os.path.join(DATA, 'manifest.json'), 'w'))
    json.dump({k: sorted(v) for k, v in sorted(card_games.items())}, open(os.path.join(DATA, 'card_games.json'), 'w'))
    json.dump({'per_set_variant_split': {f'{k[0]}|{k[1]}': dict(v) for k, v in sorted(counts.items())},
               'token_occurrences_per_card': {k: dict(sorted(v.items())) for k, v in dec_per_card.items()}},
              open(os.path.join(DATA, 'counts.json'), 'w'), indent=1)
    print(json.dumps({f'{k[0]}|{k[1]}': dict(v) for k, v in sorted(counts.items())}, indent=0))


if __name__ == '__main__':
    {'games': cmd_games, 'verifyjobs': cmd_verifyjobs, 'swapjobs': cmd_swapjobs, 'manifest': cmd_manifest}[sys.argv[1]]()
