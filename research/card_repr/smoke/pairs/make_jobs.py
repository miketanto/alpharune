"""Swap states for the Rengar mirror: test-split base_teacher games with BOTH decks
rengar.txt. Keeps the states the earlier selection already picked (jobs/swap__*.txt
filtered to R-vs-R), then tops up from the same game pool with the same rule
(at most 2 candidates per game and base card), seed 20261119, to N per base."""
import json, glob, os, random, collections, sys
H = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(H, '..', 'data')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 48
R = 'decks/rengar.txt'
old = {'punch_first': 'punch_first__amount4', 'pit_rookie': 'pit_rookie__might3', 'sabotage': 'sabotage__unit'}
games = {}
for f in sorted(glob.glob(os.path.join(DATA, 'out', 'base_teacher*.games.jsonl'))):
    for l in open(f):
        g = json.loads(l)
        if g['deck_a'] == R and g['deck_b'] == R and g['seed'] % 10 >= 8: games[g['gid']] = g
rng = random.Random(20261119)
sel = {}
for base, v in old.items():
    keep = []
    for l in open(os.path.join(DATA, 'jobs', f'swap__{v}.txt')):
        t = l.split()
        if t[2] == R and t[3] == R: keep.append((t[1], int(t[5])))
    pool = []
    for gid in sorted(games):
        cs = [c for c in games[gid]['swap_cands'] if c['base'] == base]
        pool += [(gid, c['seq']) for c in rng.sample(cs, min(2, len(cs)))]
    extra = [x for x in pool if x not in keep]; rng.shuffle(extra)
    sel[base] = keep + extra[:max(0, N - len(keep))]
    with open(os.path.join(H, 'jobs', f'{base}.txt'), 'w', newline='\n') as f:
        for k, (gid, seq) in enumerate(sel[base]):
            g = games[gid]; c = next(c for c in g['swap_cands'] if c['seq'] == seq and c['base'] == base)
            ch = g['choices'][:seq]
            f.write(' '.join([f'P{base[:2]}{k:03d}', gid, R, R, str(g['seed']), str(seq), c['fp'], base, str(len(ch))] + [str(x) for x in ch]) + '\n')
    print(base, 'kept', len(keep), 'pool', len(pool), 'total', len(sel[base]))
json.dump({b: [list(x) for x in v] for b, v in sel.items()}, open(os.path.join(H, 'jobs', 'selection.json'), 'w'))
