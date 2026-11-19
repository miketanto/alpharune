# Test 3: tokens/actions decoded from rbenv equal decision_export's JSON for the
# same seed and choices. Replays recorded smoke/data games (Rengar mirror,
# base_random) with autopay=False, so every multi-option decision is external
# and matches one recorded decision line, fed its recorded choice 'pi'.
import gzip, json, sys
from common import *
from rbenv import load_vocab
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'smoke', 'data', 'out')
NG = int(sys.argv[1]) if len(sys.argv) > 1 else 5
GROUP = sys.argv[2] if len(sys.argv) > 2 else 'base_random__s0'
vocab = load_vocab(); V = {k: i for i, k in enumerate(vocab)}
vk = lambda k: V.get(k, V['other'])

games = [json.loads(l) for l in open(os.path.join(DATA, GROUP + '.games.jsonl'))]
games = [g for g in games if g['deck_a'].endswith('rengar.txt') and g['deck_b'].endswith('rengar.txt')][:NG]
want = {g['gid']: [] for g in games}
for l in gzip.open(os.path.join(DATA, GROUP + '.decisions.jsonl.gz'), 'rt'):
    if l[8:8 + 12].split('"')[0] in want:
        d = json.loads(l)
        if d['gid'] in want: want[d['gid']].append(d)

def norm_tok(t):
    if t[0] == 'c': return ['c', vk(t[1])] + t[2:]
    if t[0] == 'b': return ['b', vk(t[1])] + t[2:]
    return ['k', vk(t[1])] + t[2:10] + [t[10], list(t[11])]

def dec_tok(r):
    if r[0] == 1: return ['c', int(r[1])] + [int(x) for x in r[2:17]]
    if r[0] == 2: return ['b', int(r[1])] + [int(x) for x in r[2:15]]
    n = int(r[15]); return ['k', int(r[1])] + [int(x) for x in r[2:10]] + [int(r[10]), [int(x) for x in r[11:11 + n]]]

def norm_act(a):
    a = dict(a)
    if 'nc' in a: a['nc'] = vk(a['nc'])
    return a

def dec_act(r):
    r = [int(x) for x in r]; a = {'t': r[0]}
    if r[1] != -1: a['src'] = r[1]
    if r[2] != -1: a['abs'] = r[2]; a['ai'] = r[3]
    for name, off, cnt in (('tg', 4, 41), ('ch', 8, 42), ('mv', 12, 43), ('mu', 16, 44)):
        if r[cnt] > 0: a[name] = r[off:off + r[cnt]]
    if r[45] > 0: a['dm'] = [[r[18 + 2 * j], r[19 + 2 * j]] for j in range(r[45])]
    if r[30] >= 0: a['dst'] = r[30]
    if r[31]: a['v'] = r[32]
    if r[33] >= 0: a['ci'] = r[33]
    if r[34] > 0: a['nc'] = r[34]
    if r[35] >= 0: a['bf'] = r[35]
    if r[36] != 0: a['ps'] = r[36]
    if r[37]: a['alt'] = 1
    if r[38]: a['plc'] = 1
    if r[39]: a['dec'] = 1
    if r[0] == 16: a['first'] = r[40]
    return a

tot = bad = 0
for g in games:
    recs = want[g['gid']]
    env = RBEnv(1, g['deck_a'].split('/')[-1][:-4], g['deck_b'].split('/')[-1][:-4], seed=g['seed'], autopay=False)
    res = []
    for d in recs:
        o = env.observe(); assert o['need'][0]
        tok = [dec_tok(o['tok'][0, t]) for t in range(env.dims['T']) if o['tok_mask'][0, t]]
        act = [dec_act(o['act'][0, a]) for a in range(env.dims['A']) if o['act_mask'][0, a]]
        ok = (tok == [norm_tok(t) for t in d['tok']] and act == [norm_act(a) for a in d['act']]
              and list(o['glob'][0]) == d['g'] and int(o['psrc'][0]) == d['psrc']
              and int(o['seat'][0]) == d['seat'])
        tot += 1
        if not ok:
            bad += 1
            if bad <= 2:
                print('MISMATCH', g['gid'], d['seq'], d['pk'])
                print(' tok', tok == [norm_tok(t) for t in d['tok']], 'act', act == [norm_act(a) for a in d['act']],
                      'g', list(o['glob'][0]) == d['g'], 'psrc', int(o['psrc'][0]), d['psrc'], 'seat', int(o['seat'][0]), d['seat'])
        env.step(np.array([d['pi']], np.int32))
        res += env.results()
    assert res and res[0]['winner'] == g['winner'], (res, g['winner'])
    env.close()
print('agreement: games', len(games), 'decisions', tot, 'mismatches', bad)
assert bad == 0
