# Test 3 (schema v2): tokens/actions decoded from rbenv equal decision_export's v2 JSON for the
# same seed and choices. Replays recorded smoke/data games (Rengar mirror,
# base_random) with autopay=False, so every multi-option decision is external
# and matches one recorded decision line, fed its recorded choice 'pi'.
import gzip, json, sys
from common import *
from rbenv import load_vocab
# v2 data: research/card_repr/audit_gate1/agree/v2.* (decision_export play, schema v2; the
# smoke/data/out files are schema v1 and no longer comparable).
# Schema v4 (gate 3 completion): the same 18 games re-exported by the rebuilt decision_export into
# research/card_repr/gate3/agree/v4.* (act 49..55 "oc", 56..60 "mg", TriggerOrder src/ci, MoveJoin dst/dec).
# The v2 files no longer match (they lack those fields); pass 'v2' and the audit_gate1 dir to compare them anyway.
NG = int(sys.argv[1]) if len(sys.argv) > 1 else 100
GROUP = sys.argv[2] if len(sys.argv) > 2 else 'v4'
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
                    *(('audit_gate1', 'agree') if GROUP == 'v2' else ('gate3', 'agree')))
vocab = load_vocab(); V = {k: i for i, k in enumerate(vocab)}
vk = lambda k: V.get(k, V['other'])

games = [json.loads(l) for l in open(os.path.join(DATA, GROUP + '.games.jsonl'))]
games = [g for g in games if g['deck_a'].endswith('rengar.txt') and g['deck_b'].endswith('rengar.txt') and not g['term'].startswith('error')][:NG]
want = {g['gid']: [] for g in games}
for l in open(os.path.join(DATA, GROUP + '.decisions.jsonl')):
    if l[8:8 + 12].split('"')[0] in want:
        d = json.loads(l)
        if d['gid'] in want: want[d['gid']].append(d)

def norm_tok(t):
    if t[0] in ('c', 'b', 'p'): return [t[0], vk(t[1])] + t[2:]
    return ['k', vk(t[1])] + t[2:10] + [t[10], list(t[11])] + t[12:15]

def dec_tok(r):
    if r[0] == 1: return ['c', int(r[1])] + [int(x) for x in r[2:24]]
    if r[0] == 2: return ['b', int(r[1])] + [int(x) for x in r[2:17]]
    if r[0] == 4: return ['p', int(r[1])] + [int(x) for x in r[2:5]]
    n = int(r[15]); return ['k', int(r[1])] + [int(x) for x in r[2:10]] + [int(r[10]), [int(x) for x in r[11:11 + n]]] + [int(x) for x in r[16:19]]

def norm_act(a):
    a = dict(a)
    if 'nc' in a: a['nc'] = vk(a['nc'])
    if 'mg' in a: a['mg'] = a['mg'][:4]   # rbenv keeps the first 4 (kMaxList)
    return a

def dec_act(r):
    r = [int(x) for x in r]; a = {'t': r[0]}
    if r[1] != -1: a['src'] = r[1]
    if r[2] != -1: a['abs'] = r[2]; a['ai'] = r[3]
    for name, off, cnt in (('tg', 4, 41), ('ch', 8, 42), ('mv', 12, 43), ('mu', 16, 44)):
        if r[cnt] > 0: a[name] = r[off:off + r[cnt]]
    if r[45] > 0: a['dm'] = [[r[18 + 2 * j], r[19 + 2 * j]] for j in range(r[45])]
    if r[30] >= 0: a['dst'] = r[30]
    if r[30] >= 1: a['dc'] = r[46]; a['atk'] = r[47]; a['con'] = r[48]
    if r[31]: a['v'] = r[32]
    if r[33] >= 0: a['ci'] = r[33]
    if r[34] > 0: a['nc'] = r[34]
    if r[35] >= 0: a['bf'] = r[35]
    if r[36] != 0: a['ps'] = r[36]
    if r[37]: a['alt'] = 1
    if r[38]: a['plc'] = 1
    if r[39]: a['dec'] = 1
    if r[0] == 16: a['first'] = r[40]
    if len(r) > 49 and r[49] >= 0: a['oc'] = r[49:56]          # v3 optional-cost consequences
    if len(r) > 60 and r[60] > 0: a['mg'] = r[56:56 + min(4, r[60])]   # v4 units already moving (MoveJoin)
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
