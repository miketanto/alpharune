"""Gate 2 (TESTPLAN.md): can the input pipeline CARRY each mechanical fact of an in-scope card
from its B2 vector?  No training of the policy; only closed-form ridge probes.

Scope: Rengar deck (legend, champion, main deck, battlefields, runes), token:mech,
token:sand_soldier, and every synthetic variant in B2_code/out/vectors.json.

Pipeline stages measured (selfplay/learner/model.py):
  raw    8192-dim B2 vector
  proj   ((x - mu_base) / scale) @ P   P ~ N(0,1) [8192,256], seed 1234   (frozen table R)
  init   W_card(R) + E_special          nn.Linear(256,64) default init, 5 seeds
  ln     norm1 LayerNorm of the token (card_emb + state emb + kind emb), init weights, 5 seeds
  trained W_card from the self-play B2 checkpoints (runs/cx_B2_s*), applied to the current R
Controls: random (model's 'random' table, randn 256, seed 777) and id (learned one-hot).

Usage: python gate2.py      -> results.json (+ stdout summary)
"""
import copy, importlib.util, json, os, re, sys, itertools
import numpy as np
import torch
torch.Tensor.numpy = lambda self, *a, **k: np.array(self.detach().cpu().tolist())   # torch built vs numpy 1.x here

HERE = os.path.dirname(os.path.abspath(__file__))
CR = os.path.normpath(os.path.join(HERE, '..'))
REPO = os.path.normpath(os.path.join(CR, '..', '..'))
sys.path.insert(0, os.path.join(CR, 'B2_code'))
_spec = importlib.util.spec_from_file_location('b2x', os.path.join(CR, 'B2_code', 'extract.py'))
X = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(X)          # main() is guarded; this only loads the encoder + edits
B = X.B
sys.path.insert(0, os.path.join(CR, 'selfplay', 'learner'))
import model as M  # noqa: E402

REAL = ['rengar_pridestalker', 'rengar_trophy_hunter',
        'noxus_hopeful', 'pit_rookie', 'sabotage', 'punch_first', 'inferna', 'grim_apothecary',
        'kinkou_initiate', 'irresistible_faefolk', 'first_mate', 'thrill_of_the_hunt', 'rampage',
        'pakaa_cub', 'ferrous_forerunner', 'kai_sa_survivor', 'nidalee_cat_form', 'darius_trifarian',
        'pyke_dockside_butcher', 'emperor_s_dais', 'seat_of_power', 'star_spring', 'body_rune',
        'fury_rune', 'token:mech', 'token:sand_soldier']
SEEDS = [0, 1, 2, 3, 4]
RAW = json.load(open(os.path.join(CR, 'B2_code', 'out', 'vectors.json')))['vectors']
SYN = sorted(k for k in RAW if k.startswith('synthetic:'))
KEYS = REAL + SYN


def load_graph(key):
    if key.startswith('synthetic:'):
        sid = key.split(':', 1)[1]
        base = sid.split('__')[0]
        bg = json.load(open(os.path.join(CR, 'B2_code', 'cards', base + '.json'), encoding='utf-8'))
        g = X.score_edit(bg, sid) if sid in X.SCORE_VARIANTS else X.syn_edit(bg, sid)
    else:
        g = json.load(open(os.path.join(CR, 'B2_code', 'cards', key.replace(':', '__') + '.json'),
                           encoding='utf-8'))
    return X.strip({k: v for k, v in g.items() if k != '_classes'})


G = {k: load_graph(k) for k in KEYS}
PATHS = {k: B.graph_paths(G[k]) for k in KEYS}
VR = {k: np.array(B.vectorize(G[k]), dtype=np.float64) for k in KEYS}
rebuild_mismatch = [k for k in KEYS if not np.allclose(VR[k], np.array(RAW[k], dtype=np.float64))]

# ---------------------------------------------------------------- pipeline stages

CJ = json.load(open(os.path.join(CR, 'cards.json'), encoding='utf-8'))
BASE_KEYS = [c['key'] for c in CJ['cards']]
Mb = np.array([RAW[k] for k in BASE_KEYS], dtype=np.float64)
MU = Mb.mean(0)
SCALE = np.linalg.norm(Mb - MU, axis=1).mean()
RAW_DIM = Mb.shape[1]   # 8192 (hash) + structured block (v2)
P = torch.randn(RAW_DIM, 256, generator=torch.Generator().manual_seed(1234), dtype=torch.float64).numpy()
VOCAB, _ = M.load_vocab()
VIDX = {k: i for i, k in enumerate(VOCAB)}
missing_in_vocab = [k for k in KEYS if k not in VIDX]


def proj(v):
    return ((v - MU) / SCALE) @ P


DIMS = {'G': 80, 'T': 8, 'F': 24, 'A': 4, 'K': 49}


NETP = {}   # seed -> the init parameters the card path uses


def net_params(seed):
    if seed in NETP:
        return NETP[seed]
    torch.manual_seed(seed)
    net = M.Net(DIMS, VOCAB, 'B2')
    tok = torch.zeros(1, 1, DIMS['F'], dtype=torch.long); tok[..., 0] = 1
    with torch.no_grad():
        fields = tok[..., 2:]
        sidx_s = 0 * net.kstride + net.field_off + (fields[..., :net.ncat].clamp(-1, 10) + 1)
        st = net.E_state(sidx_s).sum(-2) + net.W_num(M.slog(fields)).view(1, 1, M.N_KIND, M.D)[:, :, 0]
        ctx = (st + net.E_kind(torch.tensor([[1]]))).double().numpy()[0, 0]
    ln = net.enc.layers[0].norm1
    d = {'W': net.W_card.weight.detach().double().numpy(), 'Es': net.E_special.weight.detach().double().numpy(),
         'sidx': M.special_index(VOCAB).numpy(), 'ctx': ctx,
         'ln1': (ln.weight.detach().double().numpy(), ln.bias.detach().double().numpy()),
         'card_norm': getattr(net, 'card_norm', False)}
    if d['card_norm']:
        d['cln'] = (net.card_ln.weight.detach().double().numpy(), net.card_ln.bias.detach().double().numpy(),
                    float(net.card_scale.detach()))
    NETP[seed] = d
    return d


def _ln(t, g, b):
    return (t - t.mean()) / np.sqrt(t.var() + 1e-5) * g + b


def card_embs(seed, pv, key):
    """(init_lin, init, ln) for one projected vector pv of vocab key `key`, exactly as model.Net does:
    init_lin = W_card pv + E_special; init = card LayerNorm * gain (model card_norm) of that;
    ln = first pre-norm LayerNorm of the token init + state(fields=0) + kind(1)."""
    d = net_params(seed)
    e = d['W'] @ pv + d['Es'][d['sidx'][VIDX[key]]]
    ei = _ln(e, d['cln'][0], d['cln'][1]) * d['cln'][2] if d['card_norm'] else e
    return e, ei, _ln(ei + d['ctx'], *d['ln1'])


def net_stages(seed, vecfn):
    out_lin, out_init, out_ln = {}, {}, {}
    for k in KEYS:
        out_lin[k], out_init[k], out_ln[k] = card_embs(seed, vecfn(k), k)
    return out_lin, out_init, out_ln, float(np.linalg.norm(net_params(seed)['ctx']))


STAGES = {'raw': {k: VR[k] for k in KEYS}, 'proj': {k: proj(VR[k]) for k in KEYS}}
ctx_norms, card_norms, card_norms_lin = [], [], []
for s in SEEDS:
    l_, a, b, cn = net_stages(s, lambda k: STAGES['proj'][k])
    STAGES[f'initlin_s{s}'], STAGES[f'init_s{s}'], STAGES[f'ln_s{s}'] = l_, a, b
    ctx_norms.append(cn); card_norms.append(float(np.mean([np.linalg.norm(a[k]) for k in REAL])))
    card_norms_lin.append(float(np.mean([np.linalg.norm(l_[k]) for k in REAL])))
CKPTS = []
for run in ['cx_B2_s0', 'cx_B2_s1', 'cx_B2_s2', 'cx_B2_s0_ft', 'cx_B2_s1_ft', 'cx_B2_s2_ft']:
    p = os.path.join(CR, 'selfplay', 'runs', run, 'latest.pt')
    if not os.path.exists(p):
        continue
    sd = torch.load(p, map_location='cpu', weights_only=False)['model']
    W = sd["W_card.weight"].double().numpy()
    STAGES[f'trained_{run}'] = {k: W @ STAGES['proj'][k] for k in KEYS}
    CKPTS.append(run)
g7 = torch.Generator().manual_seed(777)
RND = torch.randn(len(VOCAB), 256, generator=g7).double().numpy()
STAGES['ctrl_random'] = {k: RND[VIDX[k]] for k in KEYS}
STAGES['ctrl_id'] = {k: np.eye(len(KEYS))[i] for i, k in enumerate(KEYS)}   # one-hot, learned in the net
STAGE_GROUPS = {'raw': ['raw'], 'proj': ['proj'], 'init': [f'init_s{s}' for s in SEEDS],
                'init_lin': [f'initlin_s{s}' for s in SEEDS],
                'ln': [f'ln_s{s}' for s in SEEDS], 'trained': [f'trained_{r}' for r in CKPTS],
                'ctrl_random': ['ctrl_random'], 'ctrl_id': ['ctrl_id']}


def typical(stage):
    V = STAGES[stage]
    d = [np.linalg.norm(V[a] - V[b]) for a, b in itertools.combinations(REAL, 2)]
    return float(np.median(d))


TYP = {s: typical(s) for s in STAGES}


def rel_dist(a, b, group):
    return float(np.mean([np.linalg.norm(STAGES[s][a] - STAGES[s][b]) / TYP[s] for s in STAGE_GROUPS[group]]))


def rel_dist_vec(va_fn, group):
    """va_fn(stage) -> (va, vb) for synthetic graphs not in STAGES."""
    return float(np.mean([np.linalg.norm(np.subtract(*va_fn(s))) / TYP[s] for s in STAGE_GROUPS[group]]))

# ---------------------------------------------------------------- 1. ground-truth facts

def walk_effects(g):
    for ai, a in enumerate(g['abilities']):
        for ei, e in enumerate(a.get('effects', [])):
            yield ai, a, ei, e
        for e in (a.get('cost') or {}).get('other', []) or []:
            yield ai, a, -1, e


def facts(g):
    st = g['stats']
    f = {'type': st['type'], 'supertype': st.get('supertype') or 'none',
         'energy': st.get('energy'), 'power': st.get('power'), 'might': st.get('might'),
         'keywords': sorted(k['kw'] for k in g['keywords']),
         'kw_values': {k['kw']: k['value'] for k in g['keywords'] if 'value' in k},
         'ability_kinds': [a['kind'] for a in g['abilities']],
         'timing': sorted({a['timing'] for a in g['abilities'] if 'timing' in a and a['kind'] == 'spell'}),
         'triggers': sorted({f"{a['trigger']['event']}/{a['trigger'].get('subject')}"
                             for a in g['abilities'] if 'trigger' in a}),
         'trigger_once': sorted({a['trigger']['once'] for a in g['abilities']
                                 if 'trigger' in a and 'once' in a['trigger']}),
         'verbs': [], 'amounts': [], 'target_side': [], 'target_type': [], 'conditions': [],
         'durations': [], 'chooser': [], 'dest': [], 'optional': [], 'play_modes': [],
         'play_where': []}
    for a in g['abilities']:
        for c in a.get('condition', []) + (a.get('trigger', {}) or {}).get('condition', []):
            f['conditions'].append(('ability', c['c'], bool(c.get('neg'))))
        if a['kind'] == 'play_mode':
            f['play_modes'].append(a['mode'])
            if 'where' in a:
                f['play_where'].append((a['mode'], a['where'].get('zone')))
    for ai, a, ei, e in walk_effects(g):
        v = e['verb']
        f['verbs'].append(v)
        am = e.get('amount') or {}
        if 'n' in am:
            f['amounts'].append((v, am['n']))
        t = e.get('target') or {}
        if 'side' in t and t.get('type') != 'ref':
            f['target_side'].append((v, t['side']))
        if 'type' in t:
            f['target_type'].append((v, t['type'] if isinstance(t['type'], str) else '|'.join(t['type'])))
        for c in e.get('condition', []):
            f['conditions'].append((v, c['c'], bool(c.get('neg'))))
        if 'duration' in e:
            f['durations'].append((v, e['duration']))
        ch = e.get('chooser') or t.get('chooser')
        if ch:
            f['chooser'].append((v, ch))
        if 'dest' in e:
            f['dest'].append((v, e['dest']))
        if e.get('optional'):
            f['optional'].append(v)
    return f


FACTS = {k: facts(G[k]) for k in KEYS}

# cross-check against the engine CardDef (regex over the class file's def_ block)
CT = {'Unit': 'unit', 'Spell': 'spell', 'Gear': 'gear', 'Rune': 'rune', 'Battlefield': 'battlefield',
      'Legend': 'legend'}


def carddef(key):
    cls = X.scan.CLASSES.get(key)
    if not cls:
        return None
    src = open(os.path.join(REPO, cls[0][1]), encoding='utf-8', errors='replace').read()
    i = src.rfind('CardDef d;')
    blk = src[i:] if i >= 0 else src
    out = {}
    m = re.search(r'd\.card_type\s*=\s*CardType::(\w+)', blk)
    out['type'] = CT.get(m.group(1), m.group(1).lower()) if m else None
    for fld, name in (('energy_cost', 'energy'), ('power_cost', 'power'), ('might', 'might')):
        m = re.search(r'd\.%s\s*=\s*(-?\d+)' % fld, blk)
        out[name] = int(m.group(1)) if m else None
    out['keywords'] = sorted({k.lower() for k in re.findall(r'Keyword::(\w+)', blk)})
    return out


crosscheck = {}
for k in REAL:
    if k.startswith('token:'):
        continue
    d = carddef(k)
    if not d:
        crosscheck[k] = 'no class'
        continue
    f = FACTS[k]
    bad = []
    if d['type'] and d['type'] != f['type']:
        bad.append(f"type {d['type']} vs {f['type']}")
    for n in ('energy', 'power', 'might'):
        gv, dv = f[n], d[n]
        if dv is None and gv in (None, 0):
            continue
        if dv != gv:
            bad.append(f'{n} def={dv} graph={gv}')
    dk = set(d['keywords']) - {'action'}
    gk = set(f['keywords']) - {'action'}
    if dk != gk:
        bad.append(f'keywords def={sorted(dk)} graph={sorted(gk)}')
    crosscheck[k] = bad or 'ok'

# ---------------------------------------------------------------- atoms (binary facts) + decisive paths

def atoms(k):
    f = FACTS[k]
    A = {}
    A[f"type={f['type']}"] = 'stats/type=' + f['type']
    A[f"supertype={f['supertype']}"] = ('stats/supertype=' + f['supertype']) if f['supertype'] != 'none' else None
    for kw in f['keywords']:
        A[f'kw={kw}'] = 'kw:' + kw
    for kw, v in f['kw_values'].items():
        A[f'kwval={kw}:{v}'] = f'kw:{kw}/value={v}'
    for t in f['timing']:
        A[f'timing={t}'] = f'kind:spell@{t}'
    for ab in G[k]['abilities']:
        kd = ab['kind']
        A[f'ability={kd}'] = 'kind:' + kd + ('@' + ab['timing'] if isinstance(ab.get('timing'), str) else '')
    for t in f['triggers']:
        ev, sub = t.split('/')
        A[f'trigger={t}'] = f'trigger/event={ev}'
    for o in f['trigger_once']:
        A[f'once={o}'] = f'once={o}'
    for v in set(f['verbs']):
        A[f'verb={v}'] = 'verb:' + v
    for v, s in set(f['target_side']):
        A[f'side={v}:{s}'] = f'verb:{v}/target/side={s}'
    for v, t in set(f['target_type']):
        A[f'ttype={v}:{t}'] = f'verb:{v}/target/type={t}' if '|' not in t else None
    for where, c, neg in set(f['conditions']):
        A[f"cond={c}{'(neg)' if neg else ''}"] = 'c:' + c + ('/neg' if neg else '')
    for v, d in set(f['durations']):
        A[f'dur={v}:{d}'] = f'verb:{v}/duration={d}'
    for v, c in set(f['chooser']):
        A[f'chooser={v}:{c}'] = f'verb:{v}/chooser={c}'
    for v, d in set(f['dest']):
        A[f'dest={v}:{d}'] = f'verb:{v}/dest={d}'
    for v in set(f['optional']):
        A[f'optional={v}'] = f'verb:{v}/optional'
    for m in f['play_modes']:
        A[f'playmode={m}'] = f'kind:play_mode@reaction/mode={m}'
    for m, z in f['play_where']:
        A[f'playwhere={m}:{z}'] = f'where/zone={z}'
    for v, n in set(f['amounts']):
        A[f'amount={v}:{n}'] = f'verb:{v}/amount/n={n}'
    return A


ATOMS = {k: atoms(k) for k in KEYS}
ATOM_FAMILY = lambda a: a.split('=')[0]
ALL_ATOMS = sorted({a for k in KEYS for a in ATOMS[k]})
DECISIVE = {}
for k in KEYS:
    for a, p in ATOMS[k].items():
        if p:
            DECISIVE.setdefault(a, p)

# hashed-slot collisions among all paths of the scope
slot_paths = {}
for k in KEYS:
    for p in PATHS[k]:
        slot_paths.setdefault(B.h(p)[0], set()).add(p)


def deterministic(a):
    """Is the decisive path present exactly in the cards that have the fact, and is its slot unique?"""
    p = DECISIVE.get(a)
    pos = [k for k in KEYS if a in ATOMS[k]]
    if not p:
        return {'path': None, 'present_iff_fact': False, 'slot_unique': None, 'share': 0.0}
    has = [k for k in KEYS if p in PATHS[k]]
    i, _ = B.h(p)
    others = sorted(slot_paths.get(i, set()) - {p})
    share = [abs(PATHS[k][p]) / np.linalg.norm(VR[k]) for k in pos if p in PATHS[k]]
    return {'path': p, 'present_iff_fact': sorted(has) == sorted(pos),
            'missing_in': sorted(set(pos) - set(has)), 'extra_in': sorted(set(has) - set(pos)),
            'slot_unique': not others, 'slot_shared_with': others[:4],
            'share': float(np.mean(share)) if share else 0.0}

# ---------------------------------------------------------------- 3. LOO ridge probes

def loo_ridge(Xm, y, lam_rel=1e-2):
    """Kernel ridge with intercept; exact leave-one-out predictions."""
    Xc = Xm - Xm.mean(0)
    K = Xc @ Xc.T
    n = len(y)
    lam = lam_rel * np.trace(K) / n + 1e-12
    Kc = K + lam * np.eye(n)
    # with intercept: centre y; hat matrix of centred ridge + 1/n
    H = K @ np.linalg.inv(Kc)
    H = H + (np.eye(n) - H) @ np.ones((n, n)) / n
    yhat = H @ y
    loo = y - (y - yhat) / (1 - np.diag(H))
    return loo


def probe_binary(stage, a, keys=KEYS):
    y = np.array([1.0 if a in ATOMS[k] else -1.0 for k in keys])
    Xm = np.array([STAGES[stage][k] for k in keys])
    pred = loo_ridge(Xm, y)
    pos, neg = y > 0, y < 0
    bacc = 0.5 * ((pred[pos] > 0).mean() + (pred[neg] <= 0).mean())
    return float(bacc)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def probe_numeric(stage, getter):
    ks = [k for k in KEYS if getter(k) is not None]
    y = np.array([float(getter(k)) for k in ks])
    Xm = np.array([STAGES[stage][k] for k in ks])
    pred = loo_ridge(Xm, y)
    r2 = 1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return {'n': len(ks), 'loo_r2': float(r2), 'loo_spearman': spearman(pred, y),
            'loo_mae': float(np.abs(pred - y).mean())}


def numeric_getters():
    gs = {'energy': lambda k: FACTS[k]['energy'], 'power': lambda k: FACTS[k]['power'],
          'might': lambda k: FACTS[k]['might'] if FACTS[k]['type'] == 'unit' else None}
    for kw in ('assault',):
        gs[f'kw_value:{kw}'] = (lambda kw: lambda k: FACTS[k]['kw_values'].get(kw, 0)
                                if FACTS[k]['type'] == 'unit' else None)(kw)
    gs['amount:GiveMight'] = lambda k: next((n for v, n in FACTS[k]['amounts'] if v == 'GiveMight'), None)
    gs['amount:any_first'] = lambda k: FACTS[k]['amounts'][0][1] if FACTS[k]['amounts'] else None
    return gs


NUM_PATH = {'energy': 'stats/energy', 'power': 'stats/power', 'might': 'stats/might',
            'kw_value:assault': 'kw:assault/value', 'amount:GiveMight': 'verb:GiveMight/amount/n',
            'amount:Score': 'verb:Score/amount/n', 'amount:any_first': None}


def numeric_deterministic(name, getter):
    p = NUM_PATH.get(name)
    if not p:
        return None
    ks = [k for k in KEYS if getter(k) is not None]
    vals, ys = [], []
    i, s = B.h(p)
    for k in ks:
        vals.append(VR[k][i] * s)
        ys.append(float(getter(k)))
    vals, ys = np.array(vals), np.array(ys)
    others = sorted(slot_paths.get(i, set()) - {p})
    exact_linear = bool(np.allclose(vals, ys))
    return {'path': p, 'slot_unique': not others, 'slot_shared_with': others[:4],
            'exactly_linear': exact_linear, 'monotonic_spearman': spearman(vals, ys),
            'encoding': 'raw value' if p.startswith('stats/') else 'log2(1+|n|) (signed), summed over effects'}

# ---------------------------------------------------------------- 4. C2 structural edits

def first_ab(g, kinds=('spell', 'triggered', 'static', 'activated')):
    return [a for a in g['abilities'] if a['kind'] in kinds and a.get('effects')]


def edits():
    out = []

    def add(name, family, base, fn):
        g = copy.deepcopy(G[base])
        try:
            ok = fn(g)
        except (KeyError, IndexError, StopIteration) as e:
            ok = False
        if ok is not False:
            out.append((name, family, base, g))

    # condition gating: Rampage's paid branch moved from GiveMight to the damage
    def ramp_gate(g):
        sp = [a for a in g['abilities'] if a['kind'] == 'spell'][0]
        gm = [e for e in sp['effects'] if e['verb'] == 'GiveMight'][0]
        dd = [e for e in sp['effects'] if e['verb'] == 'DealDamage'][0]
        dd['condition'] = gm.pop('condition')
    add('rampage: paid-condition gates the damage instead of the +2', 'condition_gating', 'rampage', ramp_gate)

    def ramp_ungate(g):
        sp = [a for a in g['abilities'] if a['kind'] == 'spell'][0]
        [e for e in sp['effects'] if e['verb'] == 'GiveMight'][0].pop('condition')
    add('rampage: +2 unconditional (paid branch removed)', 'condition_gating', 'rampage', ramp_ungate)

    # condition at trigger (intervening if) vs at effect: Star Spring
    def ss_cond(g):
        a = g['abilities'][0]
        c = a['trigger'].pop('condition')
        a['effects'][0]['condition'] = c
    add('star_spring: not-token condition on the effect instead of the trigger', 'condition_gating',
        'star_spring', ss_cond)

    # binding: which chosen object each effect refers to (Rampage A/B swapped on the selectors)
    def ramp_bind(g):
        sp = [a for a in g['abilities'] if a['kind'] == 'spell'][0]
        ct = [e for e in sp['effects'] if e['verb'] == 'ChooseTarget']
        ct[0]['target']['bind'], ct[1]['target']['bind'] = 'B', 'A'
    add('rampage: bind letters swapped (the +2 and the self-damage now hit the ENEMY unit)', 'binding',
        'rampage', ramp_bind)

    # effect order, every multi-effect ability
    for k in REAL:
        for ai, a in enumerate(G[k]['abilities']):
            effs = a.get('effects', [])
            if len(effs) >= 2 and effs[0] != effs[-1]:
                def rev(g, ai=ai):
                    g['abilities'][ai]['effects'] = g['abilities'][ai]['effects'][::-1]
                add(f'{k}: effect order reversed in ability {ai} '
                    f"({' -> '.join(e['verb'] for e in effs)})", 'effect_order', k, rev)
                break
    # ability order (multi-ability cards)
    for k in ['inferna', 'rampage', 'pyke_dockside_butcher', 'kai_sa_survivor']:
        if len(G[k]['abilities']) >= 2:
            add(f'{k}: ability order reversed', 'ability_order', k,
                lambda g: g.__setitem__('abilities', g['abilities'][::-1]))
    # duration
    for k in REAL:
        for ai, a, ei, e in walk_effects(G[k]):
            if ei >= 0 and e.get('duration') == 'turn':
                def dur(g, ai=ai, ei=ei):
                    g['abilities'][ai]['effects'][ei].pop('duration')
                add(f"{k}: '{e['verb']}' this turn -> permanent", 'duration', k, dur)
                break
    # chooser
    def thrill_ch(g):
        e = [e for e in g['abilities'][0]['effects'] if e['verb'] == 'PlayIgnoringCost'][0]
        e['chooser'] = 'controller'
    add('thrill_of_the_hunt: location chosen by controller instead of owner', 'chooser',
        'thrill_of_the_hunt', thrill_ch)
    for k in ['rengar_pridestalker', 'punch_first', 'first_mate']:
        def ch_opp(g):
            for ai, a, ei, e in walk_effects(g):
                if ei >= 0 and isinstance(e.get('target'), dict) and e['target'].get('type') == 'unit':
                    e['target']['chooser'] = 'opponent'
                    return True
            return False
        add(f'{k}: target chosen by the opponent', 'chooser', k, ch_opp)
    # trigger vs static vs activated
    for k in ['rengar_pridestalker', 'pit_rookie', 'first_mate']:
        def to_static(g):
            a = [a for a in g['abilities'] if a['kind'] == 'triggered'][0]
            a['kind'] = 'static'; a.pop('trigger')
        def to_act(g):
            a = [a for a in g['abilities'] if a['kind'] == 'triggered'][0]
            a['kind'] = 'activated'; a.pop('trigger'); a['cost'] = {'exhaust': True}
        add(f'{k}: triggered -> static', 'ability_kind', k, to_static)
        add(f'{k}: triggered -> activated (exhaust)', 'ability_kind', k, to_act)
    def inf_trig(g):
        a = [a for a in g['abilities'] if a['kind'] == 'static'][0]
        a['kind'] = 'triggered'; a['trigger'] = {'event': 'attack_or_defend', 'subject': 'self'}
    add('inferna: static Assault -> triggered on attack', 'ability_kind', 'inferna', inf_trig)
    # trigger event / subject
    def ev(g):
        [a for a in g['abilities'] if a['kind'] == 'triggered'][0]['trigger']['subject'] = 'self'
    add('rengar_pridestalker: trigger subject you -> self', 'trigger', 'rengar_pridestalker', ev)
    # target side
    for k in REAL:
        for ai, a, ei, e in walk_effects(G[k]):
            t = e.get('target') or {}
            if ei >= 0 and t.get('side') in ('friendly', 'enemy', 'any') and t.get('type') not in ('ref', 'self'):
                new = {'friendly': 'enemy', 'enemy': 'friendly', 'any': 'enemy'}[t['side']]
                def side(g, ai=ai, ei=ei, new=new):
                    g['abilities'][ai]['effects'][ei]['target']['side'] = new
                add(f"{k}: {e['verb']} target side {t['side']} -> {new}", 'target_side', k, side)
                break
    # condition polarity
    def pol(g):
        e = [e for e in g['abilities'][0]['effects'] if e['verb'] == 'PlayIgnoringCost'][0]
        e['condition'][0]['neg'] = False
    add('thrill_of_the_hunt: replay only if the unit IS a token (neg flipped)', 'condition_polarity',
        'thrill_of_the_hunt', pol)
    # optional
    def opt(g):
        g['abilities'][0]['effects'][0].pop('optional')
    add('star_spring: "may move" -> must move', 'optional', 'star_spring', opt)
    # amount (a numeric control for scale)
    def amt(g):
        [a for a in g['abilities'] if a['kind'] == 'triggered'][0]['effects'][0]['amount']['n'] = 2
    add('rengar_pridestalker: +1 -> +2', 'amount', 'rengar_pridestalker', amt)

    # v2 additions: keyword value and Might as single edits (numbers)
    def kwv(g):
        k = [k for k in g['keywords'] if k['kw'] == 'assault'][0]
        k['value'] = 3
        for a in g['abilities']:
            for e in a.get('effects', []):
                if e['verb'] == 'GiveMight' and (e.get('amount') or {}).get('n') == 2:
                    e['amount']['n'] = 3
    add('inferna: [Assault 2] -> [Assault 3]', 'kw_value', 'inferna', kwv)

    def mt(g):
        g['stats']['might'] += 1
    add('kai_sa_survivor: Might +1', 'might', 'kai_sa_survivor', mt)
    return out


def structural():
    res = []
    for name, fam, base, g in edits():
        vb = VR[base]
        ve = np.array(B.vectorize(g), dtype=np.float64)
        same_graph = json.dumps(g, sort_keys=True) == json.dumps(G[base], sort_keys=True)
        nd = int((np.abs(vb - ve) > 1e-9).sum())
        pb, pe = proj(vb), proj(ve)
        r = {'edit': name, 'family': fam, 'base': base, 'graph_changed': not same_graph,
             'dims_differ': nd, 'raw_rel': float(np.linalg.norm(vb - ve) / TYP['raw']),
             'proj_rel': float(np.linalg.norm(pb - pe) / TYP['proj']),
             'share_of_norm': float(np.linalg.norm(vb - ve) / np.linalg.norm(vb))}
        # init/ln/trained through the same linear maps
        inits, lns, lins = [], [], []
        for s in SEEDS:
            lb, ib, nb = card_embs(s, pb, base)
            le, ie, ne = card_embs(s, pe, base)
            lins.append(np.linalg.norm(lb - le) / TYP[f'initlin_s{s}'])
            inits.append(np.linalg.norm(ib - ie) / TYP[f'init_s{s}'])
            lns.append(np.linalg.norm(nb - ne) / TYP[f'ln_s{s}'])
        r['init_lin_rel'] = float(np.mean(lins))
        r['init_rel'] = float(np.mean(inits)); r['ln_rel'] = float(np.mean(lns))
        tr = []
        for run in CKPTS:
            sd = torch.load(os.path.join(CR, 'selfplay', 'runs', run, 'latest.pt'), map_location='cpu',
                            weights_only=False)['model']
            W = sd["W_card.weight"].double().numpy()
            tr.append(np.linalg.norm(W @ (pb - pe)) / TYP[f'trained_{run}'])
        r['trained_rel'] = float(np.mean(tr)) if tr else None
        r['verdict'] = verdict_rel(r['init_rel'], nd)
        res.append(r)
    return res


def verdict_rel(rel, nd=None):
    if nd == 0 or rel < 1e-9:
        return 'FAIL (identical)'
    if rel < 0.05:
        return 'FAIL (crushed)'
    if rel < 0.25:
        return 'WEAK'
    return 'PASS'

# ---------------------------------------------------------------- 2. C1 pairwise distinguishability

def pairwise():
    out = []
    for a, b in itertools.combinations(KEYS, 2):
        same = json.dumps(G[a], sort_keys=True) == json.dumps(G[b], sort_keys=True)
        nd = int((np.abs(VR[a] - VR[b]) > 1e-9).sum())
        r = {'a': a, 'b': b, 'same_mechanics': same, 'dims_differ': nd}
        for grp in ('raw', 'proj', 'init_lin', 'init', 'ln', 'trained', 'ctrl_random'):
            if STAGE_GROUPS[grp]:
                r[grp] = rel_dist(a, b, grp)
        r['share_of_norm'] = float(np.linalg.norm(VR[a] - VR[b]) /
                                   max(np.linalg.norm(VR[a]), np.linalg.norm(VR[b])))
        r['pass'] = (r['init'] < 1e-9) if same else (r['init'] >= 0.25)
        r['verdict'] = 'PASS (identical)' if same and r['init'] < 1e-9 else \
            ('FAIL (same mechanics, different vector)' if same else verdict_rel(r['init'], nd))
        out.append(r)
    return out


def main():
    res = {'scope': KEYS, 'n_real': len(REAL), 'n_synthetic': len(SYN),
           'rebuild_mismatch_vs_vectors_json': rebuild_mismatch, 'missing_in_vocab': missing_in_vocab,
           'typical_distance': TYP, 'checkpoints': CKPTS,
           'raw_dim': int(RAW_DIM),
           'init_norms': {'card_emb_mean': float(np.mean(card_norms)), 'card_emb_lin_mean': float(np.mean(card_norms_lin)),
                          'token_context_mean': float(np.mean(ctx_norms))},
           'token_share_at_init': {
               'typical_card_distance_over_token_norm': float(np.mean([TYP[f'init_s{s}'] for s in SEEDS]) / np.mean(ctx_norms)),
               'ln_typical_distance_over_ln_norm': float(np.mean([TYP[f'ln_s{s}'] for s in SEEDS]) / 8.0)},
           'facts': {k: FACTS[k] for k in KEYS}, 'carddef_crosscheck': crosscheck}
    # C1
    pw = pairwise()
    res['c1_pairs_total'] = len(pw)
    res['c1_same_mechanics_pairs'] = [p for p in pw if p['same_mechanics']]
    diff = [p for p in pw if not p['same_mechanics']]
    res['c1_summary'] = {
        'pairs_differing_mechanics': len(diff),
        'identical_vectors_despite_different_mechanics': [(p['a'], p['b']) for p in diff if p['dims_differ'] == 0],
        'counts_by_verdict': {v: sum(1 for p in diff if p['verdict'] == v) for v in
                              ('PASS', 'WEAK', 'FAIL (crushed)', 'FAIL (identical)')},
        'ratio_init_over_raw_median': float(np.median([p['init'] / p['raw'] for p in diff if p['raw'] > 0])),
        'ratio_trained_over_raw_median': float(np.median([p['trained'] / p['raw'] for p in diff if p['raw'] > 0]))
        if CKPTS else None,
        'random_control_min_rel': float(min(p['ctrl_random'] for p in pw))}
    res['c1_at_risk'] = sorted(diff, key=lambda p: p['init'])[:25]
    known = [p for p in pw if {p['a'], p['b']} == {'synthetic:punch_first__ctrl_scores',
                                                     'synthetic:punch_first__opp_of_ctrl_scores'}]
    res['c1_known_case'] = known[0] if known else None
    # C1/C3 atoms
    atom_rows = []
    for a in ALL_ATOMS:
        npos = sum(1 for k in KEYS if a in ATOMS[k])
        row = {'atom': a, 'family': ATOM_FAMILY(a), 'n_pos': npos, 'det': deterministic(a)}
        if 2 <= npos <= len(KEYS) - 2:
            for grp in ('raw', 'proj', 'init', 'trained', 'ctrl_random', 'ctrl_id'):
                if STAGE_GROUPS[grp]:
                    row[f'loo_bacc_{grp}'] = float(np.mean([probe_binary(s, a) for s in STAGE_GROUPS[grp]]))
        atom_rows.append(row)
    res['atoms'] = atom_rows
    fam = {}
    for r in atom_rows:
        f = fam.setdefault(r['family'], {'n_atoms': 0, 'det_present_iff': 0, 'det_slot_unique': 0,
                                         'share': [], 'loo_init': [], 'loo_raw': [], 'loo_trained': [],
                                         'loo_random': [], 'loo_id': [], 'n_loo': 0, 'fails': []})
        f['n_atoms'] += 1
        f['det_present_iff'] += int(r['det']['present_iff_fact'])
        f['det_slot_unique'] += int(bool(r['det']['slot_unique']))
        f['share'].append(r['det']['share'])
        if not r['det']['present_iff_fact']:
            f['fails'].append(r['atom'])
        if 'loo_bacc_init' in r:
            f['n_loo'] += 1
            f['loo_init'].append(r['loo_bacc_init']); f['loo_raw'].append(r['loo_bacc_raw'])
            if 'loo_bacc_trained' in r: f['loo_trained'].append(r['loo_bacc_trained'])
            f['loo_random'].append(r['loo_bacc_ctrl_random']); f['loo_id'].append(r['loo_bacc_ctrl_id'])
    for f in fam.values():
        for k in ('share', 'loo_init', 'loo_raw', 'loo_trained', 'loo_random', 'loo_id'):
            f[k] = float(np.mean(f[k])) if f[k] else None
    res['atom_families'] = fam
    # C3 numbers
    nums = {}
    for name, gt in numeric_getters().items():
        nums[name] = {'det': numeric_deterministic(name, gt)}
        for grp in ('raw', 'proj', 'init', 'trained', 'ctrl_random', 'ctrl_id'):
            if STAGE_GROUPS[grp]:
                rr = [probe_numeric(s, gt) for s in STAGE_GROUPS[grp]]
                nums[name][grp] = {k: float(np.mean([x[k] for x in rr])) for k in rr[0]}
    res['c3_numbers'] = nums
    # C2
    res['c2_structure'] = structural()
    # C4 (code check; rbenv check in rbenv_fields_200.txt)
    c4 = {}
    for k in REAL:
        cls = X.scan.CLASSES.get(k)
        if not cls:
            continue
        src = open(os.path.join(REPO, cls[0][1]), encoding='utf-8', errors='replace').read()
        hits = sorted(set(re.findall(r'\b(pickMode|pickValueAs|pickValue|chooseMode|chosen_value)\b', src)))
        if hits:
            c4[k] = {'calls': hits, 'prompt_kinds': sorted(set(re.findall(r'PromptKind::(\w+)', src))),
                     'graph_modes': FACTS[k]['play_modes'], 'graph_dest': FACTS[k]['dest']}
    res['c4_modal_classes'] = c4
    json.dump(res, open(os.path.join(HERE, 'results.json'), 'w'), indent=1, default=str)
    return res


if __name__ == '__main__':
    r = main()
    print('rebuild mismatch', r['rebuild_mismatch_vs_vectors_json'], 'missing vocab', r['missing_in_vocab'])
    print('crosscheck', {k: v for k, v in r['carddef_crosscheck'].items() if v != 'ok'})
    print('init norms', r['init_norms'], r['token_share_at_init'])
    print('C1', json.dumps(r['c1_summary'], indent=1))
    print('known', {k: r['c1_known_case'][k] for k in ('dims_differ', 'raw', 'proj', 'init', 'ln', 'trained',
                                                        'share_of_norm')})
    print('same-mech pairs', [(p['a'], p['b'], p['verdict']) for p in r['c1_same_mechanics_pairs']])
    for p in r['c1_at_risk'][:15]:
        print('  risk %-45s %-45s nd=%4d raw=%.3f init=%.3f trained=%s' % (
            p['a'], p['b'], p['dims_differ'], p['raw'], p['init'],
            '%.3f' % p['trained'] if 'trained' in p else '-'))
    for f, v in r['atom_families'].items():
        print('  fam %-12s n=%2d det=%2d uniq=%2d share=%.3f loo(n=%d) raw=%s init=%s tr=%s rnd=%s id=%s fails=%s' % (
            f, v['n_atoms'], v['det_present_iff'], v['det_slot_unique'], v['share'], v['n_loo'],
            *['%.2f' % v[k] if v[k] is not None else '-' for k in ('loo_raw', 'loo_init', 'loo_trained',
                                                                  'loo_random', 'loo_id')], v['fails'][:4]))
    for n, v in r['c3_numbers'].items():
        print('  num %-18s det=%s' % (n, v['det'] and {k: v['det'][k] for k in ('slot_unique', 'exactly_linear',
                                                                               'monotonic_spearman')}),
              {g: '%.2f/%.2f' % (v[g]['loo_r2'], v[g]['loo_spearman']) for g in v if g != 'det'})
    for s in r['c2_structure']:
        print('  c2 %-16s nd=%3d raw=%.3f init=%.3f ln=%.3f tr=%s %-18s %s' % (
            s['family'], s['dims_differ'], s['raw_rel'], s['init_rel'], s['ln_rel'],
            '%.3f' % s['trained_rel'] if s['trained_rel'] is not None else '-', s['verdict'], s['edit']))
    print('C4', r['c4_modal_classes'])
