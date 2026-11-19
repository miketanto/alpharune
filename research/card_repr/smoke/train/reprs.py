"""Build R(card) for every condition, all mapped to the same 256-dim input.

Pipeline, identical for every condition (seeded, recorded in reprs.json):
  raw vector per key -> subtract the mean over the 51 base cards -> scale so the mean
  L2 norm over base cards is 1 -> fixed Gaussian projection P[d_in, 256] (seed 1234,
  N(0, 1)) so every output dim has unit RMS on average.
Keys without a card (pad, hidden, token:*, keyword_ability, other) get R = 0; the
network gives them a small learned 'special' embedding that is the same in every
condition.
Conditions: random (iid N(0,1) 256-dim per key, seed 777), stats (printed facts),
A, B, B2, C (if wanted), B2A (B2 and A each normalised, then concatenated),
Bshuf (B vectors permuted across all card keys by a fixed derangement, seed 4242),
id (trainable; initialised from the random condition's table; UNK row handled in model).
"""
import json, os, re, random, math
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
from prep import VOCAB

SPECIAL = ['<pad>', 'hidden', 'token:mech', 'token:sand_soldier', 'keyword_ability', 'other']
PROJ_SEED, RAND_SEED, SHUF_SEED, DOUT = 1234, 777, 4242, 256
KW = ['Action', 'Reaction', 'Accelerate', 'Ambush', 'Hidden', 'Deathknell', 'Flow', 'Ganking',
      'Assault', 'Hunt', 'Level', 'Deflect', 'Legion', 'Shield', 'Tank', 'Vision', 'Temporary',
      'Weaponmaster', 'Quick-Draw', 'Backline']
TYPES = ['unit', 'spell', 'rune', 'battlefield', 'gear', 'legend']
DOMS = ['body', 'calm', 'fury', 'order', 'mind', 'chaos']


def leading_keywords(text):
    """Keywords printed for the card itself: bracketed keywords that open a line."""
    out = {}
    for line in (text or '').split('\n'):
        m = re.match(r'^\s*((?:\[[A-Za-z\-]+(?: \d+)?\]\s*)+)', line)
        if not m:
            continue
        for kw in re.findall(r'\[([A-Za-z\-]+)(?: (\d+))?\]', m.group(1)):
            if kw[0] in KW:
                out[kw[0]] = int(kw[1]) if kw[1] else 1
    return out


def stats_table():
    cards = json.load(open(os.path.join(ROOT, 'cards.json')))
    etext = json.load(open(os.path.join(ROOT, 'engine_text.json')))
    facts = {}
    for c in cards['cards']:
        if c['printings']:
            p = c['printings'][0]
            text = None
            et = etext.get(c['key'], {})
            if p['id'] in et:
                text = et[p['id']].get('ability_text')
            text = text if text is not None else p['ability_text']
            f = dict(type=p['card_type'], sup=p['super_type'], dom=p['domains'],
                     e=p['energy_cost'] or 0, p=p['power_cost'] or 0, m=p['might'],
                     kw=leading_keywords(text))
        else:  # legends carry no printing in cards.json
            et = etext.get(c['key'], {})
            text = next(iter(et.values()))['ability_text'] if et else ''
            f = dict(type='legend', sup=None, dom=[], e=0, p=0, m=None, kw=leading_keywords(text))
        facts[c['key']] = f
    for s in cards['t2_minimal_pairs_synthetic']:
        f = json.loads(json.dumps(facts[s['base']]))
        if s['field'] == 'energy_cost':
            f['e'] = int(re.search(r'energy cost (\d+)', s['change']).group(1))
        elif s['field'] == 'power_cost':
            f['p'] = int(re.search(r'power cost (\d+)', s['change']).group(1))
        elif s['field'] == 'might':
            f['m'] = int(re.search(r'Might (\d+)', s['change']).group(1))
        elif s['field'] == 'timing':
            f['kw'].pop('Action', None)
            f['kw']['Reaction'] = 1
        facts['synthetic:' + s['id']] = f
    return facts


def stats_vec(f):
    v = [1.0 if f['type'] == t else 0.0 for t in TYPES]
    v += [1.0 if f['sup'] == 'champion' else 0.0, 1.0 if f['sup'] == 'signature' else 0.0]
    v += [1.0 if d in f['dom'] else 0.0 for d in DOMS]
    v += [float(f['e']), float(f['p']), float(f['m'] or 0), 1.0 if f['m'] is not None else 0.0]
    v += [float(f['kw'].get(k, 0)) for k in KW]
    return v


def load_vecs(path):
    return json.load(open(path))['vectors']


def normalise(raw, base_keys):
    """raw: dict key->list. Center on base cards, scale mean base norm to 1."""
    d = len(next(iter(raw.values())))
    M = torch.tensor([raw[k] for k in base_keys], dtype=torch.float64)
    mu = M.mean(0)
    out = {k: torch.tensor(v, dtype=torch.float64) - mu for k, v in raw.items()}
    scale = torch.stack([out[k] for k in base_keys]).norm(dim=1).mean()
    return {k: v / scale for k, v in out.items()}


def project(nv, seed):
    d = len(next(iter(nv.values())))
    g = torch.Generator().manual_seed(seed)
    P = torch.randn(d, DOUT, generator=g, dtype=torch.float64)
    return {k: (v @ P) for k, v in nv.items()}


def build():
    cards = json.load(open(os.path.join(ROOT, 'cards.json')))
    base_keys = [c['key'] for c in cards['cards']]
    card_keys = [k for k in VOCAB if k not in SPECIAL]
    raws = {}
    raws['A'] = load_vecs(os.path.join(ROOT, 'A_text', 'out', 'vectors.json'))
    raws['B'] = load_vecs(os.path.join(ROOT, 'B_graph', 'out', 'vectors.json'))
    raws['B2'] = load_vecs(os.path.join(ROOT, 'B2_code', 'out', 'vectors.json'))
    facts = stats_table()
    raws['stats'] = {k: stats_vec(facts[k]) for k in card_keys}
    g = torch.Generator().manual_seed(RAND_SEED)
    raws['random'] = {k: torch.randn(DOUT, generator=g).tolist() for k in card_keys}
    for c in raws:
        missing = [k for k in card_keys if k not in raws[c]]
        assert not missing, (c, missing)
    norm = {c: normalise({k: raws[c][k] for k in card_keys}, base_keys) for c in raws}
    norm['B2A'] = {k: torch.cat([norm['B2'][k], norm['A'][k]]) for k in card_keys}
    # E3 control: B vectors permuted across card keys (fixed derangement)
    rng = random.Random(SHUF_SEED)
    while True:
        perm = card_keys[:]
        rng.shuffle(perm)
        if all(a != b for a, b in zip(card_keys, perm)):
            break
    norm['Bshuf'] = {k: norm['B'][p] for k, p in zip(card_keys, perm)}
    tables, info = {}, {}
    for c, nv in norm.items():
        pv = project(nv, PROJ_SEED)
        T = torch.zeros(len(VOCAB), DOUT)
        for k in card_keys:
            T[VOCAB.index(k)] = pv[k].float()
        tables[c] = T
        # diagnostics: relative distance base vs variant after projection
        rel = {}
        for s in cards['t2_minimal_pairs_synthetic'] + [{'id': None, 'base': 'punch_first', 'v': 'primal_strength'}]:
            vk = 'synthetic:' + s['id'] if s['id'] else s['v']
            a, b = pv[s['base']], pv[vk]
            # nearest base card to the variant (excluding nothing)
            dists = sorted(((pv[k] - b).norm().item(), k) for k in base_keys)
            rel[vk] = {'dist_to_base_over_mean_card_norm': ((a - b).norm() / torch.stack([pv[k] for k in base_keys]).norm(dim=1).mean()).item(),
                       'rank_of_base_among_51': [k for _, k in dists].index(s['base']) + 1}
        info[c] = rel
    tables['id_init'] = tables['random'].clone()
    torch.save({'tables': tables, 'vocab': VOCAB, 'special': SPECIAL,
                'perm': dict(zip(card_keys, perm))}, os.path.join(HERE, 'packed', 'reprs.pt'))
    json.dump({'seeds': {'projection': PROJ_SEED, 'random': RAND_SEED, 'shuffle': SHUF_SEED},
               'stats_facts': facts, 'variant_vs_base': info,
               'bshuf_perm': dict(zip(card_keys, perm))},
              open(os.path.join(HERE, 'reprs.json'), 'w'), indent=1)
    for c in info:
        print(c, {k.split(':')[-1]: (round(v['dist_to_base_over_mean_card_norm'], 3), v['rank_of_base_among_51']) for k, v in info[c].items()})


if __name__ == '__main__':
    build()
