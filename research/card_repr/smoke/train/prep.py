"""Pack the decision dataset (smoke/data) into compact torch tensors.

Reads every TEACHER decision file (base_teacher, var_teacher__*) and the swap-test
records. Writes packed/teacher.pt and packed/swap.pt (int16 / float32 ragged arrays,
built with array.array + torch.frombuffer: torch 2.0 cannot talk to numpy 2).

Token row (TOKW = 19 int16):  kind(1 c, 2 b, 3 k, 0 pad), key index, 15 fields, is_psrc, n_refs
  c fields: controller owner zone loc exhausted stunned damage might buffs temp_might
            is_token facedown combat n_attachments attached(0/1)
  b fields: slot controller contested contested_by showdown combat attacker staged
            facedown_self facedown_opp scored_self scored_opp is_token 0 0
  k fields: controller pos_from_top resolving is_spell is_ability is_activated finalized
            is_equip has_source n_chosen 0 0 0 0 0
Action row (ACTW = 14 int16): t src abs dst v ai ps alt plc dec first bf has_nc dmg_sum
  (src/abs = token index or -1, dst = -1 none else code), refs ragged (tg ch mv mu dm ci).
"""
import array, gzip, json, glob, os, sys, time
from multiprocessing import Pool
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
OUT = os.path.join(HERE, 'packed')
TOKW, ACTW = 19, 14


def clip16(x):
    try:
        x = int(x)
    except Exception:
        x = 1 if x else 0
    return max(-32000, min(32000, x))


def build_vocab():
    cards = json.load(open(os.path.join(HERE, '..', '..', 'cards.json')))
    keys = [c['key'] for c in cards['cards']]
    keys += ['synthetic:' + s['id'] for s in cards['t2_minimal_pairs_synthetic']]
    special = ['hidden', 'token:mech', 'token:sand_soldier', 'keyword_ability', 'other']
    return ['<pad>'] + keys + special


VOCAB = build_vocab()
KIDX = {k: i for i, k in enumerate(VOCAB)}


def key_index(k):
    if k in KIDX:
        return KIDX[k]
    return KIDX['other']


def encode(d, tok_a, act_a, ref_a, f_a, rec):
    """Append one decision to the arrays; rec gets the per-decision scalars."""
    toks = d['tok']
    nt = len(toks)
    psrc = d.get('psrc', -1)
    nref = [0] * nt
    acts = d['act']
    for a in acts:
        for f in ('src', 'abs'):
            if f in a and 0 <= a[f] < nt:
                nref[a[f]] += 1
    for i, t in enumerate(toks):
        kind = t[0]
        if kind == 'c':
            f = list(t[2:16]) + [1 if t[16] >= 0 else 0]
            kc = 1
        elif kind == 'b':
            f = list(t[2:15]) + [0, 0]
            kc = 2
        else:
            src = t[10] if len(t) > 10 else -1
            ch = t[11] if len(t) > 11 else []
            f = list(t[2:10]) + [1 if (isinstance(src, int) and src >= 0) else 0,
                                 len(ch) if isinstance(ch, list) else 0, 0, 0, 0, 0, 0]
            kc = 3
        row = [kc, key_index(t[1])] + [clip16(x) for x in f[:15]]
        row += [0] * (17 - len(row))
        row += [1 if i == psrc else 0, nref[i]]
        tok_a.extend(row)
    for a in acts:
        refs = []
        for f in ('tg', 'ch', 'mv', 'mu'):
            for x in a.get(f, []) or []:
                if isinstance(x, int) and 0 <= x < nt:
                    refs.append(x)
        dms = 0
        for p in a.get('dm', []) or []:
            if 0 <= p[0] < nt:
                refs.append(p[0])
            dms += p[1]
        if 'ci' in a and isinstance(a['ci'], int) and 0 <= a['ci'] < nt:
            refs.append(a['ci'])
        row = [clip16(a.get('t', 0)), clip16(a.get('src', -1)), clip16(a.get('abs', -1)),
               clip16(a.get('dst', -1)), clip16(a.get('v', -1)), clip16(a.get('ai', -1)),
               clip16(a.get('ps', 0)), clip16(a.get('alt', 0)), clip16(a.get('plc', 0)),
               clip16(a.get('dec', 0)), clip16(a.get('first', 0)), clip16(a.get('bf', -1)),
               1 if 'nc' in a else 0, clip16(dms)]
        act_a.extend(row)
        refs = refs[:16]
        ref_a.extend(refs)
        rec['nref'].append(len(refs))
    na = len(acts)
    vis = d.get('vis')
    q = d.get('q')
    searched = 1 if (d.get('searched') and vis) else 0
    if searched:
        s = float(sum(vis))
        f_a.extend([v / s if s > 0 else 1.0 / na for v in vis])
        f_a.extend([float(x) for x in q])
    else:
        f_a.extend([1.0 / na] * na)
        f_a.extend([0.0] * na)
    return nt, na, searched


def parse_file(fn):
    tok_a, act_a, ref_a = array.array('h'), array.array('h'), array.array('h')
    f_a, g_a = array.array('f'), array.array('h')
    rec = {'nref': [], 'nt': [], 'na': [], 'pi': [], 'z': [], 'searched': [], 'gid': [],
           'pk': [], 'rv': []}
    for line in gzip.open(fn, 'rt'):
        d = json.loads(line)
        if not d['teacher'].startswith('ismcts'):
            continue
        nt, na, s = encode(d, tok_a, act_a, ref_a, f_a, rec)
        g_a.extend([clip16(x) for x in d['g']])
        rec['nt'].append(nt); rec['na'].append(na); rec['pi'].append(d['pi'])
        rec['z'].append(d['z']); rec['searched'].append(s); rec['gid'].append(d['gid'])
        rec['pk'].append(d['pk']); rec['rv'].append(float(d.get('rv', 0.0) or 0.0))
    return fn, tok_a.tobytes(), act_a.tobytes(), ref_a.tobytes(), f_a.tobytes(), g_a.tobytes(), rec


def t16(b):
    return torch.frombuffer(bytearray(b), dtype=torch.int16).clone() if len(b) else torch.zeros(0, dtype=torch.int16)


def tf32(b):
    return torch.frombuffer(bytearray(b), dtype=torch.float32).clone() if len(b) else torch.zeros(0)


def pack(parts):
    """parts: list of (tok bytes, act bytes, ref bytes, f bytes, g bytes, rec)."""
    tok = torch.cat([t16(p[0]) for p in parts]).view(-1, TOKW)
    act = torch.cat([t16(p[1]) for p in parts]).view(-1, ACTW)
    ref = torch.cat([t16(p[2]) for p in parts])
    fl = torch.cat([tf32(p[3]) for p in parts])
    g = torch.cat([t16(p[4]) for p in parts]).view(-1, 72)
    rec = {k: [] for k in parts[0][5]}
    for p in parts:
        for k in rec:
            rec[k].extend(p[5][k])
    nt = torch.tensor(rec['nt'], dtype=torch.long)
    na = torch.tensor(rec['na'], dtype=torch.long)
    nr = torch.tensor(rec['nref'], dtype=torch.long)
    tok_off = torch.cumsum(nt, 0) - nt
    act_off = torch.cumsum(na, 0) - na
    ref_off = torch.cumsum(nr, 0) - nr
    # per decision float block = vis (na) then q (na)
    f_off = 2 * act_off
    idx = act_off.repeat_interleave(na) * 2 + (torch.arange(int(na.sum())) - act_off.repeat_interleave(na))
    vis = fl[idx]
    q = fl[idx + na.repeat_interleave(na)]
    return dict(tok=tok, act=act, ref=ref, g=g, vis=vis, q=q, nt=nt, na=na, nref=nr,
                tok_off=tok_off, act_off=act_off, ref_off=ref_off,
                pi=torch.tensor(rec['pi'], dtype=torch.long),
                z=torch.tensor(rec['z'], dtype=torch.float32),
                rv=torch.tensor(rec['rv'], dtype=torch.float32),
                searched=torch.tensor(rec['searched'], dtype=torch.bool),
                gid=rec['gid'], pk=rec['pk'])


def swap_records():
    """Each swap record -> two pseudo-decisions (orig, swap) with teacher summaries."""
    parts_o, parts_s, meta = [], [], []
    for line in open(os.path.join(DATA, 'swap', 'swap_tests.jsonl')):
        r = json.loads(line)
        m = {k: r[k] for k in ('jid', 'gid', 'seq', 'base', 'variant', 'sims', 'n_swapped', 'fp_ok')}
        for ver, parts in (('orig', parts_o), ('swap', parts_s)):
            v = r['versions'][ver]
            d = {'tok': v['tok'], 'act': v['act'], 'psrc': v['psrc'], 'g': v['g'], 'pi': 0, 'z': 0,
                 'teacher': 'ismcts', 'gid': r['gid'], 'pk': v['pk']}
            na = len(v['act'])
            ca = v['card_acts']
            se = v['searches']
            vis_mass, rvs, qsoft, vis_mean = [], [], [], [0.0] * na
            se = [dict(s, vis=(s['vis'] if len(s['vis']) == na else [1.0 / na] * na),
                       q=(s['q'] if len(s['q']) == na else [0.0] * na)) for s in se]
            for s in se:
                tot = float(sum(s['vis']))
                vis_mass.append(sum(s['vis'][i] for i in ca) / tot if tot else 0.0)
                rvs.append(float(s['rv']))
                for i in range(na):
                    vis_mean[i] += s['vis'][i] / tot / len(se)
            d['searched'] = 1
            d['vis'] = vis_mean
            d['q'] = [sum(s['q'][i] for s in se) / len(se) for i in range(na)]
            d['rv'] = sum(rvs) / len(rvs)
            tok_a, act_a, ref_a = array.array('h'), array.array('h'), array.array('h')
            f_a, g_a = array.array('f'), array.array('h')
            rec = {'nref': [], 'nt': [], 'na': [], 'pi': [], 'z': [], 'searched': [], 'gid': [],
                   'pk': [], 'rv': []}
            nt, na2, s_ = encode(d, tok_a, act_a, ref_a, f_a, rec)
            g_a.extend([clip16(x) for x in d['g']])
            rec['nt'].append(nt); rec['na'].append(na2); rec['pi'].append(0); rec['z'].append(0)
            rec['searched'].append(1); rec['gid'].append(r['gid']); rec['pk'].append(v['pk'])
            rec['rv'].append(d['rv'])
            parts.append((tok_a.tobytes(), act_a.tobytes(), ref_a.tobytes(), f_a.tobytes(), g_a.tobytes(), rec))
            m[ver] = {'card_acts': ca, 'n_act': na, 'vis_mass': sum(vis_mass) / len(vis_mass),
                      'vis_mass_seeds': vis_mass, 'rv': sum(rvs) / len(rvs), 'rv_seeds': rvs,
                      'q_card_max': max([d['q'][i] for i in ca]) if ca else None,
                      'q_other_max': max([d['q'][i] for i in range(na) if i not in ca]) if na > len(ca) else None}
        meta.append(m)
    return pack(parts_o), pack(parts_s), meta


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    man = json.load(open(os.path.join(DATA, 'manifest.json')))
    games = {g['gid']: g for g in man['games']}
    files = sorted(glob.glob(os.path.join(DATA, 'out', 'base_teacher__s*.decisions.jsonl.gz')) +
                   glob.glob(os.path.join(DATA, 'out', 'var_teacher__*.decisions.jsonl.gz')))
    with Pool(8) as pool:
        res = pool.map(parse_file, files, chunksize=1)
    res.sort(key=lambda r: r[0])
    P = pack([r[1:] for r in res])
    gids = P.pop('gid')
    split_map = {'train': 0, 'val': 1, 'test': 2}
    variants = sorted({g['variant'] for g in man['games']})
    P['split'] = torch.tensor([split_map[games[x]['split']] for x in gids], dtype=torch.long)
    P['variant'] = torch.tensor([variants.index(games[x]['variant']) for x in gids], dtype=torch.long)
    gid_list = sorted(set(gids))
    gmap = {x: i for i, x in enumerate(gid_list)}
    P['game'] = torch.tensor([gmap[x] for x in gids], dtype=torch.long)
    pks = sorted(set(P['pk']))
    P['pk'] = torch.tensor([pks.index(x) for x in P['pk']], dtype=torch.long)
    P['meta'] = {'vocab': VOCAB, 'variants': variants, 'gid_list': gid_list, 'pk_names': pks,
                 'files': [os.path.basename(r[0]) for r in res]}
    torch.save(P, os.path.join(OUT, 'teacher.pt'))
    print('teacher decisions', len(P['pi']), 'tokens', P['tok'].shape, 'secs', time.time() - t0, flush=True)
    O, S, meta = swap_records()
    for X in (O, S):
        X.pop('gid')
        X['pk'] = torch.tensor([pks.index(x) if x in pks else 0 for x in X['pk']], dtype=torch.long)
    torch.save({'orig': O, 'swap': S, 'meta': meta}, os.path.join(OUT, 'swap.pt'))
    print('swap records', len(meta), 'secs', time.time() - t0)


if __name__ == '__main__':
    main()
