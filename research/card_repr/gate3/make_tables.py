"""Builds the gate-3 tables (tables.txt) and results.json from inventory.json, path_*.json, learn_results.jsonl,
exposure.json, eval_*.json."""
import json, os, glob
HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda f: json.load(open(os.path.join(HERE, f)))
inv = J('inventory.json')
P = {t: J(f'path_{t}.json') for t in ('untrained', 'trained', 'untrained_v3', 'trained_v3')
     if os.path.exists(os.path.join(HERE, f'path_{t}.json'))}
L = {}
for fn in ('learn_results.jsonl', 'learn_v3.jsonl', 'learn_v3b.jsonl'):
    if os.path.exists(os.path.join(HERE, fn)):
        for l in open(os.path.join(HERE, fn)):
            r = json.loads(l); L[(r['pk'], r['mode'], r['tag'])] = r['final_acc']
REL = {  # option-relevant fields per pk: (group, field, label)
    'MainPhase': [('ref', 'f19', 'eff_energy'), ('ref', 'f20', 'eff_power'), ('ref', 'f21', 'affordable'), ('ref', 'key', 'card'),
                  ('act', 'a30', 'dst'), ('act', 'a46', 'dst_ctrl'), ('act', 'a47', 'starts_combat'), ('act', 'a48', 'contest')],
    'ChainPriority': [('ref', 'f19', 'eff_energy'), ('ref', 'f21', 'affordable'), ('ref', 'key', 'card'), ('act', 'a46', 'dst_ctrl')],
    'ShowdownFocus': [('ref', 'f19', 'eff_energy'), ('ref', 'f21', 'affordable'), ('ref', 'key', 'card'), ('act', 'a46', 'dst_ctrl')],
    'Target': [('ref', 'f2', 'side'), ('ref', 'f9', 'Might'), ('ref', 'f8', 'damage'), ('ref', 'f14', 'combat')],
    'TargetFirst': [('ref', 'f2', 'side'), ('ref', 'f9', 'Might'), ('ref', 'f8', 'damage'), ('ref', 'f14', 'combat')],
    'TargetSecond': [('ref', 'f2', 'side'), ('ref', 'f9', 'Might'), ('ref', 'f8', 'damage'), ('ref', 'f14', 'combat')],
    'ChooseObject': [('ref', 'key', 'card'), ('ref', 'f4', 'zone'), ('ref', 'f2', 'side')],
    'May': [('act', 'a39', 'decline'), ('ask', 'f9', 'ask.Might'), ('ask', 'f6', 'ask.exhausted')],
    'Accelerate': [('act', 'a39', 'decline'), ('ask', 'f19', 'ask.eff_energy'), ('ask', 'f21', 'ask.affordable')],
    'AdditionalCost': [('act', 'a39', 'decline'), ('ask', 'f19', 'ask.eff_energy'), ('ask', 'f21', 'ask.affordable')],
    'Location': [('act', 'a30', 'dst'), ('act', 'a46', 'dst_ctrl'), ('act', 'a47', 'starts_combat'), ('act', 'a48', 'contest')],
    'MoveJoin': [('ref', 'f9', 'Might'), ('ref', 'f6', 'exhausted'), ('ref', 'f5', 'loc'), ('act', 'a31', 'has_v(stop)')],
    'CombatDamage': [('ref', 'f9', 'Might'), ('ref', 'f8', 'damage'), ('act', 'a19', 'dmg_amount')],
    'CombatDamageNext': [('ref', 'f9', 'Might'), ('ref', 'f8', 'damage')],
    'TriggerOrder': [('act', 'a32', 'v (index)')],
    'Mulligan': [('ref', 'key', 'card'), ('ref', 'f19', 'eff_energy')],
    'NextStaged': [('ref', 'key', 'bf')],
}
GRP = {'ref': 'ref_token_fields', 'act': 'act_fields', 'ask': 'ask_token_fields'}


def fld(t, pk, g, f):
    m = P[t].get(pk, {}).get(GRP[g], {}).get(f)
    return None if m is None else m['tv_mean']


lines = []
hdr = f"{'pk':16s}{'/game':>7s}{'psrc':>11s} | {'askTV u':>8s}{'askTV t':>8s}{'fix u':>8s}{'fix t':>8s} | {'learn':>11s} | relevant fields TV (trained, old head) "
lines.append(hdr); lines.append('-' * len(hdr))
table = {}
for pk, I in inv['pk'].items():
    ps = '/'.join(f'{k}' for k in I['psrc_kind'])
    a = [P[t].get(pk, {}).get('ask', {}).get('tv_mean') for t in ('untrained', 'trained', 'untrained_v3', 'trained_v3') if t in P]
    lb, lf = L.get((pk, 'card', 'base')), L.get((pk, 'card', 'v3fix'), L.get((pk, 'card', 'askopt')))
    rel = []
    for g, f, nm in REL.get(pk, []):
        v = fld('trained', pk, g, f); rel.append((nm, v))
    fs = lambda x: '   -   ' if x is None else f'{x:8.4f}'
    lr = '    -      ' if lb is None else f'{lb:.2f}->{lf if lf is not None else float("nan"):.2f}'
    lines.append(f"{pk:16s}{I['per_game']:7.2f}{ps:>11s} | " + ''.join(fs(x) for x in a + [None] * (4 - len(a))) + f" | {lr:>11s} | "
                 + ', '.join(f'{n}={"n/a" if v is None else f"{v:.3f}"}' for n, v in rel))
    table[pk] = dict(per_game=I['per_game'], n=I['n'], psrc=I['psrc_kind'], ask_tv=dict(zip(['untrained', 'trained', 'untrained_fix', 'trained_fix'], a)),
                     learn_card=dict(base=lb, askopt=lf), relevant=dict(rel))
open(os.path.join(HERE, 'tables.txt'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
res = dict(inventory=inv, per_pk=table, learn={f'{k[0]}|{k[1]}|{k[2]}': v for k, v in L.items()},
           path={t: P[t] for t in P})
for f in ('exposure.json', 'verify_optcost.json') + tuple(os.path.basename(x) for x in glob.glob(os.path.join(HERE, 'eval_*.json'))):
    if os.path.exists(os.path.join(HERE, f)):
        res[f[:-5]] = J(f)
json.dump(res, open(os.path.join(HERE, 'results.json'), 'w'), indent=1)
