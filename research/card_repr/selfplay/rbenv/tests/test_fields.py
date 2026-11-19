# Schema v2 (gate 1): every new field decoded from the int32 observation and checked
# against an independent read of the engine state (rbenv.cpp verifyEnv), over random
# games of the Rengar mirror. Usage: test_fields.py [n_games]
# Schema v3/v4 (gate 3): also checks act 49..55 (optional-cost fields only at Accelerate / AdditionalCost),
# TriggerOrder options (each names a distinct pending chain item, 33, whose source token is the option's 1)
# and MoveJoin options (destination 30 + 46..48, the units already moving 56..60, "no more" = dec 39).
import sys, time
from common import *
NAMES = ['decisions', 'pile_mismatch', 'pile_tokens', 'known_hand_mismatch', 'known_opp_hand_tokens',
         'cost_mismatch', 'hand_cost_checked', 'noxus_hopeful_checked', 'offered_but_unaffordable',
         'affordable_main', 'affordable_main_offered', 'hidden_turn_mismatch', 'hidden_turn_seen',
         'starspring_stamp_seen', 'chain_paid_seen', 'chain_ordinal_seen', 'location_mapped',
         'location_unmapped', 'move_conseq_mismatch', 'moves_checked', 'sabotage_opp_hand',
         'sabotage_opp_hand_tokens', 'max_tokens']
n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
env = RBEnv(32, 'rengar', 'rengar', seed=7000)
assert env.schema_version() == 4, env.schema_version()
PK_ACC, PK_ADD, PK_TO, PK_MJ = 13, 15, 17, 20
v4 = dict(oc_outside=0, oc_rows=0, to_prompts=0, to_opts=0, to_unmapped=0, to_src_bad=0, to_dup=0,
          mj_prompts=0, mj_opts=0, mj_no_dst=0, mj_no_group=0, mj_dec_bad=0, mj_cons_bad=0, mj_group_bad=0)
rng = np.random.default_rng(1)
acc = np.zeros(23, np.int64)
res = []
t = time.time()
while len(res) < n:
    o = env.observe()
    env.verify(acc)
    for e in np.nonzero(o['need'])[0]:
        pk = int(o['pk'][e]); A_ = o['act'][e][o['act_mask'][e]].astype(int); tk = o['tok'][e]
        oc = A_[:, 49] >= 0
        if pk in (PK_ACC, PK_ADD): v4['oc_rows'] += int(oc.sum())
        elif oc.any(): v4['oc_outside'] += 1
        if pk == PK_TO:
            v4['to_prompts'] += 1; v4['to_opts'] += len(A_)
            ci = A_[:, 33]
            v4['to_unmapped'] += int((ci < 0).sum())
            ok = ci[ci >= 0]
            if len(set(ok.tolist())) != len(ok): v4['to_dup'] += 1
            for r in A_:
                if r[33] >= 0 and (tk[r[33], 0] != 3 or tk[r[33], 10] != r[1]): v4['to_src_bad'] += 1
        if pk == PK_MJ:
            v4['mj_prompts'] += 1; v4['mj_opts'] += len(A_)
            v4['mj_no_dst'] += int((A_[:, 30] < 0).sum())
            v4['mj_no_group'] += int(((A_[:, 60] < 1) | (A_[:, 56] < 0)).sum())
            stop = A_[:, 42] == 0
            if stop.sum() != 1 or (A_[stop, 39] != 1).any() or (A_[~stop, 39] != 0).any(): v4['mj_dec_bad'] += 1
            if len(set(map(tuple, A_[:, [30, 46, 47, 48]].tolist()))) != 1: v4['mj_cons_bad'] += 1
            for r in A_:
                grp = [g for g in r[56:56 + max(0, min(4, r[60]))] if g >= 0]
                if any(tk[g, 0] != 1 or tk[g, 2] != 1 for g in grp) or (r[42] > 0 and r[8] in grp): v4['mj_group_bad'] += 1
    env.step(random_choices(o, rng))
    res += env.results()
r = dict(zip(NAMES, map(int, acc)))
print('games', len(res), 'errors', sum(1 for x in res if x['error']), 'secs %.1f' % (time.time() - t))
for k, v in r.items(): print('  %-28s %d' % (k, v))
print('counters', env.counters())
print('v4', v4)
bad = [k for k in ('oc_outside', 'to_unmapped', 'to_src_bad', 'to_dup', 'mj_no_dst', 'mj_no_group', 'mj_dec_bad',
                   'mj_cons_bad', 'mj_group_bad') if v4[k]]
bad += [k for k in ('pile_mismatch', 'known_hand_mismatch', 'cost_mismatch', 'offered_but_unaffordable',
                   'hidden_turn_mismatch', 'move_conseq_mismatch', 'location_unmapped') if r[k]]
print('FAIL' if bad else 'OK', bad)
