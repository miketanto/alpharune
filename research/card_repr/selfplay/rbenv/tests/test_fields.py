# Schema v2 (gate 1): every new field decoded from the int32 observation and checked
# against an independent read of the engine state (rbenv.cpp verifyEnv), over random
# games of the Rengar mirror. Usage: test_fields.py [n_games]
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
assert env.schema_version() == 2, env.schema_version()
rng = np.random.default_rng(1)
acc = np.zeros(23, np.int64)
res = []
t = time.time()
while len(res) < n:
    o = env.observe()
    env.verify(acc)
    env.step(random_choices(o, rng))
    res += env.results()
r = dict(zip(NAMES, map(int, acc)))
print('games', len(res), 'errors', sum(1 for x in res if x['error']), 'secs %.1f' % (time.time() - t))
for k, v in r.items(): print('  %-28s %d' % (k, v))
print('counters', env.counters())
bad = [k for k in ('pile_mismatch', 'known_hand_mismatch', 'cost_mismatch', 'offered_but_unaffordable',
                   'hidden_turn_mismatch', 'move_conseq_mismatch', 'location_unmapped') if r[k]]
print('FAIL' if bad else 'OK', bad)
