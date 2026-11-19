# Several variants in one process: mixed-variant decks play, keys are known, determinism holds.
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'cardexp'))
from common import *
from rbenv import load_vocab
from make_exp_decks import ALL_VARIANTS
voc = load_vocab()
deck = lambda n: f'rengar__{n}'
seen = {}
for name in ('exp_train', 'exp_heldout', 'exp_heldout_sab1'):
    env = RBEnv(16, deck(name), deck(name), seed=11, variant=ALL_VARIANTS)
    ids = set()
    rng = np.random.default_rng(3)
    res = []
    while len(res) < 40:
        o = env.observe()
        ids |= set(np.unique(o['tok'][..., 1][o['tok_mask']]).tolist())
        env.step(random_choices(o, rng)); res += env.results()
    errs = sum(1 for r in res if r['error'])
    syn = sorted(voc[i] for i in ids if voc[i].startswith('synthetic:'))
    print(name, 'games', len(res), 'errors', errs, 'counters', env.counters(), 'synthetic keys seen', syn)
    assert errs == 0 and env.counters()['unk_key'] == 0
    env.close()
def run():
    env = RBEnv(8, deck('exp_train'), deck('exp_heldout'), seed=77, variant=ALL_VARIANTS)
    rec = []
    res, _ = play(env, 24, np.random.default_rng(5), record=rec)
    return rec, res
a, ra = run(); b, rb = run()
assert len(a) == len(b)
for (oa, ca), (ob, cb) in zip(a, b):
    for k in oa: assert np.array_equal(oa[k], ob[k]), k
key = lambda r: [(x['env'], x['game_seed'], x['winner'], x['length']) for x in r]
assert key(ra) == key(rb)
print('multivariant determinism ok:', len(a), 'steps,', len(ra), 'games identical')
