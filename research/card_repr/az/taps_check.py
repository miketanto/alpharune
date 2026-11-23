"""Checks of the network rune taps (TAPS.md). RB_WSL_COPY = a float-engine copy.

  python3 az/taps_check.py parity|random

parity: the same games with taps off and with taps offered to both seats but never taken, uniform random other moves
  (chosen by option number among the moves that are not taps): every decision must be the same one, with the same
  globals, history, prompt, tokens (the offered runes' tokens come last) and the same options in the same order.
random: games with uniform random moves, taps included: no errors, no truncation, every tap possible.
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'selfplay', 'rbenv'))
import rbenv
from rbenv import RBEnv

V = rbenv.load_vocab(); V = V if isinstance(V, list) else V['vocab']
RUNE = np.array([k.endswith('_rune') for k in V])
PAIRS = [tuple(p.split(':')) for p in os.environ['TAPS_PAIRS'].split(',')] if os.environ.get('TAPS_PAIRS') else [('azir', 'azir'), ('azir', 'rengar'), ('rengar', 'azir'), ('azir', 'diana'), ('diana', 'azir')]


def taps(o, i):
    """mask over the options of env i: a rune's own ability"""
    src = o['act'][i, :, 2]
    ok = o['act_mask'][i] & (src >= 0)
    r = np.zeros(len(src), bool)
    r[ok] = RUNE[o['tok'][i, src[ok], 1]]
    return r


def parity():
    bad = dec = withtap = 0
    for a, b in PAIRS:
        n, want = 1, 6
        for seed in range(50, 50 + want):
            E0, E1 = RBEnv(n, a, b, seed), RBEnv(n, a, b, seed, runes=3)
            rng = np.random.default_rng(seed)
            for step in range(3000):
                o0, o1 = E0.observe(), E1.observe()
                t1 = taps(o1, 0)
                moves = np.nonzero(o1['act_mask'][0] & ~t1)[0]
                n0 = int(o0['act_mask'][0].sum())
                withtap += bool(t1.any()); dec += 1
                nt = int(o0['tok_mask'][0].sum())
                same = (len(moves) == n0 and o0['nlegal'][0] + t1.sum() == o1['nlegal'][0] and o0['seat'][0] == o1['seat'][0]
                        and o0['pk'][0] == o1['pk'][0] and o0['psrc'][0] == o1['psrc'][0]
                        and np.array_equal(o0['glob'], o1['glob']) and np.array_equal(o0['hist'], o1['hist'])
                        and np.array_equal(o0['tok'][0, :nt], o1['tok'][0, :nt])
                        and np.array_equal(o0['act'][0, :n0], o1['act'][0, moves]) if n0 < 96 else True)
                if not same:
                    bad += 1
                    if bad < 6:
                        print('DIFF', a, b, seed, 'step', step, 'pk', o0['pk'][0], o1['pk'][0], 'n', n0, len(moves), t1.sum(),
                              [k for k in ('glob', 'hist') if not np.array_equal(o0[k], o1[k])], np.argwhere(o0['act'][0, :n0] != o1['act'][0, moves][:n0])[:4].tolist() if len(moves) == n0 else '')
                    if len(moves) != n0:
                        break
                c = int(rng.integers(0, n0))
                E0.step(np.array([c])); E1.step(np.array([moves[c]]))
                r0, r1 = E0.results(), E1.results()
                if r0 or r1:
                    k = ('winner', 'turns', 'n_ext', 'error')
                    if [tuple(r[x] for x in k) for r in r0] != [tuple(r[x] for x in k) for r in r1]:
                        bad += 1; print('RESULT', r0, r1)
                    break
            E0.close(); E1.close()
    print(f'parity: decisions {dec}, offering taps {withtap}, problems {bad}')
    return bad


def random_games():
    tot = dict(games=0, err=0, ex=0, rec=0, dec=0, offer=0, maxtok=0, maxopt=0)
    cnt = {}
    for a, b in PAIRS:
        E = RBEnv(40, a, b, 777, runes=3)
        rng = np.random.default_rng(3)
        done = set()
        while len(done) < 40:
            o = E.observe()
            ch = np.zeros(40, np.int64)
            for i in range(40):
                if not o['need'][i]:
                    continue
                n = min(int(o['nlegal'][i]), 96)
                t = taps(o, i)
                ch[i] = rng.integers(0, n)
                tot['dec'] += 1; tot['offer'] += bool(t.any())
                if t.any() and int((o['act_mask'][i] & ~t).sum()) < 2 and o['nlegal'][i] <= 96:
                    raise SystemExit('asked with fewer than two moves')
                tot['maxtok'] = max(tot['maxtok'], int(o['tok_mask'][i].sum())); tot['maxopt'] = max(tot['maxopt'], int(o['nlegal'][i]))
                if t[ch[i]]:
                    tot['ex' if o['act'][i, ch[i], 3] == 0 else 'rec'] += 1
            E.step(ch)
            for r in E.results():
                if r['env'] not in done:
                    done.add(r['env']); tot['games'] += 1
                    if r['error']:
                        tot['err'] += 1; print(r)
        for k, x in E.counters().items():
            cnt[k] = cnt.get(k, 0) + x
        E.close()
    print('random', tot, {k: cnt[k] for k in ('trunc_tok', 'trunc_act', 'trunc_list', 'unk_key')})
    return tot['err']


sys.exit(bool({'parity': parity, 'random': random_games}[sys.argv[1]]()))
