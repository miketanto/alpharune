"""Collect card-check (probe) states with rule-defined right answers -> learner/probes/<name>.npz.

States come from mirror games of the relevant deck with both seats uniform-random over legal
options (biased 50% towards playing the probed card when it is offered, so its prompts occur
often). At most 3 states per game and probe. Each npz holds the observe() arrays (glob, tok,
tok_mask, act, act_mask, pk, psrc) of the states plus
  good (B,A) bool  options that are right by rule
  play (B,A) bool  options that play the probed card (conditional probes)
  cond (B,)  bool  the card's condition holds (conditional probes; all True for target probes)

Probes
  ctrl_tgt      exp_train deck, punch_first__ctrl_scores at its target prompt (pk 24, psrc = the
                card): good = friendly units. Only states with both a friendly and an enemy option.
  oppctrl_tgt   exp_heldout deck, punch_first__opp_of_ctrl_scores, same prompt: good = enemy units.
  opphand4      exp_train deck, main phase (pk 3) with sabotage__score_opphand4 playable:
                good = its play options if the opponent holds 4+ cards (glob[10]) else every other option.
  alone         exp_train deck, main phase with pit_rookie__score_alone playable: condition = the
                decider controls no units on the board (card tokens, controller 1, zone Base(3)/
                BattlefieldZone(8), unit type); good as above.
  sab1          exp_heldout_sab1 deck, main phase with sabotage__score1 playable: good = its play
                options (unconditional; held out).
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'rbenv'))
from rbenv import RBEnv, load_vocab
from make_exp_decks import ALL_VARIANTS
OUT = os.path.join(HERE, '..', 'learner', 'probes')
KEYS = ('glob', 'tok', 'tok_mask', 'act', 'act_mask', 'pk', 'psrc')
voc = load_vocab(); V = {k: i for i, k in enumerate(voc)}
cj = json.load(open(os.path.join(HERE, '..', '..', 'cards.json'), encoding='utf-8'))
ctype = {c['key']: (c['printings'][0]['card_type'] if c['printings'] else 'legend') for c in cj['cards']}
UNIT = np.zeros(len(voc), bool)
for i, k in enumerate(voc):
    base = k.replace('synthetic:', '').split('__')[0]
    UNIT[i] = k.startswith('token:') or ctype.get(base) == 'unit'
PLAY_T = (0, 5, 8)


def play_opts(o, e, key):
    a, m = o['act'][e], o['act_mask'][e]
    src = a[:, 1]
    ok = m & np.isin(a[:, 0], PLAY_T) & (src >= 0)
    ok[ok] &= o['tok'][e, src[ok], 1] == V[key]
    return ok


def target_probe(o, e, key, friendly_good):
    if o['pk'][e] != 24 or o['psrc'][e] < 0 or o['tok'][e, o['psrc'][e], 1] != V[key]:
        return None
    a, m = o['act'][e], o['act_mask'][e]
    ref = a[:, 8]
    good = np.zeros_like(m)
    side = np.zeros(len(m), int)
    for i in np.nonzero(m)[0]:
        r = ref[i]
        if r < 0 or o['tok'][e, r, 0] != 1:
            return None
        side[i] = o['tok'][e, r, 2]          # controller: 1 self, 2 opp
    sides = set(side[m].tolist())
    if not {1, 2} <= sides:
        return None
    good = m & (side == (1 if friendly_good else 2))
    return good, np.zeros_like(m), True


def cond_probe(o, e, key, cond_fn):
    if o['pk'][e] != 3:
        return None
    pl = play_opts(o, e, key)
    if not pl.any():
        return None
    c = bool(cond_fn(o, e))
    good = pl.copy() if c else (o['act_mask'][e] & ~pl)
    return good, pl, c


def no_units(o, e):
    t = o['tok'][e][o['tok_mask'][e]]
    on = (t[:, 0] == 1) & (t[:, 2] == 1) & np.isin(t[:, 4], (3, 8)) & UNIT[t[:, 1].clip(0, len(voc) - 1)]
    return not on.any()


PROBES = {
    'ctrl_tgt': ('rengar__exp_train', 'synthetic:punch_first__ctrl_scores', lambda o, e: target_probe(o, e, 'synthetic:punch_first__ctrl_scores', True)),
    'oppctrl_tgt': ('rengar__exp_heldout', 'synthetic:punch_first__opp_of_ctrl_scores', lambda o, e: target_probe(o, e, 'synthetic:punch_first__opp_of_ctrl_scores', False)),
    'opphand4': ('rengar__exp_train', 'synthetic:sabotage__score_opphand4', lambda o, e: cond_probe(o, e, 'synthetic:sabotage__score_opphand4', lambda o, e: o['glob'][e, 10] >= 4)),
    'alone': ('rengar__exp_train', 'synthetic:pit_rookie__score_alone', lambda o, e: cond_probe(o, e, 'synthetic:pit_rookie__score_alone', no_units)),
    'sab1': ('rengar__exp_heldout_sab1', 'synthetic:sabotage__score1', lambda o, e: cond_probe(o, e, 'synthetic:sabotage__score1', lambda o, e: True)),
}


def collect(name, target=260, n_envs=32, seed=4242, balance=False):
    deck, key, fn = PROBES[name]
    env = RBEnv(n_envs, deck, deck, seed, variant=ALL_VARIANTS)
    rng = np.random.default_rng(seed)
    rows, per_game, gid = [], {}, np.arange(n_envs)
    nxt = n_envs
    ncond = {True: 0, False: 0}
    while len(rows) < target:
        o = env.observe()
        ch = np.zeros(n_envs, np.int32)
        for e in np.nonzero(o['need'])[0]:
            r = fn(o, e)
            if r is not None and per_game.get(gid[e], 0) < 3:
                good, pl, c = r
                if not balance or ncond[c] < target // 2:
                    rows.append(({k: o[k][e].copy() for k in KEYS}, good.copy(), pl.copy(), c))
                    per_game[gid[e]] = per_game.get(gid[e], 0) + 1
                    ncond[c] += 1
            m = np.nonzero(o['act_mask'][e])[0]
            pl = play_opts(o, e, key)
            ch[e] = np.nonzero(pl)[0][0] if pl.any() and rng.random() < 0.5 else rng.choice(m)
        env.step(ch)
        for res in env.results():
            gid[res['env']] = nxt; nxt += 1
    env.close()
    d = {k: np.stack([r[0][k] for r in rows]) for k in KEYS}
    d['good'] = np.stack([r[1] for r in rows]); d['play'] = np.stack([r[2] for r in rows])
    d['cond'] = np.array([r[3] for r in rows])
    np.savez_compressed(os.path.join(OUT, f'{name}.npz'), **d)
    print(name, 'states', len(rows), 'games', nxt, 'cond true', int(d['cond'].sum()),
          'mean options', round(float(d['act_mask'].sum(1).mean()), 2),
          'uniform mass on good', round(float((d['good'].sum(1) / d['act_mask'].sum(1)).mean()), 3), flush=True)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for n in (sys.argv[1:] or PROBES):
        collect(n, balance=n in ('opphand4', 'alone'))
