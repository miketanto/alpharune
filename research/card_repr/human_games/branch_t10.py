"""Turn 10 of game_297763546 with other facedown cards: the human hides nothing / only one / swapped, G2 then makes the same
moves as in the real game (matched by words) up to its attack; its attack probabilities and value are read there."""
import os, sys
sys.argv = ['x', 'game_297763546.json']
import rollouts as R
import numpy as np
C = R.G['choices']
H0 = 169   # first hide decision (Hidden Blade at bf1), 170 = Guards! at bf0, 171 = end the turn


def step_text(g, pick):
    op = R.options(g)
    k = [i for i, t in op.items() if pick(t)]
    if not k:
        return False
    g.step(k[0]); return True


hide = lambda card, bf: (lambda t: 'hide' in t.lower() and card in t and t.rstrip('.').endswith(str(bf)))
variants = {'real: Blade bf1, Guards bf0': [('Hidden Blade', 1), ('Guards!', 0)], 'nothing hidden': [], 'only Blade at bf1': [('Hidden Blade', 1)],
            'only Guards at bf0': [('Guards!', 0)], 'swapped: Guards bf1, Blade bf0': [('Guards!', 1), ('Hidden Blade', 0)]}
# words of G2's real choices from #172 up to its attack decision #184
g = R.replay(C[:172])
words = []
for s, c in C[172:184]:
    words.append((s, R.options(g)[c], c)); g.step(c)
for name, hs in variants.items():
    g = R.replay(C[:H0])
    ok = all(step_text(g, hide(c, b)) for c, b in hs) and step_text(g, lambda t: t.startswith('end the turn'))
    if not ok:
        print(name, 'branch failed', list(R.options(g).values())[:12]); continue
    v0 = None
    wi = 0
    bad = None
    while True:
        st = g.status()
        op = R.options(g)
        if v0 is None and st['seat'] == 2:
            v0 = R.ev(g.observe())[1]
        if wi == len(words):
            break
        s, w, c0 = words[wi]
        if st['seat'] != s:
            if st['seat'] == 1 or s == 1:      # a pass the other variant is (not) asked for
                if s == 1:
                    wi += 1; continue
                if not step_text(g, lambda t: t.startswith('pass')):
                    bad = ('seat', wi); break
                continue
        k = [i for i, t in op.items() if t == w]
        if not k:
            bad = (w, list(op.values())[:6]); break
        g.step(c0 if op.get(c0) == w else k[0]); wi += 1
    if bad:
        print(name, 'could not follow', bad); continue
    p, v = R.ev(g.observe())
    op = R.options(g)
    m = lambda b: sum(p[i] for i, t in op.items() if t.startswith('move') and f'battlefield {b}' in t)
    print(f'{name:34s} value at turn start {v0:+.2f} | before attacking: value {v:+.2f}  P(move bf1) {m(1):.2f}  P(move bf0) {m(0):.2f}  P(end turn) '
          f"{sum(p[i] for i, t in op.items() if t.startswith('end')):.2f}  P(equip) {sum(p[i] for i, t in op.items() if t.startswith('Activate')):.2f}")
