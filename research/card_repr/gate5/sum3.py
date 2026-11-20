"""Gate 5 test 3 table: greedy win rate vs ISMCTS-200 in the Yi mirror per fine-tune budget, two 200-game batches
(seed 990001: selfplay/yi_eval for budget 0, gate5/evals/ft<U>_<R>.json; seed 1990001: evals/ft<U>_<R>_s2.json).
B2 - id: difference of pooled proportions, normal CI; p two-sided."""
import json, math, os
H = os.path.dirname(os.path.abspath(__file__))
E = os.path.join(H, 'evals')


def ld(p):
    try:
        d = json.load(open(p)); return d['wins'], d['games']
    except Exception:
        return None


def wilson(w, g, z=1.96):
    p = w / g; d = 1 + z * z / g; c = p + z * z / (2 * g); h = z * math.sqrt(p * (1 - p) / g + z * z / (4 * g * g))
    return (c - h) / d, (c + h) / d


out = {}
print('%-6s %-26s %-26s %-28s' % ('budget', 'B2 wins/games  rate [CI]', 'id wins/games  rate [CI]', 'B2 - id [95% CI]  p'))
for U in (0, 20, 50, 100):
    row = {}
    for R in ('B2', 'id'):
        a = ld(os.path.join(H, '..', 'selfplay', 'yi_eval', 'cf_%s_vs_ismcts200.json' % R)) if U == 0 else ld(os.path.join(E, 'ft%d_%s.json' % (U, R)))
        b = ld(os.path.join(E, 'ft%d_%s_s2.json' % (U, R)))
        w = sum(x[0] for x in (a, b) if x); g = sum(x[1] for x in (a, b) if x)
        row[R] = (w, g, [x for x in (a, b) if x])
    (wb, gb, sb), (wi, gi, si) = row['B2'], row['id']
    pb, pi = wb / gb, wi / gi
    se = math.sqrt(pb * (1 - pb) / gb + pi * (1 - pi) / gi)
    z = (pb - pi) / se if se else 0
    p = math.erfc(abs(z) / math.sqrt(2))
    lb = wilson(wb, gb); li = wilson(wi, gi)
    print('%-6d %3d/%3d %.3f [%.3f,%.3f] %3d/%3d %.3f [%.3f,%.3f] %+.3f [%+.3f,%+.3f] p=%.3f  batches B2 %s id %s' % (
        U, wb, gb, pb, lb[0], lb[1], wi, gi, pi, li[0], li[1], pb - pi, pb - pi - 1.96 * se, pb - pi + 1.96 * se, p,
        [round(x[0] / x[1], 3) for x in sb], [round(x[0] / x[1], 3) for x in si]))
    out[U] = dict(B2=[wb, gb], id=[wi, gi], diff=pb - pi, ci=[pb - pi - 1.96 * se, pb - pi + 1.96 * se], p=p)
json.dump(out, open(os.path.join(H, 'test3_winrates.json'), 'w'), indent=1)
