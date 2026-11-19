"""Tables for CARD_EXPERIMENT.md from runs/cx_*/{probes.json, eval_*.json}. Prints Markdown."""
import json, math, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'runs')
REPRS, SEEDS = ['B2', 'id', 'random'], [0, 1, 2]


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def wilson(w, g, z=1.96):
    if not g:
        return (float('nan'), float('nan'))
    p = w / g; d = 1 + z * z / g; c = p + z * z / (2 * g)
    h = z * math.sqrt(p * (1 - p) / g + z * z / (4 * g * g))
    return ((c - h) / d, (c + h) / d)


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return '-'
    if len(xs) == 1:
        return f'{xs[0]:.2f}'
    return f'{st.mean(xs):.2f} ({", ".join(f"{x:.2f}" for x in xs)})'


def probe(run, name, key):
    d = load(os.path.join(RUNS, run, 'probes.json')) if not run.endswith('.json') else load(os.path.join(RUNS, run))
    if not d or name not in d['probes']:
        return None
    return d['probes'][name].get(key)


ROWS = [('ctrl_tgt', 'mass_right', 'ctrl_scores target: mass on friendly'),
        ('ctrl_tgt', 'argmax_right', 'ctrl_scores target: top = friendly'),
        ('opphand4', 'play_mass_cond_true', 'opphand4: P(play) when opp hand >= 4'),
        ('opphand4', 'play_mass_cond_false', 'opphand4: P(play) when opp hand < 4'),
        ('alone', 'play_mass_cond_true', 'score_alone: P(play) when no own units'),
        ('alone', 'play_mass_cond_false', 'score_alone: P(play) when own units'),
        ('oppctrl_tgt', 'mass_right', 'HELD-OUT opp_of_ctrl target: mass on enemy'),
        ('oppctrl_tgt', 'argmax_right', 'HELD-OUT opp_of_ctrl target: top = enemy'),
        ('sab1', 'play_mass_cond_true', 'HELD-OUT sabotage__score1: P(play)'),
        ('sab1', 'argmax_right', 'HELD-OUT sabotage__score1: top = play')]


def probe_table(suffix=''):
    out = ['| probe metric | ' + ' | '.join(f'{r} mean (per seed)' for r in REPRS) + ' | untrained B2 / id / random | base-deck id run |',
           '|---|' + '---|' * (len(REPRS) + 2)]
    for name, key, label in ROWS:
        cells = [ms([probe(f'cx_{r}_s{s}{suffix}', name, key) for s in SEEDS]) for r in REPRS]
        un = ' / '.join(f'{probe(f"cx_untrained_{r}.probes.json", name, key) or float("nan"):.2f}' for r in REPRS)
        bd = probe('cx_basedeck_id.probes.json', name, key)
        out.append(f'| {label} | ' + ' | '.join(cells) + f' | {un} | {bd if bd is None else round(bd, 2)} |')
    return '\n'.join(out)


def wr_table(fname, suffix=''):
    out = ['| repr | ' + ' | '.join(f'seed {s}' for s in SEEDS) + ' | pooled (Wilson 95%) |', '|---|' + '---|' * (len(SEEDS) + 1)]
    for r in REPRS:
        cells, W, G = [], 0, 0
        for s in SEEDS:
            d = load(os.path.join(RUNS, f'cx_{r}_s{s}{suffix}', fname))
            if d:
                cells.append(f"{d['win_rate']:.2f} ({d['wins']}/{d['games']})")
                W += d['wins']; G += d['games']
            else:
                cells.append('-')
        lo, hi = wilson(W, G)
        out.append(f'| {r} | ' + ' | '.join(cells) + (f' | {W / G:.2f} [{lo:.2f}, {hi:.2f}] n={G} |' if G else ' | - |'))
    return '\n'.join(out)


if __name__ == '__main__':
    print('### Probes after training on exp_train\n'); print(probe_table())
    print('\n### Probes after the 2-minute fine-tune on exp_heldout\n'); print(probe_table('_ft'))
    print('\n### (a) Win rate vs ISMCTS-100, exp_train mirror\n'); print(wr_table('eval_train.json'))
    print('\n### (c) Zero-shot win rate vs ISMCTS-100, exp_heldout mirror\n'); print(wr_table('eval_heldout.json'))
    print('\n### (d) After 2-minute fine-tune, exp_heldout mirror\n'); print(wr_table('eval_heldout.json', '_ft'))


def ft10_curve():
    """oppctrl_tgt mass on enemy (and sab1 P(play)) vs fine-tune updates, from metrics.jsonl."""
    rows = {}
    for r in REPRS:
        p = os.path.join(RUNS, f'cx_{r}_s0_ft10', 'metrics.jsonl')
        if not os.path.exists(p):
            continue
        for ln in open(p):
            d = json.loads(ln)
            if 'probes' in d and 'oppctrl_tgt' in d['probes']:
                rows.setdefault(d['update'], {})[r] = (d['probes']['oppctrl_tgt']['mass_right'],
                                                       d['probes']['sab1'].get('play_mass_cond_true'), d['wall_s'])
    out = ['| fine-tune update | ' + ' | '.join(f'{r}: enemy mass / sab1 P(play)' for r in REPRS) + ' |',
           '|---|' + '---|' * len(REPRS)]
    for u in sorted(rows):
        out.append(f'| {u} | ' + ' | '.join(f'{rows[u][r][0]:.2f} / {rows[u][r][1]:.2f}' if r in rows[u] else '-' for r in REPRS) + ' |')
    return '\n'.join(out)


if __name__ == '__main__':
    print('\n### (d2) 10-minute fine-tune on exp_heldout from the seed-0 models\n'); print(ft10_curve())
