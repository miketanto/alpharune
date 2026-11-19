"""Markdown tables from runs/cx_real_{train,heldout,sab1}.json (real_stats.py)."""
import json, os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs')
def f(a, b): return f'{a / b:.2f} ({a}/{b})' if b else '-'
for fn, title in (('cx_real_train.json', 'exp_train mirror'), ('cx_real_heldout.json', 'exp_heldout mirror'), ('cx_real_sab1.json', 'exp_heldout_sab1 mirror')):
    d = json.load(open(os.path.join(R, fn)))
    print(f'\n#### {title} (policy on both seats, 200 games per run)\n')
    print('| run | card | offered / held (affordable) | P(play) when offered | cond true: P(play) | cond false: P(play) | target prompts (2-sided) | enemy chosen |')
    print('|---|---|---|---|---|---|---|---|')
    for run, cards in d.items():
        for c, s in cards.items():
            print(f"| {run[3:]} | {c.split('__')[1]} | {f(s['offered'], s['held'])} | {f(s['played'], s['offered'])} | "
                  f"{f(s['played_c'], s['offered_c'])} | {f(s['played_nc'], s['offered_nc'])} | {s['tgt2']} | {f(s['enemy'], s['tgt2'])} |")
