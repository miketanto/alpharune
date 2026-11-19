"""Variant deck lists: every copy of the base card replaced by its partner /
synthetic variant (names as registered by decision_export --variant; the
synthetic classes are named "Synthetic <tag>" by card_probe.cpp's derive())."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'decks')
BASE_NAME = {'punch_first': 'Punch First', 'discipline': 'Discipline', 'lonely_poro': 'Lonely Poro',
             'defy': 'Defy', 'sabotage': 'Sabotage', 'pit_rookie': 'Pit Rookie'}
VARIANTS = [  # (variant name, base key, deck-list name of the variant)
    ('primal_strength', 'punch_first', 'Primal Strength'),
    ('synthetic:punch_first__amount4', 'punch_first', 'Synthetic g'),
    ('synthetic:punch_first__energy3', 'punch_first', 'Synthetic h'),
    ('synthetic:punch_first__power0', 'punch_first', 'Synthetic i'),
    ('synthetic:punch_first__reaction', 'punch_first', 'Synthetic j'),
    ('synthetic:discipline__draw2', 'discipline', 'Synthetic k'),
    ('synthetic:discipline__enemy', 'discipline', 'Synthetic l'),
    ('synthetic:lonely_poro__draw2', 'lonely_poro', 'Synthetic m'),
    ('synthetic:defy__cost2', 'defy', 'Synthetic n'),
    ('synthetic:sabotage__unit', 'sabotage', 'Synthetic o'),
    ('synthetic:pit_rookie__might3', 'pit_rookie', 'Synthetic p'),
]
DECKS = {'rengar': 'decks/rengar.txt', 'yi_rebuild': 'decks/yi_rebuild.txt'}

def short(v):
    return v.replace('synthetic:', '')

def main():
    os.makedirs(OUT, exist_ok=True)
    made = {}
    for v, base, vname in VARIANTS:
        for dk, path in DECKS.items():
            lines = open(os.path.join(ROOT, path), encoding='utf-8').read().split('\n')
            out, n, section = [], 0, ''
            for ln in lines:
                s = ln.strip()
                if s.endswith(':'):
                    section = s
                parts = s.split(' ', 1)
                if section != 'Sideboard:' and len(parts) == 2 and parts[0].isdigit() and parts[1] == BASE_NAME[base]:
                    out.append(f'{parts[0]} {vname}')
                    n += int(parts[0])
                else:
                    out.append(ln)
            if n == 0:
                continue
            rel = f'research/card_repr/smoke/data/decks/{dk}__{short(v)}.txt'
            open(os.path.join(ROOT, rel), 'w', encoding='utf-8', newline='\n').write('\n'.join(out))
            made[(v, dk)] = (rel, n)
            print(v, dk, rel, 'copies replaced:', n)
    return made

if __name__ == '__main__':
    main()
