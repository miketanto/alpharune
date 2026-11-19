"""Mixed-variant Rengar decks for the card experiment (CARD_EXPERIMENT.md). Same substitution
logic as smoke/data/src/make_decks.py (every main-deck copy of a base card replaced by its
variant; variant deck-list names are 'Synthetic <tag>' from card_probe.cpp derive())."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
CR = os.path.normpath(os.path.join(HERE, '..', '..'))
ROOT = os.path.normpath(os.path.join(CR, '..', '..'))
OUT = os.path.join(CR, 'smoke', 'data', 'decks')
BASE_NAME = {'punch_first': 'Punch First', 'sabotage': 'Sabotage', 'pit_rookie': 'Pit Rookie'}
TAG = {'punch_first__ctrl_scores': 'q11', 'punch_first__opp_of_ctrl_scores': 'q12',
       'pit_rookie__score_alone': 'q5', 'sabotage__score_opphand4': 'q7', 'sabotage__score1': 'q6'}
DECKS = {  # name -> {base: variant}
    'exp_train': {'punch_first': 'punch_first__ctrl_scores', 'sabotage': 'sabotage__score_opphand4',
                  'pit_rookie': 'pit_rookie__score_alone'},
    'exp_heldout': {'punch_first': 'punch_first__opp_of_ctrl_scores', 'sabotage': 'sabotage__score_opphand4',
                    'pit_rookie': 'pit_rookie__score_alone'},
    'exp_both': {'punch_first': 'punch_first__ctrl_scores', 'sabotage': 'punch_first__opp_of_ctrl_scores',
                 'pit_rookie': 'pit_rookie__score_alone'},   # fix check: BOTH target-side cards trained
    'exp_heldout_sab1': {'punch_first': 'punch_first__ctrl_scores', 'sabotage': 'sabotage__score1',
                         'pit_rookie': 'pit_rookie__score_alone'},
}
VARIANTS = {k: sorted('synthetic:' + v for v in d.values()) for k, d in DECKS.items()}
ALL_VARIANTS = sorted({v for vs in VARIANTS.values() for v in vs})


def make():
    lines = open(os.path.join(ROOT, 'decks', 'rengar.txt'), encoding='utf-8').read().split('\n')
    paths = {}
    for name, sub in DECKS.items():
        out, section, n = [], '', 0
        for ln in lines:
            s = ln.strip()
            if s.endswith(':'):
                section = s
            parts = s.split(' ', 1)
            base = next((b for b, nm in BASE_NAME.items() if len(parts) == 2 and parts[1] == nm), None)
            if section != 'Sideboard:' and base in sub and parts[0].isdigit():
                out.append(f'{parts[0]} Synthetic {TAG[sub[base]]}')
                n += int(parts[0])
            else:
                out.append(ln)
        p = os.path.join(OUT, f'rengar__{name}.txt')
        open(p, 'w', encoding='utf-8', newline='\n').write('\n'.join(out))
        paths[name] = p
        print(name, p, 'copies replaced:', n)
    return paths


if __name__ == '__main__':
    make()
