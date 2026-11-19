"""In-scope card keys (Rengar main deck + legend/champion + token units) and battlefields, as vocab indices."""
import numpy as np
from rbenv import load_vocab
_vi = {k: i for i, k in enumerate(load_vocab())}
DECK_CARDS = ['rengar_pridestalker', 'rengar_trophy_hunter', 'noxus_hopeful', 'pit_rookie', 'sabotage', 'punch_first',
              'inferna', 'grim_apothecary', 'kinkou_initiate', 'irresistible_faefolk', 'first_mate', 'thrill_of_the_hunt',
              'rampage', 'pakaa_cub', 'ferrous_forerunner', 'kai_sa_survivor', 'nidalee_cat_form', 'darius_trifarian',
              'pyke_dockside_butcher', 'token:mech', 'token:sand_soldier']
BFS = ['emperor_s_dais', 'seat_of_power', 'star_spring']
SC_CARD = np.array([_vi[k] for k in DECK_CARDS]); SC_BF = np.array([_vi[k] for k in BFS])
