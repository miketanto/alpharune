B2 v2: mechanics graphs for the Azir / Diana lists' cards (2026-11-21)
====================================================================

What: hand-written B2 graphs for the 48 cards of decks/azir.txt, rengar.txt and diana.txt that had no B2
vector. They follow the v1 method (graphs_code.py; one builder per engine class, from the card's C++ code,
every node cited). The Rengar list was already fully covered.

Files
- ../v2_list.py          lists the cards of given decks without a vector -> ../v2_cards.json (key, def_id, class file)
- graphs_azir_a.py       Azir legend, units, battlefields and spells (11)
- graphs_gear_b.py       Azir and Diana gear (7), plus Stupefy, Smoke Screen, Rebuke and Flash (11 in all)
- graphs_diana_c.py      Diana legend, units, battlefields (11)
- graphs_spells_d.py     Diana spells (12)
- graphs_runes.py        Order, Chaos and Mind runes (3; rune mechanics are expanded by extract.build_class)
- ../extract_v2.py       builds them and writes ../out/vectors_v2.json (all v1 vectors unchanged + the new keys),
                         cards/<key>.json (graphs with citations) and report.json (verbs, verb check, flags)
- ../scan.py KW_MAP      + Equip, Temporary, Tank, Shield, Vision, Weaponmaster, Quick-Draw, Repeat, Empower

Checks (extract_v2.py)
- 48 built, 0 problems.
- The verb check found no engine call missing from any graph.
- No card-name words leak into the encoder input. Keyword and domain words are allowed.
- v1 vectors are bit-identical (185 entries).
- New cards are all distinct: nearest-neighbour cosine median 0.41, max 0.88.
- Pairs look sensible: Smoke Screen is nearest Eclipse (both -4 Might); B.F. Sword is nearest Brutalizer.

New vocabulary (each listed in the card's "flags"). B's encoder hashes paths, so new terms need no layout
change; they only meet other cards once those use them too.
- Verbs: Attach, Discard, CantMove, Empower, Repeat, IgnoreCost.
- Events: attack, move, hold, showdown_begin, play_spell, become_empowered.
- Zones: deck_top, trash, legend, base_or_controlled_battlefield.
- Filters / conditions: is_token, token_name, equipped_with_self, attached_this_turn, at_base, etc.

Known gaps (from the flags; none would be fixed by more vocabulary alone)
- Delayed effects ("at end of turn", "at the opponent's next hold") are written as an "after" field.
- Hwei's by-type reflexive branch is folded into one ability.
- An Equipment's own text going inactive while attached is not represented.
- Fizz still pays the Power part of the cost.

Use: RB_B2=v2 (selfplay/learner/model.py) reads vectors_v2.json; default is v1.
Switching changes how a v1-trained net sees these 48 cards: a B2 vector plus the ordinary special row,
instead of the 'other' row. Use it for a new run, or continue a checkpoint and let training adapt.
