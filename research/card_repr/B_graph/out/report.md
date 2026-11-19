# Path B: mechanics graph: build report

 _Written by the orchestrator from the builder agent's returned report: the workflow harness does not let subagents write report files._

## Method

Path B is built and every entry in the output contract is produced. The one thing missing is out/report.md: the harness refused to let this subagent write a report .md file, so its content is in this summary and the cost field, ready for the orchestrator to write.

**How the graphs are built**
- B_graph/graphs_src.py holds the hand-authored keywords and abilities for all 51 cards (about 490 lines, built with small helper functions).
- Each card was written from the card text and checked against its engine class. The two legends came from src/cards/legends/0744 and 0317.
- B_graph/build.py takes stats (type, supertype, domains, energy, power, power per domain, Might) from each printing's own fields. It writes B_graph/cards/<key>.json, the debug path lists out/paths/<key>.json, out/selfcheck.json and out/vectors.json.
- vectors.json has dim 8192 and 169 vectors: 51 card keys, 57 printing:, 51 rename:, 10 synthetic:.
- B_graph/schema.md documents the schema and the encoding.

**Schema**
- card → stats / keywords (with values) / abilities.
- Each ability has: kind (spell, triggered, static, activated, replacement, play_mode, additional_cost), timing (main, action, reaction), trigger (event, subject, condition, once), condition, cost (energy, power, power_domain, exhaust, recycle_self, optional, additional, other non-resource costs), setup_cost, where, and effects.
- Each effect has: verb (the EffectType names, plus CreateToken, Reveal, LookAt, Recall, EnterReady, ReduceCost, AllowPlayTo, CantBeChosen, AddEnergy, AddPower), target selector (type, side, zone, count, filters with neg, bind/ref), amount (n, ref, per, mult), duration, optional, and an effect-level condition.
- Keywords stay in the keyword list and are also expanded into abilities: Hidden, Ambush, Flow, Accelerate, Deathknell, Assault, Hunt, Level, Legion. Runes get exhaust and recycle activations.

**Encoding (fixed, nothing learned)**
- The graph becomes a bag of typed root-to-leaf paths. Strings become key=value. Numbers are kept as a value and also as a key=value category.
- Stats numbers stay raw. Numbers inside abilities use sign·log2(1+|v|) so a −10 does not swamp the rest.
- Every path is also emitted at all its shorter suffixes, so the same kind of change lines up across different cards.
- Paths are hashed with MD5 into 8192 signed slots, with no normalisation. The vocabulary is open: adding a card never changes existing vectors.

**Self-check (Euclidean, not the official eval)**
- Every printing and rename sits at distance 0.0 from its card key. The build checks that printing texts match once reminder text is stripped (Lonely Poro, Rengar Trophy Hunter) and that no card name appears in its graph.
- No two cards get identical vectors. Card-to-card distance runs from 2.0 to 21.0, median 12.8.
- All 10 synthetic edits land between 2.4 and 6.0 from their base.
- Real pairs:

| Pair | Distance | Partner's rank among nearest cards |
|---|---|---|
| punch_first / primal_strength | 6.8 | 1 |
| punch_first / decisive_strike | 9.5 | 3 |
| discipline / moonlight_affliction | 14.8 | 46 (cost 2 vs 7, and +2 with draw vs −10, really differ) |
| lonely_poro / loyal_poro | 4.6 | 1 |
| ferrous_forerunner / machine_evangel | 8.0 | 1 |
| sabotage / decree_of_strength | 6.3 | 1 |
| defy / hush | 10.5 | 3 |

**Hash collisions**
- 1,314 distinct paths in total, 102 of which share a slot somewhere in the vocabulary.
- Within single cards, 16 of 3,669 paths collide (0.4%).

**Where the card text and the engine disagree**
- **Zhonya's Hourglass:** the engine also heals the unit; the text does not. The graph follows the text.
- **Rengar, Trophy Hunter:** the engine only widens the Ambush location; the text allows any play to a battlefield with enemy units. The graph follows the text.
- **Emperor's Dais:** the text says "If you do, play"; the engine treats paying [1] and returning a unit as the cost. Same mechanics.
- **Seat of Power:** the text counts battlefields "you or allies" control; the engine counts only your own (1v1).
- **Hush:** has no engine class at all.
- **Darius, Trifarian:** the engine excludes tokens from "second card"; the text doesn't say so.
- **Alpha Strike:** the engine has a play-legality gate (an enemy unit at a battlefield must be choosable) that the text doesn't state.
- **Hidden reminder text:** the engine's text drops the [0] on Back Off, Pakaa Cub and Pyke. Cosmetic only.

## Could not represent / limits

- printing entries for the two legends (rengar_pridestalker, master_yi_wuju_bladesman): cards.json lists no printings for them. Their engine def_ids are unl-183-219 and ogs-019-024; no printing: vectors were emitted.
- Not expressible: the order of effects and who-deals-to-whom bindings. After encoding these are only tokens (ref=A, per=unit_killed_by_prev, source), so effect order is lost.
- Not expressible: when a choice is made (finalization vs resolution) and who makes it; only Thrill of the Hunt carries chooser=owner.
- Not expressible: Alpha Strike's split-damage division (only a flag) and play-legality gates (Alpha Strike, and Defy's need for a spell on the chain).
- Not expressible: the Darius token exclusion, and how units that die together count for Lonely/Loyal Poro.
- Not expressible: Star Spring's perspective switch, where whichever player played the unit controls the ability.
- Not expressible: rules-inherent behaviour such as 0-Might conquer and the semantics of Deflect and Ganking, which stay keyword flags with no expansion.
- Card tags (Poro, Noxus, ...) are dropped on purpose: no card in scope reads a tag. Tags on created tokens are kept.
- Condition names are atomic tokens chosen by hand. Similar conditions share nothing unless they are written as the same condition with neg (as Lonely and Loyal Poro are).

## Cost

Wall time: about 25 min end to end (22:09 to about 22:35) for one LLM agent. That covers reading the spec and about 110 KB of engine C++ across 50 card classes, and writing the schema, the graphs, the encoder, the self-check and the docs.

Hand/LLM extraction: about 6-8 min of generation for all 51 cards, roughly 8 s per card plus about 4 s per card to read its engine class. Doing it by hand would take an expert about 3-5 min per card (3-4 h for 51), plus about 1 h of schema design.

Compute: build.py runs in 1.5 s on one CPU thread (Python and numpy). No GPU, no model, no downloads. Output is a 4.2 MB dense vectors.json; a sparse format would be about 70 KB.

Marginal cost of one new card: 5-15 lines in graphs_src.py. That is about 10-20 s of LLM generation plus 2-3 min of human review, or 3-5 min by hand, then a 1.5 s rebuild. A new verb, filter or condition needs no code change, and existing vectors never change.

Marginal cost of one new set (200-300 cards): about 1-1.5 h of LLM generation plus spot review, or 15-25 h by hand. Each new keyword also needs about 10 min for its expansion helper, and genuinely new mechanics (e.g. Empower, Burn) need a schema review. No retraining, and existing vectors are unaffected.

## Files

- `B_graph\graphs_src.py`
- `B_graph\build.py`
- `B_graph\schema.md`
- `B_graph\cards`
- `B_graph\out\vectors.json`
- `B_graph\out\selfcheck.json`
- `B_graph\out\paths`
