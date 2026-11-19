# Path B2: mechanics graph from card code: build report

_Written by the orchestrator from the builder agent's returned report, after checking its main claims against the files._

## Result in one line
B2 scores almost exactly like B: 43 of 51 card graphs are identical to B's text graphs. Its value is **cited provenance** (every node points at a `file:line`) and the **text-vs-engine disagreement list**, not a different vector.

## Method
- `scan.py`: reads each class's `CardDef` (stats, keywords, line numbers) and every call in the class body; comments, string literals and trace lines are removed first so names cannot leak.
- `graphs_code.py`: one hand-written graph function per engine class (Lonely Poro 0359 and 0777 separately); every node has a `_src` citation; plumbing (resume, finalize markers, trace, fallback picks) is excluded.
- Engine-implemented keywords (Hidden, Ambush, Accelerate, Assault, Flow, runes) are expanded automatically, citing the engine code.
- `extract.py`: assembles graphs, strips `_` keys before encoding (B's encoder, unchanged), applies the synthetic edits, writes `vectors.json`, and checks that the verbs implied by each class's calls appear in its graph.
- Citations: 467 of 468 nodes (the gap is Hush's text-derived keyword).

## Vocabulary (`vocab.json`)
All 340 call sites mapped: 56 effect calls (20 functions) → verbs, 53 ability-shaping calls (selector, condition, cost, destination, chooser, "may"), 54 plumbing, 177 class hooks; 11 trigger types; 5 target-requirement fields; 13 keywords + runes. Schema extensions: `Heal` verb, `play_requires` (legality gate), `is_token` condition, token tags from code.

## Confidence
- High (direct executor calls, literal arguments): 37 cards.
- Medium (a condition or selector named from a code predicate): alpha_strike, darius, decree_of_focus, fiora, not_so_fast, star_spring, zhonya, emperor_s_dais, rampage, thrill_of_the_hunt, scuttle_crab, sabotage, decree_of_strength.
- Low: Hush (no engine class; B's text graph, flagged).
- Printings without their own class reuse the card's class (rune reprints, ven-179-166): equal by construction. The real T1 measurement is Lonely Poro's two classes: identical graphs.

## Text vs engine (`text_vs_engine.json`, 13 entries, both sides cited)
- ~~Major: **Zhonya's Hourglass**: the engine also heals the unit and has no "next time" limit.~~ **Retracted by the orchestrator: not a disagreement.** The engine's CardDef carries the current (erratum) text: "If a friendly unit would die, kill this instead. Heal that unit, exhaust it, and recall it." The engine matches it exactly; the comparison used the pre-erratum wording in `cards/card_index.json`. ~~**Rengar, Trophy Hunter**: the engine widens where he can Ambush to, but does not allow any play to a battlefield with enemy units.~~ **Retracted by the orchestrator: not a disagreement.** The comparison used the pre-erratum gallery text in `cards/card_index.json`. The current text (`errata/unleased.txt`: "I can [Ambush] to a battlefield where there are enemy units, even if you don't have units there") and CR 822.1.d (which uses this card as its example) match the engine (`ambushToEnemyBattlefields()`), and testing finding D37 made the Ambush location available at every timing, so he can be played to an enemy-only battlefield as a Reaction or in the main phase. Lesson: `card_index.json` is stale where errata exist; text-based paths (A, B) used the old wording for this card.
- Minor: Rampage (the spell deals the damage, not the units); Darius (tokens don't count as your second card); Alpha Strike (needs a choosable enemy unit to be played); Emperor's Dais (token tag Shurima in code); Thrill of the Hunt (a banished token is not played back); Fiora (doubling Might is a one-time snapshot); Defy (compares printed cost, ignoring modifiers); Wuju legend ("alone" = exactly one friendly defender on the board); Seat of Power ("allies" irrelevant in 1v1).
- Cards whose B2 vector differs from B (cos): star_spring 0.197 (condition rename only), rampage 0.173, rengar_trophy_hunter 0.162, thrill_of_the_hunt 0.078, zhonya 0.061, alpha_strike 0.039, emperor_s_dais 0.032, darius 0.029.

## Scores (eval.py; `results_B2_code.json`)

| | B2_code | B_graph |
|---|---|---|
| T1 printings / renames max cos | 0 / 0 | 0 / 0 |
| T2 real pairs distinct, median rank | 7/7, 1 | 7/7, 1 |
| T2 synthetic, all base rank 1, rel distance | 0.047–0.26 | 0.048–0.26 |
| T2 direction within-field cos / p | 0.227 / 0.069 | 0.227 / 0.069 |
| T3 beat baseline, non-prov / prov | 8/16, 5/17 | 7/16, 5/17 |
| Mean numeric R² | 0.33 | 0.32 |

## Cost
Compute: extract 1.1 s, eval ~80 s. Agent wall time ~50 min (mostly reading ~3,500 lines of card C++). By hand: ~4–6 h (vocabulary ~1.5 h, then ~4–6 min per class). New card: 5–15 graph lines + a class link; a new executor method needs one vocabulary entry.

## Limits
Graphs are still hand-written from the code; automatic parts are stats, keywords, keyword expansions, call coverage and the verb-consistency check. Not expressible: effect order, Alpha Strike's damage split, Fiora's snapshot, chain timing.
