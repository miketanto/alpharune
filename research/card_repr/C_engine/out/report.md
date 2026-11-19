# Path C: engine behaviour probes: build report

 _Written by the orchestrator from the builder agent's returned report: the workflow harness does not let subagents write report files._

## Method

Path C is built and run for all 51 card keys, but (report.md was written afterwards by the orchestrator): the harness refused my Write call for that .md file ("Subagents should return findings as text"). Its content (method, coverage, cost) is in this summary and the fields below instead.

The probe tool is research/card_repr/C_engine/card_probe.cpp, a C++ target outside the default build, built in the WSL copy. For each card it builds fixed positions and drives the card through the real engine path: executeIntent, then runChain + cleanup + runChain; combat, holding a battlefield and kills use runCombat, scoreHold and killObject.

Each block records the delta of 13 fixed role objects (my units, enemy units, enemy gear, my and the opponent's hand cards, the card itself), per-side zone and board counts, XP, score, battlefield control and event counts.
- **Legality:** what the engine offers for the card (play with no location, to base, to my battlefield, to the enemy battlefield, hide, activate) in 8 windows: empty board, standard board, after a card was played this turn, showdown with my focus, two opponent spells on the chain (Fury 3 energy / 1 power, and 4 energy / 2 power), from my trash (Flow), and facedown.
- **Execution:** each legal action is run under two fixed choice policies. FIRST takes the first option and declines every "may"; LAST takes the last option and accepts optional costs.
- **Triggers:** 11 event probes with the card already in play (die alone / with a friend, a friendly unit dies, real combats, hold, play two units, moves, activate).
- **Static and runes:** 3 with/without-card comparisons, plus which costs a single rune can pay.

Costs are the rune-pool delta of a pre-funded pool, so what was actually paid by domain is recorded, including Accelerate, Pyke's [R], Rampage's [O], Hide, Flow and the Legion discount.

Fixed baselines are subtracted: passing instead of playing in the same window, and a vanilla 3-Might unit (or nothing in play) for trigger probes. Raw blocks are kept in out/raw.json.

Results:
- **Errors:** the engine raised no exception in any probe.
- **Determinism:** runs are fully deterministic, so the T1 noise floor is 0. Every rename is at distance 0, and the two Lonely Poro engine classes give identical vectors.
- **Selfchecks:** each synthetic variant copies the engine card's code with one parameter exposed. With the original parameters, all six copies reproduce the real class's vector exactly (distance 0).
- **Minimal pairs:** every real and synthetic pair is above 0. Euclidean distances range from 4.2 (sabotage vs decree_of_strength; punch_first amount 4) to 43.1 (discipline vs moonlight_affliction).

out/vectors.json follows the contract: dim 32012, 169 entries (51 keys, 57 printings, 51 renames, 10 synthetic variants). Only 1422 of the dims are nonzero for any entry.

## Could not represent / limits

- Hush (rad-041-167): the engine has no class for it (Radiance set), so its key, printing and rename vectors come from a test-only stand-in in card_probe.cpp: its printed stats from cards.json (1 energy, 2 power, no domain, [Reaction]), Defy's code shape and Hush's own counter condition. It is labelled standin:hush in raw.json
- Printings with no engine class of their own (rad-r04, ven-r04, rad-r02, ven-r02, rad-r01, ven-r01, ven-179-166) reuse their card's class vector, so for them T1 is satisfied trivially, not measured
- No card in scope has an activated ability: the activate slots and the activate_first_ability probe are all zero
- Noxian Demolitionist's conquer-kill-gear: with 1 Might it never conquers in the attack probe, and the enemy gear costs 2. It looks like a vanilla 1-Might Body unit
- Alpha Strike's 'gain 1 XP per kill': its split damage never kills in the board probe, so only the damage is seen
- Scuttle Crab's 'look at facedown cards': no knowledge state is recorded; its draw, reveal and XP are seen
- Seen only as keyword bits or an engine flag in the static block, not as behaviour: Master Yi Tempered's Level-6 [Deflect]/[Ganking], Pyke's [Ganking], Ruin Runner's 'can't be chosen by enemy spells'. There is no probe where the opponent targets my card and no battlefield-to-battlefield move
- En Garde's not-alone branch: the target is alone under both policies
- Battlefields and legends have no play legality (they are not played); they are represented by trigger and static probes only
- Not explored: choices other than the first and last option, the card controlled by the opponent, multi-turn or end-of-turn expiry, and deck-construction facts (Signature cards, champion and legend tags)
- Rename replaces the name in the card table and on every object; the card class's own def() string is a const member and is left as is (the engine reads names only for logs)

## Cost

Wall time: about 20 minutes, 22:10 to 22:30, from reading the spec to the final vectors; most of it was engine reading, probe design and writing the tool, then 6 build/run/inspect iterations. Compute (WSL, Ninja, -j12, riftbound_core already built in that copy): the incremental build of card_probe (one compile unit plus relinking riftbound_core) took 25 s. A full run over 125 subjects (51 cards plus their renames, 6 selfchecks, 10 synthetic variants, the Hush stand-in) took 0.96 s wall and 18 MB of memory, about 8 ms per subject. Assembling the vectors in numpy takes about 3 s. A cold engine build would be several CPU-minutes but was not needed. Manual effort: one-time probe and feature design and about 640 lines of probe code (1266 with comments). Each synthetic variant needs a hand-written copy of the card's C++ with one parameter exposed (10-40 lines, about 5 minutes), because on this path a variant must be implemented, not described; Hush needed a hand-written stand-in for the same reason. One new card the engine already implements: no manual work. make_subjects.py maps printings to classes automatically, rerunning costs about 10 ms per card, and existing vectors do not change. One new card the engine does not implement: someone must first write its engine class, the normal engine work of roughly 30-150 lines plus tests. A card with a mechanic no probe exercises needs a new probe or position; that changes the vector layout and dim, so every vector is recomputed (about 1 s). One new set: dominated by implementing the set in the engine (hundreds of card classes); probing after that takes seconds, plus likely a few new windows or roles for new mechanics, each a layout change.

## Files

- `C_engine\card_probe.cpp`
- `C_engine\card_probe.cmake`
- `C_engine\make_subjects.py`
- `C_engine\assemble.py`
- `C_engine\inspect.py`
- `C_engine\build_only.sh`
- `C_engine\run_only.sh`
- `C_engine\out\vectors.json`
- `C_engine\out\raw.json`
- `C_engine\out\raw_probe.json`
- `C_engine\out\assemble_log.txt`
- `C_engine\out\subjects.txt`
- `C_engine\out\printing_map.json`
- `C_engine\out\probe_stderr.log`

## Phase 2 changes (PHASE2.md section 2)

### Fixes
- **New block `static/card` (first in the layout).** It holds the printed facts from the engine's `CardDef`:
  - energy and power cost, domain one-hot, number of domains
  - Might, might_bonus, Assault, Shield and Deflect values
  - card type and super type one-hots
  - every keyword bit
  - timing (action, reaction, main-only)
  - number of activated abilities

  It is never zero, so a card that is legal in no probe keeps its cost. `synthetic:defy__cost2` is no longer a zero vector, and `assemble.py` now asserts that no entry is all zero.
- **The LAST policy runs the last offered intent of a slot; FIRST runs the first.** Phase 1 ran the first intent under both policies, so a card whose targets are intent targets (En Garde) saw one target only.
- **Perspective global `g_me`.** It is P1 in pool H. It lets the same features be written for whichever player decides in pool S.
- **Battlefield features are guarded** for a state with fewer than 2 battlefields.
- **New side features `runes` and `runes_exhausted`** (for me and for the opponent). In real games costs are paid from runes, not from a funded pool.

### New H probes (coverage gaps from RESULTS.md)
- `board_weak_enemies` (play window): the enemy units have 1 Might. Alpha Strike now kills and gains XP (+1 under FIRST, +2 under LAST).
- `board_friends_together` (play window): F1 and F2 together at bf0. This is En Garde's not-alone branch (+1 instead of +2).
- `trigger/attack_conquer`: the card attacks beside a 5-Might friend against a lone 0-Might enemy. The enemy has a token gear (cost 0) and Garbage Grabber.
  - Noxian Demolitionist now conquers and, under LAST, kills the token gear.
  - Seat of Power draws. Emperor's Dais returns a unit and makes a Sand Soldier. Master Yi, Tempered gains 2 XP.
- `trigger/move_bf0_to_bf1` and `_xp6`: a battlefield-to-battlefield move.
  - Pyke ([Ganking]) moves.
  - Master Yi, Tempered moves only with 6 XP ([Level 6]).
- `targeted/opp_spell` and `_xp6`: the opponent, in its main phase, holds "Deal 4 to a unit at a battlefield". The card is a unit of mine at bf0. The block records:
  - whether the card is offered as a target, and how many options there are
  - what the opponent paid
  - damage to the card and whether it died

  Results: Ruin Runner is never offered. Master Yi, Tempered at 6 XP costs the opponent one extra power ([Deflect]).
- `inplay_legal/board` and `_board_xp6`: the in-play legality of every permanent, battlefields and legends included. The counts are:
  - the card's activations
  - its own Standard Moves, by destination
  - unit plays to its slot, moves to its slot, and moves from its slot to base

  The three battlefields and the two legends block nothing, so their block equals the plain board's (measured, not assumed).

### Result
- `out/vectors.json` keeps the same contract: 169 entries, dim 40635 (was 32012).
- Rename and selfcheck distances are all 0, and the two Lonely Poro classes are identical.
- No engine errors. The run takes 1.3 s for 125 subjects.
- `synthetic:defy__cost2` vs `defy`: Euclidean distance 9.22. It differs in the static block (cost) and in legality.
