# Where card representation can make training invalid: failure catalogue and test plan (2026-11-19)

**Principle.** A training result says something about card understanding only if the model *could* have understood. A card reaches the network in four ways:
- (1) its printed identity R(card) (B2 / id / random);
- (2) the live fields of its token;
- (3) how actions refer to it;
- (4) global features.

A gap in any of the four makes a failure look like a learning failure; a later gap can be hidden by luck. So we first prove the pipeline can carry each fact (gates 0–3), and only then measure understanding (gates 4–5).

Ground truth used below: the Rengar deck (`decks/rengar.txt`) and the token layout in `smoke/data/SCHEMA.txt`, which `selfplay/rbenv` reproduces: F=17 = kind, key + 15 situational fields. Today a card token carries:
- controller, owner;
- zone, location;
- exhausted, stunned;
- damage, Might, buffs, temp Might;
- is-token, facedown;
- combat role;
- attachments.

**It carries no keywords, no counters, and no cost.**

## A. Observation gaps: the fact is not in the input at all

| # | Gap | Today | Rengar-deck cases | Fix |
|---|---|---|---|---|
| A1 | **Trash contents** (both trashes are public) | sizes only | Flow cards castable from the trash (Brittle Steel, sideboard; Yi deck); what the opponent has already used (closed-decklist inference) | trash objects as tokens (zone=4 already exists in the schema) |
| A2 | **Banishment contents** | sizes only | Thrill of the Hunt banishes and replays; "banished with" links | banishment tokens |
| A3 | **Live keywords and values** | absent: only the printed set via R(card) | Inferna [Assault 2] active only while attacking; granted or lost keywords ("this turn" grants, auras); text inactive on attached gear (CR 718.2); [Legion] active or not (Noxus Hopeful); Empowered; Level | per-token multi-hot of *effective* keywords + values (Assault / Shield / Deflect N) + a "rules text inactive" flag |
| A4 | **Per-object memory** | absent | "first time each turn" stamps (Star Spring), the Rampage paid flag, Pyke's counters, pending delayed abilities ("at end of turn"), tracked "banished with" objects | flags for each memory kind the cards in scope read |
| A5 | **Turn event counts used by conditions** | only cards played this turn (g69/70) | Darius (2nd card played ✓), Kinkou (derivable ✓); missing: units died, moves, draws, power spent this turn | add every count any in-scope card's condition reads (see gate 0) |
| A6 | **Effective cost now** | only legality | Noxus Hopeful Legion −2, cost modifiers, Accelerate +[1][R]: needed to plan "play now or next turn" | per hand card: effective Energy / Power cost and "affordable now" |
| A7 | **Consequences of a move** | implicit | moving into an enemy-held battlefield = combat as attacker; Faefolk pulls an enemy unit; Ganking moves between battlefields | action fields: destination controller, "starts combat as attacker", "contests" |
| A8 | **Known cards in the opponent's hand** | absent; a revealed card appears only while an action names it | Sabotage / Decree reveal the hand, then forget it | tokens for known opponent hand cards (knowledge tracker), zone=hand, owner=opp |
| A9 | **Deck knowledge** | sizes only | own remaining deck composition (known); opponent pool under open vs closed decklist | remaining-deck multiset per player, with closed decklist = the deck model's pool (the user's later experiment) |
| A10 | Runes | counts only | low impact while auto-pay handles payment | none for now |

## B. Identity failures: the object behaves unlike its printed card

| # | Failure | Rengar-deck cases | Fix / test |
|---|---|---|---|
| B1 | **Token units with no card vector** | `token:mech` (Ferrous Forerunner's Deathknell), `token:sand_soldier` (Emperor's Dais): no cards.json key, so a zero / OOV vector | B2 graphs for token definitions |
| B2 | Copies, `becomeCard`, control change | owner ≠ controller is visible ✓; copy effects take the copied card's R | the token's key = what it currently *is*; test with a copy card if one enters scope |
| B3 | Battlefield replaced by a token battlefield | Dais / replacement effects | battlefield token key follows the slot's current card |
| B4 | Reprints / printings | T1 passes (same vector) | — |

| B5 | **Cross-card modifiers**: one card changes what *other specific cards* are or do (e.g. Azir giving Sand Soldiers [Weaponmaster]; "Your Mechs have [Shield]"; cost reductions for a tag) | not in the Rengar main deck as far as known; Emperor's Dais makes Sand Soldiers | the *result* must show on the affected token (A3, effective keywords); the *source's* R must encode "affects others with tag X: grants Y", so the model can value the source before the effect exists; the affected token must carry the tags the selector reads |

**Scope for now (user, 2026-11-19): make it work for the Rengar deck first.** B5 and Flow (A1) are listed so they aren't forgotten, but are tested only once the Rengar deck passes.

## C. Representation-content failures: R(card) itself

| # | Failure | Seen? | Test |
|---|---|---|---|
| C1 | **A decisive small difference drowned** in the vector | **yes**: target side = 8 of 8,192 dims; ctrl vs opp_ctrl 7× closer than typical pairs after the projection | for each schema field, a linear probe on the *model's input layer* must recover it; distances after the projection |
| C2 | **Structure lost** in the bag of paths: which condition gates which effect, effect order ("then"), duration ("this turn"), who chooses, trigger vs static vs activated | not tested | synthetic pairs differing only in structure must get separable vectors (e.g. Rampage's paid-vs-unpaid branch) |
| C3 | **Numbers** compared as numbers (cost 2 vs 5, Might 3 vs 8, Assault 2) | partly (numbers kept in B / B2 but hashed) | probe monotonic recovery of each number |
| C4 | **Mode / value alignment**: an action's `v` (mode index, X) means different things per card | Thrill of the Hunt (pickMode) | R's mode list in engine option order; test that a model can tell modes apart |
| C5 | **Text vs engine disagreement** | found: Rampage, Darius, Thrill, Fiora, Alpha Strike, Defy | R follows the engine (B2); keep the disagreement list current |

## D. Learning and wiring failures that look like representation failures

| # | Failure | Seen? | Test |
|---|---|---|---|
| D1 | **A choice head ignores the asking card** | **yes** (target prompt; fixed by the ask head) | for *every* prompt kind (target, mode, X, location, may, accelerate, move-join, …): swapping the asking card must change the logits (gradient path exists), checked on an untrained and a briefly trained net |
| D2 | **Avoidance collapse**: the policy stops playing a card it misuses and never learns its use | **yes** (fine-tunes dropped opp_ctrl to 0–1% play) | forced-play evaluation; an entropy floor or exploration on cards under test |
| D3 | **Rare card / rare context**: no gradient | likely (Shield-while-defending, Deathknell, variants appear only via deck swaps) | count exposures per (card, context) in training data; require a minimum before testing |
| D4 | Frozen vectors collapse in the projection | yes (C1) | as C1 |

## E. Evaluation failures

- **E1** Probe states taken from random play are off the policy's distribution: use states from the trained policy's own games.
- **E2** Hand-defined right answers don't scale to real mechanics: use counterfactual playouts as ground truth (gate 4).
- **E3** Seat effects: evaluate both seats.

## Gates (in order; a later result is valid only if the earlier gates pass)

0. **Observation audit (no training).**
   - List every fact the in-scope cards' rules and conditions read; take it from the B2 vocabulary (selectors, conditions, counts, durations) plus the keywords in the Rengar deck.
   - Mark each fact as observable / derivable / missing.
   - Output: the gap table, filled in from code rather than by hand.
1. **Observation fixes.**
   - A1–A8 and B1 for in-scope cards: trash and banishment tokens, effective keywords and values, the inactive flag, memory flags, event counts, effective cost, known opponent hand cards, token-unit vectors.
   - Re-run rbenv's determinism and agreement tests.
   - This changes the input format, so every earlier checkpoint is invalidated.
2. **Representation unit tests (no training).** For C1–C4, every field and structure difference in scope is recoverable from the model's input layer for B2. id and random fail by construction; that's the control.
3. **Wiring tests.** For D1 on every prompt kind: the asking card and each relevant token field can move the decision (non-zero sensitivity) for an untrained network and after 5 minutes of training.
4. **Context understanding (trained).**
   - Matrix of mechanic × context with counterfactual edits on real policy states:
     - toggle the mechanic: Shield on/off, Assault on/off, Hidden vs revealed, the Flow card in the trash vs not;
     - toggle the context: the same unit at a battlefield vs in base, attacking vs defending, the condition true vs false.
   - **Ground truth:** win-rate difference from N playouts of each edited state with the trained policy.
   - **Pass:** the model's value and policy shifts agree in sign, and roughly in size, with the playout difference where it is large, and stay near zero where it is near zero.
   - Rengar mechanics: Ambush, Assault, Hidden, Accelerate, Legion, Deathknell, Ganking, Reaction, conditional triggers (Darius, Kinkou, Star Spring), Sabotage's hand reveal. Flow needs Brittle Steel or a Yi-deck card (not in the Rengar main deck).
5. **Generalisation.**
   - Held-out *real-mechanic* cards built from mechanics seen in training, combined in new ways.
   - Compare B2 vs id vs random, 3 seeds: zero-shot use (gate-4 metrics) and games to learn.
   - Train on contrasts so that reading each field is rewarded.

## Open decisions

- Which in-scope card set for gates 0–4: the Rengar main deck plus the sideboard (adds Flow via Brittle Steel, and gear)?
- Do the observation fixes (gate 1) also go into `decision_export` (the teacher-data path), or only into rbenv?
- Gate-4 playout budget per edited state (cost vs noise).
