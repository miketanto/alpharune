# Card representation, phase 1: results (A vs B vs C)

The scorer is `eval/eval.py`, unchanged. It was run with seed 0 and 2000 bootstraps on 51 base cards.

| Run | Vector file | Dim | Entries |
|---|---|---|---|
| A_text | `A_text/out/vectors.json` (bge-small-en-v1.5, primary) | 768 | 171 |
| A_text_mpnet | `A_text/out/vectors_paraphrase-mpnet-base-v2.json` | 1536 | 171 |
| B_graph | `B_graph/out/vectors.json` | 8192 | 169 |
| C_engine | `C_engine/out/vectors.json` | 32012 | 169 |

- Raw results are in `results_<run>.json`.
- A supplementary T3 run with different preprocessing is in `results_t3global_<run>.json`, produced by `t3_sensitivity.py`. It is not the official score (see T3).
- **No bugs were fixed in eval.py, and no metric was changed.**
- No path wrote `out/report.md`, because the harness refused report files from subagents. Method and cost below are taken from each builder's returned summary.
- Missing entries: none in any file. A also emits printing vectors for the two legends that `cards.json` does not list. These are harmless extras.

Distances are cosine (`cos`). `rel` means the distance divided by that path's median pairwise distance between base cards, so paths of different size can be compared. "r" is the partner's pessimistic rank among the 51 cards, where 1 is best. Median pairwise base cos is 0.231 for A, 0.430 for A_mpnet, 0.736 for B and 1.000 for C.

---

## T1: same mechanics give the same vector

| | A_text | A_mpnet | B_graph | C_engine |
|---|---|---|---|---|
| printings (n=13): max cos / rank-1 fraction | 0.000 / 1.00 | 0.000 / 1.00 | 0.000 / 1.00 | 0.000 / 1.00 |
| renames (n=51): max cos / rank-1 fraction | 0.000 / 1.00 | 0.000 / 1.00 | 0.000 / 1.00 | 0.000 / 1.00 |
| noise floor (all / printings only) | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

**Suspicious: T1 passes by construction on every path, so it is not evidence that one path is better.**
- **A** replaces the name with `<SELF>` and strips reminder text. `build.py` then asserts that the input strings are byte-identical across printings and renames, so the vectors are identical before any model runs.
- **B** builds one graph per card key and never represents the name or printing id.
- **C:** the 7 printings with no engine class of their own (rad-/ven- runes r01, r02, r04 and ven-179-166) reuse their card's class vector, so they are identical by construction. The engine never reads names, so renames cannot change anything.
- **The only T1 result that is a real measurement:** in C, the two Lonely Poro printings are two separate engine classes, and they give identical vectors. In A, the Lonely Poro and Rengar, Trophy Hunter printings differ only in reminder text. Without the strip rule they would sit 0.037 cos apart (A's own diagnostics), which is more than most one-detail edits in A.

**What this means for the request about the Body, Calm and Fury runes:**
- All three paths treat the three printings of each rune as one card.
- A and B do it by design: they have no input that can differ between printings.
- C does it because no other class exists for those printings.

## T2 (real pairs): one detail different gives a different vector

Each cell is cos_rel / partner rank. Every pair is distinct and above the noise floor on every path. That is trivial here, because every floor is 0.

| Pair (change) | A_text | A_mpnet | B_graph | C_engine |
|---|---|---|---|---|
| punch_first / primal_strength (+5 vs +7, and cost) | 0.044 / r1 | 0.043 / r1 | 0.291 / r1 | 0.469 / r3 |
| punch_first / decisive_strike (one unit vs all friendly) | 0.144 / r2 | 0.183 / r2 | 0.603 / r2 | 0.591 / r4 |
| discipline / moonlight_affliction (+2 and draw vs −10, cost 2 vs 7) | 0.110 / r2 | 0.099 / r1 | 1.012 / r43 | 0.896 / r18 |
| lonely_poro / loyal_poro (died alone vs not) | 0.063 / r1 | 0.034 / r1 | 0.156 / r1 | 0.201 / r6 |
| ferrous_forerunner / machine_evangel (2×3-Might Mech vs 3×1-Might Recruit) | 0.217 / r1 | 0.187 / r1 | 0.178 / r1 | 0.046 / r1 |
| sabotage / decree_of_strength (non-unit vs Mind card) | 0.161 / r1 | 0.138 / r1 | 0.331 / r1 | 0.068 / r1 |
| defy / hush (counter condition) | 0.143 / r1 | 0.119 / r1 | 0.761 / r1 | **1.000 / r50** |
| median partner rank | 1 | 1 | 1 | 4 |

- **discipline / moonlight_affliction:** a low rank is not clearly a failure. The two cards really differ: cost 2 vs 7, and a buff plus draw vs a −10 debuff.
- A ranks them 2nd because the sentence templates match.
- B ranks them 43rd because almost every field differs.

**C, defy / hush: suspicious.** The two vectors share no nonzero dimension (dot product 0).
- In C, Defy is legal only in the probe window with a 3-energy/1-power opposing spell, and the Hush stand-in only in the 4-energy/2-power window.
- Their features therefore land in disjoint blocks.
- On top of that, Hush is a test-only stand-in written into `card_probe.cpp`, because the engine has no Hush class.

## T2 (synthetic pairs and direction)

Each cell is cos_rel / rank of the base among the 51 cards, seen from the variant.

| Variant [field] | A_text | A_mpnet | B_graph | C_engine |
|---|---|---|---|---|
| punch_first__amount4 [amount] | 0.029 / r1 | 0.029 / r1 | 0.063 / r1 | 0.004 / r1 |
| punch_first__energy3 [energy_cost] | 0.011 / r1 | 0.004 / r1 | 0.078 / r1 | 0.021 / r1 |
| punch_first__power0 [power_cost] | 0.009 / r1 | 0.007 / r1 | 0.217 / r1 | 0.023 / r1 |
| discipline__draw2 [amount] | 0.006 / r1 | 0.007 / r1 | 0.056 / r1 | 0.046 / r1 |
| discipline__enemy [target_side] | 0.025 / r1 | 0.053 / r1 | 0.053 / r1 | 0.245 / r1 |
| lonely_poro__draw2 [amount] | 0.016 / r1 | 0.011 / r1 | 0.084 / r1 | 0.021 / r1 |
| defy__cost2 [condition_threshold] | 0.008 / r1 | 0.006 / r1 | 0.098 / r1 | **1.000 / r51** (zero vector) |
| sabotage__unit [selector] | 0.030 / r1 | 0.008 / r1 | 0.047 / r1 | 0.089 / r1 |
| pit_rookie__might3 [might] | **0.002** / r1 | **0.002** / r1 | 0.048 / r1 | 0.145 / r10 |
| punch_first__reaction [timing] | 0.349 / **r7** | 0.383 / **r8** | 0.262 / r1 | 0.225 / r2 |

**C, defy__cost2: suspicious, and it is a C defect, not a pass.**
- `synthetic:defy__cost2` is an all-zero vector.
- With a [2] threshold, Defy cannot counter either opposing spell in the probes (3/1 and 4/2), so it is never legal.
- C subtracts a "pass" baseline, so a card that is never legal becomes 0.
- eval.py gives a zero vector cos distance 1.0, which reads as "distinct, above floor".
- In C, an unplayable card loses everything about itself, including its own printed energy and power cost. That breaks the SPEC rule that costs are first-class.

**A's sizes are badly uneven:**
- Swapping one timing word (Action to Reaction) moves Punch First 0.35 rel. That is past 6 other cards, so the base drops to rank 7.
- Tripling the energy cost moves it 0.011, about 30 times less.
- A Might change on Pit Rookie moves it 0.002, about 3% of the distance to the nearest other card.
- So A does register numbers, but only barely, and a model reading A would see "cost 1" and "cost 3" as nearly the same card.

**B's distances are sizeable for every field**, from 0.047 to 0.26 rel, and every base ranks first.

### Direction

The question: do changes to the same field point the same way? Mean within-field cos is compared against a permutation of the field labels.

| | A_text | A_mpnet | B_graph | C_engine |
|---|---|---|---|---|
| within-field cos (raw) / shuffled / p | 0.307 / 0.027 / 0.017 | 0.178 / 0.029 / 0.059 | 0.227 / 0.019 / 0.069 | 0.000 / −0.001 / 0.83 |
| draw2 (Discipline) vs draw2 (Lonely Poro) | 0.76 | 0.55 | 0.75 | **0.00** |
| +4 Might (Punch First) vs draw2 | 0.00 / 0.16 | −0.09 / 0.07 | −0.03 / −0.03 | 0.00 / 0.00 |
| energy3 vs power0 (both cost changes) | 0.16 | 0.45 | 0.00 | 0.00 |

**Caveats on this test:**
- Only the "amount" field has more than one variant, so the test rests on 3 cosines.
- The "amount" group mixes a Might amount with a draw amount. Those arguably should not align, so the meaningful cell is draw vs draw.
- On draw vs draw, A and B both align strongly (about 0.75). C is exactly orthogonal: Discipline draws as a spell resolving, and Lonely Poro draws in a death trigger probe, so the same change lands in different feature blocks.
- This is C's structural weakness. Its features are indexed by probe window × action × role, so the same mechanic on two different cards shares no dimensions.
- None of the p-values survives a strict threshold except A_text's raw value of 0.017.

## T3: mechanics recoverable by leave-one-card-out linear probes

Official settings: per-fold per-dimension z-scoring; kernel ridge for numbers; logistic with C=1 for classes.

Each cell shows R² for numeric labels or balanced accuracy for class labels, then the 95% CI of the difference from the baseline. **Bold** means the CI lower bound is above 0, so the label beats the baseline. Prov = provisional label.

| Label | Prov | A_text | A_mpnet | B_graph | C_engine | Baseline |
|---|---|---|---|---|---|---|
| energy_cost | | **0.38 [.21,.67]** | 0.21 [−.08,.53] | **0.27 [.10,.52]** | −0.61 [−1.8,.32] | R² −0.05 |
| power_cost | | **0.45 [.31,.68]** | **0.32 [.16,.59]** | **0.52 [.44,.72]** | −0.17 [−.74,.27] | R² −0.05 |
| might | | −0.28 [−.92,.23] | −0.24 [−.74,.27] | **0.35 [.13,.68]** | **0.86 [.85,1.2]** | R² −0.09 |
| n_domains | | **0.68** | **0.68** | 0.14 | −0.37 | R² −0.04 |
| n_effects | y | 0.03 | 0.22 | 0.27 | 0.12 | R² −0.04 |
| card_type | | **0.74** | **0.75** | **0.48** | **0.49** | bacc 0.17 |
| super_type | | **0.78** | **0.99** | 0.37 | 0.48 | 0.33 |
| timing | | **1.00** | **0.95** | **0.68** | **0.78** | 0.25 |
| target_side | y | **0.56** | **0.54** | **0.47** | **0.40** | 0.20 |
| domain_body | | **0.93** | **0.93** | **0.84** | 0.57 | 0.50 |
| domain_calm | | **1.00** | **1.00** | **0.79** | 0.62 | 0.50 |
| domain_fury | | **1.00** | **1.00** | 0.64 | 0.55 | 0.50 |
| domain_order (3 positives) | | **0.92** | **0.85** | 0.50 | 0.49 | 0.50 |
| eff_counter (3) | y | 0.78 | **0.82** | 0.50 | 0.64 | 0.50 |
| eff_draw | y | **0.77** | **0.84** | **0.89** | **0.78** | 0.50 |
| eff_kill | y | 0.68 | 0.67 | 0.67 | 0.48 | 0.50 |
| eff_might_change | y | **0.90** | **0.84** | **0.79** | 0.55 | 0.50 |
| eff_move (3) | y | 0.72 | 0.64 | 0.50 | 0.47 | 0.50 |
| eff_ready (3) | y | 0.41 | **0.83** | 0.50 | 0.81 | 0.50 |
| eff_token (3) | y | **0.95** | **0.84** | 0.50 | 0.83 | 0.50 |
| eff_xp (3) | y | 0.73 | 0.62 | 0.50 | 0.50 | 0.50 |
| has_additional_cost | y | **0.88** | **0.88** | **0.81** | **0.75** | 0.50 |
| has_optional_additional_cost (4) | y | 0.86 | **0.90** | 0.75 | 0.62 | 0.50 |
| has_passive_aura (3) | y | 0.76 | **0.88** | 0.50 | 0.82 | 0.50 |
| has_trigger | y | **0.86** | **0.84** | **0.94** | **0.82** | 0.50 |
| kw_action | | **0.90** | **0.89** | 0.70 | 0.70 | 0.50 |
| kw_ambush (4) | | **1.00** | **0.96** | 0.62 | **1.00** | 0.50 |
| kw_deathknell | | **1.00** | **1.00** | 0.80 | 0.69 | 0.50 |
| kw_hidden (4) | | 0.88 | **1.00** | 0.88 | 0.88 | 0.50 |
| kw_reaction | | **1.00** | **1.00** | 0.69 | **0.88** | 0.50 |
| needs_play_target | y | 0.55 | 0.61 | 0.61 | 0.64 | 0.50 |
| trig_WhenIDie | y | **1.00** | **1.00** | 0.80 | 0.69 | 0.50 |
| trig_WhenYouPlayMe | y | **0.73** | **0.83** | 0.61 | 0.61 | 0.50 |
| **Beat baseline: non-provisional / provisional (of 16 / 17)** | | **14 / 8** | **14 / 12** | 7 / 5 | 5 / 4 | — |
| Mean numeric R² | | 0.25 | 0.24 | 0.31 | −0.03 | −0.05 |

24 labels are skipped for every path, because they are constant or have 2 or fewer positives. Full CIs are in the JSON files.

**Suspicious: A's T3 lead is largely circular.**
- The non-provisional labels are domains, keywords, timing, type and supertype, taken from `card_index.json`.
- A's stats line writes those same fields out as words: domains, keywords, timing, type, supertype.
- A probe that reads "Calm" or "Deathknell" back out of an embedding of a string containing "Calm" or "Deathknell" is a check on the text model, not evidence about mechanics.
- B also copies stats from the printing but encodes them in a way the official preprocessing penalises (next point).
- The provisional effect labels (`eff_*`, trig_*) come from regexes over the engine C++. C is built from the same engine, but it still does not win them.

**Suspicious: the official preprocessing hurts B and C.**
- eval.py z-scores every dimension. B (687 dims used by only one card) and C (651 such dims) are sparse.
- Z-scoring gives each idiosyncratic one-card dimension the same weight as a shared one, so the linear kernel turns nearly diagonal.
- The probes then fall back to the prior: B shows bacc exactly 0.50 [0,0] on every 3-positive label.
- This is the same rule for every path, so the official numbers stand. But it measures "linear and dense", not "contains the information".
- Supplementary run (`t3_sensitivity.py`, centre + one global scale, 500 bootstraps): numeric R² changes as below.

| R² under global scale | A_text | A_mpnet | B_graph | C_engine | random |
|---|---|---|---|---|---|
| energy_cost | 0.12 | −0.06 | **0.93** | **0.66** | 0.20 |
| power_cost | **0.31** | 0.06 | **0.80** | −0.39 | −2.67 |
| might | −0.60 | −0.72 | **0.92** | **0.98** | −0.29 |
| n_effects | 0.06 | 0.18 | **0.61** | −0.18 | −3.57 |

- Under its own scaling, B recovers energy, power and Might almost exactly. A does not recover Might under either preprocessing.
- The class labels of the supplementary run are not comparable to the official ones: logistic C=1 at a different scale is much stronger regularisation. They are left out here.

---

## Cost

| | A_text | B_graph | C_engine | eval (this scorer) |
|---|---|---|---|---|
| Agent wall time to build | ~35 min | ~25 min | ~20 min | ~25 min to build + ~15 min this scoring step |
| Compute | 4.7 s primary model (~70 s for all 8 variants); CPU, <1 GB | 1.5 s, one thread, numpy | 25 s incremental build of the probe target, plus 0.96 s for 125 subjects and ~3 s to assemble; needs the WSL build of riftbound_core | 85–116 s per vector file (4 files in parallel: 116 s wall); supplementary T3 ~1 min |
| One-off manual effort (human-equivalent) | 1–1.5 h: glossary of 14 keywords + 4 rules terms, icon map, cost regexes | 3–4 h by hand to author 51 graphs + ~1 h schema (LLM: ~8 min + review) | probe and feature design, ~640 lines of C++; every synthetic variant is a hand-written C++ copy (~5 min each) | 1–1.5 h to hand-check the gold labels (not done yet) |
| Marginal cost, one new card | ~0 if its keywords are in the glossary (~20 ms to embed) | 5–15 graph lines: ~3–5 min by hand, or ~20 s LLM + 2–3 min review | 0 if the engine implements it (~10 ms); otherwise the engine class must be written first (30–150 lines + tests) | ~2 s to relabel + 1–2 min checking |
| Marginal cost, one new set | 5–15 min per new keyword or term; <1 min to embed ~220 cards | 15–25 h by hand, or 1–1.5 h LLM + review; ~10 min per new keyword helper; a schema review for new mechanics | dominated by implementing the set in the engine; a new mechanic may need a new probe, which changes the layout and recomputes every vector (~1 s) | 15–30 min regex review + seconds of compute |
| Old vectors stable when cards are added? | yes (fixed model) | yes (open hashed vocabulary) | no, if a probe or role is added (layout and dim change) | — |
| Could not represent | Numbers are weak (Might, costs, amounts move less than 0.03 rel). Tags dropped. Conditional keywords flattened into the stats line. Power domain ambiguous on two-domain cards. Scuttle Crab's 0-Might reminder lost. | Effect order and bindings. When and by whom choices are made. Alpha Strike's damage split and legality gates. Darius's token exclusion. Star Spring's perspective switch. Deflect and Ganking kept as flags only. Conditions are atomic hand tokens. Follows the text where text and engine disagree (Zhonya's, Rengar Trophy Hunter, Seat of Power). | Anything not exercised by a probe: Noxian Demolitionist looks vanilla, Alpha Strike's XP, Scuttle Crab's look-at, Yi Tempered's Level 6, Ganking and Deflect, Ruin Runner's untargetability, En Garde's not-alone branch. Activate slots all zero. Hush is a hand stand-in. **A card that is never legal in the probes becomes a zero vector and loses its own cost** (defy__cost2). Battlefields and legends have no legality block. | — |

My own cost for this scoring step:
- About 15 minutes of agent wall time.
- About 2 minutes of compute for the 4 official runs (in parallel) and about 1 minute for the supplementary run.
- No manual effort beyond reading the outputs.

---

## Verdict per path

**A (text embedding): strongest on the official numbers, weakest on what matters for costs.**
- **Holds:** T1 (by construction). T2 ranks, where every synthetic base except the timing change is rank 1 and real pairs have median rank 1. Direction for same-verb amounts (draw2/draw2 cos 0.76). T3, with 14/16 + 8/17 for bge and 14/16 + 12/17 for mpnet.
- **Fails:** numbers.
  - Pit Rookie Might 2→3 moves 0.002 rel, and Punch First energy 1→3 moves 0.011.
  - Yet the one word Action→Reaction moves 0.35 and drops the base to rank 7.
  - Might is not linearly recoverable under either preprocessing (R² −0.28 / −0.60).
- **Why:** sentence encoders weight words, not magnitudes. The T3 wins on domains, keywords and timing are read back from A's own stats line, so they are partly circular.
- **mpnet vs bge:** mpnet does better on T3 and worse on direction (p 0.06 vs 0.017). Neither fixes numbers.

**B (mechanics graph, fixed hashed encoding): the most balanced, and the only path where costs and Might are genuinely first-class.**
- **Holds:**
  - T1 (by construction).
  - All 17 T2 pairs are distinct, with sizeable distances on every field: cost 0.08–0.22 rel, Might 0.05, timing 0.26.
  - Every synthetic base ranks first. Draw-amount changes align (0.75).
  - Under its own scaling, B recovers energy, power, Might and the number of effects (R² 0.93 / 0.80 / 0.92 / 0.61).
- **Fails:**
  - Official T3 on class labels (7/16, 5/17). Its sparse, card-specific hash dims are dominated after z-scoring, and every 3-positive label collapses to the prior.
  - discipline / moonlight_affliction ranks 43, which is defensible.
  - Effect order, bindings and choice timing are not representable.
  - It depends on hand or LLM authoring that follows the card text over the engine where the two disagree (Zhonya's, Rengar Trophy Hunter, Seat of Power).
- **Why:** an explicit schema keeps numbers as numbers. The bag-of-paths encoding is good for distances, but it needs a learned or dense projection before a linear probe or a network can use it well.

**C (engine probes): ground truth for behaviour, but as built it does not generalise across cards.**
- **Holds:**
  - Genuine T1 on the two Lonely Poro engine classes.
  - Might is recoverable (R² 0.86 official, 0.98 global).
  - Real pairs that differ in an observable outcome separate well: Mech vs Recruit tokens, target side 0.245 rel.
- **Fails:**
  - **Direction:** within-field cos 0.000, and draw2 vs draw2 orthogonal.
  - **Costs in official T3:** energy R² −0.61.
  - **defy / hush:** orthogonal, rank 50.
  - **defy__cost2:** a zero vector, so the card's own cost is lost when it is not legal in any probe.
  - **Silent mechanics:** Noxian Demolitionist, Alpha Strike's XP, Level 6, Ganking and Deflect never fire in the probes.
- **Why:** features are indexed by window × action × role. The same effect on two cards lands in different blocks unless the cards share a play pattern, and anything the fixed probes do not trigger is invisible.
- C also needs every card, and every hypothetical variant, implemented in C++ first.

**Recommendation for the minimal network:**
- Use B's schema as the primary input, because costs, power domain, Might, timing and effects are explicit and stable when cards are added.
- Feed it through a learned dense projection instead of raw hashed slots, so that sparse paths share strength.
- Optionally concatenate A's text embedding as a secondary channel for wording and semantics that B's atomic condition tokens miss.
- Use C as a verifier and label source, for example to check B's graphs against engine behaviour, not as the representation.

**Before trusting any of this:**
- T1 is non-discriminative (all pass by construction).
- T2 floors are 0 everywhere, so "above floor" is trivial.
- T3's non-provisional labels overlap the stats that A and B copy in.
- T3's preprocessing favours dense vectors.
- Hand-checked gold labels and more synthetic variants per field (at least 3 each for cost, Might, timing and target) are needed.

---

## Addendum (phase 2): re-scored on current (erratum) card text, plus B2 and fixed C

**Why:** `cards/card_index.json` holds pre-erratum gallery text for some cards. A and B were built from it.
The engine's `CardDef` carries the current text (`engine_text.json`, extracted from the 51 card classes).
Real errata in scope: **Rengar, Trophy Hunter** ("I can [Ambush] to a battlefield where there are enemy
units", CR 822.1.d) and **Zhonya's Hourglass** ("If a friendly unit would die, kill this instead. Heal that
unit, exhaust it, and recall it"). Cosmetic only: Emperor's Dais rewording, Seat of Power typo, `[0]` in
three Hidden reminders. Pre-erratum results are kept in `eval/before_errata/`.

Changes: A reads text from `engine_text.json` (Hush keeps card_index text: no engine class). B's two
erratum graphs were rewritten from the erratum text. C is the phase-2 version (static-facts block, more
H probes, see `C_engine/out/report.md`). Old C results were overwritten; their numbers are in the tables above.

| Run | T2 real-pair partner ranks | T3 beat baseline (non-prov / prov) | Mean numeric R² |
|---|---|---|---|
| A (before → erratum) | 1,2,2,1,1,1,1 → same | 14/16, 8/17 → 15/16, 9/17 | 0.253 → 0.241 |
| A-mpnet (before → erratum) | 1,2,1,1,1,1,1 → same | 14/16, 12/17 → same | 0.239 → 0.251 |
| B (before → erratum) | 1,2,43,1,1,1,1 → 1,2,42,1,1,1,1 | 7/16, 5/17 → 8/16, 5/17 | 0.310 → 0.313 |
| B2 (from code) | 1,2,42,1,1,1,1 | 8/16, 5/17 | 0.314 |
| C (phase 2) | 1,5,50,5,1,1,7 | 6/16, 6/17 | 0.337 |

- **Errata change little in the aggregate** (2 of 51 cards). A's erratum vectors move 0.009–0.018 cos,
  B's 0.06–0.16. Conclusions above stand.
- **B and B2 now differ on 6 cards** (was 8): Star Spring (condition naming only), Rampage, Thrill of
  the Hunt, Alpha Strike, Emperor's Dais, Darius — the remaining real text-vs-engine differences.
- **C phase 2:** defy/hush is no longer orthogonal (rank 50 → 7); `synthetic:defy__cost2` is no longer a
  zero vector (cos_rel 0.73, base rank 3). discipline/moonlight_affliction ranks 50; Lonely/Loyal Poro
  and punch_first/decisive_strike rank 5. C still separates amount changes weakly (Punch First +4: 0.004).

**Retracted text-vs-engine findings (B2 report):** both "major" disagreements (Rengar, Trophy Hunter and
Zhonya's Hourglass) came from the stale card_index text. The engine matches the current erratum text for both.
