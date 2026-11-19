# Shared scorer and provisional labels: build report

 _Written by the orchestrator from the builder agent's returned report: the workflow harness does not let subagents write report files._

## Method

Built the shared scorer and provisional labels in research/card_repr/eval/. The deliverables are eval.py, build_labels.py, labels.json, make_selftest.py, README.md, the selftest/ vectors and the results_selftest_*.json files. report.md was not written: the harness refused to let this subagent write a report file, so its cost content is in the `cost` field below and still needs to be placed in report.md.

**eval.py (T1, T2, T3).** It uses numpy and scipy only.
- **Distances:** cosine, and L2 after per-dimension z-scoring over the 51 base cards. Both are also normalised by the median pairwise base distance, so paths of different dimension can be compared.
- **Ranks are pessimistic:** ties count against the vector, so a constant vector ranks last.
- **T1:** covers 13 printing entries and 51 renames. It reports two noise floors: one over printings and renames, one over printings only.
- **T2:** covers 7 real and 10 synthetic pairs. Each gets its distance, whether it is distinct, its ratio to each floor and the partner's rank. The direction test compares difference vectors grouped by field against a field-label permutation baseline and random card-pair differences.
- **T3:** leave-one-card-out probes on base keys only. Numeric labels use kernel ridge with alpha from inner LOO; categorical and binary labels use L2 multinomial logistic regression (C=1, solved in the dual, so cost does not depend on dimension) and nearest-centroid. Baselines are majority class and mean. Each score has a bootstrap 95% CI, and a label counts as beating the baseline when the paired CI's lower bound is above 0.
- **Missing or invalid entries** are reported, not fatal.

**labels.json.** 51 cards × 59 labels, each carrying {value, source, provisional}.
- **Static fields** come from cards.json/card_index, or from the CardDef for the two legends: type, supertype, domains, energy, power, might, and bracketed keywords with reminder text stripped.
- **Engine facts** come from a regex scan of each card's C++ class: declared TriggerTypes, hooks (passive aura, replacement, optional additional cost, cost reduction, play-time target) and 16 effect categories with n_effects. The effect categories are marked provisional.
- **target_side and has_additional_cost** are text-derived and marked provisional.
- **Timing** is reaction (for [Reaction] or [Ambush]), action, main_only or not_played.

**Self-tests.**
- **Random vectors:** fail T1 (cos floor 1.25, no base at rank 1), sit at chance on T2 (none above the floor), and beat the baseline on 0 of 33 T3 labels.
- **One-hot by name:** T1 printings pass (distance 0, all rank 1) and renames fail (distance 1, rank 51). Every T2 pair is distinct and above the printings-only floor, but not above the overall floor of 1.0. T3 sits exactly at the baseline (0 of 33).
- **Oracle (labels as the vector), a positive control:** passes everything and beats the baseline on 24 of 33 labels. It also shows that labels with only 3–4 positives are underpowered with 51 cards.

The scorer was never run on or tuned to A_text/, B_graph/ or C_engine/ outputs. No engine files were edited and nothing was committed.

## Could not represent / limits

- Hush (rad-041-167) has no engine class in src/cards, so its engine labels (triggers, hooks, effects) are text-derived and marked provisional.
- target_side is text-derived, not engine-derived: target filters are spread over lambdas and helpers, so it is provisional.
- eff_* labels count executor/helper calls in the class body; a call may be conditional or aimed at the card itself (Zhonya's kills itself), so they are provisional.
- has_activated_ability and domain_chaos are constant in this 51-card scope, and 22 other labels have fewer than 3 positives; T3 skips all 24 and lists them.
- Direction test: only field 'amount' has more than one synthetic variant (3 cosines); every other field has a single variant and gets cross-field cosines only.

## Cost

Agent wall time about 25 minutes: reading the contract, exploring the engine via the graph and source, writing 3 scripts, running the self-tests and writing the README. Compute is CPU only, with no GPU, network or downloads: build_labels.py takes 2 s, and eval.py takes 15–46 s per vector file on one core (about 2 minutes for all three self-tests), most of it in T3 logistic L-BFGS. Manual effort still needed: hand-checking a gold sheet of 51×59 labels, mainly the 17 provisional effect labels, target_side and has_additional_cost, at about 1–2 minutes per card (roughly 1–1.5 hours); the 24 rare labels may also be merged into coarser groups. Marginal cost of one new card: add it to cards.json and rerun build_labels.py (2 s). Its labels come out automatically if it has an engine class; a new effect verb needs one regex line in EFFECTS. Allow 1–2 minutes of hand checking, and a re-score of under 1 minute; T3 cost grows about as the cube of the number of cards and does not depend on dimension. Marginal cost of one new set: the same per card, plus 15–30 minutes reviewing the trigger and effect regex table for new mechanics. New TriggerTypes are picked up automatically. T3 stays within minutes for a few hundred cards.

## Files

- `eval\eval.py`
- `eval\build_labels.py`
- `eval\labels.json`
- `eval\make_selftest.py`
- `eval\README.md`
- `eval\results_selftest_random.json`
- `eval\results_selftest_onehot.json`
- `eval\results_selftest_oracle.json`
- `eval\selftest\random\out\vectors.json`
- `eval\selftest\onehot\out\vectors.json`
- `eval\selftest\oracle\out\vectors.json`
