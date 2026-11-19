# eval/: the shared scorer (T1, T2, T3) and provisional labels

```
python build_labels.py                         # -> labels.json (2 s)
python eval.py ../A_text/out/vectors.json --name A_text   # -> results_A_text.json
python make_selftest.py                        # -> selftest/{random,onehot,oracle}/out/vectors.json
```

`eval.py` uses only numpy and scipy, with fixed settings for every path (seed 0, 2000 bootstrap
resamples, logistic C = 1, ridge alpha from a fixed grid chosen by inner LOO). It was written and
checked only against the three self-test files below, and not against any A/B/C output.

## What is measured

**Distances.** `cos` is the cosine distance (1 − cos). `l2s` is Euclidean distance after
per-dimension z-scoring, with the mean and sd taken over the 51 base-card vectors. A dimension
whose sd is 0 is left unscaled. Each distance is also given divided by the median pairwise
base-card distance (`*_rel`), so paths with different dimensions can be compared. **Ranks are
pessimistic**: rank = 1 + the number of other base cards at a distance ≤ the partner's distance.
Ties count against the vector, so a constant vector ranks last.

- **T1**: every `printing:<id>` of a card with more than one printing (13 entries: 3 runes ×3,
  Lonely Poro ×2, Rengar Trophy Hunter ×2) and every `rename:<key>` (51). For each: the
  distance to the base, the base's rank among all 51 cards, and the fraction at rank 1 and
  "near zero" (< 1 % of the median pairwise distance). The **noise floor** is the maximum T1
  distance. The floor from printings only is reported as well.
- **T2**: 7 real pairs and 10 synthetic pairs. For each: the distance, `distinct` (> 1e-9), the
  ratio to the noise floor and whether it is above the floor, the same against the
  printings-only floor, and the partner's rank (for a synthetic pair, the base's rank among all
  51 cards). **Direction test**: difference vectors (synthetic − base), in raw and in
  standardised space, grouped by `field`. Reported: the mean within-field pairwise cosine, the
  same with field labels shuffled (2000 permutations, with a p-value), the mean |cos| between
  random card-pair differences, and per field the cosine to other fields. Only `amount` has more
  than one member (3), so the within-field statistic is the amount group alone. The other fields
  give only cross-field cosines.
- **T3**: leave-one-card-out over the base cards only. Features are z-scored inside each fold, and
  a feature that is constant in the training fold is dropped.
  - Numeric labels use a linear-kernel ridge, reported as R² and MAE. The baseline is the
    training-fold mean.
  - Categorical and binary labels use multinomial L2 logistic regression, solved in the
    span of the rows so the cost does not depend on dimension, plus a nearest-centroid
    classifier. They are reported as accuracy and balanced accuracy. The baseline is the
    training-fold majority class.
  - Bootstrap 95 % CIs over cards come with each score. `beats_baseline` means the paired
    bootstrap CI of (model − baseline) has a lower bound above 0: R² for numeric labels,
    balanced accuracy for the others.
  - A label is skipped when fewer than 8 cards have a value, when it is constant, or when its
    second-largest class has fewer than 3 cards. The reason is listed under `t3.skipped`.
  - Summaries are reported separately for non-provisional labels (static card data, engine
    trigger declarations) and provisional ones (engine effect calls, text-derived).

Missing or invalid entries (wrong length, NaN, non-numeric) are listed in `missing_expected`
and `invalid` and skipped. A test that needs a missing entry lists it under `t1.missing` or
`t2.missing`. Fewer than 4 base vectors produces an error record, not a crash.

## labels.json

51 cards × 59 labels (57 probe-able + 2 list-valued kept for reference: keywords, trigger_types). Each label carries `{value, source, provisional}`.

| source | labels | provisional |
|---|---|---|
| `card_index` (cards.json printing 0 = card_index.json) | card_type, super_type, domain_* (6), n_domains, energy_cost, power_cost (null → 0 for unit/spell/gear; N/A for rune/legend/battlefield), might (units only) | no |
| `card_index:bracket_tokens` | keywords, kw_* (14), timing (reaction if [Reaction] or [Ambush], action if [Action], main_only, not_played). Reminder text "( … )" is stripped first | no |
| `card_def` | the same static fields for Rengar, Pridestalker and Master Yi, Wuju Bladesman (from `src/cards/legends/0744_pridestalker.cpp` and `0317_wuju_bladesman_starter.cpp`) | no |
| `engine_cpp:<file>` | trigger_types, has_trigger, trig_* (4), has_activated_ability, has_passive_aura, has_replacement, has_optional_additional_cost, has_cost_reduction, needs_play_target | no |
| `engine_cpp:<file>` | eff_* (16: draw, damage, might_change, buff, token, counter, recycle, kill, stun, move, recall, ready, bounce, banish, xp, play_free), n_effects. These come from executor and helper calls in the class body: a call that is present may be conditional | **yes** |
| `text_derived` | target_side (friendly / enemy / both / any / none), has_additional_cost | **yes** |

Known gaps:
- **Hush** (rad-041-167) has no engine class in `src/cards`, so its engine labels are text-derived
  and marked provisional.
- No card in scope has an activated ability (`has_activated_ability` is constant), and none is
  Chaos. Both labels are skipped by T3.
- 24 of the 57 probe-able labels are too rare to probe with 51 cards and are skipped.
- **Low power.** Even the oracle (below) cannot beat the baseline significantly on several labels
  that have 3–4 positives (domain_order, eff_ready, eff_token, eff_xp, kw_hidden, kw_ambush,
  has_passive_aura, …). Treat those as low-power labels for every path, not as a failure of a
  path.

## Self-test results (seed 0, 2000 bootstraps)

| | random N(0,1), dim 64 | one-hot by name, dim 112 | oracle (labels as vector), dim 74 |
|---|---|---|---|
| T1 printings: cos max / rank-1 fraction | 1.199 / 0.00 | **0.000 / 1.00** | 0.000 / 1.00 |
| T1 renames: cos max / rank-1 fraction | 1.254 / 0.00 | **1.000 / 0.00** (fails) | 0.000 / 1.00 |
| noise floor cos (all / printings only) | 1.254 / 1.199 | 1.000 / 0.000 | 0.000 / 0.000 |
| T2 real: distinct / > floor / > printing floor (cos) | 1.00 / 0.00 / 0.00 | **1.00** / 0.00 / **1.00** | 1.00 / 1.00 / 1.00 |
| T2 synthetic: distinct / > floor / > printing floor (cos) | 1.00 / 0.00 / 0.00 | **1.00** / 0.00 / **1.00** | 1.00 / 1.00 / 1.00 |
| T2 median partner rank (real / synthetic) | 21 / 30.5 | 50 / 51 | 1 / 1 |
| T2 direction, amount-group within cos (raw), shuffled mean, p | 0.028, 0.113, 0.76 | 0.000, 0.079, 1.00 | 1.000, 0.069, 0.009 |
| T3 non-provisional (16 labels): beat baseline | 0 | 0 | 13 |
| T3 mean R² (baseline) | −1.29 (−0.06) | −0.06 (−0.06) | 0.90 (−0.06) |
| T3 mean balanced acc logistic (majority) | 0.415 (0.438) | 0.438 (0.438) | 0.885 (0.438) |
| T3 provisional (17 labels): beat baseline | 0 | 0 | 11 |

**Reading the one-hot row.** It behaves as expected:
- **T1:** printings pass, because they share the name. Renames fail, because each is a new
  dimension at cos distance 1 from everything.
- **T2:** every pair is distinct and above the printings-only floor. It is not above the overall
  floor, because the renames have already pushed that floor to 1.0. So the "distinctness" claim
  must always be read together with T1. Partners rank last, since every card is equidistant
  from every other.
- **T3:** it sits exactly at the baseline. A held-out card's own dimension is constant in the
  training fold, so it is dropped and the probe predicts the prior.

**l2s caveat.** The l2s floor ratio for one-hot real pairs reads 1.0 above the floor. This is an
artefact of l2s: rename dimensions are constant over the base cards, so they are left
unscaled, while base-card dimensions are divided by a small sd. Prefer the `cos` columns when
a path has dimensions that are unused by the base cards.

The oracle is a positive control only. It is built from `labels.json`, so its T3 score is
circular. Its value is to show the scorer can register a pass, and where the power ceiling lies.

Runtime: 15–46 s per vector file on one CPU core. T3 logistic fits dominate.
