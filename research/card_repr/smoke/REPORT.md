# Card-representation smoke test: overnight report (2026-11-19)

Branch `research/card-repr-phase1`; nothing was committed. All outputs are under `research/card_repr/smoke/` (`data/`, `train/`, `effect/`, `audit/`). No fix pass ran after the audit, so this report uses the pre-fix results and states the audit's corrections next to them.

## 1. The answer

A minimal network learns something from this teacher data with every card input: it beats uniform and kind-frequency baselines on move agreement, best-q choice and value. But no representation (A text, B mechanics graph, B2 code graph, B2A) helps it generalise to an unseen similar card better than a random vector or a learned id. On the 11 held-out variants every paired difference against random is zero within noise, for example B−random q-best +0.008 ± 0.049. That null is mostly a measurement failure, not evidence against the embeddings. The ISMCTS teacher's visit labels are nearly uniform and change between search seeds. Its response to swapping in a near-identical card sits at the search's own noise floor.

In a separate effect-prediction test (card → mean state deltas), A, B and B2 put an unseen variant next to its base card: held-out error 0.08–0.09 against 0.72 for random and 0.80 for id. That holds by construction, because the input vectors are close. They do not predict *how* the variant differs: copying the base card's prediction scores better (0.050), and the predicted difference is no better than predicting "no difference". **Evidence that these embeddings let a network generalise correctly between similar-but-different cards: none yet. Evidence that they make the network *treat* such cards as similar: yes.**

## 2. What was run

**Data** (`smoke/data/`). The decisions come from a new research tool, `decision_export`, built in WSL. It #includes the engine's own `agents/mcts_agent.cpp`, and nothing under `src/` was edited.

**Teacher:** ISMCTS with 100 simulations, uct_c 2.0, score-difference heuristic, on both seats.

**Games played:**
- 1000 base games (Rengar and Yi decks, mirrors and cross).
- 200 games for each of the 11 variant decks: primal_strength plus the 10 synthetic one-detail variants. Every variant deck passes DeckValidator.
- 300 + 440 random-player games, labelled separately.
- 3,940 games in all, with 0 engine errors.

**Teacher decisions:** 336,592 train / 49,352 val / 95,549 test. The split is by seed % 10 (0–6 / 7 / 8–9).

**Replay check:** every game was replayed from its seed and recorded choices. 597,994 decisions were checked, with 0 mismatches.

**Swap set:**
- 432 states from test-split base games (punch_first 144; discipline, lonely_poro, sabotage, pit_rookie 72 each).
- Pairing each state with every variant of its base card gives 1,080 state–variant pairs.
- Each pair was searched with 8 fixed seeds at 100 and at 400 simulations.
- defy and defy__cost2 were dropped from the swap test, because Reaction timing cannot be restarted from a snapshot.

**Decision network** (`smoke/train/`, per NETWORK.md):
- 148.7k parameters; AdamW, batch 256, 3,000 steps, cosine schedule.
- Heads:
  - a policy head on visit distributions;
  - a policy head on softmax(q/0.05), with the temperature fixed before any result was seen;
  - a value head.
- Each head keeps its best checkpoint on validation games of its own regime.
- Card conditions, which differ only in R(card):
  - controls: id, random, stats;
  - representations: A, B, B2, B2A;
  - E3 control: Bshuf, a derangement of B, base regime only.
- All R vectors are centred and projected to 256 dimensions (projection seed 1234).
- Two regimes:
  - **base**: trained on base games only; all 11 variants unseen. An assertion confirms no variant key appears in training tokens.
  - **all**: trained on base and variant games.
- Seeds 0, 1, 2. That gives 45 runs, all of which finished, taking 4h01m of training.
- Protocol changes, applied to every condition:
  - steps cut from 4,000 to 3,000 for wall time;
  - per-head checkpoint selection added after the value head overfit in the first 3 runs. Those runs are kept in `runs_superseded/` and not used.

**Effect prediction** (`smoke/effect/`):
- An MLP: R(card) through a learned projection to 64 dims, plus situation kind (play or trigger).
- It predicts 48 z-scaled state-delta fields for each (card, kind). Costs are excluded from the targets.
- Conditions: id, random, stats, A, B, B2, shuffled-B. C (engine probes) was left out because it would leak pool H.
- Seeds 0–2; bootstrap seed 0 with 2,000 resamples.

**Audit** (`smoke/audit/`): independent re-derivations and checks. The game-clustered bootstrap uses seed 12345 with 1,000 resamples.

## 3. Results

All ± values are 95% t-intervals over 3 training seeds unless stated otherwise. **These seed CIs leave out test-set sampling noise.** A game-clustered bootstrap gives about ±0.015 on E1 card-source top-1, roughly 7 times the seed CI (see section 4).

### E0: does it learn at all? (all regime, full test set, n = 95,549)

| condition | NLL (pi) | top-1 vs teacher move | q-head picks best-q | value Brier |
|---|---|---|---|---|
| uniform baseline | 1.3566 | 0.300 | 0.392 | 0.250 |
| kind-frequency baseline | 1.3497 | 0.309 | 0.394 | – |
| id | ~1.347 | 0.3520 ± 0.0044 | 0.51–0.52 | 0.143–0.146 |
| random | ~1.347 | 0.3507 ± 0.0036 | 0.51–0.52 | 0.143–0.146 |
| A | 1.3464 ± 0.0004 | 0.3523 ± 0.0029 | 0.5134 ± 0.0039 | 0.1453 ± 0.0025 |
| B | 1.3466 ± 0.0010 | 0.3529 ± 0.0032 | 0.5112 ± 0.0022 | 0.1454 ± 0.0022 |
| B2 | 1.3467 ± 0.0006 | 0.3525 ± 0.0018 | 0.5131 ± 0.0109 | 0.1450 ± 0.0006 |
| B2A | 1.3465 ± 0.0010 | 0.3528 ± 0.0026 | 0.5120 ± 0.0050 | 0.1432 ± 0.0035 |

Every condition beats both baselines, so minimal learning passes. The NLL gain is tiny because the visit targets are nearly uniform (KL to them is about 0.002). No condition separates from another. The audit re-derived top-1 independently from `preds.pt` and matched it to 1e-8.

### E1: generalising to the unseen variant (base regime, pooled over 11 pairs, 1,451 decisions where the variant is a legal source)

Absolute values (`table_E1_pooled.csv`):

| condition | variant q-best | card-play agreement vs pi | card-play agreement, q-head |
|---|---|---|---|
| id | 0.548 ± 0.029 | 0.696 ± 0.063 | 0.732 ± 0.044 |
| random | 0.551 ± 0.034 | 0.739 ± 0.018 | 0.755 ± 0.043 |
| A | 0.566 ± 0.004 | 0.726 ± 0.012 | 0.747 ± 0.008 |
| B | 0.559 ± 0.017 | 0.742 ± 0.030 | 0.755 ± 0.009 |
| B2 | 0.557 ± 0.017 | 0.723 ± 0.026 | 0.734 ± 0.012 |
| B2A | 0.554 ± 0.035 | 0.717 ± 0.021 | 0.751 ± 0.020 |
| Bshuf | 0.557 ± 0.031 | 0.709 ± 0.062 | 0.734 ± 0.023 |

Always predicting the majority answer for card-play against pi scores 0.738, and the audit reproduced this. The audit also scored the q-head against the stable q-argmax label (n = 819 decisions where the q label is clear). Every condition, random and id included, scores about 0.85 against a majority baseline of 0.779. So the network does discriminate card plays, but from context and legality cues rather than from card content.

Paired differences against random, base regime (`table_paired.csv`, seed CI):

| comparison | variant NLL | variant top-1 | variant q-best | card-play q |
|---|---|---|---|---|
| A − random | +0.0005 ± 0.0009 | **+0.0106 ± 0.0020** | +0.016 ± 0.033 | −0.008 ± 0.036 |
| B − random | +0.0002 ± 0.0029 | −0.0007 ± 0.0209 | +0.008 ± 0.049 | −0.000 ± 0.048 |
| B2 − random | +0.0002 ± 0.0012 | +0.0012 ± 0.0157 | +0.006 ± 0.043 | −0.022 ± 0.033 |
| B2A − random | +0.0009 ± 0.0009 | +0.0040 ± 0.0133 | +0.003 ± 0.035 | −0.004 ± 0.060 |
| stats − random | −0.0010 ± 0.0019 | +0.0065 ± 0.0224 | +0.015 ± 0.055 | −0.014 ± 0.065 |
| B − stats | +0.0012 ± 0.0033 | −0.0073 ± 0.0109 | −0.006 ± 0.011 | +0.013 ± 0.025 |

The bolded A−random top-1 cell excludes 0 by seed CI, and so does B−stats card q-best agreement (−0.0065 ± 0.0046). The audit re-ran both kinds of comparison with a game-clustered bootstrap over the 293 test games, on the 1,451 card-source decisions:

| comparison | top-1 difference | 95% bootstrap interval |
|---|---|---|
| A − random | +0.005 | [−0.010, 0.019] |
| B − random | +0.005 | [−0.009, 0.019] |
| stats − random | +0.013 | [−0.003, 0.029] |

With about 108 cells tested, a few seed-CI exclusions are expected by chance. **Treat E1 as a null.**

Seen versus unseen: all-regime models score about 0.02 higher on variant q-best than base-regime models in every condition, random included (B 0.582 vs 0.559, random 0.576 vs 0.551). That is a data-volume effect, not a representation effect.

### E2: swap test (both cards seen; network change vs teacher change on the same state)

- **Full set:** the correlation between the network's change in P(play card) and the teacher's is about 0.999 for every condition. This is **entirely legality**: in some swapped states the card has no legal play, so both probabilities go to 0. Example: primal_strength's number of card actions differs in 120 of 144 states; only 24 are legal-same.
- **Legal-same subset:** per-pair correlations range from about −0.3 to +0.4, with CIs of ±0.2 to ±1.0. Examples for A in the all regime at 100 sims:
  - amount4 0.03 ± 0.02
  - discipline__draw2 0.00 ± 1.05
  - sabotage__unit −0.21 ± 0.66
  - pit_rookie__might3 0.07 ± 0.35

  Sign agreement on the few teacher changes that are significant (1–17 per pair at 100 sims) is near 0.5. The most positive cells, B2 and B on discipline__draw2 (0.43 ± 0.28 and 0.37 ± 0.53), do not survive correction for multiple comparisons.
- **Teacher noise floor** (audit): the same original state, searched with the same 8 seeds in different variant processes, gives different results in 328 of 434 states. Registering an unrelated card perturbs the search about as much as the swap effects do (mean |Δ root value| 0.0002–0.006). On the stable q label, the majority play/no-play decision flips on a swap in only 0–9% of states.
- **Not interpretable on this data.**

### E3: shuffled-B control

B − Bshuf on E1:

| metric | difference |
|---|---|
| card-play agreement | +0.033 ± 0.036 |
| variant NLL | +0.0001 ± 0.0019 |
| variant q-best | +0.002 ± 0.047 |

All are zero within noise, which matches the null E1. There is nothing above the control to explain.

### Supplementary: key-swap sensitivity (`keyswap.json`)

At test time, the variant's key is replaced with its base card's key, and the change in the network's output is measured:

| condition | \|dP(card)\| | \|d value\| |
|---|---|---|
| A | 0.0023 ± 0.0004 | – |
| B | 0.0043 ± 0.0011 | 0.012 |
| B2 | 0.0039 ± 0.0016 | – |
| stats | 0.0037 ± 0.0020 | – |
| random | 0.0170 ± 0.0032 | 0.073 |
| id | 0.0159 ± 0.0025 | – |
| Bshuf | 0.0174 ± 0.0015 | – |

The audit's point: this ordering follows directly from the input distances ‖R(v) − R(b)‖. It is an input sanity check, not evidence that the network uses the similarity correctly.

### Effect prediction (`effect/results.json`, error in z², bootstrap 95% CIs)

**Evaluations on seen cards** cannot rank representations:
- (1) S-train → S-test: every content-free and content-based condition equals the per-card-mean oracle at 0.085 [0.019, 0.19] (pure memorisation); stats scores 0.230.
- (3) S-train → H: all conditions about 1.07, equal to the oracle (the shift between pools dominates).
- H → S-test: about 1.04 for all conditions; stats 1.11.

**Held-out variant (leave-one-pair-out, pool S):**

| condition | error on held-out variant | − random | strict fold (all 11 held out) |
|---|---|---|---|
| copy base card's prediction | 0.050 [0.008, 0.107] | – | – |
| B | 0.082 [0.031, 0.141] | −0.637 [−0.871, −0.442] | 0.079 |
| B2 | 0.088 [0.037, 0.152] | −0.632 | 0.080 |
| A | 0.094 [0.042, 0.146] | −0.626 | 0.090 |
| stats | 0.313 [0.100, 0.628] | −0.406 [−0.804, 0.052] | 0.550 |
| random | 0.719 [0.475, 1.001] | – | 0.714 |
| id | 0.802 [0.515, 1.140] | +0.083 | 0.878 |
| shuffled-B | 1.280 [0.821, 1.944] | +0.561 | 1.441 |

**Split by pair type** (audit, `audit_effect.json`, error on held-out variant):

| pairs | B | B2 | A | stats | random | copy base |
|---|---|---|---|---|---|---|
| cost/timing (primal_strength, energy3, power0, reaction) | 0.038 | 0.044 | 0.072 | 0.702 | 0.431 | 0.006 |
| other 7 pairs | 0.107 | 0.112 | 0.106 | **0.091** | 0.884 | 0.075 |

On the 7 non-cost pairs, stats is at least as good as A, B and B2. Copying the base card's prediction beats every model on both sets.

**Predicted difference against true difference (variant − base), on pairs with a measurable difference** (post-hoc filter; S n = 7, H n = 6):
- Cosine in pool S is about 0 for every condition (B −0.08 [−0.46, 0.29]).
- In pool H, where base and variant run in paired situations:

  | condition | cosine |
  |---|---|
  | B | 0.086 [−0.22, 0.39] |
  | B2 | 0.016 |
  | A | −0.046 |

  B − random is +0.167 [0.009, 0.323] in the leave-one-pair-out folds, but +0.069 [−0.05, 0.22] (not significant) in the strict fold.
- Error in the size of the predicted difference (1.0 = predicting no difference), in H:

  | condition | error |
  |---|---|
  | B | 1.46 [0.98, 1.95] |
  | B2 | 1.58 |
  | A | 2.73 |
  | stats | 5.6 |
  | random | 35 |
  | id | 40 |
  | shuffled-B | 93 |

  **No condition beats predicting no difference.**
- Per pair:
  - pit_rookie__might3: cosine 0.6–0.9 for A, B and B2.
  - punch_first__amount4: 0.46–0.54 for B.
  - discipline__draw2: −0.4 to −0.7 for A, B and B2 (wrong direction).
  - sabotage__unit: −0.9 for B.

## 4. Audit issues and how each was handled

No blocking bug was found. The plumbing checks below passed:
- The split is by seed, with no game shared across splits.
- Swap states come only from test-split base games.
- Variant keys are absent from base-regime training tokens.
- Tokens carry no printed stats.
- The game result z matches the winner for both seats (481,493 of 481,493).
- pi = argmax(visits), and every option is visited.
- E0 top-1 and the E1 majority baseline were reproduced exactly.
- Bshuf is a true derangement.
- The effect experiment excludes held-out variants from its training folds.

Issues with the conclusions (no fix pass ran, so each is handled in this report's wording, not by a re-run):

1. **(major) "A/B/B2 clearly beat stats" in effect prediction is an artifact of how the targets were built.** Costs were removed from the targets, while stats feeds raw cost and timing into the model without per-feature scaling. That pushes stats away from the base card exactly on the cost/timing pairs. On the other 7 pairs stats (0.091) is at least as good as B (0.107). *Handled:* the claim is withdrawn here. *Not re-run:* standardised stats features or pool-neutral cost targets.
2. **(major) Effect "generalisation" is nearest-neighbour copying.** The 9× gain over random measures how close R(v) is to R(b), and copying the base card's prediction beats every model. *Handled:* the headline here is the difference metric, which is null.
3. **(major) Seed-only CIs understate the noise.** *Handled:* game-bootstrap intervals are reported for the E1 cells that looked significant; they include 0. *Not done:* a full two-level bootstrap for every E1/E3 cell, and a multiple-comparison correction or a single pre-registered primary metric.
4. **(major) The played-move label pi is mostly search noise.** Play/no-play is unanimous across 8 search seeds in only 37–47% of states at 100 sims (56–70% at 400 sims). The q-argmax is unanimous in 93–99%. *Handled:* this report gives the q-based card-play metric alongside the pi metric, and corrects the earlier claim that the network "does not discriminate the card's plays at all": it does, on the q label, but equally for every condition.
5. **(major) Swap-test teacher responses sit at or below the search's own perturbation floor**, and primal_strength's legality changes in 120 of 144 states, not 20. *Handled:* E2 is reported here only as legality plus noise. Nine one-option swap states were silently filled with uniform by `prep.py`; this was noted, not fixed.
6. **(minor) Variant identity reaches every condition through legal-action structure.** The reaction variant appears at priority; the enemy-target and cost variants change which actions exist; might3 shows its might on the board. Only amount4, discipline__draw2, lonely_poro__draw2 and sabotage__unit are "effect-only" pairs that cleanly test R(card). *Not done:* separate E1/E2 tables for those 4 pairs.
7. **(minor) The key-swap result restates input distances.** It is presented in section 3 as a sanity check only.
8. **(minor) The two experiments use different stats extractors**: text and keywords in `train/reprs.py`, engine CardDef in `effect_pred.py`. Their stats numbers are not comparable.
9. **(minor) E1 compares variant and base cards on different decision populations** (decks, opponents, prompt kinds), and per-pair n is small (defy__cost2: 50 card-source decisions). *Not done:* a paired base-vs-variant comparison on swap states.

**Still unverified or weak:**
- Policy heads were still improving at step 3,000; a longer run might change E0/E1 magnitudes.
- Only 3 seeds.
- The effect "measurable pair" filter was defined after the first results were seen.
- Only 6–7 pairs are measurable.
- The synthetic variants were checked by reading the code (card_probe copies), not by an independent test.
- Swap searches start from snapshots, not from the in-game agent's exact mirror.
- Only hand copies of the card were swapped.

## 5. Recommended next step

Fix the teacher label before spending more compute on representations. This run's experiments do not discriminate between representations.

1. **Make a repeatable target.** Use q-based policy targets, or many more simulations with a lower uct_c. Measure label repeatability (agreement across search seeds) on a held-out sample and report it as the ceiling for every metric.
2. **Pick one pre-registered primary metric.** Suggested: q-label card-play agreement on the 4 effect-only pairs, scored on a *paired* swap set where the teacher's q-argmax actually flips under the swap, with two-level (game × seed) bootstrap CIs.
3. **Re-run the effect test with fair inputs.** Standardise each stats feature (or use one shared engine-based stats extractor) and add pool-neutral cost targets. Report variant error against copying the base card, and the predicted-difference cosine and size error as the headline.

Until the effect test beats "copy the base card" on effect-only pairs, the embeddings should be treated as giving correct *similarity*, with no demonstrated ability to capture *differences*.
