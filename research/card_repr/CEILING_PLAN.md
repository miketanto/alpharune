# Can the current ceiling be pushed? Experiment plan (2026-11-20)

**Question.** The Rengar league's main agent stopped improving after about update 1,400. Two remedies are on the table: a larger network, and search. Each needs proof that it raises the ceiling before we scale it.

**Two levels of proof:**
- **Inference proof:** playing with the remedy is stronger.
- **Training proof:** training with the remedy produces a network that is stronger without it. This is the one that justifies scaling.

**Measurement rule (applies to every experiment below).**
- Strength is measured by greedy head-to-head games between networks: 2,000 games per pair, seat-balanced, which gives about ±2 points.
- ISMCTS anchors and Elo against fixed opponents are not used: every agent now beats them all 100%, so they can't tell agents apart.
- Tool: `selfplay/league/h2h.py` (and `cross_eval.py` for different decks).
- "Better" means at least 55% against the plateaued main agent, with the CI above 52%.

## E0. Is the Rengar plateau real? RESULT: no — it is slow growth, not a ceiling

2,000 greedy games per pair, Rengar mirror (`league/overnight/h2h_rengar.txt`):

| pair | newer wins | ~Elo |
|---|---|---|
| @1200 vs @200 | 74.4% ±1.9 | +185 |
| @1400 vs @1200 | 56.7% ±2.2 | +47 |
| @1800 vs @1400 | 54.0% ±2.2 | +28 |
| @2080 vs @1800 | 55.9% ±2.2 | +41 |
| @2080 vs @1400 | 57.4% ±2.2 | +52 |

- **Every later snapshot is significantly stronger than the earlier one,** but the gains are small: about +50 Elo over the last 680 updates.
- **The steps are not additive:** 1,400 → 1,800 → 2,080 implies about +69 Elo, but 2,080 against 1,400 measures +52. Part of each gain is specific to beating the previous snapshot.
- **The "flat Elo" from update 1,400 on was the same artefact as Azir:** fixed opponents beaten 100%, plus 100-game rating series.
- **Consequence for E2/E3:** the control arm keeps improving slowly, so a remedy must beat the *control arm at equal compute*, not just update 2,080. The question is now whether the slope can be raised (E2: search teacher; E3: capacity), not whether a wall can be broken.

### Original E0 design

- **Games:** 2,000 games per pair between snapshots 200, 1,200, 1,400, 1,800 and 2,080 of `ov_main`.
- **Plateau:** consecutive snapshots within about 50 ± 2%, while earlier gaps are clearly larger.
- **If it isn't a plateau,** the "ceiling" was a measurement artefact, the same one that made Azir look flat. Then the plan below becomes "how fast can we go" rather than "can we break through".

## E1. Search raises play strength (inference proof): done

- Search M1 (`search/REPORT.txt`) with network prior and value, against the plateaued main agent playing greedy, 400 games per point:
  - 64 simulations: **64.2%**;
  - 200 simulations: **71.0%**;
  - still rising at 200.
- So the plateaued network's weakness is not in what it knows: a few hundred lookahead simulations on its own value extract more.
- **To strengthen this proof (cheap):** rerun at 2,000 games per point, and add 800 simulations.

## E2. Search raises the trainable ceiling (training proof): the key experiment

- **Setup:** continue the Rengar league from update 2,080 in two arms with the same compute, seeds and league.
  - **A (control):** plain PPO, exactly as before. We expect about 50% against update 2,080 if it really is a plateau.
  - **B (search teacher):** on a sample of the learner's decisions (e.g. 5–10%), run the search (64 simulations, Gumbel root when M2 exists, otherwise PUCT visit counts). Add a cross-entropy loss pulling the policy towards the search's improved policy π′, weighted against the PPO loss.
    - The search is the C++ `rbsearch` path with the GPU evaluator, batched with the actors.
- **Pass:** B (greedy, no search) beats update 2,080 by at least 55% and beats A by at least 55%, both at 2,000 games. A's result is the measured noise floor.
- **Cost:**
  - Searching 5% of decisions at 64 simulations is roughly 3 additional simulations per decision, which should cost about 1.5–2x the time per update (to be measured).
  - About 3–4 h of engineering: wiring π′ into the PPO batch, plus the M2 Gumbel root (optional).
  - About 2 × 2 GPU-hours to run both arms.
- **Risk:** the improved policy is only as good as the value head (explained variance 0.68–0.75). Track the value loss in both arms.

## E3. A larger network raises the ceiling (training proof)

- **Today's network:** width D=64, 2 layers, 209k parameters (`learner/model.py`: module constant `D`, hard-coded depth 2).
- **Engineering (about 1 h):**
  - Make width and depth arguments.
  - Store them in the checkpoint and detect them in every loader: `verify_eval`, `league/common.load_net`, `viz`, `search/rbsearch.load_net`, `match`.
- **Design:** the arms differ only in size: same recipe (league + PPO), same seeds, same opponent pool.
  - **S:** D=64 × 2 layers, from scratch.
  - **L:** D=128 × 4 layers (about 8x the parameters), from scratch.
  - Both start from scratch because L can't be initialised from S's weights. This also gives the clean capacity comparison.
  - Budget: 800 updates each, taking snapshots every 100.
- **Pass:**
  - At equal updates, L beats S by at least 55% from some point on.
  - L's head-to-head gains between consecutive snapshots stay clearly above 50% after S's have shrunk to about 50%.
  - Report both per update and per GPU-hour, because L is slower per update.
- **Cost:** about 3 h for S plus about 5–6 h for L (GPU-bound, run one after the other), i.e. an overnight run.
- **Cheaper first probe (optional, about 1 h):** distil a fixed set of search targets (π′ from 200 simulations, plus outcomes, on about 50k positions from the plateaued agent's games) into S and L supervised. If L doesn't fit the targets clearly better (held-out policy cross-entropy, value MSE), capacity isn't the bottleneck and E3 can be skipped.

## Order

1. **E0** (now, minutes).
2. **E1** strengthening (about 30 min) and the **E3** capacity probe (about 1 h). Both are cheap and point to which remedy matters.
3. **E2** (search teacher), daytime.
4. **E3** full runs, overnight.

One GPU training at a time. Each result goes into a REPORT next to its code.
