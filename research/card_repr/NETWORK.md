# Minimal network for the decision test (draft for discussion)

Purpose: test whether a card representation R lets a model learn "a card that does
this is played like this in that context", **without** the network's capacity or
tricks confounding the answer. Everything below is fixed across conditions; the
only thing that changes between runs is the per-card input vector.

## 1. Design rules (what keeps representation and network apart)

1. **One network, many inputs.** Same architecture, size, data, optimiser, steps and
   seeds for every condition. Only `R(card)` changes.
2. **R is frozen.** The card vector enters through one learned linear projection
   and is never fine-tuned. Fine-tuning per card would let the network memorise card
   identity and hide what R itself carries.
3. **No card id anywhere.** No id embedding, no name, no printing — except in the
   explicit id baseline below.
4. **Small on purpose.** Capacity just big enough to model card-in-context
   interactions; a big network can paper over a weak R on seen cards.
5. **Held-out cards decide.** The headline number is on decisions involving cards
   the network never saw in training.

## 2. Architecture

```
per token:   x_i = W_card · R(card_i)  +  W_state · s_i  +  E_kind[kind_i]
encoder:     2-layer pre-norm transformer, d = 64, 4 heads, no positional encoding
             (tokens are a set: global, battlefields, entities on board / in hand / on chain)
value head:  global token → MLP(64→64→1) → tanh          (win / loss from decider's view)
policy head: for each legal action a:
               h_a = E_akind[kind_a] + h_source + mean(h_targets) + h_dest + W_v · scalars_a
               logit_a = MLP(64→64→1)(h_a)
             softmax over the legal actions only (pointer-style, no fixed action vocabulary)
```

- Token layout: the v6 layout in `src/ml/token_layout.h` (global / battlefield /
  entity / own-deck tokens, `ActionTokens` with source/target/dest refs), with the
  card-id slot replaced by `R(card)`. `s_i` keeps only **situational** state (zone,
  controller, ready/exhausted, damage, current Might, buffs, attachments) — the card's
  printed facts (cost, printed Might, keywords) live in R, not in `s_i`, so the
  stats-only baseline is a real baseline.
- About 100k parameters: trains on CPU in this machine's 16 cores.
- Why a transformer and not DeepSets: "play this removal on *that* unit" is a pairwise
  relation between tokens; sum/max pooling cannot express it. Two layers is the
  minimum for card → board → action reasoning. Encoder depth is ablated separately
  (1 vs 2 layers, DeepSets) with R fixed, so network effects are measured, not guessed.

## 3. Conditions (same network, different R)

| Condition | R(card) | What it shows |
|---|---|---|
| id | learned one-hot per card id (trainable) | memorisation ceiling on seen cards; ~chance on unseen |
| stats | printed cost, power, Might, domains, keywords only | how far printed numbers alone go |
| random | fixed random vector per card | control: distinguishes cards, means nothing |
| A | text embedding | |
| B | mechanics graph vector | |
| C | engine behaviour vector | |
| B+C | concatenation | whether they complement each other |

## 4. Data and targets

- Positions from ISMCTS-200 games on the two decks: Rengar mirror, Yi mirror, Rengar
  vs Yi (both seats), fixed seeds, stored as seed + action history (replayable).
- Targets: the teacher's visit distribution over legal actions (policy) and the
  game result (value).
- Caveat: the teacher is ISMCTS with a heuristic evaluator (beats a random agent
  85% of the time). "Played correctly" means "played like this teacher".
- Engineering prerequisite: nothing fills the v6 `Observation` / `ActionTokens` today.
  A C++ exporter (state → tokens + action refs, to a binary or npz file) is needed
  before any training. Same exporter for every condition: R is joined in Python.

## 5. Splits and metrics

- **Held-out cards:** leave-k-cards-out folds; every decision whose legal actions or
  board involve a held-out card is test-only. Report separately for held-out cards
  that have a mechanically close seen partner (the T2 pairs) and those that don't.
- Metrics: teacher-agreement (top-1, KL) on decisions involving seen vs held-out
  cards; value Brier score; the **generalisation gap** (seen − held-out) per condition.
- Swap tests: fixed positions, swap one card for a mechanically similar / different
  card; policy and value should move the way mechanics say.
- Pass for checkpoint 1: some R closes most of the gap between `random` and `id`-on-seen-cards on **held-out** cards,
  with 3 seeds and confidence intervals, while `id` stays near chance there.
