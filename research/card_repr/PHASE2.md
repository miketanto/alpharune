# Phase 2: code-derived graph (B2), verified by engine behaviour (C), situation sets A/B tested

Phase 1 result (eval/RESULTS.md): B (mechanics graph) is the most balanced; A (text)
cannot represent numbers; C (engine probes) is ground truth but does not generalise
across cards. B was written from card text and followed the text where text and
engine disagree. The agent will play in the engine, so the representation should
describe what the engine does.

## Goal

A card representation built from **the card's code** (B2) and **checked against what
the card does in situations** (C), heading towards a representation *learned* from
both — without overfitting to the particular situations used.

## 1. B2: mechanics graph from the card code

- Source of truth: the card's C++ class in `src/cards/<type>/` plus the engine code it
  calls (EffectExecutor methods, `card_helpers.h`, keyword handling in the engine).
- Same schema as B (`B_graph/schema.md`), extended where the code needs it.
- Effect vocabulary mapped once: executor / helper calls → schema verbs;
  `TargetRequirements` fields → selectors; `TriggerType` → triggers; `CardDef` → stats.
- Every field cites the source line it came from. Names never enter the graph (class
  names, trace strings and card-specific checks are stripped).
- Separate engine classes for one card (e.g. Lonely Poro 0359 / 0777) are extracted
  independently, so T1 is measured, not true by construction.
- By-product: a list of every place card text and engine code disagree.

## 2. C as verifier, with two situation pools

C runs each card in situations and records legality + state change. It is used to
**check** B2, not as the representation.

- Fix first: a card never legal in the probes becomes a zero vector (it lost its own
  cost). Static facts must survive.
- **Pool H (hand-designed):** C's canonical probe positions, extended for the coverage
  gaps phase 1 found (conquer, kills for XP, level 6, opponent targeting, not-alone).
- **Pool S (sampled):** positions taken from real games of the Rengar / Yi decks
  (random-player and ISMCTS agents, fixed seeds) at moments where the card can be
  played, triggered or activated. Split by game seed into S-train / S-test.
- Verification per card: every B2 effect (verb, amount, side, condition) must show up
  as the predicted state change in at least one situation where its condition holds;
  every observed state change must be explained by a B2 effect. Report
  agree / disagree / never-exercised, per pool.

## 3. A/B test of situation sets (guarding against overfitting to samples)

Anything learned or verified from situations is reported **across pools**, never only
on the pool it was built from:

| Built / trained on | Evaluated on | Question |
|---|---|---|
| H | S-test | do hand-designed probes generalise to real play? |
| S-train | H | do sampled situations cover the designed edge cases? |
| S-train | S-test | in-distribution baseline |
| S-train (k seeds) | S-test | variance across resamples |

A result that holds on its own pool but not across pools is reported as overfitting to
that pool. Pools are fixed by seed and committed so every result is reproducible.

## 4. Later: a representation learned from graph + situations

- Encoder f(graph of card) → z (small graph network or set encoder over B2's typed paths).
- Trained to predict the card's state change in a situation: g(z, situation) ≈ Δstate
  from C (a behaviour-prediction objective, no game result needed).
- Generalisation tests: held-out situations (the A/B pools above) and held-out
  cards (the T2 partners). Only then does z go into the decision-test network
  (`NETWORK.md`) as R(card).

## Deliverables of this phase

`B2_code/` (graphs with citations, vectors, text-vs-engine disagreements),
`C_engine/` fixes + `situations/` (pool H, pool S with train/test split),
`verify/` (B2-vs-C report per pool), B2 scored with `eval/eval.py`.
