# Playtesting agents: history first, then a zoo of models, then search (2026-11-21)

**Goal (user).** Hand the system a decklist. Competent agents play it against other legends on their own lists, and the results say how the deck fares.

**What that requires:**
- **Equal pilots.** Agents must pilot every deck about equally well, or the matchup numbers measure piloting, not decks.
- **Fast onboarding.** A new deck is brought up to strength by a short fine-tune against the open field.

**Order (user decision):**
1. Attention to previous actions.
2. A zoo of models (generalist, archetype specialists, deck specialists) to measure transfer.
3. Network-guided search.

The architecture is frozen after step 1, so no zoo compute is spent on a network that is about to change.

## Step 1. History tokens and a hand-prediction head (rbenv schema v5)

### What the network sees
- The last N = 32 **public** actions, from the observing player's point of view. Each action becomes one token with these fields:
  - who acted (self / opponent);
  - the action kind (play, hide, move, activate, pass priority, pass focus, end turn, mulligan count, battlefield choice);
  - the card, if it is public (otherwise a "hidden card" key);
  - the battlefield and destination;
  - the Energy / Power the actor had left;
  - turn offset and phase.
- Passes matter: "passed with 3 open and a reaction-heavy deck" is exactly the evidence about the opponent's hand that the network cannot see today.

### Leak rule
- A history token may contain only what that player was shown. It follows the engine's own rules:
  - `knowledge::identityVisible`;
  - the log audience;
  - a mulligan gives its count, not the cards.
- Test, in the spirit of the engine's `HiddenOrderLeak`: two games that differ only in cards the observer cannot see must give identical history tokens at every decision.

### Network
- History tokens are encoded like other tokens, with their own kind embedding.
- They are joined to the board tokens by a **cross-attention block with a residual connection whose output projection starts at zero**.
  - At initialisation the network computes exactly what G2 computes, so **G2's weights load unchanged** and nothing already learned is thrown away.

### Hand-prediction head
- An auxiliary loss: predict how many copies of each card are in the opponent's hand.
- The label comes from the engine's true state. It is used **only as a training target, never as an input**.
- Scoring uses the card-embedding table, so it works for cards and legends not trained on.
- This gives a direct measurement:
  - prediction accuracy (log-loss / top-k recall) of the head;
  - against a baseline that knows only the opponent's decklist and the visible cards.

### Test (A/B at equal updates)
- **Both arms:**
  - start from G2 (g2_main@1000);
  - use the same 6-deck field;
  - Rengar cells oversampled (repeated entries in `--matchups`, already supported by `g1/selfplay_g1.py`);
  - run 300 updates.
- **The arms:**
  - **Arm A (control):** today's network. Arm A can run now; arm B, once built, gets the same config and seed.
  - **Arm B:** history tokens plus the hand-prediction head.
- **Measures:**
  - B vs A head-to-head, per deck;
  - Rengar recovery in both arms (this also settles the "sampling or capacity" question from G2);
  - hand-prediction accuracy over the course of a game.

## Step 2. The zoo: generalist, archetype specialists, deck specialists

### Who plays what
- Models differ only in **which decks they pilot**.
- **Every model trains against the full open field.** Field opponents are generalist snapshots piloting their own lists.
  - A specialist that has only ever seen its own mirror is useless for playtesting.

### Design (sized to fit the GPU in a few days)
- **Pool:**
  - 12 trained decks, 2 per archetype bucket (aggro, tempo, midrange, holding, control, ramp; `deck_archetypes.md`);
  - 6 held-out decks, 1 per bucket, never trained before step 2c.
- **Models (equal compute per deck, about 300 updates per deck):**
  - **G**: one generalist, all 12 decks (3,600 updates).
  - **A_k**: 6 archetype specialists, 2 decks each (600 updates each).
  - **S_d**: 12 deck specialists (300 updates each).
- **Budget:** about 10,800 updates, roughly 45 h at 15 s/update.

### Questions and tests
- **2a. Is a specialist better than the generalist on its own deck?**
  - Pilot swap on all 12 decks, against the same field references.
  - Report the mean and the worst deck.
- **2b. Does archetype grouping beat both extremes?**
  - Compare A_k against G and against S_d on the same decks.
- **2c. Does knowledge transfer to a new deck?** This is the playtesting onboarding test.
  - For each held-out deck H, fine-tune against the open field from four starts:
    1. G;
    2. H's archetype specialist;
    3. the nearest deck specialist;
    4. scratch.
  - Measure the updates needed to reach the bar: 50% head-to-head against a 300-update H specialist.
  - Budget: 6 × 4 × ≤150 updates, about 15 h.
  - The best start becomes the onboarding recipe for new decklists.
- **Matchup-table stability.**
  - For the trained decks, compute the deck-vs-deck table at two checkpoints, 300 updates apart.
  - Report the largest change. Playtest numbers are only reported once it is small.
- **Exploiter per deck.**
  - Run a short main exploiter against each deck's best pilot.
  - Report its best win rate: a pilot that is easy to exploit gives a misleading matchup.

## Step 3. Network-guided search (after step 1)

- **Search:** ISMCTS / PUCT with the network's policy as prior and its value at the leaves, in the style of `search/rbsearch.py`.
- **Hand sampling:** opponent hands are drawn from the **hand-prediction head** instead of uniformly from the deck model. This is where opponent hand modelling and the opponent's possible actions meet:
  - the search plays out the opponent's replies under the believed hands;
  - this lets it play around a likely reaction.
- **Uses:**
  - a stronger pilot at playtest time (search costs only at inference);
  - a teacher for later distillation. E2 showed a 64-simulation search teacher did not help PPO, but that search had no hand model.

## Concept probes (alongside, CPU only)

- Hand-built position suites with a known right answer:
  - lethal or not;
  - hold vs conquer;
  - racing to 8;
  - passing with a reaction open;
  - committing vs holding back;
  - mulligans.
- Each checkpoint gets a score per concept, for points/tempo understanding.

## Choosing to go first or second

- The engine picks the first player at random (CR 115, Duel mode), so there is no decision to learn.
- Adding the choice is an engine change in `src/` and needs the owner's OK.
- Until then, the value head can answer it: compare V(go first) with V(go second) at the start position for each matchup.
