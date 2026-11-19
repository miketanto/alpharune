# Card experiment: does B2 let a self-play network understand cards? (2026-11-19)

## Answer

**Not answerable with the network as it was. The zero-shot test exposed a design bug in the policy head.
That bug is now fixed and the fix is verified. The B2-vs-id-vs-random comparison has to be rerun with it.**

1. **Design bug: the asking card is ignored at target prompts.** The card being played is visible to the network:
   - it is a token, the prompt's psrc, marked by `E_psrc`;
   - the transformer sees it.

   But nothing connects it directly to the option scores. Every trained policy, B2, id and random alike, makes
   its target choice independently of the asking card. If the asking token's card key is swapped (to the
   held-out card, ctrl_scores, base Punch First, `hidden` or a unit), or the psrc marker is removed, the mass on
   enemy units moves by at most 0.002. The network learned "at a target prompt, pick my own unit".
2. **The bug matters even when the deck demands the card.** Control experiment: an id network trained for
   15 min on a deck holding BOTH ctrl_scores (right side: friendly) and opp_of_ctrl_scores (right side: enemy).
   - The old head still targets friendly with both (enemy mass 0.04 for opp_of_ctrl).
   - In real games it learns to stop playing opp_of_ctrl: 13 plays in 4,355 decisions where it was legal.
3. **Fix: `Net(ask_head=True)`** (`learner/model.py`, `selfplay.py --ask-head`). It adds a bilinear term between
   the asking token's state and each option's referenced tokens. On the same deck and budget the network learns
   both sides:
   - probes: ctrl_scores friendly 0.99-1.00, opp_of_ctrl enemy 0.98-0.99 from about update 90 (8 min);
   - real games: 259/259 friendly for ctrl_scores and 271/271 enemy for opp_of_ctrl;
   - card swap: putting ctrl_scores' key on the opp_of_ctrl states moves enemy mass from 0.99 to 0.003.

   It now reads the card.
4. **Zero-shot results of the main experiment (old head, so not a fair test of B2).** Every repr targets
   friendly with the held-out opp_of_ctrl_scores: 0-2% enemy mass, 0-2 enemy picks in 246-364 real target
   prompts per run. B2 is no better than id or random. The 2-minute fine-tunes raise the held-out win rate
   (about 0.52-0.58 to 0.63-0.66 vs ISMCTS-100) only by **no longer playing the card** (B2: 9% to 1% of the
   decisions where it can be played; id seed 1: 9% to 0.2%), never by targeting enemies.
5. **Training-deck cards (old head).**
   - score_alone is played conditionally by every repr (real games: 0.68-0.85 with no own units vs
     0.34-0.55 with units). random is at least as good as B2.
   - score_opphand4 is affordable (offered in 86-91% of the decisions where it is held) and played (~0.17 per
     decision), but blind to its condition: the same rate with 4+ cards in the opponent's hand and without.
   - Held-out sabotage__score1 is played at the same rate (0.16-0.22).
   - This is not the target-prompt bug: the play decision is a main-phase option whose own card token is its
     source. The signal for this condition is simply weak: a wasted play costs 1 Energy + 1 Power.
6. **Strength (old head), pooled over 2 seeds, 400 games each vs ISMCTS-100 (Wilson 95%):**
   - training deck: B2 0.73 [0.68, 0.77], id 0.74 [0.70, 0.79], random 0.71 [0.66, 0.75];
   - held-out deck zero-shot: 0.54 / 0.54 / 0.51;
   - after the 2-minute fine-tune: 0.64 / 0.64 / 0.60.

   No repr difference is resolved.

**Next:** rerun the main comparison (B2 / id / random × seeds, train on exp_train, zero-shot on exp_heldout)
with `--ask-head`. Better still, train on side contrasts and hold out a new instance of one (see "What to do
next"). Only then does "does B2 transfer the side field?" have a chance to show.

## Design (written before any run)

Question: with no search, does a mechanics-based card input (B2) let a self-play policy
(1) play a card differently from its partner card, and (2) use an UNSEEN card correctly, better
than learned card ids (`id`) or frozen random vectors (`random`)?

Game: Rengar mirror only. Both seats play the same deck.

Decks (`cardexp/make_exp_decks.py`, written to `smoke/data/decks/`; same substitution as `make_decks.py`,
every main-deck copy of the base card replaced):

| deck | punch_first (3) | sabotage (3) | pit_rookie (3) | role |
|---|---|---|---|---|
| `rengar__exp_train` | `punch_first__ctrl_scores` (q11) | `sabotage__score_opphand4` (q7) | `pit_rookie__score_alone` (q5) | training |
| `rengar__exp_heldout` | `punch_first__opp_of_ctrl_scores` (q12) | q7 | q5 | held-out (main test) |
| `rengar__exp_heldout_sab1` | q11 | `sabotage__score1` (q6) | q5 | held-out, unconditional (probe only) |

Card texts (C_engine/card_probe.cpp):
- ctrl_scores: "Choose a unit. Its controller scores 1." Right target = a friendly unit.
- opp_of_ctrl_scores: "Choose a unit. Its controller's opponent scores 1." Right target = an enemy unit.
  It is never seen in training. Its B2 graph differs from ctrl_scores only in the Score target's
  `side` (friendly -> enemy).
- score_opphand4 (spell): "Score 1 if the opponent has 4+ cards in hand." Right = play iff opp hand >= 4.
- score_alone (unit): "When you play me, score 1 if you control no other units."
- sabotage__score1: "Score 1." Right = play it whenever it can be played.

All five variants are registered in every process (rbenv now takes a variant list). They have
fixed vocab rows, so an unseen card keeps its untrained `id` row, its random vector, or its B2 vector.

Conditions: repr in {id, B2, random}, 2 seeds each. The same PPO settings and wall-clock budget for all:
128 envs, 8192 samples per update, minibatch 1024, 3 epochs, opponent pool (pool_every 2), 10 min of
training on `exp_train`. Fine-tuning (sample efficiency): a copy of each final model trained for
2 more minutes on `exp_heldout` (fresh optimizer, same settings).

Card checks (`cardexp/collect_probes.py` -> `learner/probes/*.npz`, 260 states each, from uniform-random
mirror games of the relevant deck, at most 3 states per game). Metrics: policy mass on the rule-right
options (`mass_right`) and how often the top choice is right (`argmax_right`); for the
conditional cards, the policy's probability of playing the card when the condition holds vs not.
- `ctrl_tgt` (train): target prompt of ctrl_scores, only states with both a friendly and an enemy option. good = friendly.
- `oppctrl_tgt` (HELD-OUT): the same for opp_of_ctrl_scores. good = enemy.
- `opphand4` (train): main-phase states where score_opphand4 can be played, balanced 130 condition-true / 130 false.
- `alone` (train): main-phase states where score_alone can be played, 130/130.
- `sab1` (held-out, unconditional): main-phase states where sabotage__score1 can be played.

Evaluation per run (all ISMCTS games through `verify_eval.py --procs N`, one game per env slot):
- (a) win rate vs ISMCTS-100 on `exp_train`, 200 games;
- (b) probes on the training variants;
- (c) zero-shot on `exp_heldout`: `oppctrl_tgt` probe + win rate vs ISMCTS-100 (100 games, cut from 200 for time);
- (d) after the 2-minute fine-tune: (c) again.

Controls/reference points: the untrained network of each repr, and the earlier base-deck `id` run
(`runs/real_smoke_id`, never saw any variant) on the same probes.

Pre-registered prediction: if B2 transfers mechanics, B2 should beat id/random on `oppctrl_tgt` zero-shot
(mass on enemy units clearly above the uniform 0.50), while all three should be similar on the training probes.

## What was built (steps 1-4)

- **B2 and B vectors for the 12 score-family variants** (`B2_code/extract.py`, `SCORE_VARIANTS` + `score_edit`).
  Each variant is a graph edit of its base card's graph that keeps stats and keywords and replaces the effect:
  - a new verb `Score {target: player selector, amount 1}`;
  - conditions `you_control_units` (neg), `you_control_other_units` (neg), `opponent_hand_size_ge 4`,
    `friendly_unit_at_battlefield`, `opponent_score_gt_yours`;
  - ctrl / opp_of_ctrl: `ChooseTarget unit any -> A`, then `Score {side: friendly|enemy, relative_to: A}`;
  - kill: `Kill`; minus5: GiveMight -5.

  The same edit is applied to path B's base graphs, and the result is written into `B_graph/out/vectors.json`.
  Note: re-running `B_graph/build.py` alone drops these 12 keys; re-run `B2_code/extract.py` after it.
  Existing vectors are unchanged (0 of the old keys differ).
  - **Nearest base card.** The punch_first and pit_rookie variants (except minus5) are nearest their base card.
    sabotage__score1 / score_opphand4 and punch_first__minus5 are nearest `charm`: Sabotage's whole effect
    is replaced, so its stats dominate, and charm is a cheap Body spell.
  - **ctrl vs opp_of_ctrl.** They differ only in the paths ending in `side=friendly` vs `side=enemy` under
    the Score target (4 paths each; 8 nonzero dims; distance 2.83 vs 9.24 to punch_first).
- **model.py** loads B2/B for the whole vocab from the source `vectors.json`. It uses the pipeline of
  `smoke/train/reprs.py`: center on the 51 base cards, scale, then a seed-1234 Gaussian projection to 256.
  It equals the packed `reprs.pt` exactly on the old keys, and no vocab key is missing (the score variants
  were zero rows before).
- **Several variants per process.** `rbenv_init` takes a comma-separated list, and `RBEnv(variant=[...])`
  takes a list. Tested in `rbenv/tests/test_multivariant.py`:
  - three mixed decks, 40 random games each: 0 errors, 0 unknown keys;
  - same seed and choices give identical games;
  - `test_determinism.py` still passes.

  `selfplay.py` gained `--variants` and `--init` (fine-tuning). `verify_eval.py` gained `--deck`,
  `--variants` and `--out`.
- **Probes**: `cardexp/collect_probes.py` writes `learner/probes/*.npz` (`good`, plus `play` / `cond` for
  conditional cards). `probes.py` reports the play rate by condition, and `probe_eval.py` evaluates a checkpoint.
- Real-game statistics: `cardexp/real_stats.py`, which reads the policy's own choices in mirror games.
  Diagnostics: `cardexp/check_probes.py` (decoding and card-swap sensitivity) and `cardexp/emb_diag.py`.
- Runs: `runs/cx_<repr>_s<seed>` (10 min), `_ft` (+2 min on held-out), `cx_<repr>_s0_ft10` (+10 min on held-out).
  Training: about 170 PPO updates of 8192 decisions each in 10 min (about 60 games/s, ~20k games per run; log.csv per run).

## Checks (coordinator questions)

1. **Is the oppctrl_tgt `good` mask right?** Yes. Decoded states (`check_probes.py`):
   - pk 24 (Target);
   - the asking token is `synthetic:punch_first__opp_of_ctrl_scores` (zone hand: the card being played,
     controller self);
   - each option is a MakeChoice naming one unit token: own units have controller 1, enemy units controller 2;
   - `good` is True exactly on the controller-2 options. Example: option 0 = own Pit Rookie (good False),
     option 1 = enemy Irresistible Faefolk (good True).
2. **What the policy does with the card in real games** (`real_stats.py`, mirror of the held-out deck, policy on
   both seats, 200 games per run; tables below):
   - before fine-tuning it plays opp_of_ctrl in about 9-16% of the main-phase decisions where it can be played;
   - at two-sided target prompts it picks an enemy unit 0-2 times out of 246-364;
   - after the 2-minute fine-tune B2 plays it in 1% of those decisions (id seed 1: 0.2%; random: 5-11%),
     and still targets friendly (0 of 38-40).

   So fine-tuning learns to **stop playing** the card, not to target enemies. That is where the held-out
   win-rate gain comes from.
3. **Does the network see which card asks?** In principle yes:
   - the asking card is a token (psrc), marked by `E_psrc`, and visible to the transformer;
   - every option's score includes the chosen unit's token, attended over the whole set.

   In practice the trained policies ignore it. With the asking token's key replaced by ctrl_scores,
   punch_first, `hidden` or noxus_hopeful, or with the psrc marker removed, the mass on enemy units changes
   by at most 0.002 for every run tested (B2/id/random seed 0, before and after the fine-tune;
   `runs/cx_check_sensitivity.json`).

   The policy head has no DIRECT path from the asking card to the option score: the asking card's effect must
   pass through attention into the chosen unit's token or the global token. Training never rewarded such a
   path, because every target prompt in the training deck has the same right side. **This is a design bug of
   the experiment, as the coordinator suspected.** It is fixed and verified: see "Fix check" below.
4. **opphand4 almost never played in the probes: preference or affordability?** In real games the card is
   affordable: its play is offered in 86-91% of the main-phase decisions where it is in hand. It is played in
   about 17% of those decisions (about 1.9 plays per game across both seats), at the same rate whether the
   opponent holds 4+ cards or not.

   The low probe rates (0.04-0.10) come from the probe states: they are taken from uniform-random games
   (odd boards, many options) and differ from the policy's own games. The real-game rate is the one to
   trust: played, but blind to the condition. Playing it uselessly costs 1 Energy + 1 Power, a mild loss,
   so the PPO signal for the condition is weak in 10 minutes.

## Results (partial: the user stopped the remaining work at 12:53; my first kill missed the eval chain and the
10-minute fine-tunes, which ran on until 13:10 and were then killed. What they finished is kept.)

What finished:
- training: id, B2, random for seeds 0 and 1, plus id and B2 seed 2 (random seed 2 was killed mid-run and
  is excluded);
- 2-minute fine-tunes for all of those;
- probes for all of them;
- ISMCTS-100 evaluations (200 games each): (a), (c), (d) for seeds 0 and 1 of all three reprs.

Not run: seed-2 ISMCTS evaluations (seed-2 probes are in the probe tables), the id 10-minute fine-tune,
the rest of the random one, and random seed 2.
CIs are Wilson 95% on games pooled over seeds. Probe numbers are the mean over seeds, with the per-seed values
in brackets. "untrained" means freshly initialised networks (each call re-initialises, so these move by about ±0.1).

### Probes after training on exp_train

| probe metric | B2 mean (per seed) | id mean (per seed) | random mean (per seed) | untrained B2 / id / random | base-deck id run |
|---|---|---|---|---|---|
| ctrl_scores target: mass on friendly | 0.99 (1.00, 0.99, 0.99) | 1.00 (0.99, 1.00, 1.00) | 0.98 (0.98, 0.98) | 0.45 / 0.47 / 0.48 | 0.9 |
| ctrl_scores target: top = friendly | 1.00 (1.00, 1.00, 1.00) | 1.00 (1.00, 1.00, 1.00) | 1.00 (1.00, 1.00) | 0.07 / 0.16 / 0.42 | 1.0 |
| opphand4: P(play) when opp hand >= 4 | 0.05 (0.05, 0.07, 0.03) | 0.13 (0.10, 0.08, 0.21) | 0.04 (0.04, 0.03) | 0.27 / 0.31 / 0.27 | 0.05 |
| opphand4: P(play) when opp hand < 4 | 0.01 (0.01, 0.01, 0.01) | 0.02 (0.01, 0.02, 0.04) | 0.01 (0.01, 0.01) | 0.15 / 0.16 / 0.11 | 0.01 |
| score_alone: P(play) when no own units | 0.77 (0.69, 0.81, 0.81) | 0.66 (0.64, 0.78, 0.56) | 0.80 (0.81, 0.80) | 0.23 / 0.29 / 0.39 | 0.4 |
| score_alone: P(play) when own units | 0.30 (0.25, 0.24, 0.41) | 0.24 (0.23, 0.30, 0.19) | 0.33 (0.32, 0.33) | 0.16 / 0.18 / 0.20 | 0.16 |
| HELD-OUT opp_of_ctrl target: mass on enemy | 0.01 (0.00, 0.02, 0.01) | 0.00 (0.01, 0.00, 0.00) | 0.02 (0.02, 0.01) | 0.54 / 0.52 / 0.51 | 0.12 |
| HELD-OUT opp_of_ctrl target: top = enemy | 0.00 (0.00, 0.00, 0.00) | 0.00 (0.00, 0.00, 0.00) | 0.00 (0.00, 0.00) | 0.93 / 0.81 / 0.55 | 0.02 |
| HELD-OUT sabotage__score1: P(play) | 0.03 (0.03, 0.04, 0.02) | 0.10 (0.09, 0.06, 0.14) | 0.09 (0.09, 0.09) | 0.24 / 0.25 / 0.24 | 0.05 |
| HELD-OUT sabotage__score1: top = play | 0.00 (0.00, 0.00, 0.00) | 0.09 (0.13, 0.01, 0.13) | 0.13 (0.13, 0.13) | nan / 0.58 / 0.13 | 0.0 |

### Probes after the 2-minute fine-tune on exp_heldout

| probe metric | B2 mean (per seed) | id mean (per seed) | random mean (per seed) | untrained B2 / id / random | base-deck id run |
|---|---|---|---|---|---|
| ctrl_scores target: mass on friendly | 0.99 (1.00, 0.99, 0.99) | 1.00 (0.99, 1.00, 1.00) | 0.98 (0.98, 0.98) | 0.45 / 0.47 / 0.48 | 0.9 |
| ctrl_scores target: top = friendly | 1.00 (1.00, 1.00, 1.00) | 1.00 (1.00, 1.00, 1.00) | 1.00 (1.00, 1.00) | 0.07 / 0.16 / 0.42 | 1.0 |
| opphand4: P(play) when opp hand >= 4 | 0.04 (0.04, 0.04, 0.04) | 0.10 (0.10, 0.08, 0.13) | 0.04 (0.06, 0.03) | 0.27 / 0.31 / 0.27 | 0.05 |
| opphand4: P(play) when opp hand < 4 | 0.01 (0.01, 0.01, 0.02) | 0.03 (0.03, 0.02, 0.03) | 0.01 (0.01, 0.01) | 0.15 / 0.16 / 0.11 | 0.01 |
| score_alone: P(play) when no own units | 0.79 (0.71, 0.83, 0.82) | 0.65 (0.65, 0.77, 0.55) | 0.83 (0.81, 0.85) | 0.23 / 0.29 / 0.39 | 0.4 |
| score_alone: P(play) when own units | 0.30 (0.26, 0.23, 0.42) | 0.25 (0.27, 0.32, 0.17) | 0.34 (0.32, 0.35) | 0.16 / 0.18 / 0.20 | 0.16 |
| HELD-OUT opp_of_ctrl target: mass on enemy | 0.01 (0.00, 0.01, 0.01) | 0.01 (0.01, 0.00, 0.00) | 0.02 (0.02, 0.02) | 0.54 / 0.52 / 0.51 | 0.12 |
| HELD-OUT opp_of_ctrl target: top = enemy | 0.00 (0.00, 0.00, 0.00) | 0.00 (0.00, 0.00, 0.00) | 0.00 (0.00, 0.00) | 0.93 / 0.81 / 0.55 | 0.02 |
| HELD-OUT sabotage__score1: P(play) | 0.02 (0.02, 0.02, 0.02) | 0.06 (0.04, 0.03, 0.10) | 0.09 (0.12, 0.06) | 0.24 / 0.25 / 0.24 | 0.05 |
| HELD-OUT sabotage__score1: top = play | 0.00 (0.00, 0.00, 0.00) | 0.05 (0.00, 0.00, 0.13) | 0.08 (0.13, 0.02) | nan / 0.58 / 0.13 | 0.0 |

### (a) Win rate vs ISMCTS-100, exp_train mirror

| repr | seed 0 | seed 1 | seed 2 | pooled (Wilson 95%) |
|---|---|---|---|---|
| B2 | 0.73 (147/200) | 0.72 (144/200) | - | 0.73 [0.68, 0.77] n=400 |
| id | 0.81 (162/200) | 0.68 (136/200) | - | 0.74 [0.70, 0.79] n=400 |
| random | 0.75 (150/200) | 0.67 (133/200) | - | 0.71 [0.66, 0.75] n=400 |

### (c) Zero-shot win rate vs ISMCTS-100, exp_heldout mirror

| repr | seed 0 | seed 1 | seed 2 | pooled (Wilson 95%) |
|---|---|---|---|---|
| B2 | 0.52 (103/200) | 0.56 (111/200) | - | 0.54 [0.49, 0.58] n=400 |
| id | 0.58 (116/200) | 0.51 (101/200) | - | 0.54 [0.49, 0.59] n=400 |
| random | 0.56 (113/200) | 0.46 (91/200) | - | 0.51 [0.46, 0.56] n=400 |

### (d) After 2-minute fine-tune, exp_heldout mirror

| repr | seed 0 | seed 1 | seed 2 | pooled (Wilson 95%) |
|---|---|---|---|---|
| B2 | 0.65 (129/200) | 0.63 (126/200) | - | 0.64 [0.59, 0.68] n=400 |
| id | 0.63 (126/200) | 0.65 (129/200) | - | 0.64 [0.59, 0.68] n=400 |
| random | 0.66 (132/200) | 0.54 (107/200) | - | 0.60 [0.55, 0.64] n=400 |

### (d2) 10-minute fine-tune on exp_heldout from the seed-0 models (B2 complete; random killed at update 90; id never started)

Cells: oppctrl_tgt mass on enemy / sab1 P(play), from the probes logged every 10 updates.

| fine-tune update | B2: enemy mass / sab1 P(play) | id: enemy mass / sab1 P(play) | random: enemy mass / sab1 P(play) |
|---|---|---|---|
| 10 | 0.00 / 0.05 | - | 0.03 / 0.10 |
| 20 | 0.00 / 0.04 | - | 0.03 / 0.10 |
| 30 | 0.00 / 0.03 | - | 0.04 / 0.11 |
| 40 | 0.00 / 0.03 | - | 0.02 / 0.11 |
| 50 | 0.00 / 0.03 | - | 0.04 / 0.09 |
| 60 | 0.00 / 0.03 | - | 0.02 / 0.11 |
| 70 | 0.00 / 0.04 | - | 0.02 / 0.10 |
| 80 | 0.00 / 0.03 | - | 0.02 / 0.10 |
| 90 | 0.01 / 0.02 | - | 0.02 / 0.07 |
| 100 | 0.03 / 0.02 | - | - |
| 110 | 0.06 / 0.01 | - | - |
| 120 | 0.08 / 0.02 | - | - |

With the old head, 10 more minutes on the held-out deck barely move the target side (B2: 0.00 -> 0.08 enemy mass by update 120).

### Real-game play and target-side rates (cardexp/real_stats.py)

Counts are main-phase decisions (several per turn) where the card is in the decider's hand. "offered" means
its play is legal there, "P(play)" is the policy's choice rate at those decisions, and the condition is
judged at decision time. Target prompts are counted only when there is both a friendly and an enemy option.

#### exp_train mirror (policy on both seats, 200 games per run)

| run | card | offered / held (affordable) | P(play) when offered | cond true: P(play) | cond false: P(play) | target prompts (2-sided) | enemy chosen |
|---|---|---|---|---|---|---|---|
| B2_s0 | ctrl_scores | 0.82 (3222/3919) | 0.09 (303/3222) | - | - | 318 | 0.00 (0/318) |
| B2_s0 | score_opphand4 | 0.87 (2175/2498) | 0.18 (385/2175) | 0.17 (222/1326) | 0.19 (163/849) | 0 | - |
| B2_s0 | score_alone | 0.75 (882/1171) | 0.45 (394/882) | 0.72 (171/238) | 0.35 (223/644) | 0 | - |
| B2_s1 | ctrl_scores | 0.86 (3333/3893) | 0.09 (306/3333) | - | - | 279 | 0.00 (0/279) |
| B2_s1 | score_opphand4 | 0.90 (2097/2321) | 0.19 (395/2097) | 0.19 (228/1229) | 0.19 (167/868) | 0 | - |
| B2_s1 | score_alone | 0.82 (635/777) | 0.61 (385/635) | 0.82 (219/268) | 0.45 (166/367) | 0 | - |
| id_s0 | ctrl_scores | 0.83 (3451/4147) | 0.08 (268/3451) | - | - | 238 | 0.00 (0/238) |
| id_s0 | score_opphand4 | 0.86 (1928/2239) | 0.19 (369/1928) | 0.19 (216/1120) | 0.19 (153/808) | 0 | - |
| id_s0 | score_alone | 0.73 (763/1051) | 0.49 (376/763) | 0.72 (180/249) | 0.38 (196/514) | 0 | - |
| id_s1 | ctrl_scores | 0.80 (3056/3803) | 0.09 (283/3056) | - | - | 281 | 0.00 (0/281) |
| id_s1 | score_opphand4 | 0.86 (1697/1969) | 0.21 (357/1697) | 0.20 (200/998) | 0.22 (157/699) | 0 | - |
| id_s1 | score_alone | 0.82 (670/820) | 0.58 (391/670) | 0.74 (185/249) | 0.49 (206/421) | 0 | - |
| random_s0 | ctrl_scores | 0.82 (3008/3673) | 0.07 (217/3008) | - | - | 316 | 0.01 (3/316) |
| random_s0 | score_opphand4 | 0.86 (2122/2469) | 0.16 (347/2122) | 0.16 (207/1289) | 0.17 (140/833) | 0 | - |
| random_s0 | score_alone | 0.77 (676/875) | 0.58 (393/676) | 0.84 (203/241) | 0.44 (190/435) | 0 | - |
| random_s1 | ctrl_scores | 0.85 (3312/3913) | 0.09 (309/3312) | - | - | 289 | 0.00 (1/289) |
| random_s1 | score_opphand4 | 0.89 (2149/2404) | 0.18 (379/2149) | 0.15 (204/1331) | 0.21 (175/818) | 0 | - |
| random_s1 | score_alone | 0.81 (594/733) | 0.64 (380/594) | 0.79 (174/221) | 0.55 (206/373) | 0 | - |

#### exp_heldout mirror (policy on both seats, 200 games per run)

| run | card | offered / held (affordable) | P(play) when offered | cond true: P(play) | cond false: P(play) | target prompts (2-sided) | enemy chosen |
|---|---|---|---|---|---|---|---|
| B2_s0 | opp_of_ctrl_scores | 0.82 (3006/3676) | 0.09 (273/3006) | - | - | 283 | 0.00 (0/283) |
| B2_s0 | score_opphand4 | 0.86 (2182/2548) | 0.17 (373/2182) | 0.16 (208/1305) | 0.19 (165/877) | 0 | - |
| B2_s0 | score_alone | 0.75 (850/1127) | 0.45 (380/850) | 0.73 (173/238) | 0.34 (207/612) | 0 | - |
| B2_s1 | opp_of_ctrl_scores | 0.86 (3260/3808) | 0.09 (281/3260) | - | - | 246 | 0.00 (1/246) |
| B2_s1 | score_opphand4 | 0.91 (2189/2417) | 0.18 (397/2189) | 0.18 (234/1315) | 0.19 (163/874) | 0 | - |
| B2_s1 | score_alone | 0.83 (673/813) | 0.58 (393/673) | 0.80 (197/245) | 0.46 (196/428) | 0 | - |
| id_s0 | opp_of_ctrl_scores | 0.80 (2141/2683) | 0.16 (352/2141) | - | - | 309 | 0.00 (0/309) |
| id_s0 | score_opphand4 | 0.86 (2019/2347) | 0.18 (357/2019) | 0.17 (172/995) | 0.18 (185/1024) | 0 | - |
| id_s0 | score_alone | 0.74 (796/1078) | 0.48 (383/796) | 0.68 (165/244) | 0.39 (218/552) | 0 | - |
| id_s1 | opp_of_ctrl_scores | 0.81 (3131/3868) | 0.09 (293/3131) | - | - | 286 | 0.00 (0/286) |
| id_s1 | score_opphand4 | 0.85 (1762/2068) | 0.21 (365/1762) | 0.19 (196/1025) | 0.23 (169/737) | 0 | - |
| id_s1 | score_alone | 0.79 (669/847) | 0.59 (392/669) | 0.75 (181/242) | 0.49 (211/427) | 0 | - |
| random_s0 | opp_of_ctrl_scores | 0.78 (1748/2241) | 0.13 (232/1748) | - | - | 354 | 0.00 (1/354) |
| random_s0 | score_opphand4 | 0.87 (2358/2705) | 0.14 (338/2358) | 0.13 (172/1296) | 0.16 (166/1062) | 0 | - |
| random_s0 | score_alone | 0.75 (751/1003) | 0.51 (385/751) | 0.81 (208/257) | 0.36 (177/494) | 0 | - |
| random_s1 | opp_of_ctrl_scores | 0.78 (1418/1807) | 0.15 (215/1418) | - | - | 364 | 0.01 (2/364) |
| random_s1 | score_opphand4 | 0.88 (2295/2609) | 0.15 (335/2295) | 0.12 (156/1286) | 0.18 (179/1009) | 0 | - |
| random_s1 | score_alone | 0.82 (602/733) | 0.63 (378/602) | 0.79 (188/238) | 0.52 (190/364) | 0 | - |
| B2_s0_ft | opp_of_ctrl_scores | 0.86 (4715/5493) | 0.01 (34/4715) | - | - | 38 | 0.00 (0/38) |
| B2_s0_ft | score_opphand4 | 0.87 (2302/2637) | 0.17 (399/2302) | 0.16 (225/1371) | 0.19 (174/931) | 0 | - |
| B2_s0_ft | score_alone | 0.77 (951/1241) | 0.47 (447/951) | 0.78 (204/261) | 0.35 (243/690) | 0 | - |
| B2_s1_ft | opp_of_ctrl_scores | 0.90 (4755/5283) | 0.01 (41/4755) | - | - | 40 | 0.00 (0/40) |
| B2_s1_ft | score_opphand4 | 0.92 (2356/2567) | 0.17 (412/2356) | 0.16 (222/1415) | 0.20 (190/941) | 0 | - |
| B2_s1_ft | score_alone | 0.84 (749/888) | 0.58 (435/749) | 0.78 (217/279) | 0.46 (218/470) | 0 | - |
| id_s0_ft | opp_of_ctrl_scores | 0.80 (3010/3749) | 0.09 (260/3010) | - | - | 235 | 0.01 (2/235) |
| id_s0_ft | score_opphand4 | 0.84 (1633/1936) | 0.21 (344/1633) | 0.19 (211/1090) | 0.24 (133/543) | 0 | - |
| id_s0_ft | score_alone | 0.72 (782/1081) | 0.50 (393/782) | 0.75 (183/243) | 0.39 (210/539) | 0 | - |
| id_s1_ft | opp_of_ctrl_scores | 0.86 (4777/5537) | 0.00 (10/4777) | - | - | 13 | 0.00 (0/13) |
| id_s1_ft | score_opphand4 | 0.86 (1824/2122) | 0.20 (360/1824) | 0.20 (204/1031) | 0.20 (156/793) | 0 | - |
| id_s1_ft | score_alone | 0.81 (743/919) | 0.60 (444/743) | 0.78 (208/266) | 0.49 (236/477) | 0 | - |
| random_s0_ft | opp_of_ctrl_scores | 0.84 (3432/4101) | 0.05 (164/3432) | - | - | 196 | 0.01 (1/196) |
| random_s0_ft | score_opphand4 | 0.87 (2185/2525) | 0.18 (384/2185) | 0.17 (230/1335) | 0.18 (154/850) | 0 | - |
| random_s0_ft | score_alone | 0.76 (793/1049) | 0.53 (422/793) | 0.85 (225/266) | 0.37 (197/527) | 0 | - |
| random_s1_ft | opp_of_ctrl_scores | 0.84 (2640/3138) | 0.11 (288/2640) | - | - | 279 | 0.00 (1/279) |
| random_s1_ft | score_opphand4 | 0.90 (2226/2480) | 0.16 (351/2226) | 0.15 (200/1379) | 0.18 (151/847) | 0 | - |
| random_s1_ft | score_alone | 0.82 (608/743) | 0.66 (401/608) | 0.81 (207/254) | 0.55 (194/354) | 0 | - |

#### exp_heldout_sab1 mirror (policy on both seats, 200 games per run)

| run | card | offered / held (affordable) | P(play) when offered | cond true: P(play) | cond false: P(play) | target prompts (2-sided) | enemy chosen |
|---|---|---|---|---|---|---|---|
| B2_s0 | score1 | 0.87 (2167/2495) | 0.16 (345/2167) | 0.16 (345/2167) | - | 0 | - |
| B2_s1 | score1 | 0.90 (2142/2376) | 0.16 (352/2142) | 0.16 (352/2142) | - | 0 | - |
| id_s0 | score1 | 0.86 (1881/2180) | 0.19 (364/1881) | 0.19 (364/1881) | - | 0 | - |
| id_s1 | score1 | 0.85 (1694/1990) | 0.20 (333/1694) | 0.20 (333/1694) | - | 0 | - |
| random_s0 | score1 | 0.86 (1825/2126) | 0.19 (345/1825) | 0.19 (345/1825) | - | 0 | - |
| random_s1 | score1 | 0.90 (1763/1963) | 0.22 (380/1763) | 0.22 (380/1763) | - | 0 | - |

## Caveats

- Training is short: 10 minutes, about 20k games per run, and the seed spread is large (id train-deck win
  rate 0.81 vs 0.68 across seeds). Strength differences between reprs are not resolved.
- The probe states come from uniform-random games. That explains why probe play rates for the spells are
  lower than in the policy's own games. The real-game tables are the better read of behaviour. The
  target-prompt probes agree with real games.
- score_alone's condition is checked at decision time. The engine checks it as the trigger fires, which is
  the same here (no unit can enter in between).
- Mirror self-play: in the held-out mirror both seats misuse opp_of_ctrl, so its cost partly cancels in
  self-play. Against ISMCTS (which plays it right) it does not, which is why the held-out win rate drops
  (about 0.75 to 0.52-0.58).
- B2 for these variants is hand-written graph edits (no engine code was scanned for the new classes). The
  B vectors for them are the same edit applied to path B's graphs.
- Untrained probe baselines are re-initialised on each call (not seeded), so treat them as about 0.5 ± 0.1.

## What to do next

1. **Use the ask head (`--ask-head`) in every run from now on**: a direct path from the asking card (psrc token) to each
   option's score. Without it, a target choice cannot easily depend on the card, even when the training deck
   requires it (fix check below).
2. **Train on the contrast, hold out a NEW instance of it.** Zero-shot transfer of a side field can only work
   if training rewards reading that field. Put pairs in training where the side decides the right target:
   - ctrl_scores vs opp_of_ctrl_scores;
   - a friendly-vs-enemy GiveMight pair (punch_first +5 vs minus5, discipline vs discipline__enemy).

   Then hold out a third card whose only new feature is a known side value on a new verb or base card. id
   and random have no shared feature to transfer through, so that is where B2 can win.
3. **Conditions need more signal**: opphand4's wasted play costs little, and the networks never learned its
   condition in 10 minutes. Use conditions with a larger payoff gap, or train longer, before reading
   anything into B2 vs id on conditional cards.
4. Evaluate in the policy's own state distribution (real-game rates, as here) rather than on probe states from
   random play, or collect probe states from the trained policy's games.

## Fix check: the ask head (2026-11-19, 12:55-13:08)

Change (`learner/model.py`, opt-in `Net(..., ask_head=True)`, `selfplay.py --ask-head`,
`probe_eval.py --ask-head`):
- h_ask = the transformer state of the psrc token (zero when the prompt has no source);
- every option's score input gets `W_ask(h_ask) * (h_ref + W_ask_src(h_src))`.

So the asking card can raise or lower each candidate object directly. Old checkpoints load unchanged (the
flag defaults to off).

Test: deck `rengar__exp_both`, with Punch First -> ctrl_scores (3), Sabotage -> opp_of_ctrl_scores (3),
Pit Rookie -> score_alone (3). Both target-side cards are TRAINED. id repr, seed 0, the same PPO settings,
two runs at the same time (5 min, then resumed for 9 more; about 150 updates in 13.5 min).
Run dirs: `runs/cx_fix_both_id_ask`, `runs/cx_fix_both_id_noask`.

Probe curve (`ctrl_tgt` mass on friendly / `oppctrl_tgt` mass on enemy; uniform = 0.5 / 0.5):

| update | ask head | old head |
|---|---|---|
| 10 | 0.59 / 0.43 | 0.52 / 0.48 |
| 30 | 0.78 / 0.27 | 0.66 / 0.35 |
| 60 | 0.83 / 0.45 | 0.83 / 0.18 |
| 70 | 0.87 / 0.77 | 0.90 / 0.11 |
| 90 | 0.96 / 0.98 | 0.92 / 0.09 |
| 120 | 0.99 / 0.98 | 0.95 / 0.06 |
| 150 | 1.00 / 0.99 | 0.96 / 0.04 |

Real games (`real_stats.py rengar__exp_both`, mirror, 200 games):

| run | ctrl_scores: enemy picks / 2-sided prompts | opp_of_ctrl: enemy picks / 2-sided prompts | opp_of_ctrl played / offered |
|---|---|---|---|
| ask head | 0 / 259 (all friendly: right) | 271 / 271 (all enemy: right) | 218 / 3210 |
| old head | 1 / 281 | 0 / 8 | 13 / 4355 (learned to avoid the card) |

Card-swap sensitivity on the `oppctrl_tgt` states, mass on enemy (`runs/cx_check_sensitivity_fix.json`):

| asking key | ask head | old head |
|---|---|---|
| opp_of_ctrl_scores | 0.99 | 0.04 |
| ctrl_scores | 0.003 | 0.04 |
| punch_first (+5) | 0.67 | 0.04 |
| hidden | 0.25 | 0.04 |
| a unit (noxus_hopeful) | 0.19 | 0.04 |
| no psrc marker | 0.18 | 0.04 |

With the ask head the target side follows the asking card. Without it the choice is the same whatever card
asks. The fix is verified. The main B2 / id / random zero-shot comparison was NOT rerun with it (user stop).
