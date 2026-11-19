Play-changing variant pairs (Rengar mirror)
==========================================

Question: can we find card variant pairs where the teacher's RIGHT WAY TO PLAY
the card differs (not just its value)? The earlier synthetic variants changed
the teacher's best move in only 0-9% of states.

Setup
-----
- Rengar mirror only: decks/rengar.txt on both sides. Bases: punch_first,
  pit_rookie, sabotage. Discipline and lonely_poro were dropped.
- States (jobs/<base>.txt, jobs/selection.json, make_jobs.py): main-phase
  decisions from TEST-split base_teacher games with BOTH decks rengar.txt,
  where the base card is in the decider's hand and its play is offered. We kept
  the earlier selection's R-vs-R states (punch_first 34, pit_rookie 18,
  sabotage 15) and topped each base up to 48 states from the same pool, with
  the same rule (at most 2 per game and base), seed 20261119.
- Teacher: decision_export swap (IsMctsSearch, uct_c 2.0, score-difference
  leaf). Root stage: sims 200, 12 seeds. Target stage: sims 200, 8 seeds.
  Ran on WSL cores 8-15 with at most 6 processes (build.sh, run.sh,
  run_tgt.sh, run_tgt2.sh, run_rob.sh, run_rob400.sh).
- Variant code: ScoreSpellVar, PunchKillVar, PunchScoreTargetVar, ScoreUnitVar (and the
  existing PunchFirstVar for minus5) in C_engine/card_probe.cpp, registered only in
  smoke/data/src/decision_export.cpp (ids 1631-1642). They are not in the
  probe's registerSynthetics, so card_probe output is unchanged. src/ was not
  touched. Each variant keeps the base CardDef (cost, domains, keywords, type);
  only the effect changes.
- decision_export swap has a new --tgt flag. After the root searches it plays
  the first card action and answers the decider's non-target prompts (e.g.
  payment) with their first option. It then runs the same K teacher searches
  at the target prompt ("tgt" in each version).

Metrics (analyze.py, analyze_tgt.py). q is averaged over seeds.
- play decision: "play" if the best card action's q is strictly above the best
  non-card action's q; otherwise "noplay"; "illegal" if no card action exists.
- (a) play_flip_rate: orig vs variant decision differs, over states where both
  are legal.
- (b) dst_flip_rate: when both versions play, the best card action's
  destination changes. Spell targets are not part of the main-phase action.
  They are chosen at a later prompt, so for spells (b) is measured by the --tgt
  stage: target_flip_rate and target_side_flip_rate (friendly vs enemy).
- (c) noise: the same rates between the two seed halves of the ORIGINAL state.
  We also check orig decisions across processes (orig is re-searched in each
  variant's process).
- (d) swap_no_legal_play_rate: the variant has no legal play when orig did.
- For conditional variants: swap play rate when the condition holds vs when it
  does not. The condition is read from the observation, using "board card with
  Might > 0" as a proxy for a unit.

IMPORTANT caveat
----------------
The teacher's leaf value is score difference only (agents/mcts_agent.cpp,
RiftboundHeuristicEvaluator). An effect changes the teacher's choice only if it
moves points within the search horizon. That explains the earlier null results
(+4 vs +5, draw 2 vs 1). It is also why the variants that pass here all score
points, directly or under a condition. The seed-half noise floor is near zero
because q is almost identical across seeds at these budgets. So the noise
floor understates real teacher noise; see the sims 100 and 400 re-runs in
results.json.

Results: results.json has one entry per pair, with keep/drop and the reason.
Raw data: raw_s200/, raw_s100/, raw_s400/ and raw_tgt/. To regenerate, run
analyze.py <rawdir>, then analyze_tgt.py, then make_results.py.

Summary (sims 200; n = 48 states per pair; noise = seed-half flip rate of orig)
variant                       (a) play flip  noise  cond: plays if true / if false   s100  s400  keep
punch_first__score1             0.85         0.00   -                                0.42  0.90  keep
punch_first__score_nounits      0.23         0.00   10/10 vs 0/38                    0.19  0.23  keep
punch_first__score_bfunit       0.31         0.00   16/21 vs 0/27                    0.06  0.35  keep
punch_first__score_behind       0.29         0.00   13/14 vs 0/34                    0.15  0.27  keep
pit_rookie__score1              0.35         0.00   -                                0.25  0.58  keep
pit_rookie__score_alone         0.21         0.00   10/11 vs 0/37                    0.23  0.23  keep
sabotage__score1                0.83         0.04   -                                0.62  0.81  keep
sabotage__score_opphand4        0.58         0.04   30/33 vs 1/15                    0.50  0.67  keep
punch_first__kill               0.00         0.00   -                                0.00  0.04  drop
(d) legality: no variant lost its legal play in any state (all rates 0.00).

Conditional pairs: the overall flip rate is capped by how often the condition
holds (10-33 of 48 states). Inside those states the variant is played 85-100%
of the time, and almost never outside them. This is the "read the state to know
whether to play" signal, so they are kept even though some are below 0.30
overall.

Target stage (punch_first, 36 states with 2+ targets, 32 with both a friendly
and an enemy target; noise = seed-half flip of the ORIGINAL card's best target)
variant                       best target is enemy   side flip vs orig   noise (side)  keep
(orig punch_first, +5)          0.56                   -                   0.25         -
punch_first__ctrl_scores        0.00                   0.56                0.25         keep
punch_first__opp_of_ctrl_scores 1.00                   0.44                0.25         keep
punch_first__kill               0.81                   0.44                0.25         drop (weak)
punch_first__minus5             0.69                   0.12                0.25         drop
ctrl_scores vs opp_of_ctrl_scores: the best target side flips in 32/32
two-sided states. The base card's own target choice is close to noise, so a
variant pair is the cleaner test than variant vs base.
(b) by destination (units) could not be measured: the base Pit Rookie is
never the teacher's best play, so "both versions play" never happened.

Not done / caveats
- Every passing variant works through points, because of the score-only leaf.
  Kill/debuff effects (kill, minus5), which should change play for a human, do
  not change the teacher's play at these budgets. A teacher with a richer leaf
  value would be needed for those.
- The seed-half noise floor is near zero because q barely depends on the seed
  at the root (the worlds differ little). Budget sensitivity is the larger
  effect: compare sims 100 / 200 / 400.
- The --tgt stage answers non-target prompts before the target (payment) with
  their first option, and searches only the first card action's target prompt.
- Discipline and lonely_poro were dropped (Rengar mirror only). No new base
  cards were added.
