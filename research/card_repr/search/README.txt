search: network-guided determinized PUCT (SEARCH_PLAN.md milestones M0 + M1)
==========================================================================
Run under WSL python3 (torch 2.7.1+cu126). Results and discussion: REPORT.txt.

Files
  rbsearch.cpp   C++ search + evaluation harness. #included at the end of
                 selfplay/rbenv/rbenv.cpp (same TU: Env, encodeRaw, applyTracked, card keys),
                 so it is built into librbenv.so. Nothing in src/ was changed.
  rbsearch.py    ctypes wrapper (Search class, load_net, batched GPU Evaluator, dummy evaluator).
  m0.py          M0: leak test (`leak`) and ms-per-simulation benchmark (`bench`).
  eval_search.py M1: searched network vs the same network greedy (and vs a C++ opponent).
  out/           JSON results / logs of the runs in REPORT.txt.

Build (same procedure as rbenv; build.sh now also copies search/rbsearch.cpp to the WSL copy)
  wsl -e bash /mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/selfplay/rbenv/build.sh
  -> ~/riftbound/alpharune/build_pic/librbenv.so

Changes outside this directory (all research code, none in src/)
  selfplay/rbenv/rbenv.cpp  `encode` split into `encodeRaw` (writes into caller-owned buffers and
                            counters; no Env / Handle: thread-safe on distinct states) + a thin
                            `encode` wrapper; `#include` of search/rbsearch.cpp at the end.
                            rbenv observations unchanged: tests/test_agreement_v2.py 0 mismatches
                            (18 games, 3263 decisions), test_determinism ok.
  selfplay/rbenv/build.sh   copies search/rbsearch.cpp.
  smoke/data/src/decision_export.cpp
                            `g_strict_vis` / `visibleTo` (OFF by default: training, the exporter and
                            rbenv are unchanged). With it on, an opponent's hand / facedown / deck
                            card whose identity the observer knows but whose SLOT it does not
                            (CardKnowledge::zone_confirmed false, or a deck card known only within a
                            group) is encoded "hidden". knowledge::identityVisible (src/) reports such
                            a card visible, so the default observation pins its identity to its real
                            slot -- a small leak in the trained network's input (REPORT.txt, M0).
                            The search harness turns strict visibility on for BOTH seats.

Leak test (M0)
  python3 m0.py leak --games 200 --worlds 8 --seed 880001 --out out/leak.json
    Network (league_02080, greedy) plays both seats; at every external decision, 8 worlds are built
    for the decider exactly as a simulation builds them, and encode(world, decider) must be
    byte-identical to encode(truth, decider) (glob, tok, tok mask, act, act mask, psrc, pk, nlegal).
    Also counts WorldSampler::check problems, how many of the opponent's hidden cards a world
    redrew, and a negative control (the same comparison for the OPPONENT's view, whose hidden cards
    were redrawn, must differ). --loose: the default (non-strict) encoder. --deck-model decklist.

Benchmark (M0)
  python3 m0.py bench --games 8 --sims 64 --threads 0 --decisions 200     # single thread
  python3 m0.py bench --games 64 --sims 64 --threads 14 --decisions 1500
    Dummy evaluator (uniform prior, value 0). C++ timers per phase: build (fresh engine, resume the
    last main-phase snapshot, replay the choices since), edit (world draw), descend, encode,
    teardown; wall time per simulation.

Evaluation (M1)
  python3 eval_search.py --sims 64 --games 400 --out out/n64_pool.json
  python3 eval_search.py --sims 0  --games 400 --out out/n0.json        # greedy vs greedy row
  python3 eval_search.py --sims 64 --games 200 --opponent ismcts:sims=200   # vs ISMCTS-200
  options: --cpuct 2.0 --fpu 0.0 (FPU reduction) --vl 1.0 (virtual loss) --leaves 4 (leaves in
  flight per game) --threads 14 --deck-model pool|decklist --perfect (diagnostics only)
  --loose (non-strict encoder) --seed 990001.
  Game slot i plays seed (seed + i // 2); the search plays seat 1 in even slots and seat 2 in odd
  slots (seat-balanced, common seeds); each slot plays exactly one game; all slots run at once.
  win_rate = games won by the searching side; pairs_won_0_1_2 = per seed, how many of its two games
  the search won (a pair the search does not affect is 1-1). stats.changed = searched decisions
  whose played option differs from the network's argmax.

The search (rbsearch.cpp)
  - Leaf-queue loop driven from Python: collect() advances every game (thread pool, parallel over
    games) until it needs the network -> rows() -> GPU forward -> feed(logp, v). A greedy decision
    is one row; a searching game contributes up to --leaves leaves per round (virtual loss; the
    root's first expansion is a round of its own).
  - Every external decision (>= 2 options, not a rune payment) of the search seat is searched
    (mulligan and prompts included) with the full budget. Inside simulations single-option
    decisions and rune payments are auto-taken (legal[0]), as rbenv does in training.
  - One world per simulation: a fresh GameEngine resumed from the live game's last top-level
    main-phase snapshot (GameEngine::resumeFromSnapshot) plus the choices since (from the start of
    the game before the first such decision); then WorldSampler::sample for the root player through
    GameEngine::editHiddenInformation (as IsMctsAgent::makeWorld; engine RNG reseeded).
    deck_model decklist additionally redraws the opponent's unknown cards from its decklist.
  - Nodes keyed by openspiel::infoSetIdFor(state, legal, decider, ROOT player) (single-observer, as
    IsMctsSearch); children by openspiel::actionIds; root children by index into the live legal
    list, world options matched by Intent equality. A node is (re)expanded -- the simulation stops
    there and the leaf is evaluated -- when it is new or offers an option it has no prior for
    (opponent nodes in other worlds); priors kept from the first evaluation that saw the option.
  - PUCT: Q + c * P * sqrt(max(1, sum N)) / (1 + N); unvisited Q = node's first value - fpu.
    Leaf value = the net's value for the leaf's decider, sign-flipped to each node's decider;
    terminal = +-1. No rollouts. Options past A = 96 are never searched (the net cannot score them).
  - Move played: most visited root option, ties by the higher prior (so N = 1 = greedy exactly).

E2 search teacher (CEILING_PLAN.md E2; results e2/REPORT.txt)
--------------------------------------------------------------
  gen_targets.py   generator process: self-play with the CURRENT main weights (file --weights, reloaded
                   when it changes), both seats sample from the net; each external decision is searched
                   with prob --p-search (N=64 PUCT, strict visibility inside the search). Per searched
                   decision it writes the observation encoded with the TRAINING encoder (loose, as rbenv),
                   net prior P, root visits N, root Q, v0, visit-weighted root value, outcome z, weights
                   update, to rolling chunk files <out>/c_<seq>.npz (newest --keep kept).
                   --holdout N writes N records to <out>/holdout.npz and exits (e2/holdout.npz: league_02080).
  rbsearch.cpp     seat_mode 2 ("gen"): both seats sample (temperature 1); searched decisions record a
                   GenRec; the move played there is sampled from the root prior (positions stay on the
                   net's own distribution). rbs_gen_take(i) returns a finished game's records with z,
                   rbs_restart(i, seed) refills a finished slot (continuous generation, no stragglers).
  decision_export.cpp  thread_local g_vis_force overrides g_strict_vis per thread (generator: stored
                   observation loose, simulations strict). rbenv / training unchanged (default -1).
  learner/selfplay.py  --search-buf DIR: SearchBuf (newest --search-window records); every PPO minibatch
                   adds --search-coef * CE(pi', pi_net) on --search-mb buffer records; pi' from
                   e2/targets.py (--search-target cq = completed-Q improved policy, or visits).
                   --export-weights PATH --export-every K: the weights the generator reloads.
                   --search-holdout e2/holdout.npz: CE / KL to fixed targets logged (metrics.jsonl).
  league/overnight.py  --stop-main-at N, --no-rate, --main-extra (main-only trainer options),
                   --train-extra (both roles).
  e2/run_arm.sh A|B N  one arm (fresh league dir league/e2_<arm>, runs e2a_* / e2b_*; arm B also starts
                   the generator on cores 12-15, buffer in ~/e2buf/e2b). e2/eval_e2.py: head-to-heads.
