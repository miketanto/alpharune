league: league-training infrastructure for Riftbound no-search self-play (2026-11-19)
=====================================================================================
Goal: let the main training move from "latest + uniform past checkpoints" (learner/selfplay.py today) to a
league: a registry of frozen agents with roles, an opponent sampler (uniform / PFSP / OSFP / AlphaStar roles),
exploiter roles with snapshot / reset rules, network-vs-network evaluation and a rating table.
Nothing here edits learner/ or src/. The training-loop hook is an UNAPPLIED patch (selfplay_integration.patch).
Run everything under WSL python3 (as the rest of selfplay/). Evaluation pins itself to CPU cores 12-15.

Files
  common.py        paths, checkpoint inspection (options + schema detected as verify_eval.py does), net loading,
                   to_dev / act_rows (copied from learner/selfplay.py so league code does not import the trainer)
  model_snapshot.py  verbatim copy of learner/model.py at 19:14 (md5 0011b2e1...). learner/model.py is being
                   edited by another agent; league code uses this copy by default. LEAGUE_MODEL=live switches to
                   learner/model.py (what the trainer integration effectively uses, since the trainer builds the
                   nets through its own model.Net).
  registry.py      checkpoint registry (JSON)
  match.py         network-vs-network games (both seats external) and network-vs-anchor games (via verify_eval.py)
  anchor_games.py  anchor-vs-anchor games (C++ agent vs C++ agent) via the WSL build's `riftbound` CLI,
                   seat-balanced, so the anchors are linked directly and not only through the networks
  ratings.py       incremental results store (JSONL) + Bradley-Terry/Elo fit with CIs, tables
  opponents.py     samplers: Uniform, PFSP (squared/hard/linear/variance/even), OSFP, RoleSampler (AlphaStar roles)
  exploiter.py     ExploiterConfig + RoleController: snapshot / reset / target-refresh rules per role
  integration.py   LeagueOpponents: the trainer-facing object (per-game opponent assignment, reporting,
                   snapshots, OSFP periods, exploiter resets, state_dict for resume)
  configs/         main_uniform.json (today's behaviour), main_osfp.json, main_pfsp.json, main_exploiter.json,
                   league_exploiter.json
  selfplay_integration.patch   unified diff against learner/selfplay.py (md5 7d4a17b7... at 19:30), NOT applied
  demo.py          the demo round robin; registry_demo.json, results_demo.jsonl, ratings_demo.txt/.json its output

1. Registry (registry.py)
-------------------------
Registry(path).  Member fields: id, kind (net | anchor), path (relative to selfplay/), spec (anchors), repr,
opts (ask_head, card_norm, ask_opt, opt_cost, opt_refs: detected from state-dict keys exactly as verify_eval.py
does), schema (1..4 from the stored dims: F 17 / T 64 -> v1, K 49 -> v2, 56 -> v3, 61 -> v4; from W_num's shape
if a checkpoint has no dims), loadable (schema >= 2; v1 checkpoints do not load on the current env, rbenv
README), why, parent (previous snapshot of the run, or 'file:<init>' from the run's config.json), role
(main | main-exploiter | league-exploiter | anchor | historical), run, update, created (file mtime), frozen.
Fixed anchors: add_anchors() registers random-player, ismcts:sims=25, ismcts:sims=100, ismcts:sims=200.
  reg.add_checkpoint(path, role, parent, name)   reg.add_run('gate4_id', role='main', which='latest'|'all'|[50,100])
  reg.members(role=..., kind=..., loadable=...)  reg.abspath(id)  reg.table()  reg.save()
  CLI: python3 registry.py add-run gate4_id --role main ; python3 registry.py anchors ; python3 registry.py list

2. Evaluation (match.py)
------------------------
play_nets(ckA, ckB, n): rbenv with opponent=None, both networks on CPU (1 thread per process), greedy by
default (--sample to draw from the policy), --procs processes (default 4) pinned to cores 12-15. One game per
env slot (its first game; later games in that slot are finished with the first legal option and ignored), so
the sample is unbiased as in verify_eval.py. Seats alternate by global slot index: A is seat 1 in even slots,
seat 2 in odd ones, so with an even game count each net plays each seat n/2 times (the record keeps A's seat-1
games and wins, so seat bias is visible). Every pair uses the same game seeds (common random numbers).
play_anchor(ck, spec, n): runs verify_eval.py (taskset -c 12-15) and parses its JSON; network = seat 1 there, so
anchor games are NOT seat-balanced (documented in every table). If verify_eval.py fails (e.g. learner files
mid-edit) the identical loop runs natively (record field via = 'native').
play(reg, a, b, n) is the registry-level entry; CLI: python3 match.py <id A> <id B|anchor:spec> <n>.
anchor_games.py A B n: anchor vs anchor through ~/riftbound/alpharune/build/riftbound (rbenv cannot seat two
C++ agents); half the games each way round. Caveat: there the agents also answer rune-payment prompts
themselves, while rbenv auto-pays them for everyone.
Speed (demo): net vs net 100 games ~7 s; vs random-player ~7 s; vs ismcts:sims=25 ~30-60 s;
vs ismcts:sims=100 and 200 several minutes per 100 games (the search dominates).

3. Ratings (ratings.py)
-----------------------
ResultStore: append-only JSONL of match records; matrix() sums all records of a pair, so adding games is just
appending (incremental); pair_games() lets a scheduler skip pairs already played.
fit_bt: Bradley-Terry on the Elo scale, P(i beats j) = 1/(1+10^((Rj-Ri)/400)), draws = 1/2.
  Anchor: anchor:random-player = 0 (the C++ random-player, pass 0.2). Documented choice: it is cheap, fixed and
  always in the matrix. Any member can be the anchor (fit_bt(..., anchor=...)).
  Prior: one virtual drawn game per PLAYED pair (prior_games=1). Without it a 100-0 pair has no finite MLE; a
  member whose every result is a sweep gets a rating set by the prior (flagged "bound set by prior").
  Fit: Newton on the penalised likelihood. CIs: Laplace (inverse Hessian, anchor fixed: the CI of R_i - R_anchor)
  and a parametric bootstrap (each pair's wins redrawn ~ Binomial(n, p_obs), refit, 2.5/97.5 percentiles).
  Members not connected to the anchor through games get no rating.
  `python3 ratings.py selftest` recovers known ratings (0/300/500/800/1200) from simulated games.
ratings_table(fit), winrate_matrix(names, W, N, order).

4. Opponent sampling (opponents.py)
-----------------------------------
API (per game or per batch): add(id), sample(n) -> ids ('self' = current learner), update(id, score: 1/0/0.5),
probs() -> {id: p}, win_rates(), state_dict()/load_state_dict(). Win rates: EMA (decay 0.99) with a 2-game
prior at 0.5.
  Uniform(p_self=0.5, window=20)          today's behaviour.
  PFSP(p_self, weighting, power, window)  P(B) ~ f(P[learner beats B]) + 1e-3 floor; weighting 'squared'
                                          (1-p)^2 = AlphaStar f_hard, 'hard' (1-p)^k, 'linear', 'variance'
                                          p(1-p) = f_var, 'even'.
  OSFP(p_self=0.6, optimism=1.0, xi=0.55, max_periods=6, period_samples, f=None)
     Sources: Xiao et al. 2023, "Mastering Strategy Card Game (Hearthstone) with Improved Techniques",
     arXiv 2303.05197, Sec. II-C, Appendix VII-4 Algorithm 1 and Table IV (p = 0.6, xi = 0.55, c = 6, LP =
     3.2e8 samples); which follows Xi et al. 2023, "Mastering Strategy Card Game (Legends of Code and Magic) via
     End-to-End Policy and Optimistic Smooth Fictitious Play", arXiv 2303.04096, Sec. 2.1 Eq. 6 (smooth best
     response to F~_k + F_k: the optimistic extra copy of the latest payoff), Sec. 2.2 Eq. 16 (alpha_i = 1 for
     i < k, alpha_k = 2), Appendix B Algorithm 1.
     Implemented: with prob p_self the opponent is the current learner (the "latest policy" of smooth FSP),
     otherwise H[i] with weight alpha_i, alpha_k = 1 + optimism for the newest historical model (optimism = 1
     = the paper; 0 = plain smooth FSP). Learning periods: per game vs H[i], G[i] += +-1, C[i] += 1; at the end
     of a period (tick(n) counts learner decisions to period_samples) end_of_period() says "add the learner to H"
     if G[i]/C[i] > xi for every i, or if more than max_periods periods passed without an addition; G, C reset.
     NOT specified by the papers (our choices, marked as such): the function f in "i ~ f(...,(G[i],C[i]),...)"
     (default f = 1; f='pfsp' multiplies by (1-p)^2 of the period win rate), the period length in our units
     (default 163,840 decisions = 10 updates), and xi: Hearthstone Table IV says 0.55, the LoCM Table 7
     reportedly 0.7 (read from a summary; check before relying on it). xi = 0.55 on G/C means win rate > 77.5%.
     The RL step is untouched (PPO + GAE here; the paper uses V-trace targets with a PPO-clip objective, Sec.
     IV-F Eq. 7).
  RoleSampler(role)  AlphaStar (Vinyals et al. 2019, Nature 575, Methods "League composition"):
     main: 35% self-play, 50% PFSP squared vs every frozen member, 15% PFSP vs forgotten main snapshots
       (p < 0.3) and main-exploiter snapshots (self-play if there are none);
     main-exploiter: the current main agent(s); if its win rate vs them < 0.2, half PFSP variance vs main
       snapshots; never self-play;
     league-exploiter: PFSP squared vs every frozen member; never self-play.
  make_sampler(cfg) builds one from a config's "opponents" block. `python3 opponents.py` prints a simulation.

5. Exploiters and roles (exploiter.py, configs/)
------------------------------------------------
ExploiterConfig: role, init (checkpoint to reset to; '' = fresh net), snapshot_every (main), win_threshold
(0.7), min_games (200 per target since the last reset), max_updates (main-exploiter 60, league-exploiter 40;
AlphaStar's 4e9 / 2e9 steps rescaled to our ~150-update runs), reset_prob (main-exploiter 1.0, league-exploiter
0.25, main 0), target_refresh (main-exploiter: reload the main run's latest.pt every 10 updates), target.
RoleController.after_update(update, winrates, games) -> {snapshot, reset, refresh_target, reason}.
configs/main_exploiter.json: set "target_ckpt" to the main run's latest.pt; the exploiter trains only vs that
frozen copy (no self-play, only its own seat's decisions are trained on), snapshots into the registry with role
main-exploiter when it beats it >= 70% (>= 200 games) or after 60 updates, then resets to init with a fresh
optimizer. configs/league_exploiter.json seeds its pool with every loadable registered net.

6. Integration (integration.py + selfplay_integration.patch, NOT applied)
-------------------------------------------------------------------------
LeagueOpponents(cfg_path, dims, make_net, device, run_dir, run_name, seed):
  begin_update(u)      draw n_slots opponents from the sampler for this update, load their nets (cache);
  assign(env)          per new game: None (self-play) or a member id (sampler's SELF prob, then the slots'
                       relative weights); a game keeps its opponent across updates;
  net(id)              frozen net of that member;   report(env, score|None)  per finished game;
  end_update(u, net, n_samples, save_fn)  OSFP period / snapshot rules / exploiter resets; registers snapshots
                       (league_<u>.pt in the run dir) in the registry; returns {snapshot, reset, init};
  state_dict()         saved in latest.pt as 'league' by the patch; log_row() goes to metrics.jsonl.
The patch adds `--league <config>` to learner/selfplay.py. Without it the trainer behaves as today (the only
difference: latest.pt gains a 'league': None key). With it: per-env opponent ids replace opp_self/pool_net; opponent rows are grouped by member and
evaluated with that member's net (at most n_slots + 1 forward passes per env step); results are reported per
game; after each PPO update end_update() runs, and on an exploiter reset the learner is reloaded from init (or
re-initialised) with a fresh Adam. Apply with:  cd <repo root> && patch -p1 --dry-run < research/card_repr/
selfplay/league/selfplay_integration.patch  (then without --dry-run). It was made against selfplay.py md5
7d4a17b7...; the other agent is editing that file, so expect to re-base the hunks by hand (they are small).
Tested (2026-11-19 19:20-19:30, CPU only, CUDA hidden, cores 12-15, on a scratch copy of the patched
selfplay.py with the snapshot model.py; nothing in learner/ touched):
  * `patch -p1 --dry-run` applies cleanly to the current learner/selfplay.py (md5 7d4a17b7...).
  * --league with an OSFP config (2 frozen gate4_id snapshots, period = 1 update): 3 updates, 32 envs; per-env
    opponents, per-game reports, a learning period ending and the learner joining H (registered as t_osfp@1);
    a fresh learner's logged win rates vs gate4_id@50/@100 were 0.06-0.3 as expected. (The first run showed
    the report placed after the seat flip, i.e. inverted outcomes; fixed in the patch before it was written.)
  * --league with a main-exploiter config (target = gate4_id latest.pt, max_updates 2): only the target is
    played, no self-play rows trained, target refreshed, snapshot + reset at update 2 with a fresh optimizer.
  * without --league: 2 updates, pool behaviour as before.
`python3 integration.py` runs the same call pattern without the trainer (16 envs, 2 frozen gate4_id
snapshots, OSFP; writes only to a temp dir).

7. Demo / tests
---------------
Full output: ratings_demo.txt (table, win-rate matrix, fit check, every match record); ratings_demo.json (fit);
results_demo.jsonl (7,100 games, 75 records); registry_demo.json; logs demo_run.log, demo_run2.log,
anchor_run.log. Reproduce: python3 demo.py run [--hist --all200]; python3 anchor_games.py A B n --results
results_demo.jsonl; python3 demo.py report.
Played (greedy, CPU cores 12-15, 19:18-20:28): every pair of the 9 loadable nets (latest of gate1_sanity_id_ask,
gate2v2_sanity_B2_ask, gate3_B2_v3, gate4_B2, gate4_id + intermediate gate4_id / gate4_B2 @50 and @100), 100
games each; each net vs random-player, ismcts:sims=25 and ismcts:sims=100 (100 games) and ismcts:sims=200 (60
games); anchors among themselves: random-player vs ismcts25 (100), ismcts25 vs ismcts100 (100), ismcts100 vs
ismcts200 (60). 0 errors, 0 draws. No pairs had to be cut.
SKIPPED: real_smoke_id (schema v1: T 64, F 17, K 46; does not load on the current v4 env).

  rank member                    Elo   95% CI (bootstrap)
     1 gate4_id@150              974   [915, 1048]
     2 gate4_B2@150              964   [904, 1041]
     3 gate4_B2@100              924   [859,  997]
     4 gate4_id@100              868   [815,  937]
     5 gate1_sanity_id_ask@100   743   [685,  816]
     6 gate4_B2@50               680   [630,  749]
     7 gate4_id@50               680   [623,  745]
     8 gate3_B2_v3@50            665   [614,  733]
     9 gate2v2_sanity_B2_ask@70  647   [589,  719]
    10 ismcts:sims=200           498   [431,  569]
    11 ismcts:sims=100           396   [334,  467]
    12 ismcts:sims=25            287   [232,  349]
    13 random-player               0   (anchor)
Observations:
  * gate4_id and gate4_B2 final checkpoints are tied (52-48 head to head); B2 is ahead at update 100
    (924 vs 868, 54-46 head to head), level at 50. Every greedy net beats ISMCTS-200 (73-95%).
  * The fit check finds no pair worse than |z| = 2.4 over 75 pairs: one rating line describes these games.
  * Seat balance: over 3,600 net-vs-net games seat 1 won 49.4%; anchor-vs-anchor games are seat-balanced.
    Net-vs-anchor games are all network-on-seat-1 (verify_eval.py), so anchor ratings carry that bias, if any.
  * The CIs are relative to random-player, which every net beats 97-100%; the nets' ratings vs each other are
    tighter than these CIs suggest (net-net pairs are many and close).

8. What's left
--------------
* Apply / re-base the patch after the counterfactual-pair work lands in selfplay.py; then a short real run
  with configs/main_osfp.json and one with main_pfsp.json, comparing against main_uniform.json at equal games.
* Anchors as training opponents: an RBEnv takes one C++ opponent spec per handle, so ISMCTS/random opponents
  in training need a second env handle in the actor (not done; they are rating anchors only).
* Run several learners (main + exploiters) at once: today each is a separate trainer process sharing the
  registry JSON; registry writes are atomic (os.replace) but not locked -- add a file lock before running
  concurrent writers.
* Periodic rating job: a cron-like loop that adds new snapshots and plays only missing / thin pairs
  (demo.py's _play already skips pairs with enough games); choose pairs by rating uncertainty.
* Network-vs-anchor games are seat-1-only (verify_eval.py's convention); a seat-balanced variant needs the C++
  opponent on seat 1, which rbenv does not offer.
* The OSFP f() and xi are under-specified in the papers (see 4); tune period_samples to our update size.

9. Overnight driver (overnight.py / overnight.sh), 2026-11-19 22:15-
--------------------------------------------------------------------
The patch is now APPLIED to learner/selfplay.py (it applied cleanly on top of the cf-pair version, md5 7d4a17b7;
pre-patch copy: league/selfplay.pre_league.py.bak). Also added: --gpu-mem-frac (0 = no cap, default).
League code changes: opponents.OSFPMix (kind 'osfp_mix': p_osfp x OSFP over the learner's own history + rest
PFSP over the league); integration.LeagueOpponents.sync() (config 'sync_roles': pick up registry members of
those roles written by other processes, drop pruned members), 'seed_history' (OSFPMix history seeds); with an
OSFP sampler, periodic snapshot_every snapshots are registered for ratings but only OSFP-admitted ones join the
history; exploiter end_update rows log reason / wr_since_reset / games_since_reset.
Launch (foreground, exits at the deadline):  bash league/overnight.sh --deadline 07:30
Design, defaults and outputs: the docstring of overnight.py. Summary only: python3 league/overnight.py summary
Defaults chosen after the dry runs (league/overnight_dry, overnight_dry2; runs/dry*_main, dry*_exp): n_envs 256,
n_slots 1 (one opponent net per update besides the learner: main 26-36 games/s vs 22-29 with 2 slots, measured
with the gate-5 jobs loading the CPU), main 40 / exploiter 20 updates per phase, snapshot every 50 main updates,
rating every 45 min (100 games per pair, ismcts 100/200 included), training stops 25 min before the deadline for
a final rating, at most 30 league_*.pt files.
