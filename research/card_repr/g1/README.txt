G1: Rengar + Azir generalist (GENERALIST_PLAN.md section 4 "G1", measurement rules section 5). 2026-11-20.
Built while E2 was running: nothing E2 uses was edited or rebuilt (learner/selfplay.py, league/*.py, rbenv,
librbenv.so, learner/model.py are imported read-only). All G1 code is in this directory. No C++ change was needed.

Launch (WSL; one GPU training at a time, so after E2 has finished; the script refuses while an overnight.py /
learner/selfplay.py training is running unless G1_FORCE=1):
    bash /mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/g1/run_g1.sh
  nohup in the background; driver output selfplay/league/g1/driver.out, progress selfplay/league/g1/log.txt.
  Runs: selfplay/runs/g1_main (snapshots league_00050.pt ... league_00500.pt), selfplay/runs/g1_exp.
  Stops at 500 main updates (the deadline is only a +12 h safety stop). --fg runs it in the foreground.
After training:
    bash /mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/g1/eval_g1.sh   [--cores 8-15]
  -> g1/eval/REPORT.txt, report.json, pairs.jsonl (cache: a rerun plays only the missing pairs).

Files
  selfplay_g1.py   copy of learner/selfplay.py (league path only). Change: MultiEnv = one RBEnv handle per ORDERED
                   matchup (--matchups, default rengar:rengar,rengar:azir,azir:rengar,azir:azir), n_envs split
                   equally (64 slots each at 256). Observations of the handles are concatenated, so there is one
                   learner forward per step as before; choices are split back. The learner seat alternates per game
                   in each slot, so the learner plays every (own deck, opponent deck, seat) combination, uniform over
                   slots. League opponents are nets, so they play whichever deck their seat holds. PPO unchanged.
                   Removed (unused by the league recipe): CF data, E2 search teacher / holdout, in-training C++ eval
                   and probes, the non-league pool. Logs per matchup cell "<learner deck>_v_<opponent deck>":
                   log.csv wr_<cell> (learner score vs league opponents, draws 1/2), n_<cell>, dec_<cell>;
                   metrics.jsonl 'matchups' with seat-1 / seat-2 splits.
  overnight_g1.py  copy of league/overnight.py: same MAIN (OSFPMix 0.8 OSFP own history seeded with ov_main@2080
                   + 0.2 PFSP (1-p)^2 over cf_id@230, gate4_B2@150, gate4_id@150 and exploiter snapshots; snapshot
                   every 50) and MAIN EXPLOITER (vs frozen main copy, 70%/200 games or 60 updates, reset to
                   ov_main@2080), 40 main / 20 exploiter updates per phase, gpu_mem_frac 0.4, train cores 0-11 --
                   the E2 settings. Both roles use selfplay_g1.py on the mixed grid. No rating cycles (as E2:
                   match.py ratings are single-deck and would be the wrong games). After every phase the
                   per-matchup win rate vs pool is written to log.txt and matchups.csv; SUMMARY.txt has it per
                   50-update block. Main snapshots at multiples of 50 are never pruned. --extra-members
                   name=path,... adds fixed nets to the PFSP pool (default none, as in E2).
  run_g1.sh        the launch command above.
  eval_g1.py/.sh   P1 pilot swap per deck, forgetting, midpoint (method in the eval_g1.py docstring). Greedy,
                   8 procs, 2000 games per pair, both seat orders on the same deals, Wilson CIs.
                   S_R = runs/e2a_main/league_00500.pt (ov_main@2080 + 500 updates, budget-matched with G), else
                   runs/e2a_main/latest.pt, else FALLBACK ov_main@2080 (printed in the report). S_A = az_main@300.
                   A player against itself on the same deck is exactly 0.5 by design and is not played.
                   d_X = mean over refs (S_R on Rengar, S_A on Azir) of WR(G on X) - WR(S_X on X); bar -0.03;
                   mean over decks and worst deck reported. ~12 pairs x 2000 games, roughly 30-60 min.
  smoke_g1.sh      tiny CPU smoke test (16 envs, 384 samples/update, 3 main + 2 exploiter updates). Its logs are
                   in smoke/ (the smoke checkpoints were deleted). eval_smoke/ is the eval script's smoke run
                   (16 games per pair with the smoke checkpoints; numbers are meaningless, it shows the plumbing).

Choices and limits (read before interpreting results)
  * League sampling is ONE OSFP/PFSP table pooled over the four matchups (league/integration.py is shared with E2
    and was not copied). The plan's "PFSP per matchup" and per-matchup advantage normalisation are not done.
    Matchups are uniform over env slots, not over games or samples: longer games give a cell more decisions
    (dec_<cell> shows the split).
  * One main exploiter on the mixed grid. Per-legend exploiters would be a small change: a --learner-deck option
    in selfplay_g1.py (fix the learner's seat to the seat holding that deck, instead of alternating) plus one
    exploiter run/config per legend in overnight_g1.py's phase rotation. Not implemented.
  * Win rates in the training log are sampled play vs whatever league opponent was drawn (mostly self / own
    history): a health signal per cell, not an evaluation. Use eval_g1.sh for numbers.
  * Throughput at full size is not measured (the GPU was in use by E2). Estimate from E2 (Rengar ~10.5 s/update)
    and the Azir league (~14 s/update): ~12 s/update main, ~13 s exploiter, i.e. ~8 min per 40 main updates plus
    ~4.5 min for the 20 exploiter updates, ~2.7 h for 500 main updates.
