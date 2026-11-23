# Rune taps for the network (2026-11-22)

A network seat may use its runes' own abilities (exhaust: 1 floating Energy; recycle: 1 floating Power) as moves.
Engine: `feature/float-recycled-rune-energy` (`GameEngine::setManualRunes`), WSL build copy `~/riftbound/float_train`,
selected by `RB_WSL_COPY` (rbenv.py, build.sh). The old copy `~/riftbound/alpharune` is untouched; a library whose
engine has no taps refuses `runes`.

## What changed
- Inputs: no format change. An offered tap already has its rune's token (card, ready / exhausted), the ability index,
  and the pool in the globals. Taps are not recorded in the action history (the pool and runes show them).
- Decisions: a network seat is asked only what it would be asked without taps. A decision whose options are one move
  plus taps is answered with that move (`soleOption`, rbenv.cpp). Cost: it cannot tap in a window where it can do
  nothing else (e.g. floating in response when no card or ability is playable).
- Search (`rbsearch.cpp`, `Cfg::runes`: 0 none, 1 both seats, 2 the search seat, 3 the other): taps inside the worlds
  too. A human's taps (tools/play_web, `manual_seat` without `tap_seat`) stay out of the worlds.
- Model: `tap_bias` (2 learned logit offsets, start -3, `RB_TAP_INIT`). A checkpoint without it loads and computes as
  before where no tap is offered.
- Root noise: a tap gets a tenth of a move's Dirichlet noise (`--dir-tap 0.1`), see below.

## Checks
- `taps_check.py parity`: 3,793 decisions, offering taps that are never taken changes nothing the network sees.
- `taps_check.py random`: 200 random games with taps, 0 errors; 14 options past the 96-option limit in 46,240 decisions.
- Leak test with taps forced (`m0.py leak --runes 1`): 54,996 worlds, 0 mismatches.
- Speed: generator records/s the same with and without taps (two 4-minute runs).

## Found on the way
- AZ @9944 (never saw taps) would put 46% of its policy on them; at -3 it is 5%.
- Recipe 1's target is visits, and root noise becomes visits: in 272 steps the policy mass on taps went 5% -> 17%,
  recycles included, though the search rated them about 0.03 below the main other move. Hence `--dir-tap 0.1`.
- Recipe 2 (completed-Q target) on the same 69,632 records, 270 steps, then 400 games vs G2 greedy (mirror, search with taps):
  recipe 1 0.695, recipe 2 0.535 (8 exhausts a game), recipe 2 + anchor 0.1 0.635; untrained AZ @9944 0.628.
  These three used `train_az2.py` with zmix 0.5. One run each: only recipe 2's drop is clearly outside the noise.
- Baseline (floating engine, AZ @9944 search vs G2 greedy, 400 games): mirror 0.678 without taps, 0.628 with; Rengar
  0.610 / 0.605. It taps about 0.02 times a game untrained.

- Recipe 1 (visits) with `--dir-tap 0.1`, closed loop 21 min (392 steps): taps' share of visits 8.9% -> 10.9%
  (before the change 15.2% -> 21.5%); 1.75 taps a game.
- That run's network lost 19 points: 0.483 vs G2 greedy without taps (@9944: 0.678). Cause: the value target. Training
  started after 30k records (about 400 games), and the value head memorised their results (error on unseen records
  0.58 -> 0.78; the earlier run started with a 200k buffer). Same records, 390 steps, `train_az2.py --pi visits`:
  zmix 0 (game result only) 0.503 / 0.533 with taps; zmix 0.5 (half the search's root value) 0.673 / 0.645.
  So the run uses `train_az2.py --pi visits --zmix 0.5 --min-records 100000` (run_taps.sh, AZ_TRAIN_ARGS overrides).

## Run
    AZ_HOURS=8 nohup setsid bash az/run_taps.sh > /dev/null 2>&1 &     # run_taps, buf_taps, from run_b2/az_009944.pt
    bash az/eval_taps.sh run_taps/az_004000.pt                          # taps off / on, mirror and Rengar
    AZ_RUNES=0 AZ_RUN=run_notaps AZ_BUF=buf_notaps ... run_taps.sh      # the control: new engine, no taps
Without the control, a gain over @9944 cannot be credited to taps rather than to the floating engine itself.
The web client cannot let the network tap yet (it passes no tap seat to the search).
