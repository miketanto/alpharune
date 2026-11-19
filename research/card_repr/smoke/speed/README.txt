Pipeline speed on this machine (2026-11-19)
===========================================
Ryzen 7 5800X (8 cores / 16 threads), 16 GB RAM, RTX 3060 12 GB. All measurements on WSL cpus 8-15
(= physical cores 4-7 with their SMT siblings; WSL cpus 2k and 2k+1 share a core) or Windows
affinity 8-15, while another job ran 8 game processes on cpus 0-7 at ~100% the whole time.
Windows available RAM was 0.8-1.8 GB throughout. Every rate is the median of 3 repeats (min/max
in results.json). Numbers: results.json; raw timings: raw/.

Game generation (WSL, speed_export = decision_export + timing, one process per physical core)
  random-player              ~81 games/s per process (290k games/h); 3/4 of it is writing the export
  teacher ismcts, games/h per process (ms per search):
     sims 25   1981  (11 ms)      sims 200   210  (119 ms)
     sims 50    975  (25 ms)      sims 400    92  (270 ms)
     sims 100   413-486 (49-53 ms)            ~0.5-0.7 ms per simulation, linear in sims
  decisions per game ~250-285, multi-option ~140-170; every multi-option decision is searched
  sims 100 scaling (ms/search per process): P=1 49.0, P=2 50.0, P=4 52.8, P=8 65.5
     -> aggregate ~490 / 950 / 1820 / 2910 games/h. 4 -> 8 processes on 4 cores (SMT) = 1.61x.
  auto-pay (rune prompts answered with legal[0], no search): rune prompts are 42% of searched
     decisions and 44.6% of search time -> 1.8x if games kept their length; measured on the same 24
     seeds 2.3x (413 -> 948 games/h; searches/game 168 -> 78; games ~10% shorter).
  cross-check: smoke/data's 3,200 sims-100 games: 7.19 s/game mean, 150 searches -> 47.8 ms/search.

Training (smoke/train/train.py model, losses and optimizer; 49k-decision base_teacher subset)
  samples/s, d=64, bs 256: torch 2.7.1 Linux (WSL) 1 thr 1680, 2 thr 2210, 4 thr 2930, 8 thr 3600
                           torch 2.0 Windows (as the existing runs) 2 thr 1140, 8 thr 1650
  bs 64 and 1024 are slower per sample than 256 on CPU (padding to the batch's longest decision).
  d=128 costs 1.7-2.4x d=64 on CPU (bs 1024 at d=128 not measured: memory guard).
  data batching is 1-3% of a CPU step; prep (JSON -> packed) 5.5k decisions/s, 17k with 4 procs.
  GPU (WSL python3 torch 2.7.1+cu126, already installed): d=64 bs 256 13.4k samples/s (52 steps/s),
     bs 1024 24.2k, bs 4096 24.4k; d=128 11.7k / 18.2k / 17.8k. Feeding (CPU batch + copy) 5-9%;
     batches already on the GPU add 15-25%. train.py needs a device fix to run on cuda (results.json).
  inference d=64: CPU 1 thread batch 1 1.9 ms; GPU batch 1 6.2 ms, batch 256 37k positions/s.
  existing 45 runs (3000 steps, bs 256, 3 x 2 threads): 0.29 s/step, 873 s each; best policy step
     median 2750 (783 s), but validation policy loss moved 0.0003 over the last 1000 steps (flat);
     best value step median 750 (220 s), then value overfits (+0.12 MSE by the end).

Projections (hours; assumptions in results.json "assumptions")
  10k teacher games, sims 100:  8 procs (4 cores + SMT) 3.4 | 8 physical cores 2.8 | 14 procs 2.0
                                with auto-pay: 8 procs 1.5 | 14 procs 0.85 (1.1 by the derived 1.8x)
  imitation run, 1M decisions (2.3 epochs, bs 256): CPU Linux 8 thr 0.19 | CPU Windows torch 2.0 0.41
                                | existing launch setup 0.73 | GPU 0.05 (bs 1024: 0.03); prep 0.02
                                needs ~1.1 GB RAM for the packed data
  self-play iteration, 1k games + 1 pass over a 1M-decision window:
        8 procs 0.34 + train (GPU 0.02 / CPU 0.08)  -> 0.37-0.42; with auto-pay 0.17-0.23
       14 procs 0.20 + train                         -> 0.22-0.28; with auto-pay 0.11-0.17
  Bottleneck: game generation (engine search on CPU), not training. A network in the search loop
  (one evaluation per simulation, 15k per game) would cost 29 s/game on 1 CPU thread or 93 s on GPU
  at batch 1, against ~10 s of engine search today; only batched GPU evaluation (~0.4 s/game of GPU
  time at batch 256) avoids that, and IsMctsSearch has no such evaluator.

Files
  src/make_speed_export.py  derives src/speed_export.cpp from smoke/data/src/decision_export.cpp
                            (timing per prompt kind; SPEED_AUTOPAY=1; SPEED_NOREC=1)
  src/speed_export.cmake, src/build.sh   WSL build (target speed_export in the WSL copy only)
  src/run_job.sh, src/phase_sweep.sh, src/phase_scale.sh   the game runs; lists/ the game lists
  src/bench_prep.py, src/bench_train.py  prep and training/inference benchmarks (bench_prep.py writes
                            the subset data/subset_teacher.pt that bench_train.py reads; deleted after)
  src/guard.sh              kills a benchmark if Windows available RAM falls below a threshold
  src/analyze_existing.py, src/analyze_games.py, src/make_results.py   -> raw/*.json, results.json
  src/wsl_load.sh, raw/wsl_load.csv, raw/win_cpu_load.csv   load logs (5 s)
  raw/jobs.csv, raw/games/  per-job wall time and per-game timings; logs/  benchmark stdout
