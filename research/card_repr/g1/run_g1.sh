#!/usr/bin/env bash
# G1 launch (GENERALIST_PLAN.md section 4): Rengar+Azir generalist, league recipe on the mixed 2x2 matchup grid.
#   bash research/card_repr/g1/run_g1.sh              # starts in the background (nohup), logs into selfplay/league/g1
#   bash research/card_repr/g1/run_g1.sh --fg [...]   # foreground; extra args go to overnight_g1.py run
# Output: selfplay/league/g1/{log.txt,matchups.csv,throughput.csv,SUMMARY.txt,train_*.log,driver.out},
#         selfplay/runs/g1_main (league_00050.pt ... league_00500.pt), selfplay/runs/g1_exp.
# Only one GPU training at a time: do not start this while E2 (league/overnight.py) is still training.
set -u
CR=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr
SP=$CR/selfplay
OUT=league/g1
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RB_LEAGUE_INIT=$SP/runs/ov_main/league_02080.pt RB_LEAGUE_BASE=ov_main@2080
unset RB_DECK
FG=0
if [ "${1:-}" = "--fg" ]; then FG=1; shift; fi
if pgrep -f "league/overnight.py run" >/dev/null || pgrep -f "learner/selfplay.py" >/dev/null; then
  echo "WARNING: an overnight.py / selfplay.py training is still running (E2?). One GPU training at a time." >&2
  [ "${G1_FORCE:-0}" = 1 ] || { echo "refusing to start; set G1_FORCE=1 to override" >&2; exit 1; }
fi
cd $SP || exit 1
mkdir -p $OUT
DL=$(date -d '+12 hours' +%H:%M)          # safety deadline only; the run stops at 500 main updates
CMD=(python3 $CR/g1/overnight_g1.py run --deadline "$DL" --tag g1 --out $OUT --stop-main-at 500 \
     --snapshot-every 50 --main-updates 40 --exp-updates 20 --gpu-mem-frac 0.4 --train-cores 0-11 "$@")
if [ $FG = 1 ]; then
  exec "${CMD[@]}"
fi
nohup "${CMD[@]}" > $OUT/driver.out 2>&1 &
echo "G1 started, pid $!; tail -f $SP/$OUT/log.txt"
