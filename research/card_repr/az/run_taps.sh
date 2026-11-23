#!/usr/bin/env bash
# AlphaZero Azir run in which both seats may tap their runes (TAPS.md). Floating engine: ~/riftbound/float_train.
#   AZ_HOURS=8 nohup setsid bash az/run_taps.sh > /dev/null 2>&1 &
# Generator cores 0-11 + trainer cores 12-13, both on the GPU; stops after AZ_HOURS hours (default 8). Restarts the
# generator if it dies. A run directory with checkpoints is resumed (use the same AZ_RUN / AZ_BUF).
set -u
AZ=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/az
cd $AZ || exit 1
export RB_B2=v2 RB_WSL_COPY=${RB_WSL_COPY:-$HOME/riftbound/float_train}
HOURS=${AZ_HOURS:-8}; RUN=${AZ_RUN:-run_taps}; BUF=${AZ_BUF:-buf_taps}
INIT=${AZ_INIT:-run_b2/az_009944.pt}
# Policy target = visits; value target = half game result, half search value; first step after 100k records (TAPS.md:
# game results alone on a small buffer cost 17 points of win rate in 390 steps).
TRAINER=${AZ_TRAINER:-train_az2.py}
TARGS=${AZ_TRAIN_ARGS:---pi visits --zmix 0.5 --min-records 100000}
MATCHUPS=${AZ_MATCHUPS:-azir:azir,azir:rengar,rengar:azir,azir:diana,diana:azir,azir:azir}   # e.g. kennen:kennen
RUNES=${AZ_RUNES:-1}                    # 0 = the control: the same run without taps
export RB_TAP_INIT=${RB_TAP_INIT:--3.0}
mkdir -p $RUN $BUF
if pgrep -f "[t]rain_az2?\.py|[g]en_az\.py" > /dev/null; then echo "another az run is active" | tee -a $RUN/run.log; exit 1; fi
MIN=$(python3 -c "print(int($HOURS*60))"); END=$(( $(date +%s) + MIN * 60 ))
echo "$(date '+%F %T') start: init $INIT, $HOURS h, run $RUN, buf $BUF, trainer $TRAINER $TARGS, runes $RUNES, matchups $MATCHUPS, tap init $RB_TAP_INIT, copy $RB_WSL_COPY" >> $RUN/run.log
rm -f $BUF/tmp_*.npz
taskset -c 12-13 python3 $TRAINER --init $INIT --buf $BUF --run $RUN --max-hours $HOURS $TARGS >> $RUN/train.log 2>&1 &
TP=$!
for i in $(seq 1 60); do [ -f $RUN/gen_weights.pt ] && break; sleep 2; done
[ -f $RUN/gen_weights.pt ] || { echo "$(date '+%F %T') no gen_weights.pt: trainer failed, see train.log" >> $RUN/run.log; kill $TP 2>/dev/null; exit 1; }
gen() {
  taskset -c 0-11 python3 gen_az.py --weights $RUN/gen_weights.pt --out $BUF --threads 12 --runes $RUNES --matchups $MATCHUPS \
    --seed $((60000000 + $(date +%s) % 10000000)) --max-minutes $1 >> $RUN/gen.log 2>&1 &
  GP=$!
}
gen $MIN
trap 'kill $TP $GP 2>/dev/null' EXIT INT TERM
echo "$(date '+%F %T') trainer pid $TP, generator pid $GP" >> $RUN/run.log
n=0
while kill -0 $TP 2>/dev/null; do
  if ! kill -0 $GP 2>/dev/null; then
    left=$(( (END - $(date +%s)) / 60 ))
    if [ $left -gt 5 ]; then
      echo "$(date '+%F %T') generator down; restarting for $left min" >> $RUN/run.log
      rm -f $BUF/tmp_*.npz; gen $left
    fi
  fi
  n=$((n+1)); [ $((n % 10)) -eq 0 ] && echo "$(date '+%T') mem_avail_MB $(free -m | awk '/Mem/{print $7}')" >> $RUN/mem.log
  sleep 30
done
wait $TP; echo "$(date '+%F %T') trainer done rc=$?" >> $RUN/run.log
