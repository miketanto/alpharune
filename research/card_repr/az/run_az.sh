#!/usr/bin/env bash
# AlphaZero Azir run (az/README.txt): generator (search self-play, cores 0-11) + trainer (cores 12-13), both
# on the GPU; cores 14-15 stay free for light checks. Stops after AZ_HOURS hours (default 6).
#   nohup bash az/run_az.sh > az/run/run.log 2>&1 &
set -u
AZ=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/az
cd $AZ || exit 1
HOURS=${AZ_HOURS:-6}
RUN=${AZ_RUN:-run}
BUF=${AZ_BUF:-buf}
INIT=${AZ_INIT:-../selfplay/runs/g2_main/league_01000.pt}
mkdir -p $RUN $BUF
echo "$(date '+%F %T') az run start: init $INIT, $HOURS h, run $RUN, buf $BUF"
# the trainer writes $RUN/gen_weights.pt from INIT before the generator starts
taskset -c 12-13 python3 train_az.py --init $INIT --buf $BUF --run $RUN --max-hours $HOURS > $RUN/train.log 2>&1 &
TP=$!
for i in $(seq 1 60); do [ -f $RUN/gen_weights.pt ] && break; sleep 2; done
taskset -c 0-11 python3 gen_az.py --weights $RUN/gen_weights.pt --out $BUF --threads 12 \
  --max-minutes $(python3 -c "print($HOURS*60)") > $RUN/gen.log 2>&1 &
GP=$!
echo "$(date '+%F %T') trainer pid $TP, generator pid $GP"
wait $TP
echo "$(date '+%F %T') trainer done rc=$?"
kill $GP 2>/dev/null
wait $GP 2>/dev/null
echo "$(date '+%F %T') az run done"
