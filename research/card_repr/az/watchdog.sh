#!/usr/bin/env bash
# watchdog: while the trainer runs, restart the generator if it died; log memory every 5 min. Exits when the trainer ends.
AZ=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/az
cd $AZ
RUN=${AZ_RUN:-run_b2}; END=${AZ_END_EPOCH:?}
n=0
while pgrep -f "train_az.py" > /dev/null; do
  if ! pgrep -f "gen_az.py" > /dev/null; then
    left=$(( (END - $(date +%s)) / 60 ))
    if [ $left -gt 5 ]; then
      echo "$(date '+%F %T') generator down; restarting for $left min" | tee -a $RUN/watchdog.log
      rm -f buf/*.tmp.npz buf/tmp_*.npz
      RB_B2=v2 nohup taskset -c 0-11 python3 gen_az.py --weights $RUN/gen_weights.pt --out buf --threads 12 --seed $((29000000 + RANDOM * 100)) --max-minutes $left >> $RUN/gen.log 2>&1 < /dev/null &
    fi
  fi
  n=$((n+1)); [ $((n % 5)) -eq 0 ] && echo "$(date '+%T') mem_avail_MB $(free -m | awk '/Mem/{print $7}')" >> $RUN/watchdog.log
  sleep 60
done
echo "$(date '+%F %T') trainer ended" | tee -a $RUN/watchdog.log