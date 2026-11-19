#!/usr/bin/env bash
# Plays the annotated games in parallel (one process each), cores 8-15. Results: replays/run_log.txt
cd "$(dirname "$0")"
OUT=replays
mkdir -p $OUT
for spec in "$@"; do   # spec = seed:netseat
  s=${spec%%:*}; seat=${spec##*:}
  taskset -c 8-15 python3 play_viz.py --seed $s --net-seat $seat --out $OUT > $OUT/log_seed${s}.txt 2>&1 &
done
wait
cat $OUT/log_seed*.txt
