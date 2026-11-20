#!/usr/bin/env bash
# Plays the annotated v4 games (gate-4 B2 network, GREEDY) at nice 10, 3 processes at a time, cores 8-15.
# usage: bash run_games.sh 7001:1 7002:1 ...   (seed:netseat). Results: replays_v4/log_seed*.txt
cd "$(dirname "$0")"
OUT=replays_v4
mkdir -p $OUT
n=0
for spec in "$@"; do
  s=${spec%%:*}; seat=${spec##*:}
  CUDA_VISIBLE_DEVICES= nice -n 10 taskset -c 8-15 python3 play_viz.py --seed $s --net-seat $seat --greedy --out $OUT > $OUT/log_seed${s}.txt 2>&1 &
  n=$((n+1)); if [ $n -ge 3 ]; then wait; n=0; fi
done
wait
cat $OUT/log_seed*.txt
