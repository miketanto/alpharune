#!/usr/bin/env bash
# One process per variant on cores 8-15, at most 6 at a time.
set -uo pipefail
H=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/smoke/pairs
SIMS=${SIMS:-200}; SEEDS=${SEEDS:-12}; P=${P:-6}
VARS=${VARS:-"punch_first__score1 punch_first__score_nounits punch_first__score_bfunit punch_first__score_behind punch_first__kill pit_rookie__score1 pit_rookie__score_alone sabotage__score1 sabotage__score_opphand4"}
OUT=$HOME/dx/pairs; mkdir -p $OUT
cd $HOME/riftbound/alpharune
for v in $VARS; do
  base=${v%%__*}
  taskset -c 8-15 ./build/decision_export swap $H/jobs/$base.txt $OUT/${v}_s$SIMS.jsonl --variant synthetic:$v --sims $SIMS --seeds $SEEDS 2> $OUT/${v}_s$SIMS.log &
  while [ $(jobs -r | wc -l) -ge $P ]; do sleep 1; done
done
wait
mkdir -p $H/raw_s$SIMS; cp $OUT/*_s$SIMS.jsonl $H/raw_s$SIMS/
wc -l $H/raw_s$SIMS/*.jsonl
