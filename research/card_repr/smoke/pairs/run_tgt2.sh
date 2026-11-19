#!/usr/bin/env bash
# Target-stage runs (--tgt): root search + search at the target prompt after playing the card.
TV="punch_first__ctrl_scores punch_first__opp_of_ctrl_scores"
set -uo pipefail
H=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/smoke/pairs
SIMS=${SIMS:-200}; SEEDS=${SEEDS:-8}; TV=${TV:-"punch_first__kill punch_first__minus5"}
OUT=$HOME/dx/pairs_tgt; mkdir -p $OUT
cd $HOME/riftbound/alpharune
cp /mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/smoke/data/src/decision_export.cpp research/card_repr/smoke/data/src/
cp /mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/C_engine/card_probe.cpp research/card_repr/C_engine/
for part in 0 1 2; do awk -v p=$part 'NR%3==p' $H/jobs/punch_first.txt > $OUT/jobs_$part.txt; done
taskset -c 8-15 cmake --build build --target decision_export -j4 2>&1 | grep -E -A3 "error" | head -20
for v in $TV; do
  # split the 48 states over 3 processes per variant (6 total)
  for part in 0 1 2; do
    taskset -c 8-15 ./build/decision_export swap $OUT/jobs_$part.txt $OUT/${v}_p${part}.jsonl --variant synthetic:$v --sims $SIMS --seeds $SEEDS --tgt 2> $OUT/${v}_p$part.log &
  done
done
wait
mkdir -p $H/raw_tgt
for v in $TV; do cat $OUT/${v}_p*.jsonl > $H/raw_tgt/${v}_s${SIMS}_tgt.jsonl; done
wc -l $H/raw_tgt/*.jsonl
