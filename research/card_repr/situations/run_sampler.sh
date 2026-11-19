#!/usr/bin/env bash
# Plays every game of games.txt in 12 shards (WSL build copy), writes the raw
# index / records to situations/raw/ (gitignored), then verifies replays.
set -euo pipefail
SRC=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr
RAW=$SRC/situations/raw
mkdir -p $RAW
cd $HOME/riftbound/alpharune
N=12
for i in $(seq 0 $((N-1))); do
  ./build/situation_sampler play $SRC/situations/games.txt $SRC/C_engine/out/subjects.txt \
     /tmp/sit_index_$i.jsonl /tmp/sit_records_$i.jsonl $i $N 2> /tmp/sit_log_$i.txt &
done
wait
cat /tmp/sit_index_*.jsonl > $RAW/index.jsonl
cat /tmp/sit_records_*.jsonl > $RAW/records.jsonl
cat /tmp/sit_log_*.txt > $RAW/play_log.txt
grep -c "situations" $RAW/play_log.txt
grep "shard" $RAW/play_log.txt
