#!/bin/bash
# Shards 2 and 3 of 4 on CPU cores 2-7 (B2 eval done; the id training uses about two of cores 0-7).
G4=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate4
SP=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/selfplay
B2=$SP/runs/gate4_B2/latest.pt
export CUDA_VISIBLE_DEVICES=
cd /tmp
for k in 2 3; do
  if [ $k = 2 ]; then C=2-4; else C=5-7; fi
  G4_THREADS=3 taskset -c $C python3 $G4/run.py $G4/positions.json --out $G4/results_s$k.jsonl \
    --playout-ckpt $B2 --playout-repr B2 --nets B2=$B2 --n 384 --cap 40 --shard $k/4 --sample --n-envs 384 \
    > $G4/run_s$k.log 2>&1 &
done
wait
