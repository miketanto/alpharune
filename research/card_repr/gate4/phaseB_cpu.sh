#!/bin/bash
# Gate 4 phase B on CPU (the GPU is the id training's): collect positions from the final B2 policy, then shards 0 and 1 of 4 (2 and 3 go on the GPU once training is done):
# playout shards (B2 policy on both seats, SAMPLED, N=384 per variant, common random numbers).
G4=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate4
SP=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/selfplay
B2=$SP/runs/gate4_B2/latest.pt
export CUDA_VISIBLE_DEVICES=
cd /tmp
[ -f $G4/positions.json ] || taskset -c 8-15 python3 $G4/collect.py $B2 --out $G4/positions.json --games 300 --n-envs 64 > $G4/collect.log 2>&1
for k in 0 1; do
  C=$((8 + 4*k)); C2=$((C+3))
  G4_THREADS=4 taskset -c $C-$C2 python3 $G4/run.py $G4/positions.json --out $G4/results_s$k.jsonl \
    --playout-ckpt $B2 --playout-repr B2 --nets B2=$B2 --n 384 --cap 40 --shard $k/4 --sample --n-envs 384 \
    > $G4/run_s$k.log 2>&1 &
done
wait
