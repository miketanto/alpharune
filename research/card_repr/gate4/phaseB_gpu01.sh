#!/bin/bash
# 17:51: shards 0 and 1 move from CPU to the GPU (memory capped at 0.2 each; the shard files
# are resumed: positions already written are skipped).
G4=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate4
SP=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/selfplay
B2=$SP/runs/gate4_B2/latest.pt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True G4_CUDA_FRAC=0.2
cd /tmp
for k in 0 1; do
  if [ $k = 0 ]; then C=6-8; else C=9-11; fi
  G4_THREADS=2 taskset -c $C python3 $G4/run.py $G4/positions.json --out $G4/results_s$k.jsonl \
    --playout-ckpt $B2 --playout-repr B2 --nets B2=$B2 --n 384 --cap 40 --shard $k/4 --sample --n-envs 768 \
    >> $G4/run_s$k.log 2>&1 &
done
wait
