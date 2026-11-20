#!/bin/bash
# Secondary ground truth: the id net's OWN policy plays out (sampled) the positions of shard 2 (same permutation and
# caps, so mostly the same positions as results_s2.jsonl), to check id's value head against its own policy.
G4=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate4
SP=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/selfplay
ID=$SP/runs/gate4_id/latest.pt
B2=$SP/runs/gate4_B2/latest.pt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True G4_CUDA_FRAC=0.2
cd /tmp
G4_THREADS=2 taskset -c 12-14 python3 $G4/run.py $G4/positions.json --out $G4/results_idpol_s2.jsonl \
  --playout-ckpt $ID --playout-repr id --nets id=$ID,B2=$B2 --n 384 --cap 40 --shard 2/4 --sample --n-envs 768 \
  >> $G4/run_idpol_s2.log 2>&1
