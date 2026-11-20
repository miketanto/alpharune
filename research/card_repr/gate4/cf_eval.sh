#!/bin/bash
# Strength of a CF-trained net: GREEDY vs ismcts:sims=100, 200 games, --procs 6 (as gate 4). usage: cf_eval.sh B2|id CORES
SP=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/selfplay
G4=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate4
cd $SP
CUDA_VISIBLE_DEVICES= taskset -c $2 python3 verify_eval.py runs/cf_$1/latest.pt ismcts:sims=100 200 --repr $1 --procs 6 \
  --out $G4/cf_eval_$1_ismcts100.json > $G4/cf_eval_$1_ismcts100.log 2>&1
