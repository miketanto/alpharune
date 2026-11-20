#!/bin/bash
# Gate 5 sample efficiency: fine-tune a Rengar-trained CF net on the Yi mirror (SAMPLED PPO, fresh optimizer via --init),
# 100 updates, checkpoints every 10 (20 / 50 / 100 are evaluated). ONE net at a time. usage: ft.sh B2|id
SP=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/selfplay
G4=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate4
R=$1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True G4_CUDA_FRAC=0.4
cd $SP/learner
taskset -c ${CORES:-0-3} python3 $G4/sp_wrap.py --n-envs 128 --updates ${U:-100} --eval-games 32 --eval-envs 32 --eval-every 50 --threads 2 \
  --ask-head --ask-opt --repr $R --run-name g5ft_$R --init $SP/runs/cf_$R/latest.pt --seed 5 \
  --deck /mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/decks/yi_rebuild.txt --ckpt-every 10 \
  > $SP/runs/g5ft_$R.out 2>&1
