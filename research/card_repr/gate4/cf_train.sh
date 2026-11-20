#!/bin/bash
# CF-pair training: continue gate4_<repr> (update 150, optimizer + opponent pool kept) for U more PPO updates
# with the CF value + pairwise loss mixed in. ONE net at a time. usage: cf_train.sh B2|id
SP=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/selfplay
G4=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate4
R=$1; U=${U:-210}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True G4_CUDA_FRAC=0.4
mkdir -p $SP/runs/cf_$R
[ -f $SP/runs/cf_$R/latest.pt ] || cp $SP/runs/gate4_$R/latest.pt $SP/runs/cf_$R/latest.pt
cd $SP/learner
taskset -c ${CORES:-0-7} python3 $G4/sp_wrap.py --n-envs 128 --updates $U --eval-games 64 --eval-envs 32 --threads 2 \
  --ask-head --ask-opt --resume --repr $R --run-name cf_$R \
  --cf-data "$G4/cf_data/w*.npz" --cf-coef ${CFC:-0.5} --cf-pair-coef ${CFP:-1.0} --cf-pos ${CFPOS:-16} \
  ${EXTRA} >> $SP/runs/cf_$R.out 2>&1
