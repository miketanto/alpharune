#!/bin/bash
# helper for evalq2 (same evals, reverse order, cores 0-7): seed 1990001, budgets 100 and 50.
SP=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/selfplay
G5=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate5
cd $SP
export CUDA_VISIBLE_DEVICES=
for U in 100 50; do for R in id B2; do
  CK=$SP/runs/g5ft_$R/ckpt_$(printf %05d $U).pt
  OUT=$G5/evals/ft${U}_${R}_s2.json
  [ -f $OUT ] && continue
  [ -f $G5/evals/ft${U}_${R}_s2.log ] && continue
  taskset -c 0-7 python3 verify_eval.py $CK ismcts:sims=200 200 --repr $R --deck yi_rebuild --procs 8 --every 50 --seed 1990001 --out $OUT > $G5/evals/ft${U}_${R}_s2.log 2>&1
done; done
