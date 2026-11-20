#!/bin/bash
# Gate 5 test 3: GREEDY vs ismcts:sims=200, Yi mirror, 200 games (seed 990001 as the zero-shot yi_eval), per fine-tune budget.
SP=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/selfplay
G5=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate5
mkdir -p $G5/evals
cd $SP
export CUDA_VISIBLE_DEVICES=
for R in B2 id; do for U in 20 50 100; do
  CK=$SP/runs/g5ft_$R/ckpt_$(printf %05d $U).pt
  OUT=$G5/evals/ft${U}_${R}.json
  [ -f $OUT ] && continue
  while [ ! -f $CK ]; do sleep 20; done; sleep 5
  taskset -c ${CORES:-12-15} python3 verify_eval.py $CK ismcts:sims=200 200 --repr $R --deck yi_rebuild --procs ${PROCS:-5} --every 50 --out $OUT > $G5/evals/ft${U}_${R}.log 2>&1
done; done
