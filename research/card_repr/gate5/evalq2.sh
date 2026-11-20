#!/bin/bash
# Gate 5 test 3, second batch of 200 games per budget (seed 1990001), budget 0 = the zero-shot cf nets.
SP=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/selfplay
G5=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate5
cd $SP
export CUDA_VISIBLE_DEVICES=
until [ -f $G5/evals/ft100_id.json ]; do sleep 20; done
for U in 0 20 50 100; do for R in B2 id; do
  if [ $U = 0 ]; then CK=$SP/runs/cf_$R/latest.pt; else CK=$SP/runs/g5ft_$R/ckpt_$(printf %05d $U).pt; fi
  OUT=$G5/evals/ft${U}_${R}_s2.json
  [ -f $OUT ] && continue
  taskset -c ${CORES:-12-15} python3 verify_eval.py $CK ismcts:sims=200 200 --repr $R --deck yi_rebuild --procs ${PROCS:-5} --every 50 --seed 1990001 --out $OUT > $G5/evals/ft${U}_${R}_s2.log 2>&1
done; done
