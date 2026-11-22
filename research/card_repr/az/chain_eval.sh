#!/usr/bin/env bash
# Waits for run_az.sh to finish, then evaluates checkpoints (every 4000 steps + the last) with eval_az.py,
# 2000 greedy seat-balanced games per pair on 14 processes.
#   nohup bash az/chain_eval.sh > az/eval/chain.log 2>&1 &
AZ=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/az
cd $AZ || exit 1
mkdir -p eval
while pgrep -f "bash az/run_az.sh" > /dev/null; do sleep 60; done
echo "$(date '+%F %T') run finished; evaluating"
RUN=${AZ_RUN:-run}
CK=()
for f in $(ls $RUN/az_*.pt | sort); do
  s=$(basename $f .pt); s=$((10#${s#az_}))
  if [ $((s % 4000)) -eq 0 ]; then CK+=(--ckpt $f); fi
done
LAST=$(ls $RUN/az_*.pt | sort | tail -1)
CK+=(--ckpt $LAST)
CUDA_VISIBLE_DEVICES= python3 eval_az.py "${CK[@]}" --games 2000 --procs 14 --cores 0-13 > eval/REPORT.txt 2>&1
echo "$(date '+%F %T') eval done"
