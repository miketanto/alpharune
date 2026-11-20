#!/bin/bash
# CF training data: 3 CPU workers on cores 0-11 (12-15 are the league agent's), B2 policy SAMPLED, N=48.
G4=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate4
B2=$G4/../selfplay/runs/gate4_B2/latest.pt
export CUDA_VISIBLE_DEVICES=
cd /tmp
for k in 0 1 2; do
  C0=$((4*k)); C1=$((C0+3))
  taskset -c $C0-$C1 python3 $G4/cf_gen.py $B2 --out $G4/cf_data/w$k --base-seed $((5550001 + 10000*k)) \
    --minutes ${MIN:-33} --p 0.06 --threads 4 > $G4/cf_data/w$k.log 2>&1 &
done
wait
