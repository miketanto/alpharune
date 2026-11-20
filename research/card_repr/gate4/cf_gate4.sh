#!/bin/bash
# Gate 4 re-run for a CF-trained net: the SAME positions, edits and ground truth (B2-policy playouts, N=384)
# as gate 4; the net's root responses are added with add_model.py (variants rebuilt deterministically).
# usage: cf_gate4.sh NAME CKPT IN_SUFFIX OUT_SUFFIX   e.g. cf_gate4.sh B2_cf runs/cf_B2/latest.pt both b2cf
G4=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate4
export CUDA_VISIBLE_DEVICES=
cd /tmp
mkdir -p $G4/cf_eval
for f in results_s0 results_s1 results_s2 results_s3 results_rare_s0 results_rare_s1 results_rare_s2 results_rare_s3; do
  if [ $3 = both ]; then IN=$G4/${f}_both.jsonl; else IN=$G4/cf_eval/${f}_$3.jsonl; fi
  taskset -c ${CORES:-8-11} python3 $G4/add_model.py $IN $G4/cf_eval/${f}_$4.jsonl --net $1=$2 &
done
wait
