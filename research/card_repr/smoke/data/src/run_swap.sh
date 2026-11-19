#!/usr/bin/env bash
# Swap tests: one process per variant (each registers only its own variant).
set -uo pipefail
DATA=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/smoke/data
SIMS=${SIMS:-100}; SEEDS=${SEEDS:-4}; P=${P:-10}
cd $HOME/riftbound/alpharune
mkdir -p $HOME/dx/swap $HOME/dx/log
for f in $DATA/jobs/swap__*.txt; do
  b=$(basename $f .txt); short=${b#swap__}
  case $short in primal_strength) v=primal_strength;; *) v=synthetic:$short;; esac
  # split each variant's jobs in 2 halves to use the cores
  n=$(wc -l < $f); h=$(( (n+1)/2 ))
  head -n $h $f > $HOME/dx/swap/${b}_a.jobs; tail -n +$((h+1)) $f > $HOME/dx/swap/${b}_b.jobs
  for part in a b; do
    ./build/decision_export swap $HOME/dx/swap/${b}_$part.jobs $HOME/dx/swap/${b}_${part}_sims$SIMS.jsonl --variant $v --sims $SIMS --seeds $SEEDS \
       2> $HOME/dx/log/${b}_${part}_sims$SIMS.log &
    while [ $(jobs -r | wc -l) -ge $P ]; do sleep 1; done
  done
done
wait
wc -l $HOME/dx/swap/*.jsonl
