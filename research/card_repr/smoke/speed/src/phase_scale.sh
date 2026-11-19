#!/usr/bin/env bash
# Phase B: sims=100 scaling, P concurrent processes, 6 games each, 3 repeats.
#   bash phase_scale.sh "1 2 4"   then (memory permitting)   bash phase_scale.sh "8"
# Pinning: P=1 cpu 8; P=2 cpus 8,10; P=4 cpus 8,10,12,14 (one per physical core);
# P=8 cpus 8..15 (4 physical cores x 2 SMT threads). Games: lists/t100long.txt, a fresh block
# of 6 per process (offset = 100*P + 6*k + 18*... see below, never reused).
SP=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/smoke/speed
J=$SP/src/run_job.sh
for P in $1; do
  for r in 0 1 2; do
    case $P in 1) cpus="8";; 2) cpus="8 10";; 4) cpus="8 10 12 14";; 8) cpus="8 9 10 11 12 13 14 15";; esac
    base=$(( (P==1?0:(P==2?18:(P==4?54:150))) + r*6*P ))
    k=0
    for c in $cpus; do bash $J sc_P${P}_r${r}_k$k $c t100long $(( base + 6*k )) 6 0 & k=$((k+1)); done
    wait
  done
done
echo phase_scale $1 done
