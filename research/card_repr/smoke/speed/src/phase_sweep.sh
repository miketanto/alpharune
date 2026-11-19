#!/usr/bin/env bash
# Phase A: sims sweep + auto-pay + random, one process per physical core (WSL cpus 8,10,12,14;
# their SMT siblings 9,11,13,15 left idle). Each lane runs its jobs sequentially. 3 repeats per
# config; repeat r = a different block of games (the same seeds at every sims level).
SP=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/smoke/speed
J=$SP/src/run_job.sh
lane() { cpu=$1; shift; for j in "$@"; do IFS=: read tag list start count ap <<< "$j"; bash $J $tag $cpu $list $start $count ${ap:-0}; done; }
lane 8  sw_t400_r0:t400:0:4 sw_t25_r0:t25:0:16 sw_t25_r1:t25:16:16 sw_t25_r2:t25:32:16 sw_rand_r0:rand:0:300 &
lane 10 sw_t400_r1:t400:4:4 sw_t50_r0:t50:0:16 sw_t50_r1:t50:16:16 sw_t50_r2:t50:32:16 sw_rand_r1:rand:300:300 &
lane 12 sw_t400_r2:t400:8:4 sw_t100_r0:t100:0:8 sw_t100_r1:t100:8:8 sw_t100_r2:t100:16:8 sw_rand_r2:rand:600:300 &
lane 14 sw_t200_r0:t200:0:6 sw_t200_r1:t200:6:6 sw_t200_r2:t200:12:6 ap_t100_r0:t100:0:8:1 ap_t100_r1:t100:8:8:1 ap_t100_r2:t100:16:8:1 &
wait
echo phase_sweep done
