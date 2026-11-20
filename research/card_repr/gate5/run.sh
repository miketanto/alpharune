#!/bin/bash
# Gate 5 test 1: swap playouts, 2 CPU shards (B2 and id policies, N=256 each, SAMPLED). usage: run.sh MINUTES
cd /mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate5
export CUDA_VISIBLE_DEVICES= G5_THREADS=3
taskset -c 4-7 python3 swap.py run --positions pos_all.json --out res_s0.jsonl --n 256 --n-envs 512 --shard 0/2 --max-minutes $1 >> run_s0.log 2>&1 &
taskset -c 8-11 python3 swap.py run --positions pos_all.json --out res_s1.jsonl --n 256 --n-envs 512 --shard 1/2 --max-minutes $1 >> run_s1.log 2>&1 &
wait
