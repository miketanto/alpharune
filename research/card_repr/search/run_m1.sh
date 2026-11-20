#!/usr/bin/env bash
# M1 evaluation table (REPORT.txt). WSL: bash run_m1.sh  (one job at a time, nice 10)
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { name=$1; shift; echo "== $name $*"; nice -n 10 python3 eval_search.py --every 120 --out out/$name.json "$@" > out/$name.log 2>&1; tail -1 out/$name.log | cut -c1-400; }
[ -f out/n64_pool.json ] || run n64_pool --sims 64 --games 400
run n0_greedy --sims 0 --games 400
run n1_pool --sims 1 --games 400
run n16_pool --sims 16 --games 400
run n200_pool --sims 200 --games 400
run n64_decklist --sims 64 --games 400 --deck-model decklist
run n64_perfect --sims 64 --games 400 --perfect
run n64_c15 --sims 64 --games 400 --cpuct 1.5
run n64_c30 --sims 64 --games 400 --cpuct 3.0
run n1_vs_ismcts200 --sims 1 --games 200 --opponent ismcts:sims=200
run n64_vs_ismcts200 --sims 64 --games 200 --opponent ismcts:sims=200
echo ALLDONE
