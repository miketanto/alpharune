#!/usr/bin/env bash
# Runs the rbenv tests under WSL on cores 8-15.
set -euo pipefail
cd "$(dirname "$0")"
T="taskset -c 8-15 python3"
$T test_random.py
$T test_determinism.py
$T test_agreement.py 20
$T test_agreement.py 10 base_teacher__s0
$T test_opponent.py random-player 100
for n in 64 256; do for p in 1 4; do $T bench.py $n $p 10; done; done
