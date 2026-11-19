#!/usr/bin/env bash
# Copies results from the WSL scratch dirs back into data/ (gzip for decisions).
set -uo pipefail
DATA=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/smoke/data
mkdir -p $DATA/out $DATA/swap $DATA/logs
cp $HOME/dx/out/*.games.jsonl $DATA/out/
for f in $HOME/dx/out/*.decisions.jsonl; do gzip -c $f > $DATA/out/$(basename $f).gz; done
[ -d $HOME/dx/swap ] && cat $HOME/dx/swap/*.jsonl > $DATA/swap/swap_tests.jsonl 2>/dev/null
cp $HOME/dx/log/*.log $DATA/logs/ 2>/dev/null
[ -d $HOME/dx/verify ] && cat $HOME/dx/verify/*.txt > $DATA/verify_results.txt
true
