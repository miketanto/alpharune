#!/usr/bin/env bash
# G1 evaluation after training (g1/eval_g1.py): pilot swap per deck, forgetting, midpoint. ~30-60 min on 8 procs.
#   bash research/card_repr/g1/eval_g1.sh [extra eval_g1.py args, e.g. --cores 8-15 --s-rengar PATH]
# Report: research/card_repr/g1/eval/REPORT.txt (+ report.json, pairs.jsonl cache).
set -u
CR=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr
cd $CR/selfplay || exit 1
export PYTHONUNBUFFERED=1 LEAGUE_MODEL=live CUDA_VISIBLE_DEVICES=
unset RB_DECK
python3 $CR/g1/eval_g1.py --games 2000 --procs 8 "$@" 2>&1 | tee -a $CR/g1/eval/eval.log
