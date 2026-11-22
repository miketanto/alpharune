#!/usr/bin/env bash
# Tiny CPU smoke test of the G1 pipeline (a few updates on all four matchups, exploiter snapshot+reset).
set -u
CR=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr
cd $CR/selfplay || exit 1
export PYTHONUNBUFFERED=1 RB_LEAGUE_INIT=$PWD/runs/ov_main/league_02080.pt RB_LEAGUE_BASE=ov_main@2080
unset RB_DECK
DL=$(date -d '+20 min' +%H:%M)
timeout 900 nice -n 15 python3 $CR/g1/overnight_g1.py run --deadline "$DL" --tag g1smoke --out league/g1_smoke \
  --stop-main-at 3 --main-updates 2 --exp-updates 2 --snapshot-every 2 --exp-max-updates 2 --n-envs 16 \
  --samples-per-update 384 --phase-max-min 8 --cpu --nice 15 --train-cores 0-15 "$@"
