#!/usr/bin/env bash
# Rate Azir-league members by hand (same env as the Azir league driver). usage: bash league/rate_azir.sh az_main@100 [...]
cd "$(dirname "$0")/.." || exit 1
export RB_DECK=azir RB_LEAGUE_INIT="$PWD/runs/ov_main/league_02080.pt" RB_LEAGUE_BASE=rengar_main@2080 PYTHONUNBUFFERED=1
for m in "$@"; do
  nice -n 10 taskset -c 12-15 python3 league/overnight.py rate --out league/azir --tag az --member "$m" \
    --rate-games 100 --rate-procs 4 >> league/azir/rate.log 2>&1
done
grep "rate \|timeline" league/azir/log.txt | tail -20 | cut -c1-200
