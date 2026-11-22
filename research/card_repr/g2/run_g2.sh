#!/usr/bin/env bash
# G2: six-legend generalist (Rengar, Azir, Diana, Kai'Sa, Master Yi, Kennen), full 6x6 ordered grid (36 matchups,
# 8 slots each = 288 envs), league recipe of g1/overnight_g1.py, starting from the G1 generalist at 500.
#   bash g2/run_g2.sh --fg      (run from research/card_repr)
# Output: selfplay/league/g2, checkpoints selfplay/runs/g2_main. Stops at G2_DEADLINE (default 09:30) or 1000 updates.
set -u
CR=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr
SP=$CR/selfplay
OUT=league/g2
INIT=${G2_INIT:-$SP/runs/g1_main/league_00500.pt}
[ -f "$INIT" ] || { echo "no init $INIT" >&2; exit 1; }
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RB_LEAGUE_INIT=$INIT RB_LEAGUE_BASE=g1_main@500
unset RB_DECK
D="rengar azir diana kaisa yi_rebuild kennen"
MU=""; for a in $D; do for b in $D; do MU="$MU${MU:+,}$a:$b"; done; done
if pgrep -f "league/overnight.py run" >/dev/null || pgrep -f "g1/overnight_g1.py run --deadline .* --tag g1 " >/dev/null; then
  echo "another GPU training is running; refusing" >&2; exit 1
fi
cd $SP || exit 1
mkdir -p $OUT
CMD=(python3 $CR/g1/overnight_g1.py run --deadline "${G2_DEADLINE:-09:30}" --tag g2 --out $OUT --stop-main-at 1000 \
     --snapshot-every 50 --main-updates 40 --exp-updates 20 --gpu-mem-frac 0.4 --train-cores 0-11 \
     --n-envs 288 --matchups "$MU")
if [ "${1:-}" = "--fg" ]; then exec "${CMD[@]}"; fi
nohup "${CMD[@]}" > $OUT/driver.out 2>&1 &
echo "G2 started, pid $!"
