#!/usr/bin/env bash
# E2 arm driver (research/card_repr/e2/README section in REPORT.txt).
#   bash e2/run_arm.sh A 600            # control: league continued from ov_main@2080, 600 main updates
#   bash e2/run_arm.sh B 600            # + search teacher (generator process + CE loss on the main agent)
#   bash e2/run_arm.sh B 6 _dry --main-updates 3 --exp-updates 2     # smoke test (extra overnight.py options)
# Fresh league dir selfplay/league/e2_<ARM><suffix>, runs/e2<a|b><suffix>_{main,exp}. No rating cycles.
set -u
ARM=$1; N=$2; SUF=${3:-}
shift 3 2>/dev/null || shift $#
CR=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr
SP=$CR/selfplay
TAG=e2$(echo "$ARM" | tr AB ab)$SUF
OUT=league/e2_$ARM$SUF
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RB_DECK=rengar
export RB_LEAGUE_INIT=$SP/runs/ov_main/league_02080.pt RB_LEAGUE_BASE=ov_main@2080
HOLD="--search-holdout $CR/e2/holdout.npz --eval-holdout-every 10"
EXTRA=""
GP=""
if [ "$ARM" = B ]; then
  BUF=$HOME/e2buf/$TAG
  rm -rf "$BUF"; mkdir -p "$BUF"
  W=$SP/runs/${TAG}_main/gen_weights.pt
  mkdir -p "$(dirname "$W")"
  python3 -c "import torch; ck=torch.load('$RB_LEAGUE_INIT', map_location='cpu', weights_only=False); torch.save({'model': ck['model'], 'update': 0}, '$W')"
  taskset -c 12-15 python3 $CR/search/gen_targets.py --weights "$W" --out "$BUF" --threads 4 --games 96 \
      --p-search 0.3 --sims 64 --gpu-mem-frac 0.15 --seed 5000000 > $CR/e2/gen_$TAG.log 2>&1 &
  GP=$!
  echo "generator pid $GP -> $BUF"
  EXTRA="--search-buf $BUF --search-coef ${E2_COEF:-1.0} --search-mb 256 --search-window 20000 --search-target cq --export-weights $W --export-every 5"
fi
cd $SP || exit 1
DL=$(date -d '+10 hours' +%H:%M)
python3 league/overnight.py run --deadline "$DL" --tag "$TAG" --out "$OUT" --stop-main-at "$N" --no-rate --no-ismcts \
    --main-extra="$EXTRA" --train-extra="$HOLD" "$@"
rc=$?
if [ -n "$GP" ]; then kill $GP; fi
exit $rc
