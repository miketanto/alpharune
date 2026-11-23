#!/usr/bin/env bash
# Does tapping help? (TAPS.md) Azir seat searches (64 sims) against G2 greedy, seat-balanced, same seeds, floating engine.
#   bash az/eval_taps.sh run_taps/az_004000.pt [more checkpoints]   (GPU free; GAMES=400)
# Rows per checkpoint and opponent: taps off (runes 0) and on for the searching seat only (runes 2). G2 never taps.
set -u
cd /mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/search || exit 1
export RB_B2=v2 RB_WSL_COPY=${RB_WSL_COPY:-$HOME/riftbound/float_train} RB_TAP_INIT=${RB_TAP_INIT:--3.0}
GAMES=${GAMES:-400}; THREADS=${THREADS:-12}; G2=../selfplay/runs/g2_main/league_01000.pt
for ck in "$@"; do
  tag=$(echo $ck | tr '/' '_' | sed 's/\.pt$//')
  for opp in azir rengar; do for r in 0 2; do
    b=""; [ $opp != azir ] && b="--deck-b $opp"
    out=out/taps_${tag}_${opp}_r$r.json
    [ -f $out ] || taskset -c 0-13 python3 eval_search.py --sims 64 --games $GAMES --threads $THREADS --every 600 --deck azir $b \
      --ckpt ../az/$ck --opp-ckpt $G2 --runes $r --out $out > out/taps_last.log 2>&1
    python3 - $out <<'P'
import json, sys
d = json.load(open(sys.argv[1]))
n = d['games']
print(sys.argv[1].split('/')[-1], 'win', d['win_rate'], d['wilson95'], 'n', n, 'errors', d['errors'], 'fails', d['stats']['fails'], 'secs', round(d['secs']),
      'taps/game', [round(x / max(1, n), 2) for x in d['taps'][:2]] if d.get('taps') else '-', flush=True)
P
  done; done
done
