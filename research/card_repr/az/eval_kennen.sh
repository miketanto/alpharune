#!/usr/bin/env bash
# After run_kennen (TAPS.md): Kennen mirrors, 400 seat-balanced games each, floating engine.
#   bash az/eval_kennen.sh [run dir, default run_kennen]
# Rows, for the last checkpoint X and the midpoint M:
#   X greedy vs the start (AZ @9944) greedy                  is the policy alone better than where it started?
#   X search vs the start greedy, taps off / taps on          does search + taps help, against a fixed reference?
set -u
cd /mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/search || exit 1
export RB_B2=v2 RB_WSL_COPY=${RB_WSL_COPY:-$HOME/riftbound/float_train}
RUN=${1:-run_kennen}; START=../az/run_b2/az_009944.pt
cks=$(ls ../az/$RUN/az_*.pt | sort); X=$(echo "$cks" | tail -1); n=$(echo "$cks" | wc -l); M=$(echo "$cks" | sed -n "$(( (n + 1) / 2 ))p")
row() { out=out/kennen_$1.json; shift
  [ -f $out ] || taskset -c 0-13 python3 eval_search.py --games 400 --threads 12 --every 900 --deck kennen --opp-ckpt $START --out $out "$@" > out/kennen_last.log 2>&1
  python3 -c "import json,sys; d=json.load(open('$out')); t=d.get('taps'); print('$out'.split('/')[-1], d['ckpt'].split('/')[-1], 'sims', d['sims'], 'runes', d.get('runes'), 'win', d['win_rate'], d['wilson95'], 'errors', d['errors'], 'taps/game', [round(x/max(1,d['games']),2) for x in t[:2]] if t else '-')"
}
echo "== $(date '+%F %T') $RUN: last $X, midpoint $M"
for ck in $X $M; do tag=$(basename $ck .pt)
  row ${RUN}_${tag}_greedy --sims 0 --ckpt $ck
  row ${RUN}_${tag}_s64_r0 --sims 64 --ckpt $ck --runes 0
  row ${RUN}_${tag}_s64_r2 --sims 64 --ckpt $ck --runes 2
done
row ${RUN}_start_s64_r0 --sims 64 --ckpt $START --runes 0
echo "== $(date '+%F %T') done"
