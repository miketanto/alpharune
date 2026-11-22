#!/usr/bin/env bash
# Recovery after the 19:20 stop (2026-11-20): arm B stopped at main update 480 of 500.
# Equal compute: arm A runs to 480 too. Then the E2 head-to-heads, then G1 training and its evaluation.
#   nohup bash e2/chain_A_eval_G1.sh > e2/chain.log 2>&1 &
set -u
CR=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr
cd $CR || exit 1
echo "$(date '+%F %T') arm A start"
bash e2/run_arm.sh A 480
echo "$(date '+%F %T') arm A done rc=$?"
ls selfplay/runs/e2a_main/league_00480.pt || { echo "no A@480, stopping"; exit 1; }

echo "$(date '+%F %T') E2 head-to-heads start"
cd $CR/selfplay
RB_DECK=rengar CUDA_VISIBLE_DEVICES= python3 $CR/e2/eval_e2.py --games 2000 --procs 14 --out $CR/e2/h2h.jsonl \
  --net B480=runs/e2b_main/league_00480.pt --net A480=runs/e2a_main/league_00480.pt \
  --net B250=runs/e2b_main/league_00250.pt --net A250=runs/e2a_main/league_00250.pt \
  --net base=runs/ov_main/league_02080.pt \
  B480:A480 B480:base A480:base B250:A250 B250:base A250:base
echo "$(date '+%F %T') E2 head-to-heads done"

cd $CR
echo "$(date '+%F %T') G1 start"
bash g1/run_g1.sh --fg
echo "$(date '+%F %T') G1 done rc=$?"
bash g1/eval_g1.sh --s-rengar $CR/selfplay/runs/e2a_main/league_00480.pt
echo "$(date '+%F %T') all done"
