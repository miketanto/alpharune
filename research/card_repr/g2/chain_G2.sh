#!/usr/bin/env bash
# Waits for e2/chain_A_eval_G1.sh (G1 training + G1 eval) to end, then runs G2 in the foreground.
CR=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr
cd $CR || exit 1
echo "$(date '+%F %T') waiting for the G1 chain"
while pgrep -f chain_A_eval_G1.sh >/dev/null; do sleep 60; done
echo "$(date '+%F %T') G1 chain ended; last lines:"; tail -3 e2/chain.log
bash g2/run_g2.sh --fg
echo "$(date '+%F %T') G2 done rc=$?"
