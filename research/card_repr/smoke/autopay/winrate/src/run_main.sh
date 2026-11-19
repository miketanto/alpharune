#!/usr/bin/env bash
# Plays the main batches in order (each: 1000 seeds = 2000 games, 8 processes on
# cores 0-7), syncing after each. Stop early by creating ~/apw/STOP.
set -uo pipefail
D=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/smoke/autopay/winrate/src
for b in "$@"; do
  [ -e $HOME/apw/STOP ] && { echo "stopped before $b"; break; }
  date; bash $D/run_play.sh $b ${P:-8}; bash $D/sync.sh
done
date; echo all done
