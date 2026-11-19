#!/usr/bin/env bash
# Plays lists/<name>.txt in the WSL build copy as P shards (one process each),
# pinned to cores 0-7, writing ~/apw/out/<name>__s<k>.games.jsonl and logs.
# usage: run_play.sh <name> [P] [extra args, e.g. --check-mirror]
set -uo pipefail
W=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/smoke/autopay/winrate
name=$1; P=${2:-8}; shift 2 || shift $#
cd $HOME/riftbound/alpharune
mkdir -p $HOME/apw/out $HOME/apw/log
for ((s=0; s<P; s++)); do
  taskset -c 0-7 ./build/autopay_match play $W/lists/$name.txt $HOME/apw/out/${name}__s$s --shard $s $P "$@" \
     2> $HOME/apw/log/${name}__s$s.log &
done
wait
echo "done $name"
