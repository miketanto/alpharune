#!/usr/bin/env bash
# Replays jobs/verify_<name>.txt (seed + choices, no agent) in P shards, cores 0-7.
set -uo pipefail
W=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/smoke/autopay/winrate
name=$1; P=${2:-8}
cd $HOME/riftbound/alpharune
mkdir -p $HOME/apw/verify $HOME/apw/vjobs
split -n l/$P -d $W/jobs/verify_$name.txt $HOME/apw/vjobs/${name}_
for f in $HOME/apw/vjobs/${name}_*; do
  taskset -c 0-7 ./build/autopay_match verify $f > $f.out 2>&1 &
done
wait
cat $HOME/apw/vjobs/${name}_*.out > $HOME/apw/verify/verify_$name.txt
cat $HOME/apw/verify/verify_$name.txt
