#!/usr/bin/env bash
# Replays the sampled games (jobs/verify__*.txt) from seed + choice indices.
set -uo pipefail
DATA=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/smoke/data
cd $HOME/riftbound/alpharune
mkdir -p $HOME/dx/verify
while read grp v; do
  va=""; [ "$v" != "-" ] && va="--variant $v"
  ./build/decision_export verify $DATA/jobs/verify__$grp.txt $va > $HOME/dx/verify/$grp.txt 2>&1 &
  while [ $(jobs -r | wc -l) -ge ${P:-10} ]; do sleep 1; done
done < $DATA/jobs/verify_jobs.txt
wait
cat $HOME/dx/verify/*.txt
