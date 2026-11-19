#!/usr/bin/env bash
# Plays every job of jobs/play_jobs.txt in the WSL build copy, at most $P
# processes at once, writing to ~/dx/out (copied back to data/out by sync.sh).
set -uo pipefail
DATA=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/smoke/data
P=${P:-10}
cd $HOME/riftbound/alpharune
mkdir -p research/card_repr/smoke/data/decks $HOME/dx/out $HOME/dx/log
cp $DATA/decks/*.txt research/card_repr/smoke/data/decks/
run() {
  name=$1; v=$2; s=$3; ns=$4
  va=""; [ "$v" != "-" ] && va="--variant $v"
  ./build/decision_export play $DATA/lists/$name.txt $HOME/dx/out/${name}__s$s $va --shard $s $ns \
     2> $HOME/dx/log/${name}__s$s.log
  echo "done $name $s rc=$?"
}
export -f run
export DATA
xargs -P $P -L 1 bash -c 'run "$@"' _ < $DATA/jobs/play_jobs.txt
