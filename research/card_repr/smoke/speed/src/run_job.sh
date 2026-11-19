#!/usr/bin/env bash
# run_job.sh <tag> <cpu> <list> <start> <count> [autopay 0|1]
# Plays lines start..start+count-1 of lists/<list>.txt with speed_export pinned to one WSL cpu.
# Output: raw/games/<tag>.games.jsonl (per-game secs + timing), one row in raw/jobs.csv.
set -uo pipefail
SP=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/smoke/speed
tag=$1; cpu=$2; list=$3; start=$4; count=$5; ap=${6:-0}
cd $HOME/riftbound/alpharune
mkdir -p /tmp/spd
sed -n "$((start+1)),$((start+count))p" $SP/lists/$list.txt > /tmp/spd/$tag.txt
envs=""; [ "$ap" = 1 ] && export SPEED_AUTOPAY=1
la0=$(cut -d' ' -f1 /proc/loadavg)
t0=$(date +%s.%N)
taskset -c $cpu ./build/speed_export play /tmp/spd/$tag.txt /tmp/spd/$tag 2> /tmp/spd/$tag.log
rc=$?
t1=$(date +%s.%N)
la1=$(cut -d' ' -f1 /proc/loadavg)
dbytes=$(stat -c %s /tmp/spd/$tag.decisions.jsonl)
cp /tmp/spd/$tag.games.jsonl $SP/raw/games/$tag.games.jsonl
rm -f /tmp/spd/$tag.decisions.jsonl
echo "$tag,$cpu,$list,$start,$count,$ap,$t0,$t1,$(echo "$t1 - $t0" | bc),$dbytes,$la0,$la1,$rc" >> $SP/raw/jobs.csv
