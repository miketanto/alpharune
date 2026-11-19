#!/usr/bin/env bash
# Samples WSL per-CPU busy % (from /proc/stat) and loadavg every 5 s -> raw/wsl_load.csv
OUT=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/smoke/speed/raw/wsl_load.csv
[ -f $OUT ] || echo "time,loadavg1,$(seq -s, -f 'cpu%g' 0 15)" > $OUT
prev=$(grep '^cpu[0-9]' /proc/stat)
while true; do
  sleep 5
  cur=$(grep '^cpu[0-9]' /proc/stat)
  line=$(paste <(echo "$prev") <(echo "$cur") | awk '{b1=$2+$3+$4+$7+$8+$9; t1=b1+$5+$6; b2=$13+$14+$15+$18+$19+$20; t2=b2+$16+$17; d=t2-t1; printf "%.0f,", (d>0?100*(b2-b1)/d:0)}')
  echo "$(date +%s),$(cut -d' ' -f1 /proc/loadavg),${line%,}" >> $OUT
  prev=$cur
done
