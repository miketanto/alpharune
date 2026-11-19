#!/usr/bin/env bash
# Two slots, three rounds: (id,B2) s0, (random,id) s0/s1, (B2,random) s1.
D=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/selfplay/cardexp
TM=${TM:-10}; FM=${FM:-2}
slotA() { for j in "id 0" "random 0" "B2 1"; do bash $D/run_job.sh $j $TM $FM 0-2; done; }
slotB() { for j in "B2 0" "id 1" "random 1"; do bash $D/run_job.sh $j $TM $FM 3-5; done; }
slotA & slotB & wait
echo "all training done $(date)"
