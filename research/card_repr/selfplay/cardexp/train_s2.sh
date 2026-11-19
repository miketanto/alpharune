#!/usr/bin/env bash
# Third seed, after train_all.sh: (id, B2) together, then random.
D=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/selfplay/cardexp
until grep -q "all training done" $D/../runs/cx_train_all.out; do sleep 5; done
TM=${TM:-10}; FM=${FM:-2}
(bash $D/run_job.sh id 2 $TM $FM 0-2; bash $D/run_job.sh random 2 $TM $FM 0-2) & (bash $D/run_job.sh B2 2 $TM $FM 3-5) & wait
echo "s2 training done $(date)"
