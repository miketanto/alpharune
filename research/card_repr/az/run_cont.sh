#!/usr/bin/env bash
# continue run_b2 (resumes from its last checkpoint) for AZ_HOURS; generator cores 0-11, trainer 12-13
set -u
AZ=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/az
cd $AZ || exit 1
export RB_B2=v2
HOURS=${AZ_HOURS:-5.5}; RUN=run_b2; SEED=${AZ_SEED:-41000000}
echo "$(date '+%F %T') continue $RUN for $HOURS h, gen seed $SEED" >> $RUN/run.log
taskset -c 12-13 python3 train_az.py --init run/az_001000.pt --buf buf --run $RUN --max-hours $HOURS >> $RUN/train.log 2>&1 &
TP=$!
sleep 60
rm -f buf/tmp_*.npz
taskset -c 0-11 python3 gen_az.py --weights $RUN/gen_weights.pt --out buf --threads 12 --seed $SEED \
  --max-minutes $(python3 -c "print(int($HOURS*60))") >> $RUN/gen.log 2>&1 &
wait $TP
echo "$(date '+%F %T') trainer done" >> $RUN/run.log
pkill -f gen_az.py
