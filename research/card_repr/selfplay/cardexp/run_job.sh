#!/usr/bin/env bash
# One experiment run: train on the TRAIN deck (mirror self-play) for $3 minutes, then fine-tune a
# copy on the HELD-OUT deck for $4 minutes. usage: run_job.sh <repr> <seed> <train_min> <ft_min> <cpus>
set -u
R=$1; S=$2; TM=$3; FM=$4; CPUS=$5
SP=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/selfplay
VARS=synthetic:pit_rookie__score_alone,synthetic:punch_first__ctrl_scores,synthetic:punch_first__opp_of_ctrl_scores,synthetic:sabotage__score1,synthetic:sabotage__score_opphand4
COMMON="--repr $R --variants $VARS --n-envs 128 --samples-per-update 8192 --minibatch 1024 --updates 100000 --pool-every 2 --eval-every 20 --eval-games 64 --eval-envs 32 --ckpt-every 10 --threads 2 --seed $S"
cd $SP/learner
NAME=cx_${R}_s${S}
taskset -c $CPUS python3 selfplay.py $COMMON --deck rengar__exp_train --run-name $NAME --max-minutes $TM > $SP/runs/$NAME.out 2>&1
taskset -c $CPUS python3 selfplay.py $COMMON --deck rengar__exp_heldout --run-name ${NAME}_ft --max-minutes $FM --init $SP/runs/$NAME/latest.pt > $SP/runs/${NAME}_ft.out 2>&1
echo "job $NAME done $(date)"
