#!/usr/bin/env bash
# Fix check: id, 5 min on rengar__exp_both (ctrl_scores AND opp_of_ctrl_scores in the deck), with and
# without the ask head. Probes (ctrl_tgt, oppctrl_tgt) every 10 updates in metrics.jsonl.
SP=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/selfplay
VARS=synthetic:pit_rookie__score_alone,synthetic:punch_first__ctrl_scores,synthetic:punch_first__opp_of_ctrl_scores,synthetic:sabotage__score1,synthetic:sabotage__score_opphand4
C="--repr id --variants $VARS --n-envs 128 --samples-per-update 8192 --minibatch 1024 --updates 100000 --pool-every 2 --eval-every 10 --eval-games 64 --eval-envs 32 --ckpt-every 10 --threads 2 --seed 0 --deck rengar__exp_both --max-minutes ${TM:-5}"
cd $SP/learner
taskset -c 0-3 python3 selfplay.py $C --ask-head --run-name cx_fix_both_id_ask ${RESUME:+--resume} >> $SP/runs/cx_fix_both_id_ask.out 2>&1 &
taskset -c 4-7 python3 selfplay.py $C --run-name cx_fix_both_id_noask ${RESUME:+--resume} >> $SP/runs/cx_fix_both_id_noask.out 2>&1 &
wait; echo "fixcheck done $(date)"
