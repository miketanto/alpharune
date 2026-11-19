#!/usr/bin/env bash
# Longer fine-tune on the HELD-OUT deck (sample-efficiency curve): 10 min from each seed-0 model,
# probes logged every 10 updates (metrics.jsonl). Runs in the training slots once seed 2 frees them.
SP=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/selfplay
VARS=synthetic:pit_rookie__score_alone,synthetic:punch_first__ctrl_scores,synthetic:punch_first__opp_of_ctrl_scores,synthetic:sabotage__score1,synthetic:sabotage__score_opphand4
ft() { cd $SP/learner; taskset -c $3 python3 selfplay.py --repr $1 --variants $VARS --n-envs 128 --samples-per-update 8192 --minibatch 1024 \
  --updates 100000 --pool-every 2 --eval-every 10 --eval-games 64 --eval-envs 32 --ckpt-every 20 --threads 2 --seed 0 \
  --deck rengar__exp_heldout --run-name cx_$1_s0_ft10 --max-minutes $2 --init $SP/runs/cx_$1_s0/latest.pt > $SP/runs/cx_$1_s0_ft10.out 2>&1; }
w() { until grep -q "job cx_$1 done" $SP/runs/cx_train_s2.out 2>/dev/null; do sleep 5; done; }
(w B2_s2; ft B2 10 3-5; ft random 10 3-5) & (w random_s2; ft id 10 0-2) & wait
echo "ft10 done $(date)"
