#!/usr/bin/env bash
# Evaluate runs as their jobs finish (after the s0 evals): pairs in training order.
D=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/selfplay/runs
w() { until grep -q "job cx_$1 done" $D/cx_train_all.out $D/cx_train_s2.out 2>/dev/null; do sleep 10; done; }
until grep -q "evals done" $D/cx_eval_s0.out; do sleep 10; done
w random_s0; w id_s1; RUNS='random_s0 id_s1' TS=6-15 bash eval_all.sh 5 200 200 200 > $D/cx_eval_b.out 2>&1
w B2_s1; w random_s1; RUNS='B2_s1 random_s1' TS=6-15 bash eval_all.sh 5 200 200 200 > $D/cx_eval_c.out 2>&1
w B2_s2; w id_s2; RUNS='id_s2 B2_s2' TS=6-15 bash eval_all.sh 5 200 200 200 > $D/cx_eval_d.out 2>&1
w random_s2; RUNS='random_s2' TS=2-15 bash eval_all.sh 8 200 200 200 > $D/cx_eval_e.out 2>&1
echo "chain done $(date)"
