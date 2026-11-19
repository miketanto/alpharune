#!/usr/bin/env bash
# Evaluations for every run (after training). Probes first (CPU, seconds), then ISMCTS-100 games,
# interleaved over runs so partial results stay balanced. usage: eval_all.sh <procs> <nA> <nC> <nD>
P=${1:-10}; NA=${2:-100}; NC=${3:-100}; ND=${4:-60}
SP=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/selfplay
VARS=synthetic:pit_rookie__score_alone,synthetic:punch_first__ctrl_scores,synthetic:punch_first__opp_of_ctrl_scores,synthetic:sabotage__score1,synthetic:sabotage__score_opphand4
RUNS=${RUNS:-"id_s0 B2_s0 random_s0 id_s1 B2_s1 random_s1"}
TS=${TS:-2-15}
cd $SP
for r in $RUNS; do R=${r%_s*}
  CUDA_VISIBLE_DEVICES= python3 probe_eval.py runs/cx_$r/latest.pt --repr $R --out runs/cx_$r/probes.json > /dev/null
  CUDA_VISIBLE_DEVICES= python3 probe_eval.py runs/cx_${r}_ft/latest.pt --repr $R --out runs/cx_${r}_ft/probes.json > /dev/null
done
for R in id B2 random; do CUDA_VISIBLE_DEVICES= python3 probe_eval.py untrained --repr $R --out runs/cx_untrained_$R.probes.json > /dev/null; done
CUDA_VISIBLE_DEVICES= python3 probe_eval.py runs/real_smoke_id/latest.pt --repr id --out runs/cx_basedeck_id.probes.json > /dev/null
echo "probes done $(date)"
ev() { # run deck n out
  R=${1%_s*}; R=${R#cx_}; R=${R%_ft*}
  CUDA_VISIBLE_DEVICES= taskset -c $TS python3 verify_eval.py runs/$1/latest.pt ismcts:sims=100 $3 --repr $R --procs $P --every 1000 \
     --deck $2 --variants $VARS --out runs/$1/$4 | tail -n 1
}
for r in $RUNS; do ev cx_$r rengar__exp_train $NA eval_train.json; done
for r in $RUNS; do ev cx_$r rengar__exp_heldout $NC eval_heldout.json; done
for r in $RUNS; do ev cx_${r}_ft rengar__exp_heldout $ND eval_heldout.json; done
echo "evals done $(date)"
