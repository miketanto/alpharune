#!/bin/bash
# Phase C: add the id net's responses to the B2-policy results, then the tables (results.json, tables*.txt).
G4=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate4
SP=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/selfplay
export CUDA_VISIBLE_DEVICES=
cd /tmp
F=""
for f in results_s0 results_s1 results_s2 results_s3 results_rare_s0 results_rare_s1 results_rare_s2 results_rare_s3; do
  if [ -f $G4/$f.jsonl ]; then
    python3 $G4/add_model.py $G4/$f.jsonl $G4/${f}_both.jsonl --net id=$SP/runs/gate4_id/latest.pt &
    F="$F $G4/${f}_both.jsonl"
  fi
done
wait
python3 $G4/analyze.py $F --nets B2,id --json $G4/results.json --txt $G4/tables.txt > /dev/null
python3 $G4/analyze.py $G4/results_idpol_s2.jsonl --nets id,B2 --json $G4/results_idpolicy.json --txt $G4/tables_idpolicy.txt > /dev/null
python3 $G4/check_edits.py $F > $G4/check_edits.txt
cat $G4/tables.txt; echo; cat $G4/tables_idpolicy.txt; echo; cat $G4/check_edits.txt
