#!/bin/bash
# 18:20: more positions for the rare cells only (Darius, hidden, Inferna, Kinkou, Star Spring relevant):
# a second collection (another 600 sampled self-play games, different seed), edits filtered to those cells,
# 4 GPU shards (seed 4343).
G4=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate4
SP=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/selfplay
B2=$SP/runs/gate4_B2/latest.pt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True G4_CUDA_FRAC=0.2
cd /tmp
taskset -c 0-7 python3 $G4/collect.py $B2 --out $G4/positions2.json --games 600 --n-envs 64 --seed 8880001 > $G4/collect2.log 2>&1
python3 - <<'PY'
import json
G4 = '/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate4'
RARE = ('darius', 'hidden', 'inferna_assault-:can_attack', 'kinkou', 'starspring_unused:relevant')
P = json.load(open(G4 + '/positions2.json'))
out = []
for p in P:
    e = [x for x in p['edits'] if x['cell'].startswith(RARE)]
    if e:
        out.append(dict(p, edits=e))
json.dump(out, open(G4 + '/positions2_rare.json', 'w'))
print(len(out), 'positions with a rare cell')
PY
for k in 0 1 2 3; do
  C=$((4*k))-$((4*k+2))
  G4_THREADS=2 taskset -c $C python3 $G4/run.py $G4/positions2_rare.json --out $G4/results_rare_s$k.jsonl \
    --playout-ckpt $B2 --playout-repr B2 --nets B2=$B2 --n 384 --cap 80 --shard $k/4 --sample --n-envs 768 --seed 4343 \
    > $G4/run_rare_s$k.log 2>&1 &
done
wait
