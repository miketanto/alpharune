cd /mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate5
export CUDA_VISIBLE_DEVICES=
taskset -c 8-11 python3 swap.py collect --deck rengar --games 120 --n-envs 32 --p 0.06 --per-game 2 --seed 9100001 --out pos_rengar.json > collect_rengar.log 2>&1 &
taskset -c 12-15 python3 swap.py collect --deck yi --games 120 --n-envs 32 --p 0.06 --per-game 2 --seed 9200001 --out pos_yi.json > collect_yi.log 2>&1 &
wait
python3 - <<'P'
import json, random
r = json.load(open('pos_rengar.json')); y = json.load(open('pos_yi.json'))
random.seed(5); random.shuffle(r); random.shuffle(y)
inter = []
for a, b in zip(r, y): inter += [a, b]
json.dump(inter, open('pos_all.json', 'w'))
print(len(r), len(y), len(inter))
P
