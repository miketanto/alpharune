cd /mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate5
export CUDA_VISIBLE_DEVICES=
taskset -c 12-15 python3 swap.py collect --deck yi --games 8 --n-envs 8 --p 0.1 --seed 9990001 --out /tmp/g5_test_pos.json
python3 - <<'P'
import json; x=json.load(open('/tmp/g5_test_pos.json')); print(len(x)); print(json.dumps(x[0]['edits'], indent=0)[:1500] if x else None)
P
