cd /tmp; rm -f /mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate4/tmp/r3.jsonl
CUDA_VISIBLE_DEVICES= taskset -c 8-11 timeout 120 python3 /mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate4/run.py /mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate4/tmp/pos_test.json --out /mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate4/tmp/r3.jsonl --playout-ckpt /mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/selfplay/runs/gate4_B2/ckpt_00040.pt --nets B2=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/selfplay/runs/gate4_B2/ckpt_00040.pt --n 16 --screen 16 --cap 4 --shard 0/4 --sample --n-envs 64 --max-minutes 0.5
python3 -c "
import json
for l in open('/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate4/tmp/r3.jsonl'):
    r=json.loads(l)
    if r.get('skipped'): continue
    for v in r['variants'][1:3]: print(v['cell'], v['edit'], v.get('post'))
" | head
