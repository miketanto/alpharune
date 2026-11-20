cd /mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate5
export CUDA_VISIBLE_DEVICES=
head -n 3 res_s0.jsonl > /tmp/g5_r3.jsonl
R=../selfplay/runs
time python3 swap.py respond --inp /tmp/g5_r3.jsonl --out /tmp/g5_r3b.jsonl --nets B2=$R/cf_B2/latest.pt,B2_ft20=$R/g5ft_B2/ckpt_00020.pt
python3 -c "
import json
for l in open('/tmp/g5_r3b.jsonl'):
    r=json.loads(l)
    if r.get('skipped'): continue
    for v in r['variants'][:3]: print(v['edit'], {k:(round(m['v'],3), m.get('pobj')) for k,m in v['model'].items()})
"
