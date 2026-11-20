cd /mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate5
export CUDA_VISIBLE_DEVICES=
rm -f /tmp/g5_test_res.jsonl
python3 - <<'P'
import json; x=json.load(open('/tmp/g5_test_pos.json')); json.dump(x[:3], open('/tmp/g5_test_pos3.json','w'))
P
time taskset -c 12-15 python3 swap.py run --positions /tmp/g5_test_pos3.json --out /tmp/g5_test_res.jsonl --n 64 --n-envs 512 --screen 16 --screen-lo 0.0
python3 - <<'P'
import json
for l in open('/tmp/g5_test_res.jsonl'):
    r=json.loads(l)
    if r.get('skipped'): print('skip'); continue
    for v in r['variants']:
        import numpy as np
        print(v['kind'], v['pool'], v['edit'], v['ok'], v['err'], (v.get('post') or {}).get('key'), (v.get('post') or {}).get('might'),
              'B2win %.2f idwin %.2f' % (np.mean(v['wins_B2']), np.mean(v['wins_id'])) if v['ok'] else '',
              {k: round(m['v'],3) for k,m in v.get('model',{}).items()})
P
