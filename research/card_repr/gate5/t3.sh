cd /mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate5
export CUDA_VISIBLE_DEVICES=
time taskset -c 12-15 python3 zs.py collect --out /tmp/zs_test.npz --decisions 6 --sims 400 --n-envs 8 --p-seen 0.2 --p-unseen 0.5 --max-minutes 3
taskset -c 12-15 python3 zs.py score --inp /tmp/zs_test.npz --out /tmp/zs_test.json --nets B2=$PWD/../selfplay/runs/cf_B2/latest.pt,id=$PWD/../selfplay/runs/cf_id/latest.pt
python3 -c "import json;d=json.load(open('/tmp/zs_test.json'));[print(m, d['picks']['B2'][i], d['picks']['id'][i]) for i,m in enumerate(d['meta'])]"
