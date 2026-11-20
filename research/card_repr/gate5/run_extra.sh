# third playout shard on the tail of the position list (positions 300-369; shards 0/1 do not reach them in 70 min)
cd /mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate5
export CUDA_VISIBLE_DEVICES= G5_THREADS=3
python3 -c "import json;x=json.load(open('pos_all.json'));json.dump(x[300:],open('pos_extra.json','w'))"
taskset -c 12-15 python3 swap.py run --positions pos_extra.json --out res_s2.jsonl --n 256 --n-envs 512 --max-minutes $1 >> run_s2.log 2>&1
