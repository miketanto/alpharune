# third playout shard on the tail of the position list (positions 240-299)
cd /mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate5
export CUDA_VISIBLE_DEVICES= G5_THREADS=3
python3 -c "import json;x=json.load(open('pos_all.json'));json.dump(x[240:300],open('pos_extra2.json','w'))"
taskset -c 0-3 python3 swap.py run --positions pos_extra2.json --out res_s3.jsonl --n 256 --n-envs 512 --max-minutes $1 >> run_s3.log 2>&1
