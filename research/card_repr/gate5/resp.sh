# adds root responses of the cf nets (again, for P(card)) and the six fine-tuned nets to a results file. usage: resp.sh K CORES
cd /mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate5
export CUDA_VISIBLE_DEVICES= G5_THREADS=2
R=../selfplay/runs
N="B2=$R/cf_B2/latest.pt,id=$R/cf_id/latest.pt"
for U in 20 50 100; do for X in B2 id; do N="$N,${X}_ft$U=$R/g5ft_$X/ckpt_$(printf %05d $U).pt"; done; done
taskset -c $2 python3 swap.py respond --inp res_s$1.jsonl --out resp_s$1.jsonl --nets $N > resp_s$1.log 2>&1
