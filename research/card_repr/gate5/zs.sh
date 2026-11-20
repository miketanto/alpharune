cd /mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate5
export CUDA_VISIBLE_DEVICES=
taskset -c 12-15 python3 zs.py collect --out zs_obs.npz --decisions 500 --sims 400 --n-envs 16 --p-seen 0.05 --p-unseen 0.3 --max-minutes 25 > zs_collect.log 2>&1
