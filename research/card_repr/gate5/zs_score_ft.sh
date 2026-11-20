cd /mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate5
export CUDA_VISIBLE_DEVICES=
R=../selfplay/runs
N=""
for U in 20 50 100; do for X in B2 id; do N="$N,${X}_ft$U=$R/g5ft_$X/ckpt_$(printf %05d $U).pt"; done; done
python3 zs.py score --inp zs_obs.npz --out zs_scored_ft.json --nets ${N:1}
python3 zs_analyze.py zs_scored.json zs_scored_ft.json
