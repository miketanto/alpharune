cd /mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate5
export CUDA_VISIBLE_DEVICES=
R=../selfplay/runs
python3 zs.py score --inp zs_obs.npz --out zs_scored.json --nets B2=$R/cf_B2/latest.pt,id=$R/cf_id/latest.pt
python3 zs_analyze.py zs_scored.json
