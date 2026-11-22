#!/usr/bin/env bash
# after s1000_mirror finishes, replace the rest of search_check.sh with the slope points that matter
cd /mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/search
while [ ! -f out/az_s1000_mirror.json ]; do sleep 20; done
pkill -f search_check.sh; sleep 2; pkill -f "eval_search.py --sims 64 --games 400 --threads 2"
G2=../selfplay/runs/g2_main/league_01000.pt
run() { name=$1; shift; echo "== $(date +%T) $name"; CUDA_VISIBLE_DEVICES= taskset -c 14-15 python3 eval_search.py --sims 64 --games 400 --threads 2 --every 600 --opp-ckpt $G2 --out out/az_$name.json "$@" 2>&1 | grep -v Warning | tail -1 | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d['ckpt'].split('/')[-1], d['deck'], 'vs', d['deck_b'], d['win_rate'], d['wilson95'], round(d['secs']))"; }
run s9944_mirror --deck azir --ckpt ../az/run_b2/az_009944.pt
run g2_rengar --deck azir --deck-b rengar --ckpt $G2
run s9944_rengar --deck azir --deck-b rengar --ckpt ../az/run_b2/az_009944.pt
echo ALLDONE $(date +%T)
