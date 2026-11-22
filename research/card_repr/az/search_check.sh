#!/usr/bin/env bash
# search-strength check of AZ checkpoints: search seat (net X, 64 sims) vs greedy G2, Azir pilot
cd /mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/search
G2=../selfplay/runs/g2_main/league_01000.pt
run() { name=$1; shift; echo "== $(date +%T) $name"; CUDA_VISIBLE_DEVICES= taskset -c 14-15 python3 eval_search.py --sims 64 --games 400 --threads 2 --every 600 --opp-ckpt $G2 --out out/az_$name.json "$@" 2>&1 | grep -v Warning | tail -1 | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d['ckpt'].split('/')[-1], d['deck'], 'vs', d['deck_b'], d['win_rate'], d['wilson95'], round(d['secs']))"; }
run g2_mirror --deck azir --ckpt $G2
run s4000_mirror --deck azir --ckpt ../az/run_b2/az_004000.pt
run s1000_mirror --deck azir --ckpt ../az/run/az_001000.pt
run g2_rengar --deck azir --deck-b rengar --ckpt $G2
run s4000_rengar --deck azir --deck-b rengar --ckpt ../az/run_b2/az_004000.pt
echo ALLDONE $(date +%T)