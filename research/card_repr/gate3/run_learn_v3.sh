#!/bin/bash
cd "$(dirname "$0")"
S=--samples\ inventory_v3_samples.npz
NK='{"ask_opt": true}'
for pk in Accelerate AdditionalCost; do
  python3 learn.py --pk $pk --mode card --tag v3fix $S --net-kw "$NK" --out learn_v3.jsonl | tail -1
  python3 learn.py --pk $pk --mode field:x:glob:42:0:6:3 --tag v3fix $S --net-kw "$NK" --out learn_v3.jsonl | tail -1
  python3 learn.py --pk $pk --mode field:x:glob:19:0:4:2 --tag v3fix $S --net-kw "$NK" --out learn_v3.jsonl | tail -1
  python3 learn.py --pk $pk --mode field:x:optyes:53:0:6:3 --tag v3fix $S --net-kw "$NK" --out learn_v3.jsonl | tail -1
  python3 learn.py --pk $pk --mode field:x:optyes:54:0:8:4 --tag v3fix $S --net-kw "$NK" --out learn_v3.jsonl | tail -1
  python3 learn.py --pk $pk --mode field:x:optyes:51:0:2:1 --tag v3fix $S --net-kw "$NK" --out learn_v3.jsonl | tail -1
  python3 learn.py --pk $pk --mode field:x:optyes:53:0:6:3 --tag v3_noopt $S --net-kw '{"ask_opt": true, "opt_cost": false}' --out learn_v3.jsonl | tail -1
done
