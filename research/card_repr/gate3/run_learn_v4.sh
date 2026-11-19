#!/bin/bash
# Gate 3 completion: TriggerOrder / MoveJoin learnability, schema v4 vs the v3 observation (--v3), ask_opt net.
cd "$(dirname "$0")"
for m in trig_key mj_cons mj_might; do
  python3 learn_v4.py --mode $m --v3 | tail -1
  python3 learn_v4.py --mode $m | tail -1
done
for m in trig_key mj_cons; do python3 learn_v4.py --mode $m --net-kw '{"ask_opt": true, "opt_refs": false}' | tail -1; done   # control: v4 obs, no W_ci / W_mg
