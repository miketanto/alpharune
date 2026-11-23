#!/usr/bin/env bash
cd /mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/az/seal
export RB_B2=v2 RIFTBOUND_WSL_COPY=$HOME/riftbound/float_base PLAYWEB_TORCH_THREADS=1
for arm in none play play_use; do for s in 0 1 2 3; do
  python3 seal_probe.py --games 600 --arm $arm --shard $s --nshards 4 --out ${arm}_$s.json > ${arm}_$s.log 2>&1 &
done; done
wait; echo ALLDONE
