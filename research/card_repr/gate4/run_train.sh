#!/bin/bash
# Gate 4 phase A (revised 17:03 after the coordinator's resource report): the first attempt ran B2 and id
# concurrently and both were ~10x slow (GPU memory oversubscription). Both were stopped (B2 at update 33,
# latest.pt = update 30; id at 41, latest.pt = update 40) and are now RESUMED ONE AFTER THE OTHER to the same
# fixed update count (150: about 20 min each at normal speed, ~9-10 s per update solo), with CUDA memory capped at 0.4 of the GPU per process.
SP=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/selfplay
G4=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate4
U=${U:-150}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SP/learner
C="--n-envs 128 --updates $U --eval-games 64 --eval-envs 32 --threads 2 --ask-head --ask-opt --resume"
taskset -c 0-7 python3 $G4/sp_wrap.py $C --repr B2 --run-name gate4_B2 >> $SP/runs/gate4_B2.out 2>&1
taskset -c 0-7 python3 $G4/sp_wrap.py $C --repr id --run-name gate4_id >> $SP/runs/gate4_id.out 2>&1
