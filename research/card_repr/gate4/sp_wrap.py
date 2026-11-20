"""Runs learner/selfplay.py with this process's CUDA memory capped (coordinator, 2026-11-19: WDDM oversubscription)."""
import os, runpy, sys
import torch
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('G4_CUDA_FRAC', '0.4')))
sp = '/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/selfplay/learner/selfplay.py'
sys.argv = [sp] + sys.argv[1:]
sys.path.insert(0, os.path.dirname(sp))
runpy.run_path(sp, run_name='__main__')
