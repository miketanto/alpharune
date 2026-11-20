#!/usr/bin/env bash
# Overnight league driver (league/overnight.py), foreground; exits at the deadline.
#   bash league/overnight.sh --deadline 07:30 [more overnight.py run options]
# Run from research/card_repr/selfplay (the script cd's there itself).
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec python3 league/overnight.py run "$@"
