#!/usr/bin/env bash
# Copy the probe + exporter into the WSL build copy and build decision_export on cores 8-15.
set -euo pipefail
SRC=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr
DST=$HOME/riftbound/alpharune
cp $SRC/C_engine/card_probe.cpp $DST/research/card_repr/C_engine/
cp $SRC/smoke/data/src/decision_export.cpp $DST/research/card_repr/smoke/data/src/
cd $DST
taskset -c 8-15 cmake --build build --target decision_export -j6 2>&1 | grep -E "error|FAILED" | head -30 || true
ls -la build/decision_export
