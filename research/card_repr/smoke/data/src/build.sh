#!/usr/bin/env bash
# Copies the exporter (and the probe file it includes) into the WSL build copy
# and builds decision_export there. Never touches src/.
set -euo pipefail
SRC=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr
DST=$HOME/riftbound/alpharune
mkdir -p $DST/research/card_repr/C_engine $DST/research/card_repr/smoke/data/src
cp $SRC/C_engine/card_probe.cpp $DST/research/card_repr/C_engine/
cp $SRC/smoke/data/src/decision_export.cpp $SRC/smoke/data/src/decision_export.cmake $DST/research/card_repr/smoke/data/src/
mkdir -p $DST/research/card_repr/C_engine/out
cp $SRC/C_engine/out/subjects.txt $DST/research/card_repr/C_engine/out/
grep -q decision_export.cmake $DST/CMakeLists.txt || echo 'include(research/card_repr/smoke/data/src/decision_export.cmake)' >> $DST/CMakeLists.txt
cd $DST
cmake build > /dev/null
cmake --build build --target decision_export -j12 2>&1 | grep -E "error|Error|warning: unused|Linking|FAILED" | head -60
ls -la build/decision_export
