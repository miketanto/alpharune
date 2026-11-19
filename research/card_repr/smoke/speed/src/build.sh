#!/usr/bin/env bash
# Generates speed_export.cpp (make_speed_export.py, run on Windows beforehand), copies it into
# the WSL build copy and builds target speed_export there on cores 8-15. Never touches src/.
set -euo pipefail
SRC=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/smoke/speed/src
DST=$HOME/riftbound/alpharune
mkdir -p $DST/research/card_repr/smoke/speed/src
cp $SRC/speed_export.cpp $SRC/speed_export.cmake $DST/research/card_repr/smoke/speed/src/
grep -q speed_export.cmake $DST/CMakeLists.txt || echo 'include(research/card_repr/smoke/speed/src/speed_export.cmake)' >> $DST/CMakeLists.txt
cd $DST
taskset -c 8-15 cmake build > /dev/null
taskset -c 8-15 cmake --build build --target speed_export -j8 2>&1 | grep -E "error|Error|Linking|FAILED" | head -60
ls -la build/speed_export
