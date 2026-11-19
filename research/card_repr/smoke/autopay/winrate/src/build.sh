#!/usr/bin/env bash
# Copies autopay_match.cpp (and the exporter + probe it includes) into the WSL
# build copy and builds autopay_match there. Never touches src/.
set -euo pipefail
SRC=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr
DST=$HOME/riftbound/alpharune
mkdir -p $DST/research/card_repr/C_engine/out $DST/research/card_repr/smoke/data/src $DST/research/card_repr/smoke/autopay/winrate/src
cp $SRC/C_engine/card_probe.cpp $DST/research/card_repr/C_engine/
cp $SRC/C_engine/out/subjects.txt $DST/research/card_repr/C_engine/out/
cp $SRC/smoke/data/src/decision_export.cpp $DST/research/card_repr/smoke/data/src/
cp $SRC/smoke/autopay/winrate/src/autopay_match.cpp $SRC/smoke/autopay/winrate/src/autopay_match.cmake $DST/research/card_repr/smoke/autopay/winrate/src/
grep -q autopay_match.cmake $DST/CMakeLists.txt || echo 'include(research/card_repr/smoke/autopay/winrate/src/autopay_match.cmake)' >> $DST/CMakeLists.txt
cd $DST
cmake build > /dev/null
taskset -c 0-7 cmake --build build --target autopay_match -j8 2>&1 | grep -E "error|Error|Linking|FAILED" | head -60
ls -la build/autopay_match
