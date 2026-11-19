#!/usr/bin/env bash
set -euo pipefail
SRC=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/C_engine
DST=$HOME/riftbound/alpharune
mkdir -p $DST/research/card_repr/C_engine
cp $SRC/card_probe.cpp $SRC/card_probe.cmake $DST/research/card_repr/C_engine/
grep -q card_probe.cmake $DST/CMakeLists.txt || echo 'include(research/card_repr/C_engine/card_probe.cmake)' >> $DST/CMakeLists.txt
cd $DST
cmake build > /dev/null
cmake --build build --target card_probe -j12
