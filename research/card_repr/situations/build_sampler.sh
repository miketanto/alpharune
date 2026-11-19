#!/usr/bin/env bash
# Copies the probe + sampler sources into the WSL build copy and builds
# card_probe and situation_sampler there (never touches src/).
set -euo pipefail
SRC=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr
DST=$HOME/riftbound/alpharune
mkdir -p $DST/research/card_repr/C_engine $DST/research/card_repr/situations
cp $SRC/C_engine/card_probe.cpp $SRC/C_engine/card_probe.cmake $DST/research/card_repr/C_engine/
cp $SRC/situations/sampler.cpp $SRC/situations/situations.cmake $DST/research/card_repr/situations/
grep -q card_probe.cmake $DST/CMakeLists.txt || echo 'include(research/card_repr/C_engine/card_probe.cmake)' >> $DST/CMakeLists.txt
grep -q situations.cmake $DST/CMakeLists.txt || echo 'include(research/card_repr/situations/situations.cmake)' >> $DST/CMakeLists.txt
cd $DST
cmake build > /dev/null
cmake --build build --target card_probe situation_sampler -j12
