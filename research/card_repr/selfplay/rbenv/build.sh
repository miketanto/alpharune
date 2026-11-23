#!/usr/bin/env bash
# Copies rbenv sources (and the exporter + probe it #includes) into the WSL
# build copy and builds librbenv.so in build_pic (PIC build of the engine).
# Never touches src/. Cores 8-15, at most 6 jobs.
set -euo pipefail
SRC=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr
DST=${RB_WSL_COPY:-$HOME/riftbound/alpharune}   # RB_WSL_COPY: another build copy (rbenv.py reads the same variable)
mkdir -p $DST/research/card_repr/search $DST/research/card_repr/C_engine/out $DST/research/card_repr/smoke/data/src $DST/research/card_repr/selfplay/rbenv
cp $SRC/C_engine/card_probe.cpp $DST/research/card_repr/C_engine/
cp $SRC/C_engine/out/subjects.txt $DST/research/card_repr/C_engine/out/
cp $SRC/smoke/data/src/decision_export.cpp $SRC/smoke/data/src/decision_export.cmake $DST/research/card_repr/smoke/data/src/
cp $SRC/selfplay/rbenv/rbenv.cpp $SRC/selfplay/rbenv/rbenv.cmake $SRC/selfplay/rbenv/history.h $DST/research/card_repr/selfplay/rbenv/
cp $SRC/search/rbsearch.cpp $DST/research/card_repr/search/
grep -q selfplay/rbenv/rbenv.cmake $DST/CMakeLists.txt || echo 'include(research/card_repr/selfplay/rbenv/rbenv.cmake)' >> $DST/CMakeLists.txt
cd $DST
if [ ! -f build_pic/build.ninja ]; then
  taskset -c 8-15 cmake -B build_pic -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DRIFTBOUND_BUILD_JOBS=6 -DRIFTBOUND_BUILD_OPENSPIEL=ON \
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
    -DFETCHCONTENT_SOURCE_DIR_OPEN_SPIEL=$DST/build/_deps/open_spiel-src \
    -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=$DST/build/_deps/googletest-src > build_pic_configure.log
fi
taskset -c 8-15 cmake --build build_pic --target rbenv -j6 2>&1 | grep -E "error|Error|FAILED|Linking" | head -60
ls -la build_pic/librbenv.so
