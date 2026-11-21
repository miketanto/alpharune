#!/usr/bin/env bash
# Builds build_pic/libplayweb.so in the WSL build copy (~/riftbound/alpharune), the PIC
# build directory research/card_repr/selfplay/rbenv/build.sh set up. Copies only
# tools/play_web/playweb.{cpp,cmake} there; the research sources it #includes must
# already be in the copy (rbenv's build.sh puts them there). Touches no other target:
# only `playweb` is built. Cores 12-15, nice 10, 2 jobs (training may own the rest).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DST=${RIFTBOUND_WSL_COPY:-$HOME/riftbound/alpharune}
mkdir -p "$DST/tools/play_web"
cp "$HERE/playweb.cpp" "$HERE/playweb.cmake" "$DST/tools/play_web/"
test -f "$DST/research/card_repr/selfplay/rbenv/rbenv.cpp" || { echo "no rbenv sources in $DST (run rbenv/build.sh first)"; exit 1; }
test -f "$DST/build_pic/build.ninja" || { echo "no $DST/build_pic (run research/card_repr/selfplay/rbenv/build.sh first)"; exit 1; }
grep -q tools/play_web/playweb.cmake "$DST/CMakeLists.txt" || echo 'include(tools/play_web/playweb.cmake)' >> "$DST/CMakeLists.txt"
cd "$DST"
nice -n 10 taskset -c 12-15 cmake --build build_pic --target playweb -j2 2>&1 | grep -E "error|Error|FAILED|Linking|warning: unused" | head -60
ls -la build_pic/libplayweb.so
