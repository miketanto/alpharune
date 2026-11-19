#!/usr/bin/env bash
# Builds librbenv_viz.so (rbenv_viz.cpp = HEAD rbenv.cpp + HTML replay option) WITHOUT
# touching rbenv's build: compiles only rbenv_viz.cpp and links it against a private
# snapshot of the WSL build copy's PIC build (~/riftbound/viz_build = cp -a of
# ~/riftbound/alpharune/build_pic, taken once; engine sources under src/ unchanged).
# Viz sources are copied to ~/riftbound/alpharune/research/card_repr/selfplay/viz/ (a new
# folder there). Cores 8-15.
set -euo pipefail
SRC=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/selfplay/viz
REPO=$HOME/riftbound/alpharune
B=$HOME/riftbound/viz_build
[ -d $B ] || cp -a $REPO/build_pic $B
mkdir -p $REPO/research/card_repr/selfplay/viz
cp $SRC/rbenv_viz.cpp $SRC/decision_export.cpp $SRC/card_probe.cpp $REPO/research/card_repr/selfplay/viz/
cd $B
INC="-I$B/generated -I$REPO -I$REPO/src -I$REPO/build/_deps/open_spiel-src -I$REPO/build/_deps/open_spiel-src/open_spiel/json/include -I$REPO/build/_deps/open_spiel-src/open_spiel -I$REPO/build/_deps/open_spiel-src/open_spiel/abseil-cpp"
mkdir -p viz
taskset -c 8-15 /usr/bin/c++ -DBOOST_ALL_NO_LIB -DBOOST_CONTEXT_DYN_LINK -DBOOST_SYSTEM_DYN_LINK $INC -O3 -DNDEBUG -fPIC -std=gnu++20 \
  -o viz/rbenv_viz.o -c $REPO/research/card_repr/selfplay/viz/rbenv_viz.cpp
# link: rbenv's own link line with its rbenv.cpp.o swapped for ours
LINK=$(ninja -t commands rbenv | grep -- '-o librbenv.so' | sed 's/^: && //; s/ && :$//')
LINK=${LINK//CMakeFiles\/rbenv.dir\/research\/card_repr\/selfplay\/rbenv\/rbenv.cpp.o/viz\/rbenv_viz.o}
LINK=${LINK//-soname,librbenv.so/-soname,librbenv_viz.so}
LINK=${LINK//-o librbenv.so/-o librbenv_viz.so}
taskset -c 8-15 bash -c "$LINK"
ls -la $B/librbenv_viz.so
