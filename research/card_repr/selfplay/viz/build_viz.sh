#!/usr/bin/env bash
# Builds librbenv_viz4.so (rbenv_viz.cpp = working-tree rbenv.cpp, schema v4, + HTML replay option)
# WITHOUT touching rbenv's build: compiles only rbenv_viz.cpp and links it against the private
# snapshot ~/riftbound/viz_build (cp -a of ~/riftbound/alpharune/build_pic, engine src/ unchanged).
# Viz sources are copied to ~/riftbound/alpharune/research/card_repr/selfplay/viz/.
# nice -n 10, cores 8-15, one compile job. The v2 build (old replays) is v2/build_viz.sh -> librbenv_viz.so.
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
nice -n 10 taskset -c 8-15 /usr/bin/c++ -DBOOST_ALL_NO_LIB -DBOOST_CONTEXT_DYN_LINK -DBOOST_SYSTEM_DYN_LINK $INC -O3 -DNDEBUG -fPIC -std=gnu++20 \
  -o viz/rbenv_viz4.o -c $REPO/research/card_repr/selfplay/viz/rbenv_viz.cpp
# link: rbenv's own link line (from the snapshot) with its rbenv.cpp.o swapped for ours
LINK=$(ninja -t commands rbenv | grep -- '-o librbenv.so' | sed 's/^: && //; s/ && :$//')
LINK=${LINK//CMakeFiles\/rbenv.dir\/research\/card_repr\/selfplay\/rbenv\/rbenv.cpp.o/viz\/rbenv_viz4.o}
LINK=${LINK//-soname,librbenv.so/-soname,librbenv_viz4.so}
LINK=${LINK//-o librbenv.so/-o librbenv_viz4.so}
nice -n 10 taskset -c 8-15 bash -c "$LINK"
ls -la $B/librbenv_viz4.so
