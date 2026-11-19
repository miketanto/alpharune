#!/usr/bin/env bash
set -euo pipefail
SRC=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/C_engine
cd $HOME/riftbound/alpharune
SUBJ=${1:-$SRC/out/subjects.txt}
OUTF=${2:-$SRC/out/raw_probe.json}
/usr/bin/time -v ./build/card_probe $SUBJ $OUTF 2> $SRC/out/probe_stderr.log || true
tail -25 $SRC/out/probe_stderr.log
