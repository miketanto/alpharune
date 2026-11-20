#!/bin/bash
set -e
G5=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate5
python3 $G5/splice.py
bash $G5/../selfplay/rbenv/build.sh
