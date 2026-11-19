#!/usr/bin/env bash
# Copies ~/apw results back into winrate/raw (games) and winrate/logs.
set -uo pipefail
W=/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/smoke/autopay/winrate
mkdir -p $W/raw $W/logs
cp $HOME/apw/out/*.games.jsonl $W/raw/ 2>/dev/null
cp $HOME/apw/log/*.log $W/logs/ 2>/dev/null
[ -d $HOME/apw/verify ] && cp $HOME/apw/verify/*.txt $W/logs/ 2>/dev/null
true
