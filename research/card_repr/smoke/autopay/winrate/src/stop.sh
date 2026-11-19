#!/usr/bin/env bash
# Stops the running batches (finished games are already flushed).
touch ~/apw/STOP
pkill -f "bash .*/run_main.sh"
pkill -f "bash .*/run_play.sh"
pkill -x autopay_match
sleep 1
pgrep -c autopay_match
