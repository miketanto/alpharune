#!/usr/bin/env bash
# guard.sh <min_available_MB> <cmd...>: runs cmd in WSL; kills it if Windows Available memory
# (queried through powershell.exe) drops below the threshold. Logs the minimum seen.
min=$1; shift
"$@" &
pid=$!
lowest=999999
while kill -0 $pid 2>/dev/null; do
  av=$(powershell.exe -NoProfile -c "[int](Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue" 2>/dev/null | tr -d '\r')
  [ -n "$av" ] && [ "$av" -lt "$lowest" ] && lowest=$av
  if [ -n "$av" ] && [ "$av" -lt "$min" ]; then echo "GUARD: available $av MB < $min, killing" >&2; kill $pid; pkill -P $pid; fi
  sleep 3
done
wait $pid; rc=$?
echo "GUARD: lowest Windows available MB during run: $lowest rc=$rc" >&2
exit $rc
