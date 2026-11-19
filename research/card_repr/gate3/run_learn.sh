#!/bin/bash
cd "$(dirname "$0")"
for pk in May Accelerate AdditionalCost Location Target TargetFirst TargetSecond ChooseObject TriggerOrder; do
  python3 learn.py --pk $pk --mode card --tag base | tail -1
  python3 learn.py --pk $pk --mode card --tag askopt --net-kw '{"ask_opt": true}' | tail -1
done
# field-only labels (resources / asking unit state)
for kw in base askopt; do
  if [ $kw = askopt ]; then NK='{"ask_opt": true}'; else NK='{}'; fi
  python3 learn.py --pk Accelerate --mode field:x:glob:19:0:4:2 --tag $kw --net-kw "$NK" | tail -1
  python3 learn.py --pk AdditionalCost --mode field:x:glob:19:0:4:2 --tag $kw --net-kw "$NK" | tail -1
  python3 learn.py --pk May --mode field:x:psrc:9:0:6:3 --tag $kw --net-kw "$NK" | tail -1
  python3 learn.py --pk Location --mode field:x:psrc:9:0:6:3 --tag $kw --net-kw "$NK" | tail -1
  python3 learn.py --pk Target --mode field:x:psrc:9:0:6:3 --tag $kw --net-kw "$NK" | tail -1
done
