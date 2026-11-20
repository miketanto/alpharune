cd /mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate5
python3 analyze.py "resp_s*.jsonl" --json results.json > tables.txt 2>&1
python3 analyze.py "resp_s*.jsonl" --nets B2,id,B2_ft20,id_ft20,B2_ft50,id_ft50,B2_ft100,id_ft100 > tables_ft.txt 2>&1
