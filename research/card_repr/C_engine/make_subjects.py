"""Map every card key / printing of cards.json to an engine card class id.

Writes out/subjects.txt (input of card_probe) and out/printing_map.json.
The engine has one class per printing id it implements (def_id); printings
without a class of their own are recorded as aliases of the key's class.
"""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "out")

# Keys whose cards.json entry has no printing: the engine legend classes.
LEGEND_CLASSES = {"rengar_pridestalker": 744, "master_yi_wuju_bladesman": 317}


def main():
    os.makedirs(OUT, exist_ok=True)
    cards = json.load(open(os.path.join(ROOT, "research/card_repr/cards.json"), encoding="utf8"))["cards"]
    by_def = {}
    for f in glob.glob(os.path.join(ROOT, "src/cards/*/*.cpp")):
        t = open(f, encoding="utf8").read()
        for m in re.finditer(r'd\.id = (\d+);\s*d\.def_id = R"RB\(([^)]*)\)RB"', t):
            by_def[m.group(2)] = int(m.group(1))
    pmap = {}
    lines = []
    for c in cards:
        key = c["key"]
        classes = []
        printings = {}
        for p in c["printings"]:
            cid = by_def.get(p["id"])
            printings[p["id"]] = cid
            if cid is not None and cid not in classes:
                classes.append(cid)
        if key in LEGEND_CLASSES:
            classes = [LEGEND_CLASSES[key]]
        pmap[key] = {"classes": classes, "printings": printings}
        for i, cid in enumerate(classes):
            lines.append(f"class:{key}:{cid} {cid} 0")
        if classes:
            lines.append(f"rename:{key} {classes[0]} 1")
    # Hush has no engine class: its stand-in (registered by card_probe as 1650).
    pmap["hush"]["classes"] = [1650]
    lines.append("class:hush:1650 1650 0")
    lines.append("rename:hush 1650 1")
    open(os.path.join(OUT, "subjects.txt"), "w").write("\n".join(lines) + "\n")
    json.dump(pmap, open(os.path.join(OUT, "printing_map.json"), "w"), indent=1)
    print(len(lines), "subjects")


if __name__ == "__main__":
    main()
