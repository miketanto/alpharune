"""Assemble out/vectors.json (the SPEC contract) and out/raw.json from the
probe tool's output (out/raw_probe.json), and print the checks the report
quotes (rename / printing / selfcheck distances, coverage).

Vector = concatenation of every probe block, in the probe tool's fixed layout
order. Values are raw engine deltas (counts, Might, resources), no scaling.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def main():
    raw = json.load(open(os.path.join(OUT, "raw_probe.json")))
    pmap = json.load(open(os.path.join(OUT, "printing_map.json")))
    cards = json.load(open(os.path.join(ROOT, "research/card_repr/cards.json"), encoding="utf8"))
    layout = raw["layout"]
    order = [b["block"] for b in layout]
    subj = raw["subjects"]

    def vec(name):
        blocks = subj[name]["blocks"]
        return np.concatenate([np.asarray(blocks[b], dtype=float) for b in order])

    dim = int(sum(len(b["fields"]) for b in layout))
    vectors = {}
    notes = {"aliased_printings": [], "class_printing_distances": {}, "rename_distances": {},
             "selfcheck_distances": {}}
    for c in cards["cards"]:
        key = c["key"]
        classes = pmap[key]["classes"]
        if not classes:
            continue
        cvs = {cid: vec(f"class:{key}:{cid}") for cid in classes}
        kv = np.mean(list(cvs.values()), axis=0)
        vectors[key] = kv
        for pid, cid in pmap[key]["printings"].items():
            if cid is None:
                cid_use = classes[0]
                notes["aliased_printings"].append(pid)
            else:
                cid_use = cid
            vectors[f"printing:{pid}"] = cvs.get(cid_use, kv)
        if len(classes) > 1:
            a, b = classes[0], classes[1]
            notes["class_printing_distances"][key] = float(np.linalg.norm(cvs[a] - cvs[b]))
        rv = vec(f"rename:{key}")
        vectors[f"rename:{key}"] = rv
        notes["rename_distances"][key] = float(np.linalg.norm(rv - cvs[classes[0]]))
    for s in cards["t2_minimal_pairs_synthetic"]:
        vectors[f"synthetic:{s['id']}"] = vec(f"synthetic:{s['id']}")
    base_of = {"punch_first": "class:punch_first:419", "discipline": "class:discipline:58",
               "lonely_poro": "class:lonely_poro:777", "defy": "class:defy:45",
               "sabotage": "class:sabotage:156", "pit_rookie": "class:pit_rookie:136"}
    for k, real in base_of.items():
        notes["selfcheck_distances"][k] = float(np.linalg.norm(vec(f"selfcheck:{k}") - vec(real)))

    # Coverage: blocks that are all-zero for every real card; per card, how many blocks are nonzero.
    real_names = [n for n in subj if n.startswith("class:")]
    nz_blocks = {b: sum(1 for n in real_names if any(subj[n]["blocks"][b])) for b in order}
    coverage = {"blocks_nonzero_count": nz_blocks,
                "errors": raw["errors"],
                "total_seconds": raw["total_seconds"]}

    def num(x):
        x = round(float(x), 6)
        return int(x) if x == int(x) else x
    out = {"dim": dim, "vectors": {k: [num(x) for x in v] for k, v in vectors.items()}}
    with open(os.path.join(OUT, "vectors.json"), "w") as f:
        json.dump(out, f, separators=(",", ":"))
    rawout = {"layout": layout, "subjects": subj, "notes": notes, "coverage": coverage}
    with open(os.path.join(OUT, "raw.json"), "w") as f:
        json.dump(rawout, f, separators=(",", ":"))
    print("dim", dim, "entries", len(vectors))
    print("aliased printings:", notes["aliased_printings"])
    print("two-class printing distances:", notes["class_printing_distances"])
    print("max rename distance:", max(notes["rename_distances"].values()))
    print("selfcheck distances:", notes["selfcheck_distances"])
    nonzero_dims = int((np.abs(np.stack([vectors[k] for k in vectors])).sum(0) > 0).sum())
    print("dims nonzero for some entry:", nonzero_dims)
    # Phase 2 fix: no entry may be all zero (the static/card block is always set).
    zero = [k for k, v in vectors.items() if not np.any(v)]
    print("all-zero entries:", zero)
    assert not zero, zero
    print("blocks never nonzero for a real card:", [b for b, c in nz_blocks.items() if c == 0])
    # minimal pairs distances
    def d(a, b):
        return float(np.linalg.norm(vectors[a] - vectors[b]))
    for a, b, why in cards["t2_minimal_pairs_real"]:
        if a in vectors and b in vectors:
            print(f"pair {a} / {b}: {d(a, b):.2f}")
    for s in cards["t2_minimal_pairs_synthetic"]:
        print(f"synthetic {s['id']}: {d(s['base'], 'synthetic:' + s['id']):.2f}")


if __name__ == "__main__":
    main()
