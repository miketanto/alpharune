"""Self-test vector files for eval.py (written under eval/selftest/<name>/out/vectors.json).

  random  : every entry an independent N(0,1) vector, dim 64 -> should fail T1, T2 at chance, T3 at baseline
  onehot  : one-hot of the entry's NAME (printings share the card's name, a rename gets a
            new name, a synthetic variant gets its own id) -> T1 printings pass, renames fail,
            T2 distinct, T3 at baseline (a held-out card's own dimension is constant in training)
  oracle  : the labels themselves (non-list labels one-hot / numeric) for every base card; printings and
            renames copy the base; synthetics copy the base with the changed field moved.
            Only a positive control of the scorer (it is built FROM labels.json, so T3 is circular).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
cards = json.load(open(os.path.join(HERE, "..", "cards.json"), encoding="utf8"))
labels = json.load(open(os.path.join(HERE, "labels.json"), encoding="utf8"))


def entries():
    out = []
    for c in cards["cards"]:
        out.append((c["key"], c["key"]))
        for p in c["printings"]:
            out.append(("printing:" + p["id"], c["key"]))
        out.append(("rename:" + c["key"], "RENAMED " + c["key"]))
    for s in cards["t2_minimal_pairs_synthetic"]:
        out.append(("synthetic:" + s["id"], "SYN " + s["id"]))
    return out


def write(name, dim, vecs):
    d = os.path.join(HERE, "selftest", name, "out")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "vectors.json"), "w") as f:
        json.dump({"dim": dim, "vectors": {k: [float(x) for x in v] for k, v in vecs.items()}}, f)
    print("wrote", os.path.join(d, "vectors.json"), len(vecs), "vectors, dim", dim)


def main():
    rng = np.random.default_rng(12345)
    ents = entries()
    write("random", 64, {k: rng.standard_normal(64) for k, _ in ents})

    names = sorted({n for _, n in ents})
    ix = {n: i for i, n in enumerate(names)}
    vecs = {}
    for k, n in ents:
        v = np.zeros(len(names))
        v[ix[n]] = 1
        vecs[k] = v
    write("onehot", len(names), vecs)

    # oracle
    L = labels["labels"]
    kinds = labels["label_kinds"]
    cols = []
    for lab in kinds["numeric"]:
        cols.append(("num", lab, None))
    for lab in kinds["categorical"]:
        for v in sorted({str(L[k][lab]["value"]) for k in L}):
            cols.append(("cat", lab, v))
    for lab in kinds["binary"]:
        cols.append(("bin", lab, None))

    def vec_of(lv):
        out = []
        for t, lab, v in cols:
            x = lv[lab]["value"]
            if t == "num":
                out.append(float(x) if x is not None else -1.0)
            elif t == "cat":
                out.append(1.0 if str(x) == v else 0.0)
            else:
                out.append(float(x))
        return np.array(out)

    vecs = {}
    for c in cards["cards"]:
        b = vec_of(L[c["key"]])
        vecs[c["key"]] = b
        vecs["rename:" + c["key"]] = b.copy()
        for p in c["printings"]:
            vecs["printing:" + p["id"]] = b.copy()
    col = {(t, lab, v): i for i, (t, lab, v) in enumerate(cols)}
    for s in cards["t2_minimal_pairs_synthetic"]:
        v = vecs[s["base"]].copy()
        f = s["field"]
        if f in ("energy_cost", "power_cost", "might"):
            v[col[("num", f, None)]] += 1.0
        elif f == "timing":
            for (t, lab, val), i in col.items():
                if lab == "timing":
                    v[i] = 1.0 if val == "reaction" else 0.0
        elif f == "target_side":
            for (t, lab, val), i in col.items():
                if lab == "target_side":
                    v[i] = 1.0 if val == "enemy" else 0.0
        else:  # amount / selector / threshold: the labels have no such column -> extra dim
            pass
        vecs["synthetic:" + s["id"]] = v
    # one extra dim for amount/selector/threshold changes so they are not identical to the base
    extra = {"amount": 0, "selector": 1, "condition_threshold": 2}
    for k in list(vecs):
        vecs[k] = np.concatenate([vecs[k], np.zeros(3)])
    for s in cards["t2_minimal_pairs_synthetic"]:
        if s["field"] in extra:
            vecs["synthetic:" + s["id"]][len(cols) + extra[s["field"]]] = 1.0
    write("oracle", len(cols) + 3, vecs)


if __name__ == "__main__":
    main()
