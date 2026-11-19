"""Path B build: mechanics graphs -> fixed (non-learned) feature-hashed vectors.

Usage:  python build.py            (from anywhere; paths are relative to this file)

Writes
  cards/<key>.json        the graph per card (stats + keywords + abilities)
  out/vectors.json        output contract (keys, printing:, rename:, synthetic:)
  out/paths/<key>.json    the explicit sparse path features per card key (debug)
  out/selfcheck.json      quick distances (T1 / T2 sanity, not the official eval)
"""
import copy
import hashlib
import json
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import graphs_src as GS  # noqa: E402

HASH_DIM = 8192
import struct_block as SB  # noqa: E402  (structured block, gate 2 fix; concatenated after the hash)
DIM = HASH_DIM + SB.size()
ROOT = "card"
SKIP_SEGMENTS = {"abilities", "effects"}   # carried by the ab:/eff: labels already
LABEL_KEYS = ("kind", "verb", "f", "c", "kw")   # dict keys that name the node
T0 = time.time()

# ---------------------------------------------------------------- stats

def stats_from_printing(p):
    t = p["card_type"]
    has_cost = t in ("unit", "spell", "gear")
    domains = sorted(p["domains"] or [])
    s = {"type": t, "supertype": p["super_type"], "domains": domains}
    if has_cost:
        s["energy"] = p["energy_cost"] or 0
        s["power"] = p["power_cost"] or 0
        if s["power"]:
            # power cost per domain: the card's own domain(s); no domain -> any
            pd = domains or ["any"]
            s["power_by_domain"] = {d: s["power"] for d in pd}
    else:
        s["energy"] = None
        s["power"] = None
    s["might"] = p["might"] if t == "unit" else None
    if t == "unit" and s["might"] is None:
        s["might"] = 0
    return s


def legend_stats(key):
    s = dict(GS.LEGEND_STATS[key])
    s["domains"] = sorted(s["domains"])
    return s


def norm_text(t):
    """Printing text -> mechanics-relevant text: drop reminder text and glyph noise."""
    if not t:
        return ""
    t = re.sub(r"\([^()]*\)", " ", t)
    t = t.replace("�", " - ").replace("[>]", " - ").replace("—", " - ")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([.,])", r"\1", t)
    t = re.sub(r"(\s-\s)+", " - ", t)
    return t.strip()

# ---------------------------------------------------------------- graph -> paths

def node_label(d):
    for k in LABEL_KEYS:
        if k in d and isinstance(d[k], str):
            lab = f"{k}:{d[k]}"
            if k == "kind" and isinstance(d.get("timing"), str):
                lab += f"@{d['timing']}"
            return lab, k
    return None, None


def walk(node, segs, out):
    if isinstance(node, dict):
        lab, lk = node_label(node)
        if lab:
            segs = segs + [lab]
            out.append((segs, 1.0))            # presence/count of this node
        for k, v in sorted(node.items()):
            if k == lk or (lk == "kind" and k == "timing"):
                continue
            walk_value(k, v, segs, out)
    else:
        raise TypeError(node)


def walk_value(k, v, segs, out):
    if v is None or v is False:
        return
    if v is True:
        out.append((segs + [k], 1.0))
    elif isinstance(v, (int, float)):
        # numeric leaf kept as a value: raw for card stats (small bounded
        # costs / Might), signed log2(1+|v|) inside abilities so a -10 does not
        # swamp the structure
        x = float(v) if segs[-1] == "stats" or segs[-1] == "power_by_domain" else             math.copysign(math.log2(1 + abs(v)), v)
        out.append((segs + [k], x))
        out.append((segs + [f"{k}={v}"], 1.0))       # and as a category
    elif isinstance(v, str):
        out.append((segs + [f"{k}={v}"], 1.0))
    elif isinstance(v, list):
        for x in v:
            if isinstance(x, dict):
                walk(x, segs if k in SKIP_SEGMENTS else segs + [k], out)
            else:
                walk_value(k, x, segs, out)
    elif isinstance(v, dict):
        sub = segs if k in SKIP_SEGMENTS else segs + [k]
        lab, _ = node_label(v)
        if lab:
            walk(v, sub, out)
        else:
            for kk, vv in sorted(v.items()):
                walk_value(kk, vv, sub, out)
    else:
        raise TypeError((k, v))


def graph_paths(graph):
    """Typed root-to-leaf paths, each also emitted at every shorter suffix
    (multi-granularity: 'eff:GiveMight/amount/n' and 'amount/n' both exist)."""
    raw = []
    walk(graph, [ROOT], raw)
    feats = {}
    for segs, val in raw:
        for i in range(1, len(segs)):
            p = "/".join(segs[i:])
            feats[p] = feats.get(p, 0.0) + val
    return feats


def h(path):
    d = hashlib.md5(path.encode("utf-8")).digest()
    idx = int.from_bytes(d[:4], "little") % HASH_DIM
    sign = 1.0 if d[4] & 1 else -1.0
    return idx, sign


def hash_vectorize(graph):
    """The original 8192-dim path hash (pre-gate-2 vectors = exactly this)."""
    v = [0.0] * HASH_DIM
    for p, val in graph_paths(graph).items():
        i, s = h(p)
        v[i] += s * val
    return [0 if x == 0 else round(x, 6) for x in v]


def vectorize(graph):
    """Hash (8192) ++ structured block (struct_block.py, weighted). DIM = 8192 + block size."""
    return hash_vectorize(graph) + [0 if x == 0 else round(x, 6) for x in SB.vectorize(graph)]

# ---------------------------------------------------------------- synthetic edits

def _spell(g):
    return [a for a in g["abilities"] if a["kind"] == "spell"][0]


def syn_edit(base_graph, sid):
    g = copy.deepcopy(base_graph)
    if sid == "punch_first__amount4":
        _spell(g)["effects"][0]["amount"]["n"] = 4
    elif sid == "punch_first__energy3":
        g["stats"]["energy"] = 3
    elif sid == "punch_first__power0":
        g["stats"]["power"] = 0
        g["stats"].pop("power_by_domain", None)
    elif sid == "punch_first__reaction":
        _spell(g)["timing"] = "reaction"
        g["keywords"] = [{"kw": "reaction"} if k["kw"] == "action" else k for k in g["keywords"]]
    elif sid == "discipline__draw2":
        _spell(g)["effects"][1]["amount"]["n"] = 2
    elif sid == "discipline__enemy":
        _spell(g)["effects"][0]["target"]["side"] = "enemy"
    elif sid == "lonely_poro__draw2":
        g["abilities"][0]["effects"][0]["amount"]["n"] = 2
    elif sid == "defy__cost2":
        _spell(g)["effects"][0]["target"]["filters"][0]["v"] = 2
    elif sid == "sabotage__unit":
        _spell(g)["effects"][1]["target"]["filters"] = [{"f": "type", "v": "unit"}]
    elif sid == "pit_rookie__might3":
        g["stats"]["might"] = 3
    else:
        raise KeyError(sid)
    return g

# ---------------------------------------------------------------- main

def main():
    cj = json.load(open(os.path.join(HERE, "..", "cards.json"), encoding="utf-8"))
    os.makedirs(os.path.join(HERE, "cards"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "out", "paths"), exist_ok=True)
    vecs, graphs, problems = {}, {}, []

    def assemble(stats, key):
        src = GS.G[key]
        return {"stats": stats, "keywords": copy.deepcopy(src["keywords"]),
                "abilities": copy.deepcopy(src["abilities"])}

    def no_name(graph, name):
        s = json.dumps(graph).lower()
        if name and name.lower() in s:
            problems.append(f"name '{name}' leaks into graph")

    missing_src = [c["key"] for c in cj["cards"] if c["key"] not in GS.G]
    assert not missing_src, missing_src

    for c in cj["cards"]:
        key, name = c["key"], c["name"]
        prs = c["printings"]
        if prs:
            texts = {norm_text(p["ability_text"]) for p in prs}
            if len(texts) > 1:
                problems.append(f"{key}: printings differ after normalisation: {texts}")
            stats_set = {json.dumps(stats_from_printing(p), sort_keys=True) for p in prs}
            if len(stats_set) > 1:
                problems.append(f"{key}: printings differ in stats")
            base = assemble(stats_from_printing(prs[0]), key)
        else:
            base = assemble(legend_stats(key), key)
        no_name(base, name)
        graphs[key] = base
        with open(os.path.join(HERE, "cards", f"{key}.json"), "w", encoding="utf-8") as f:
            json.dump(base, f, indent=1)
        with open(os.path.join(HERE, "out", "paths", f"{key}.json"), "w", encoding="utf-8") as f:
            json.dump(dict(sorted(graph_paths(base).items())), f, indent=0)
        vecs[key] = vectorize(base)
        # per printing: rebuild the graph from that printing's own fields
        for p in prs:
            g = assemble(stats_from_printing(p), key)
            vecs[f"printing:{p['id']}"] = vectorize(g)
        # rename: the name never enters the graph; rebuild after substituting a
        # random name anywhere it could occur (it occurs nowhere) and re-encode
        renamed = json.loads(json.dumps(base).replace(name, "Zorblax Quintessa")) if name else base
        vecs[f"rename:{key}"] = vectorize(renamed)

    for s in cj["t2_minimal_pairs_synthetic"]:
        vecs[f"synthetic:{s['id']}"] = vectorize(syn_edit(graphs[s["base"]], s["id"]))

    with open(os.path.join(HERE, "out", "vectors.json"), "w", encoding="utf-8") as f:
        json.dump({"dim": DIM, "vectors": vecs}, f)

    # ---- self check (sanity only; official scoring is eval/)
    import numpy as np
    keys = [c["key"] for c in cj["cards"]]
    M = np.array([vecs[k] for k in keys])
    def d(a, b):
        return float(np.linalg.norm(np.array(vecs[a]) - np.array(vecs[b])))
    t1 = {}
    for c in cj["cards"]:
        for p in c["printings"]:
            t1[f"printing:{p['id']}"] = d(c["key"], f"printing:{p['id']}")
        t1[f"rename:{c['key']}"] = d(c["key"], f"rename:{c['key']}")
    pairs = {}
    for a, b, why in cj["t2_minimal_pairs_real"]:
        dist = np.linalg.norm(M - np.array(vecs[a]), axis=1)
        order = [keys[i] for i in np.argsort(dist) if keys[i] != a]
        pairs[f"{a}|{b}"] = {"dist": d(a, b), "rank_of_partner": order.index(b) + 1,
                             "nearest": order[:3]}
    syn = {s["id"]: d(s["base"], f"synthetic:{s['id']}") for s in cj["t2_minimal_pairs_synthetic"]}
    dists = [float(np.linalg.norm(M[i] - M[j])) for i in range(len(keys)) for j in range(i + 1, len(keys))]
    dup = [(keys[i], keys[j]) for i in range(len(keys)) for j in range(i + 1, len(keys))
           if np.allclose(M[i], M[j])]
    nnz = sum(1 for v in M.flatten() if v != 0)
    paths_total = len(set().union(*[set(graph_paths(g)) for g in graphs.values()]))
    used = sorted({h(p)[0] for g in graphs.values() for p in graph_paths(g)})
    report = {"max_t1_distance": max(t1.values()), "real_pairs": pairs, "synthetic_dist": syn,
              "pairwise_card_distance_min_median_max": [min(dists), float(np.median(dists)), max(dists)],
              "identical_card_vectors": dup, "distinct_paths": paths_total,
              "hash_slots_used": len(used), "nonzeros_total": nnz,
              "problems": problems, "n_vectors": len(vecs),
              "build_seconds": round(time.time() - T0, 2)}
    with open(os.path.join(HERE, "out", "selfcheck.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
