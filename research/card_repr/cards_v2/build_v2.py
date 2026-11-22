"""Card vocabulary v2: every card the engine registers gets its own key and a text vector.

  python build_v2.py          (Windows python: transformers 4.23 + the cached bge-small model, as A_text)

Inputs
  registry.json                     ./build/riftbound --dump-registry (955 cards, current CardDef text)
  ../C_engine/out/subjects.txt      the v1 id -> key map (51 cards): kept verbatim
  ../selfplay/rbenv/vocab.json      the v1 vocabulary (80 keys): kept verbatim as indices 0..79

Outputs (this directory)
  subjects_v2.txt   rbenv's id -> key map: the v1 lines first, then "class:<key>:<id> <id> 0" for every
                    other registered card id. Several ids share a key when their printed card is the
                    same (same name, type, costs, Might, domains, text): reprints.
  vocab_v2.json     v1 vocabulary, then the new keys (sorted). Old indices keep their meaning.
  text_v2.json      {"dim", "model", "vectors": {key: [...]}} A_text's pipeline (stats line + rules
                    text, reminder text stripped, bge-small CLS, segmented) for every card key.
  keys_v2.json      key -> {ids, name, type} (for reports).
"""
import importlib.util
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CR = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("a_build", os.path.join(CR, "A_text", "build.py"))
AB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AB)


def slug(s):
    o = ""
    for c in s:
        if c.isalnum():
            o += c.lower()
        elif o and o[-1] != "_":
            o += "_"
    return o.strip("_")


def printed(c):
    return json.dumps([c["name"], c["card_type"], c.get("super_type"), sorted(c.get("domains") or []),
                       c.get("energy_cost"), c.get("power_cost"), c.get("might"), c.get("might_bonus"),
                       (c.get("ability_text") or "").strip(), (c.get("effect_text") or "").strip()])


def main():
    t0 = time.time()
    reg = json.load(open(os.path.join(HERE, "registry.json"), encoding="utf-8"))["cards"]
    by_id = {c["card_id"]: c for c in reg}
    v1_vocab = json.load(open(os.path.join(CR, "selfplay", "rbenv", "vocab.json")))
    v1_lines = [l.rstrip("\n") for l in open(os.path.join(CR, "C_engine", "out", "subjects.txt"), encoding="utf-8")]
    id2key = {}
    for l in v1_lines:
        name, cid, rn = l.split()
        if rn == "0" and name.startswith("class:"):
            id2key[int(cid)] = name[6:name.rfind(":")]
    # a v1 key also covers every other id with the same printed card
    pr2key = {printed(by_id[i]): k for i, k in id2key.items() if i in by_id}
    used = {}                                   # key -> printed signature
    for i, k in id2key.items():
        if i in by_id:
            used.setdefault(k, printed(by_id[i]))
    new_lines = []
    for c in sorted(reg, key=lambda c: c["card_id"]):
        i = c["card_id"]
        if i in id2key:
            continue
        sig = printed(c)
        if sig in pr2key:
            k = pr2key[sig]
        else:
            k = slug(c["name"]) or f"card_{i}"
            if k in used and used[k] != sig:      # same name, different card (another set's version)
                k = f"{k}__{i}"
            pr2key[sig] = k
        used.setdefault(k, sig)
        id2key[i] = k
        new_lines.append(f"class:{k}:{i} {i} 0")
    keys_new = sorted({k for k in id2key.values()} - set(v1_vocab))
    # Tokens created without a registered token card are keyed "token:<name>" by the encoder (cardKey).
    # Names from the engine's createToken calls (src/cards) plus Reflection (Mirror Image / Deceiver) and
    # Baron Pit (battlefield token); found missing by an unknown-key scan of all 41 decks.
    tokens = [f"token:{t}" for t in ("recruit", "sprite", "mech", "bird", "sand_soldier", "gold", "tentacle",
                                       "shadow_clone", "reflection", "baron_pit")]
    keys_new += [t for t in tokens if t not in v1_vocab and t not in keys_new]
    vocab = list(v1_vocab) + keys_new
    assert len(set(vocab)) == len(vocab)
    with open(os.path.join(HERE, "subjects_v2.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(v1_lines + new_lines) + "\n")
    json.dump(vocab, open(os.path.join(HERE, "vocab_v2.json"), "w"), indent=0)
    info = {}
    for i, k in id2key.items():
        c = by_id.get(i)
        if c is None:
            continue
        e = info.setdefault(k, {"ids": [], "name": c["name"], "type": c["card_type"]})
        e["ids"].append(i)
    json.dump(info, open(os.path.join(HERE, "keys_v2.json"), "w", encoding="utf-8"), indent=1)

    # text vectors: A_text's input builder on the registry's (current) CardDef text
    items = {}
    for k, e in info.items():
        c = by_id[e["ids"][0]]
        p = {"card_type": c["card_type"], "super_type": c.get("super_type"), "domains": c.get("domains") or [],
             "energy_cost": c.get("energy_cost"), "power_cost": c.get("power_cost"), "might": c.get("might"),
             "might_bonus": c.get("might_bonus") or None, "ability_text": c.get("ability_text") or "",
             "effect_text": c.get("effect_text") or ""}
        if c["card_type"] in ("legend", "battlefield", "rune") or p["energy_cost"] is None:
            pass
        items[k] = AB.card_input(p, c["name"], AB.MODE)
    emb = AB.Embedder(AB.PRIMARY)
    vec = AB.embed_items(emb, items, AB.LAYOUT)
    out = {"dim": int(len(next(iter(vec.values())))), "model": AB.PRIMARY, "mode": AB.MODE, "layout": AB.LAYOUT,
           "vectors": {k: [round(float(x), 6) for x in v] for k, v in vec.items()}}
    json.dump(out, open(os.path.join(HERE, "text_v2.json"), "w"))
    json.dump(items, open(os.path.join(HERE, "text_inputs_v2.json"), "w", encoding="utf-8"), indent=1)
    M = np.stack([vec[k] for k in vec])
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    D = 1 - M @ M.T
    np.fill_diagonal(D, np.inf)
    dup = int((D.min(1) < 1e-6).sum())
    print(f"registry {len(reg)} cards -> {len(info)} keys ({len(keys_new)} new); vocab {len(vocab)}; "
          f"text dim {out['dim']}; nearest-key cos distance median {np.median(D.min(1)):.4f}, "
          f"{dup} keys with an identical vector; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
