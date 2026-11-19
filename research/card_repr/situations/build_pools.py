"""Build the committed situation pools and per-pool behaviour records.

Inputs
  ../C_engine/out/raw_probe.json   pool H: the canonical probes (card_probe)
  raw/index.jsonl, raw/records.jsonl  pool S: situation_sampler output (gitignored)

Outputs (committed)
  H.json                 pool H situations (probe name, kind, what it sets up)
  S_train.json, S_test.json   pool S situations, split by game seed (seed % 10 < 7 -> train)
  out/H.json, out/S_train.json, out/S_test.json   behaviour records per pool
  out/coverage.json      per card: situations per pool and kind
  out/ab_effects.json    per card: effect fields seen per pool (the first A/B signal)

Behaviour records are compact: per card, the distinct behaviours (legality +
sparse state delta) and, per situation, the index of its behaviour. The
situation order is the order of the pool's situation file.
"""
import collections
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CE = os.path.join(HERE, "..", "C_engine", "out")
RAW = os.path.join(HERE, "raw")
OUT = os.path.join(HERE, "out")

EXERCISED_S = {"play", "act", "trigger"}
NOISE = {"prompts", "error", "chain_size"}


def is_train(seed):
    return seed % 10 < 7


def common_field(f):
    """Fields comparable between H and S: side aggregates, pool, battlefields,
    events and the card itself (H's other roles F1/E1/... do not exist in S)."""
    if f in NOISE:
        return False
    head = f.split(".")[0]
    if "." not in f:
        return f.startswith("ev_")
    return head in ("me", "opp", "pool", "bf0", "bf1", "SELF")


def effects_of(rec):
    """Signed nonzero comparable fields of every executed delta of a record."""
    out = set()
    for d in rec.get("exec", {}).values():
        for f, v in d.items():
            if common_field(f) and v:
                out.add(f + ("+" if v > 0 else "-"))
    return out


def compact(records):
    """records: list of (card, behaviour dict) in situation order."""
    cards = collections.OrderedDict()
    for i, (card, beh) in enumerate(records):
        c = cards.setdefault(card, {"behaviours": [], "_keys": {}, "situations": []})
        key = json.dumps(beh, sort_keys=True)
        if key not in c["_keys"]:
            c["_keys"][key] = len(c["behaviours"])
            c["behaviours"].append(beh)
        c["situations"].append([i, c["_keys"][key]])
    for c in cards.values():
        del c["_keys"]
    return cards


def build_h():
    raw = json.load(open(os.path.join(CE, "raw_probe.json")))
    lay = {b["block"]: b["fields"] for b in raw["layout"]}
    order = [b["block"] for b in raw["layout"]]
    # situations = probe windows / events, in layout order
    sits = collections.OrderedDict()
    for b in order:
        parts = b.split("/")
        if parts[0] == "legal":
            sits.setdefault("play:" + parts[1], {"kind": "play", "probe": parts[1]})
        elif parts[0] == "exec":
            sits.setdefault("play:" + parts[1], {"kind": "play", "probe": parts[1]})
        elif parts[0] == "trigger":
            sits.setdefault("trigger:" + parts[1], {"kind": "trigger", "probe": parts[1]})
        elif parts[0] == "static" and parts[1] != "card":
            sits.setdefault("static:" + parts[1], {"kind": "static", "probe": parts[1]})
        elif parts[0] in ("targeted", "inplay_legal"):
            sits.setdefault(parts[0] + ":" + parts[1], {"kind": parts[0].replace("_legal", ""), "probe": parts[1]})
        elif b == "static/card":
            sits.setdefault("card", {"kind": "card", "probe": "printed facts (CardDef)"})
        elif b == "rune/pay":
            sits.setdefault("rune", {"kind": "rune", "probe": "what one rune pays"})

    def subject_key(name):
        if name.startswith("class:"):
            return name.split(":")[1]
        if name.startswith("synthetic:"):
            return name
        return None

    seen = set()
    records = []
    exercised = collections.Counter()
    by_kind = collections.defaultdict(collections.Counter)
    for name, s in raw["subjects"].items():
        key = subject_key(name)
        if key is None or key in seen:
            continue
        seen.add(key)
        blocks = s["blocks"]
        for sid, meta in sits.items():
            beh = {}
            if sid.startswith("play:"):
                w = meta["probe"]
                leg = {f: v for f, v in zip(lay["legal/" + w], blocks["legal/" + w]) if v}
                ex = {}
                for b in order:
                    if b.startswith("exec/" + w + "/"):
                        d = {f: v for f, v in zip(lay[b], blocks[b]) if v}
                        if d:
                            ex[b[len("exec/" + w + "/"):]] = d
                beh = {"legal": leg, "exec": ex}
                active = bool(leg)
            elif sid.startswith("trigger:"):
                ex = {}
                for pol in ("first", "last"):
                    b = f"trigger/{meta['probe']}/{pol}"
                    d = {f: v for f, v in zip(lay[b], blocks[b]) if v}
                    if d:
                        ex["resolve/" + pol] = d
                beh = {"exec": ex}
                active = bool(ex)
            else:
                b = {"card": "static/card", "rune": "rune/pay"}.get(sid, sid.replace(":", "/", 1))
                if sid.startswith("static:"):
                    b = "static/" + meta["probe"]
                d = {f: v for f, v in zip(lay[b], blocks[b]) if v}
                beh = {"exec": {"present": d}} if sid.startswith("static:") else {"legal": d}
                active = bool(d)
            records.append((key, sid, beh))
            if active and meta["kind"] not in ("card", "rune"):
                exercised[key] += 1
                by_kind[key][meta["kind"]] += 1
    h_index = {"situations": [{"sid": k, **v} for k, v in sits.items()],
               "note": "Every card is run in every H situation; behaviours in out/H.json."}
    return h_index, records, exercised, by_kind


def build_s():
    idx = [json.loads(l) for l in open(os.path.join(RAW, "index.jsonl"))]
    rec = [json.loads(l) for l in open(os.path.join(RAW, "records.jsonl"))]
    assert len(idx) == len(rec)
    games = collections.OrderedDict()
    split = {"train": [], "test": []}
    for i, r in zip(idx, rec):
        assert i["sid"] == r["sid"]
        gk = (i["deck_a"], i["deck_b"], i["agent_a"], i["agent_b"], i["game_seed"])
        if gk not in games:
            games[gk] = len(games)
        sp = "train" if is_train(i["game_seed"]) else "test"
        beh = {k: r[k] for k in ("legal", "exec") if k in r}
        split[sp].append((gk, i, r["card"], r["kind"], beh, r.get("err", 0)))
    return games, split


def main():
    os.makedirs(OUT, exist_ok=True)
    h_index, h_records, h_ex, h_kind = build_h()
    json.dump(h_index, open(os.path.join(HERE, "H.json"), "w"), indent=0)
    cards_h = compact([(k, b) for k, _, b in h_records])
    # H situation ids per card are the H.json order (one record per card per situation)
    json.dump({"pool": "H", "situation_file": "H.json", "cards": cards_h},
              open(os.path.join(OUT, "H.json"), "w"), separators=(",", ":"))

    games, split = build_s()
    fields = ["game", "decision", "turn", "prompt", "decider", "card", "class_id", "kind", "sub",
              "self_obj", "host_obj", "host_card", "fp"]
    s_ex = {}
    s_kind = {}
    s_err = {}
    for sp, rows in split.items():
        gl = sorted({g for g, *_ in rows}, key=lambda g: games[g])
        gid = {g: n for n, g in enumerate(gl)}
        sits = []
        for g, i, card, kind, beh, err in rows:
            sits.append([gid[g], i["decision"], i["turn"], i["prompt"], i["decider"], card, i["class_id"],
                         kind, i["sub"], i["self_obj"], i["host_obj"], i["host_card"], i["fp"]])
        doc = {"pool": "S_" + sp,
               "split_rule": "game seed % 10 < 7 -> train, else test",
               "games": [{"game": gid[g], "deck_a": g[0], "deck_b": g[1], "agent_a": g[2], "agent_b": g[3],
                          "game_seed": g[4]} for g in gl],
               "fields": fields, "situations": sits}
        with open(os.path.join(HERE, f"S_{sp}.json"), "w") as f:
            f.write(json.dumps({k: v for k, v in doc.items() if k != "situations"})[:-1] + ',"situations":[\n')
            f.write(",\n".join(json.dumps(s, separators=(",", ":")) for s in sits) + "\n]}\n")
        cards = compact([(card, dict(beh, kind=kind, sub=i["sub"])) for g, i, card, kind, beh, err in rows])
        json.dump({"pool": "S_" + sp, "situation_file": f"S_{sp}.json", "cards": cards},
                  open(os.path.join(OUT, f"S_{sp}.json"), "w"), separators=(",", ":"))
        kc = collections.defaultdict(collections.Counter)
        for g, i, card, kind, beh, err in rows:
            kc[card][kind + ("_sub" if i["sub"] else "")] += 1
        s_kind[sp] = kc
        s_err[sp] = sum(r[-1] for r in rows)
        s_ex[sp] = rows

    # coverage
    allcards = list(cards_h.keys())
    cov = {}
    for c in allcards:
        e = {"H_exercised": h_ex.get(c, 0), "H_by_kind": dict(h_kind.get(c, {}))}
        for sp in ("train", "test"):
            k = s_kind[sp].get(c, collections.Counter())
            e[f"S_{sp}_by_kind"] = dict(k)
            e[f"S_{sp}_exercised"] = sum(v for kk, v in k.items() if kk.split("_")[0] in EXERCISED_S)
        e["flag_lt10"] = [sp for sp in ("train", "test") if e[f"S_{sp}_exercised"] < 10]
        cov[c] = e
    json.dump({"exercised_kinds_S": sorted(EXERCISED_S), "probe_errors": s_err, "cards": cov},
              open(os.path.join(OUT, "coverage.json"), "w"), indent=1)

    # A/B: effect fields per pool
    eff = {c: {"H": set(), "S_train": set(), "S_test": set()} for c in allcards}
    hraw = {}
    for k, sid, b in h_records:
        if sid.startswith(("play:", "trigger:")):
            eff[k]["H"] |= effects_of(b)
    for sp, rows in s_ex.items():
        for g, i, card, kind, beh, err in rows:
            if card in eff:
                eff[card]["S_" + sp] |= effects_of(beh)
    ab = {}
    for c, e in eff.items():
        s_all = e["S_train"] | e["S_test"]
        ab[c] = {"only_H": sorted(e["H"] - s_all), "only_S": sorted(s_all - e["H"]),
                 "only_S_train": sorted(e["S_train"] - e["S_test"] - e["H"]),
                 "only_S_test": sorted(e["S_test"] - e["S_train"] - e["H"]),
                 "n_H": len(e["H"]), "n_S": len(s_all), "n_both": len(e["H"] & s_all)}
    json.dump(ab, open(os.path.join(OUT, "ab_effects.json"), "w"), indent=1)

    # print a table
    print(f"{'card':38s} {'H':>4s} {'Str':>6s} {'Ste':>6s}  S-train kinds")
    for c in allcards:
        e = cov[c]
        print(f"{c:38s} {e['H_exercised']:4d} {e['S_train_exercised']:6d} {e['S_test_exercised']:6d}  "
              f"{dict(sorted(e['S_train_by_kind'].items()))} {'FLAG' if e['flag_lt10'] else ''}")
    print("probe errors", s_err)
    print("situations", {sp: len(r) for sp, r in s_ex.items()}, "games", len(games))


if __name__ == "__main__":
    main()
