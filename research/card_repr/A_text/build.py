"""Path A: text-embedding card representation.

Usage:  python build.py            (writes out/vectors.json, out/vectors_<model>.json,
                                    out/inputs_<mode>.json, out/diagnostics.json)

Pipeline per card (or printing / rename / synthetic variant):
  raw printing dict -> canonical text (name -> <SELF>, reminder text stripped and
  replaced by a fixed per-keyword glossary, icons spelled out)  +  stats line
  (type, supertype, domains, energy cost, power cost, might, timing, keywords,
  additional costs) -> two segments embedded separately by a local model, each
  L2-normalised, concatenated and scaled by 1/sqrt(2) (unit norm overall).
"""
import os, re, json, time, copy, hashlib
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # research/card_repr
OUT = os.path.join(HERE, "out")
HUB = os.path.expanduser("~/.cache/huggingface/hub")
MODELS = {
    # name: (hf id, pooling)
    "bge-small-en-v1.5": ("BAAI/bge-small-en-v1.5", "cls"),          # model card: CLS + normalise
    "paraphrase-mpnet-base-v2": ("sentence-transformers/paraphrase-mpnet-base-v2", "mean"),
}
PRIMARY = "bge-small-en-v1.5"
MODE = "strip"       # chosen reminder-text mode (see report.md); "keep" is built for comparison
LAYOUT = "segmented"  # chosen layout (see report.md); "single" is built for comparison

# ---------------------------------------------------------------- card data
# Legends missing from card_index.json: transcribed from the engine CardDef
# (src/cards/legends/0744_pridestalker.cpp, 0317_wuju_bladesman_starter.cpp).
LEGENDS = {
    "rengar_pridestalker": {"id": "unl-183-219", "card_type": "legend", "super_type": None,
        "domains": ["fury", "body"], "tags": ["Rengar"], "energy_cost": None, "power_cost": None,
        "might": None, "might_bonus": None,
        "ability_text": "When you play a unit, give a unit +1 [M] this turn.", "effect_text": None},
    "master_yi_wuju_bladesman": {"id": "ogs-019-024", "card_type": "legend", "super_type": None,
        "domains": ["calm", "body"], "tags": ["Master Yi"], "energy_cost": None, "power_cost": None,
        "might": None, "might_bonus": None,
        "ability_text": "While a friendly unit defends alone, it gets +2 [M].", "effect_text": None},
}

DOMAIN_ICON = {"R": "fury", "G": "calm", "O": "body", "B": "mind", "Y": "order", "P": "chaos"}

# Fixed glossary, one entry per keyword / rules term, written once from the printed
# reminder texts.  "{dom}" is filled with the card's domain(s).  In "strip" mode every
# printed parenthetical is removed and these are appended instead, so the text no
# longer depends on which printing (or how much reminder text) the card carries.
KEYWORD_GLOSSARY = {
    "action": "Action: play on your turn or in showdowns.",
    "reaction": "Reaction: play any time, even before spells and abilities resolve.",
    "hidden": "Hidden: hide now for [A] to react with later for [0].",
    "ambush": "Ambush: you may play me as a [Reaction] to a battlefield where you have units.",
    "deathknell": "Deathknell: when I die, get the effect.",
    "flow": "Flow: you may play this from your trash for its Flow cost. Then banish it.",
    "accelerate": "Accelerate: you may pay [1] and 1 {dom} power as an additional cost to have me enter ready.",
    "assault": "Assault N: +N [M] while I'm an attacker.",
    "hunt": "Hunt N: when I conquer or hold, gain N XP.",
    "level": "Level N: while you have N or more XP, get the effect.",
    "deflect": "Deflect: opponents must pay [A] to choose me with a spell or ability.",
    "ganking": "Ganking: I can move from battlefield to battlefield.",
    "legion": "Legion: get the effect if you've played another card this turn.",
    "stun": "Stun: it doesn't deal combat damage this turn.",
}
TERM_GLOSSARY = [  # (regex on canonical text, definition)
    (r"\bbuff\b", "Buff: if it doesn't have a buff, it gets a +1 [M] buff."),
    (r"\brecall\b", "Recall: send it to base. This isn't a move."),
    (r"\balone\b", "Alone: there are no other friendly units here."),
    (r"\bwin a combat\b", "Win a combat: I win if I remain after combat."),
]
ICON_RUN = r"(?:\[(?:\d+|[ARGOBYP])\])+"


def load_cards():
    d = json.load(open(os.path.join(ROOT, "cards.json"), encoding="utf-8"))
    for c in d["cards"]:
        if not c["printings"] and c["key"] in LEGENDS:
            c["printings"] = [copy.deepcopy(LEGENDS[c["key"]])]
            c["printings_source"] = "engine CardDef"
    # Current (erratum) text: cards/card_index.json holds pre-erratum gallery wording
    # for some cards (e.g. Rengar, Trophy Hunter; Zhonya's Hourglass). Use the text in
    # the engine's CardDef: a printing's own class if it has one, else the card's class.
    eng_path = os.path.join(ROOT, "engine_text.json")
    if os.path.exists(eng_path):
        eng = json.load(open(eng_path, encoding="utf-8"))
        for c in d["cards"]:
            classes = eng.get(c["key"])
            if not classes:
                continue  # no engine class (Hush): keep card_index text
            default = next(iter(classes.values()))
            for p in c["printings"]:
                e = classes.get(p.get("id"), default)
                if e.get("ability_text"):
                    p["ability_text"] = e["ability_text"]
                    p["effect_text"] = e.get("effect_text")
                    p["text_source"] = "engine CardDef " + e["file"]
    return d


# ---------------------------------------------------------------- text canonicalisation
def substitute_self(text, name):
    """Replace every self-reference (full name, and the part before the comma) by <SELF>."""
    forms = [name]
    if "," in name:
        forms.append(name.split(",")[0].strip())
    for f in sorted(set(forms), key=len, reverse=True):
        text = re.sub(r"(?<![\w<])" + re.escape(f) + r"(?![\w>])", "<SELF>", text, flags=re.I)
    return text


def icons_to_words(text):
    def run(m):
        icons = re.findall(r"\[([^\]]+)\]", m.group(0))
        parts, energy, power = [], 0, {}
        for i in icons:
            if i.isdigit():
                energy += int(i)
            elif i == "A":
                power["any-domain"] = power.get("any-domain", 0) + 1
            elif i in DOMAIN_ICON:
                power[DOMAIN_ICON[i]] = power.get(DOMAIN_ICON[i], 0) + 1
        if any(i.isdigit() for i in icons):
            parts.append(f"{energy} energy")
        for k, v in power.items():
            parts.append(f"{v} {k} power")
        return " and ".join(parts)
    text = re.sub(ICON_RUN, run, text)
    return text.replace("[M]", "Might")


KW_RE = r"\[([A-Za-z][A-Za-z]+(?: \d+)?)\]"   # bracketed keyword (icons are 1 char)


def keywords_of(text):
    return [k.strip() for k in re.findall(KW_RE, text or "")]


def canonical_text(p, name, mode):
    raw = "\n".join(t for t in [p.get("ability_text"), p.get("effect_text")] if t)
    t = substitute_self(raw, name)
    kws = keywords_of(re.sub(r"\([^()]*\)", "", t))   # printed keywords only, not ones named in reminder text
    t = t.replace("—", "[>]")                 # printed em dash == [>] icon separator
    t = re.sub(r"\s*\[>\]\s*", ": ", t)
    t = re.sub(r"\(\s*\[[A-Z]\]\s*\)", "", t)      # icon echo e.g. "Fury ([R])"
    if mode == "strip":
        t = re.sub(r"\s*\([^()]*\)", "", t)        # reminder text
    t = re.sub(KW_RE, r"\1", t)
    t = icons_to_words(t)
    t = re.sub(r"\.{2,}", ".", t)
    t = re.sub(r"[ \t]+([.,:])", r"\1", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r":\s*:", ":", t)
    lines = [l.strip() for l in t.split("\n") if l.strip()]
    rules = " ".join(l if l.endswith((".", ":")) else l + "." for l in lines) or "No abilities."
    gloss = []
    if mode == "strip":
        dom = " or ".join(p["domains"]) or "any-domain"
        seen = set()
        for k in kws:
            base = k.split()[0].lower()
            if base in KEYWORD_GLOSSARY and base not in seen:
                seen.add(base)
                g = KEYWORD_GLOSSARY[base].replace("{dom}", dom)
                n = re.findall(r"\d+", k)
                if n:
                    g = g.replace("N", n[0])
                gloss.append(icons_to_words(re.sub(KW_RE, r"", g)))
        for rx, g in TERM_GLOSSARY:
            if re.search(rx, rules, flags=re.I):
                gloss.append(icons_to_words(g))
    if gloss:
        rules += " Keyword meanings: " + " ".join(gloss)
    return rules, kws


def additional_costs(p, kws, raw):
    raw = re.sub(r"\([^()]*\)", "", raw)
    out = []
    dom = " or ".join(p["domains"]) or "any-domain"
    low = [k.lower().split()[0] for k in kws]
    if "accelerate" in low:
        out.append(f"optional 1 energy and 1 {dom} power to enter ready (Accelerate)")
    if "hidden" in low:
        out.append("hide for 1 any-domain power, later play for 0 energy (Hidden)")
    m = re.search(r"\[Flow\]\s*(" + ICON_RUN + ")", raw)
    if m:
        out.append("alternative cost from trash " + icons_to_words(m.group(1)) + ", then banish (Flow)")
    for m in re.finditer(r"pay (" + ICON_RUN + ") as an additional cost", raw):
        out.append("optional additional cost " + icons_to_words(m.group(1)))
    m = re.search(r"cost (" + ICON_RUN + ") less", raw)
    if m:
        out.append("conditional cost reduction " + icons_to_words(m.group(1)))
    return out


def timing_of(p, kws):
    low = [k.lower().split()[0] for k in kws]
    if p["card_type"] in ("legend", "battlefield", "rune"):
        return "not played from hand"
    if "reaction" in low:
        t = "Reaction (any time, even before spells and abilities resolve)"
    elif "action" in low:
        t = "Action (your turn or showdowns)"
    else:
        t = "main phase only (your turn, outside showdowns)"
    if "ambush" in low:
        t += "; or as a Reaction to a battlefield where you have units (Ambush)"
    if "hidden" in low:
        t += "; or hidden and played later as a Reaction (Hidden)"
    return t


def stats_line(p, kws, raw):
    num = lambda v: "none" if v is None else str(v)
    pc = p.get("power_cost")
    power = "none" if pc is None else f"{pc} ({' or '.join(p['domains']) or 'any-domain'})"
    parts = [
        f"Type: {p['card_type']}",
        f"Supertype: {p.get('super_type') or 'none'}",
        f"Domains: {', '.join(p['domains']) or 'none'}",
        f"Energy cost: {num(p.get('energy_cost'))}",
        f"Power cost: {power}",
        f"Might: {num(p.get('might'))}",
    ]
    if p.get("might_bonus") is not None:
        parts.append(f"Might bonus: {p['might_bonus']}")
    parts.append(f"Timing: {timing_of(p, kws)}")
    kw = sorted(set(kws), key=str.lower)
    parts.append("Keywords: " + (", ".join(kw) if kw else "none"))
    ac = additional_costs(p, kws, raw)
    parts.append("Additional costs: " + ("; ".join(ac) if ac else "none"))
    return ". ".join(parts) + "."


def card_input(p, name, mode):
    raw = "\n".join(t for t in [p.get("ability_text"), p.get("effect_text")] if t)
    rules, kws = canonical_text(p, name, mode)
    return {"stats": stats_line(p, kws, raw), "rules": rules}


# ---------------------------------------------------------------- variants
def fake_name(key, real):
    syl = ["zor", "ba", "vel", "kri", "mon", "tha", "lu", "ques", "dar", "fen", "ix", "ol"]
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    w = lambda n, s: "".join(syl[(s >> (4 * i)) % len(syl)] for i in range(n)).capitalize()
    nm = w(3, h)
    if "," in real:
        nm += ", " + w(2, h >> 20)
    return nm


def renamed(c):
    """Copy of the card under a made-up name: printed text has the real name swapped for
    the fake one BEFORE the <SELF> substitution runs (which then uses the fake name)."""
    p = copy.deepcopy(c["printings"][0])
    nm = fake_name(c["key"], c["name"])
    pairs = [(c["name"], nm)]
    if "," in c["name"]:
        pairs.append((c["name"].split(",")[0].strip(), nm.split(",")[0]))
    for k in ("ability_text", "effect_text"):
        if p.get(k):
            for old, new in pairs:
                p[k] = re.sub(r"(?<!\w)" + re.escape(old) + r"(?!\w)", new, p[k])
    return p, nm


def text_edit(p, old, new):
    assert old in p["ability_text"], (old, p["ability_text"])
    p["ability_text"] = p["ability_text"].replace(old, new, 1)


SYNTH = {  # id -> edit on a copy of the base card's first printing (cards.json descriptions)
    "punch_first__amount4": lambda p: text_edit(p, "+5 [M]", "+4 [M]"),
    "punch_first__energy3": lambda p: p.__setitem__("energy_cost", 3),
    "punch_first__power0": lambda p: p.__setitem__("power_cost", 0),
    "discipline__draw2": lambda p: text_edit(p, "Draw 1.", "Draw 2."),
    "discipline__enemy": lambda p: text_edit(p, "Give a unit +2", "Give an enemy unit +2"),
    "lonely_poro__draw2": lambda p: text_edit(p, "draw 1.", "draw 2."),
    "defy__cost2": lambda p: text_edit(p, "no more than [4]", "no more than [2]"),
    "sabotage__unit": lambda p: text_edit(p, "non-unit card", "unit card"),
    "pit_rookie__might3": lambda p: p.__setitem__("might", 3),
    "punch_first__reaction": lambda p: text_edit(p, "[Action] (Play on your turn or in showdowns.)",
                                                 "[Reaction] (Play any time, even before spells and abilities resolve.)"),
}


def build_inputs(d, mode):
    byk = {c["key"]: c for c in d["cards"]}
    items = {}
    for c in d["cards"]:
        per = []
        for p in c["printings"]:
            x = card_input(p, c["name"], mode)
            items["printing:" + p["id"]] = x
            per.append(x)
        items[c["key"]] = per[0]   # identical for every printing in strip mode (asserted in main)
        rp, nm = renamed(c)
        items["rename:" + c["key"]] = dict(card_input(rp, nm, mode), fake_name=nm)
    for s in d["t2_minimal_pairs_synthetic"]:
        p = copy.deepcopy(byk[s["base"]]["printings"][0])
        SYNTH[s["id"]](p)
        items["synthetic:" + s["id"]] = card_input(p, byk[s["base"]]["name"], mode)
    return items


def self_test():
    """Substitution must catch full name and short name (no card in scope prints its name)."""
    t = substitute_self("When Zed, Shadow attacks, Zed gets +1. Zedd is untouched.", "Zed, Shadow")
    assert t == "When <SELF> attacks, <SELF> gets +1. Zedd is untouched.", t
    c = {"key": "zed_shadow", "name": "Zed, Shadow", "printings": [{"ability_text": "When Zed attacks, draw 1.",
         "effect_text": None, "domains": ["calm"], "card_type": "unit"}]}
    rp, nm = renamed(c)
    assert nm.split(",")[0] in rp["ability_text"] and canonical_text(rp, nm, "strip") == canonical_text(c["printings"][0], c["name"], "strip")
    p = {"ability_text": "[Deathknell] — Draw 1. (When I die, get the effect.)", "effect_text": None,
         "domains": ["calm"], "card_type": "unit"}
    q = dict(p, ability_text="[Deathknell][>] Draw 1.")
    assert canonical_text(p, "X", "strip") == canonical_text(q, "X", "strip")


# ---------------------------------------------------------------- embedding
def snapshot(hf_id):
    base = os.path.join(HUB, "models--" + hf_id.replace("/", "--"), "snapshots")
    return os.path.join(base, sorted(os.listdir(base))[0])


def read_safetensors(fn):
    # torch 2.0 here is built against numpy 1.x, so torch<->numpy bridges fail: use torch.frombuffer.
    dt = {"F32": torch.float32, "F16": torch.float16, "I64": torch.int64, "I32": torch.int32}
    with open(fn, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        hdr = json.loads(f.read(n))
        buf = f.read()
    out = {}
    for k, m in hdr.items():
        if k == "__metadata__":
            continue
        a, b = m["data_offsets"]
        t = torch.frombuffer(bytearray(buf[a:b]), dtype=dt[m["dtype"]]).reshape(m["shape"]).clone()
        out[k] = t.float() if m["dtype"] == "F16" else t
    return out


class Embedder:
    def __init__(self, name):
        hf_id, self.pool = MODELS[name]
        path = snapshot(hf_id)
        self.tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        # transformers 4.23 without the `safetensors` package cannot read model.safetensors,
        # so the file is parsed here (8-byte header length, JSON header, raw little-endian buffers).
        self.model = AutoModel.from_config(AutoConfig.from_pretrained(path, local_files_only=True))
        sd = read_safetensors(os.path.join(path, "model.safetensors"))
        prefix = self.model.base_model_prefix + "."
        sd = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        missing = [m for m in missing if not m.endswith("position_ids")]
        assert not missing, missing
        self.model.eval()

    @torch.no_grad()
    def __call__(self, texts, bs=32):
        out = []
        for i in range(0, len(texts), bs):
            b = self.tok(texts[i:i + bs], padding=True, truncation=True, max_length=512, return_tensors="pt")
            h = self.model(**b).last_hidden_state
            if self.pool == "cls":
                v = h[:, 0]
            else:
                m = b["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            out.extend(torch.nn.functional.normalize(v, dim=-1).tolist())
        return np.array(out, dtype=np.float64)


def embed_items(emb, items, layout):
    keys = list(items)
    if layout == "segmented":
        uniq = sorted({items[k]["stats"] for k in keys} | {items[k]["rules"] for k in keys})
        V = dict(zip(uniq, emb(uniq)))
        return {k: np.concatenate([V[items[k]["stats"]], V[items[k]["rules"]]]) / np.sqrt(2) for k in keys}
    s = {k: items[k]["stats"] + "\n" + items[k]["rules"] for k in keys}
    uniq = sorted(set(s.values()))
    V = dict(zip(uniq, emb(uniq)))
    return {k: V[s[k]] for k in keys}


def cos_d(a, b):
    return float(1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def diagnostics(d, vec):
    """Own sanity numbers (not the eval scorer): printing spread, nearest-card distance, synthetic shifts."""
    keys = [c["key"] for c in d["cards"]]
    M = np.stack([vec[k] for k in keys]); M = M / np.linalg.norm(M, axis=1, keepdims=True)
    D = 1 - M @ M.T; np.fill_diagonal(D, np.inf)
    out = {"nn_dist_median": float(np.median(D.min(1))), "nn_dist_min": float(D.min())}
    pd = {}
    for c in d["cards"]:
        ids = [p["id"] for p in c["printings"]]
        if len(ids) > 1:
            pd[c["key"]] = max(cos_d(vec["printing:" + a], vec["printing:" + b]) for a in ids for b in ids)
    out["printing_max_dist"] = pd
    out["rename_max_dist"] = max(cos_d(vec[k], vec["rename:" + k]) for k in keys)
    out["synthetic_dist_to_base"] = {s["id"]: cos_d(vec["synthetic:" + s["id"]], vec[s["base"]])
                                     for s in d["t2_minimal_pairs_synthetic"]}
    out["real_pair_dist"] = {f"{a}|{b}": cos_d(vec[a], vec[b]) for a, b, _ in d["t2_minimal_pairs_real"]}
    return out


def main():
    t0 = time.time()
    self_test()
    d = load_cards()
    inputs = {m: build_inputs(d, m) for m in ("strip", "keep")}
    for c in d["cards"]:   # printing / name invariance by construction in the chosen mode
        xs = {json.dumps(inputs[MODE]["printing:" + p["id"]], sort_keys=True) for p in c["printings"]}
        assert len(xs) == 1, (c["key"], xs)
        r = inputs[MODE]["rename:" + c["key"]]
        assert (r["rules"], r["stats"]) == (inputs[MODE][c["key"]]["rules"], inputs[MODE][c["key"]]["stats"]), c["key"]
    os.makedirs(OUT, exist_ok=True)
    for m, it in inputs.items():
        json.dump(it, open(os.path.join(OUT, f"inputs_{m}.json"), "w", encoding="utf-8"), indent=1)
    timing, diag = {}, {}
    for name in MODELS:
        t1 = time.time()
        emb = Embedder(name)
        timing[name + ":load_s"] = round(time.time() - t1, 2)
        for mode in ("strip", "keep"):
            for layout in ("segmented", "single"):
                t1 = time.time()
                vec = embed_items(emb, inputs[mode], layout)
                timing[f"{name}:{mode}:{layout}:embed_s"] = round(time.time() - t1, 2)
                diag[f"{name}:{mode}:{layout}"] = diagnostics(d, vec)
                if mode == MODE and layout == LAYOUT:
                    out = {"dim": int(len(next(iter(vec.values())))), "model": name, "mode": mode, "layout": layout,
                           "vectors": {k: [round(float(x), 6) for x in v] for k, v in vec.items()}}
                    fn = "vectors.json" if name == PRIMARY else f"vectors_{name}.json"
                    json.dump(out, open(os.path.join(OUT, fn), "w"))
    timing["total_s"] = round(time.time() - t0, 2)
    json.dump({"timing": timing, "diagnostics": diag}, open(os.path.join(OUT, "diagnostics.json"), "w"), indent=1)
    print(json.dumps(timing, indent=1))


if __name__ == "__main__":
    main()
