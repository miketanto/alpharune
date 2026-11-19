"""Build eval/labels.json: provisional mechanics labels for the 51 phase-1 cards.

Sources (objective only):
  * "card_index"  : fields of the first printing in cards.json (which mirrors
                    cards/card_index.json): type, supertype, domains, energy
                    cost, power cost, might, bracketed [Keyword] tokens.
  * "card_def"    : the CardDef block inside the engine class (used for the two
                    legends that are not in card_index.json).
  * "engine_cpp"  : static scan of the card's C++ class (src/cards/<type>/...):
                    declared TriggerType(s), overridden hooks, executor/helper
                    calls that perform an effect.
  * "text_derived": regex over ability_text; always marked provisional.
Every label carries {"source", "provisional"}. A hand-checked gold sheet is
meant to replace the provisional ones later.

Usage: python build_labels.py   (writes labels.json next to this file)
"""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CARDS = json.load(open(os.path.join(HERE, "..", "cards.json"), encoding="utf8"))

DOMAINS = ["fury", "calm", "mind", "body", "chaos", "order"]
# bracketed tokens that are resource symbols / glyphs, not keywords
NOT_KW = {"M", "A", "R", "B", "O", "C", "E", "N", ">", "0", "1", "2", "3", "4",
          "5", "6", "7", "8", "9"}

LEGEND_FILES = {
    "rengar_pridestalker": "src/cards/legends/0744_pridestalker.cpp",
    "master_yi_wuju_bladesman": "src/cards/legends/0317_wuju_bladesman_starter.cpp",
}

# effect category -> regexes over the class body (comments stripped)
EFFECTS = {
    "draw": [r"\bdrawCards\("],
    "damage": [r"\bdealDamage\(", r"\bsplitDamageAmong\("],
    "might_change": [r"\bgiveTemporaryMight\(", r"\bauraMight\(", r"might_bonus\s*="],
    "buff": [r"\bbuffUnit\("],
    "token": [r"\bcreateToken\("],
    "counter": [r"\bcounterChosenItem\(", r"\bcounterSpell\w*\("],
    "recycle": [r"\brecycleCards\("],
    "kill": [r"\bkillObject\(", r"\bkillLethallyDamagedTogether\(", r"\bkill\("],
    "stun": [r"\bstunUnit\w*\("],
    "move": [r"\bmoveTo\w*\(", r"\bpickMoveDest\("],
    "recall": [r"\bhealExhaustRecall\(", r"\brecallToBase\("],
    "ready": [r"\breadyObject\("],
    "bounce": [r"\bbounceToHand\("],
    "banish": [r"\bbanishObject\(", r"\bbanish\("],
    "xp": [r"\bgainXp\("],
    "play_free": [r"\bplayIgnoringCost\("],
}
HOOKS = {
    "has_activated_ability": [r"\bonActivate\(", r"\bhasActivatedAbility\(", r"TriggerType::Activated"],
    "has_passive_aura": [r"\bapplyPassiveAura\("],
    "has_replacement": [r"\bhasReplacementEffect\("],
    "has_optional_additional_cost": [r"\boptionalAdditionalCost\(", r"\bflowCost\("],
    "has_cost_reduction": [r"\bselfCostReduction\("],
    "needs_play_target": [r"\bneedsPlayTimeTarget\w*\("],
}


def find_cpp(key, printing_ids):
    if key in LEGEND_FILES:
        return os.path.join(ROOT, LEGEND_FILES[key])
    for f in glob.glob(os.path.join(ROOT, "src", "cards", "*", "*.cpp")):
        s = open(f, encoding="utf8").read()
        for pid in printing_ids:
            if "R\"RB(%s)RB\"" % pid in s:
                return f
    return None


def class_body(src):
    body = src.split("const CardDef def_")[0]
    body = re.sub(r"//.*", "", body)
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    return body


def card_def_static(src):
    """Parse static fields from a CardDef block (for the legends)."""
    out = {}
    m = re.search(r"d\.card_type\s*=\s*CardType::(\w+)", src)
    out["card_type"] = m.group(1).lower() if m else None
    m = re.search(r"d\.domains\s*=\s*\{([^}]*)\}", src)
    out["domains"] = [x.split("::")[1].lower() for x in re.findall(r"Domain::\w+", m.group(1))] if m else []
    for f in ("energy_cost", "power_cost", "might"):
        m = re.search(r"d\.%s\s*=\s*(-?\d+)" % f, src)
        out[f] = int(m.group(1)) if m else None
    m = re.search(r'd\.ability_text\s*=\s*R"RB\((.*?)\)RB"', src, flags=re.S)
    out["ability_text"] = m.group(1) if m else ""
    out["super_type"] = None
    return out


def lab(value, source, provisional):
    return {"value": value, "source": source, "provisional": provisional}


def text_target_side(text):
    t = (text or "").lower()
    t = re.sub(r"\([^)]*\)", "", t)  # drop reminder text
    fr = bool(re.search(r"friendly|you control|another unit|your other units|\bme\b|\bmy might\b", t))
    en = bool(re.search(r"enemy|opponent", t))
    anyu = bool(re.search(r"\ba unit\b|\ba gear\b|\ba spell\b", t))
    if fr and en:
        return "both"
    if en:
        return "enemy"
    if fr:
        return "friendly"
    if anyu:
        return "any"
    return "none"


def main():
    labels = {}
    notes = {}
    for c in CARDS["cards"]:
        key = c["key"]
        pids = [p["id"] for p in c["printings"]]
        cpp = find_cpp(key, pids)
        src = open(cpp, encoding="utf8").read() if cpp else None
        if c["printings"]:
            p = c["printings"][0]
            st_src = "card_index"
            st = {k: p.get(k) for k in ("card_type", "super_type", "domains", "energy_cost",
                                        "power_cost", "might", "ability_text")}
        else:
            st_src = "card_def"
            st = card_def_static(src)
        text = st.get("ability_text") or ""
        L = {}
        ctype = (st["card_type"] or "").lower()
        L["card_type"] = lab(ctype, st_src, False)
        L["super_type"] = lab(st.get("super_type") or "none", st_src, False)
        doms = [d.lower() for d in (st.get("domains") or [])]
        for d in DOMAINS:
            L["domain_" + d] = lab(int(d in doms), st_src, False)
        L["n_domains"] = lab(len(doms), st_src, False)
        playable = ctype in ("unit", "spell", "gear")
        # power/energy null on a playable card means 0; on rune/legend/battlefield N/A
        L["energy_cost"] = lab((st.get("energy_cost") or 0) if playable else None, st_src, False)
        L["power_cost"] = lab((st.get("power_cost") or 0) if playable else None, st_src, False)
        L["might"] = lab(st.get("might") if ctype == "unit" else None, st_src, False)
        # keywords: bracketed tokens outside reminder text "( ... )"
        toks = re.findall(r"\[([A-Za-z][A-Za-z ]*?)(?:\s+\d+)?\]", re.sub(r"\([^)]*\)", "", text))
        kws = sorted({t.strip() for t in toks if t.strip() not in NOT_KW})
        L["keywords"] = lab(kws, st_src + ":bracket_tokens", False)
        for k in ["Action", "Reaction", "Hidden", "Ambush", "Accelerate", "Deathknell",
                  "Flow", "Ganking", "Stun", "Legion", "Hunt", "Level", "Assault", "Deflect"]:
            L["kw_" + k.lower()] = lab(int(k in kws), st_src + ":bracket_tokens", False)
        # timing: when the card itself can be played (Ambush units are playable as a Reaction)
        if not playable:
            timing = "not_played"
        elif "Reaction" in kws or "Ambush" in kws:
            timing = "reaction"
        elif "Action" in kws:
            timing = "action"
        else:
            timing = "main_only"
        L["timing"] = lab(timing, st_src + ":bracket_tokens", False)

        if src is not None:
            body = class_body(src)
            esrc = "engine_cpp:" + os.path.relpath(cpp, ROOT).replace("\\", "/")
            trig = sorted(set(re.findall(r"TriggerType::(\w+)", body)) - {"None"})
            L["trigger_types"] = lab(trig, esrc, False)
            L["has_trigger"] = lab(int(bool(trig) or "onTrigger(" in body), esrc, False)
            for h, pats in HOOKS.items():
                L[h] = lab(int(any(re.search(pt, body) for pt in pats)), esrc, False)
            effs = []
            for e, pats in EFFECTS.items():
                v = int(any(re.search(pt, body) for pt in pats))
                L["eff_" + e] = lab(v, esrc, True)  # call present != always performed
                if v:
                    effs.append(e)
            L["n_effects"] = lab(len(effs), esrc, True)
        else:
            # no engine class (Hush): fall back to text, provisional
            notes[key] = "no engine class found for printings %s; engine labels text-derived" % pids
            tsrc = "text_derived(no engine class)"
            t = text.lower()
            L["trigger_types"] = lab([], tsrc, True)
            L["has_trigger"] = lab(int(bool(re.search(r"\bwhen\b|deathknell", t))), tsrc, True)
            for h in HOOKS:
                L[h] = lab(0, tsrc, True)
            effs = []
            tests = {"draw": r"\bdraw\b", "damage": r"damage", "might_change": r"\[m\]",
                     "buff": r"\bbuff\b", "token": r"token", "counter": r"\bcounter\b",
                     "recycle": r"recycle", "kill": r"\bkill\b", "stun": r"stun",
                     "move": r"\bmove\b", "recall": r"\brecall\b", "ready": r"\bready\b", "bounce": r"return .* hand",
                     "banish": r"banish", "xp": r"\bxp\b", "play_free": r"ignoring its cost"}
            for e, pt in tests.items():
                v = int(bool(re.search(pt, re.sub(r"\([^)]*\)", "", t))))
                L["eff_" + e] = lab(v, tsrc, True)
                if v:
                    effs.append(e)
            L["n_effects"] = lab(len(effs), tsrc, True)
        trig = L["trigger_types"]["value"]
        for tt in ["WhenYouPlayMe", "WhenIDie", "WhenIConquer", "WhenYouConquerHere"]:
            L["trig_" + tt] = lab(int(tt in trig), L["trigger_types"]["source"], L["trigger_types"]["provisional"])
        L["target_side"] = lab(text_target_side(text), "text_derived", True)
        L["has_additional_cost"] = lab(
            int(bool(re.search(r"additional cost|\[accelerate\]|\[flow\]|\[hidden\]", text.lower()))),
            "text_derived", True)
        labels[key] = L

    # label catalogue: kind of each label for the probes
    numeric = ["energy_cost", "power_cost", "might", "n_domains", "n_effects"]
    categorical = ["card_type", "super_type", "timing", "target_side"]
    list_valued = ["keywords", "trigger_types"]
    binary = sorted({k for L in labels.values() for k in L} - set(numeric) - set(categorical) - set(list_valued))
    out = {
        "_doc": __doc__,
        "label_kinds": {"numeric": numeric, "categorical": categorical, "binary": binary,
                        "list_valued_not_probed": list_valued},
        "notes": notes,
        "labels": labels,
    }
    with open(os.path.join(HERE, "labels.json"), "w", encoding="utf8") as f:
        json.dump(out, f, indent=1)
    print("wrote labels.json for", len(labels), "cards;", notes)


if __name__ == "__main__":
    main()
