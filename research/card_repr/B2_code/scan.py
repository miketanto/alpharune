"""B2: automatic scan of the card classes (no hand input).

For every engine class used by the 51 cards it reads
  - the CardDef block (stats, keywords, assault value) with line numbers,
  - every call made by the class body: executor calls (ctx.executor.X) and
    free calls (card_helpers.h / Card methods), with line numbers,
  - TriggerType::X tokens, TargetRequirements fields, Card hooks overridden.
The extractor (extract.py) uses the CardDef parse for stats/keywords and the
call list for the vocabulary-coverage check (every call must be mapped in
vocab.json either to a schema verb or to an explicit plumbing class).
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))

# card key -> [(def_id, class file)]; first entry = the class the card key uses.
# Found by grepping the def_id literal in src/cards (R"RB(<def_id>)RB").
CLASSES = {
    "alpha_strike": [("unl-192-219", "src/cards/spells/0751_alpha_strike.cpp")],
    "back_off": [("unl-042-219", "src/cards/spells/0604_back_off.cpp")],
    "body_rune": [("ogn-126-298", "src/cards/runes/0126_body_rune.cpp")],
    "brittle_steel": [("ven-003-166", "src/cards/spells/0790_brittle_steel.cpp")],
    "calm_rune": [("ogn-042-298", "src/cards/runes/0042_calm_rune.cpp")],
    "charm": [("ogn-043-298", "src/cards/spells/0043_charm.cpp")],
    "darius_trifarian": [("ogn-027-298", "src/cards/units/0027_darius_trifarian.cpp")],
    "decisive_strike": [("ogs-024-024", "src/cards/spells/0322_decisive_strike.cpp")],
    "decree_of_focus": [("ven-040-166", "src/cards/spells/0827_decree_of_focus.cpp")],
    "decree_of_strength": [("ven-085-166", "src/cards/spells/0872_decree_of_strength.cpp")],
    "defy": [("ogn-045-298", "src/cards/spells/0045_defy.cpp")],
    "disarming_rake": [("sfd-032-221", "src/cards/units/0355_disarming_rake.cpp")],
    "discipline": [("ogn-058-298", "src/cards/spells/0058_discipline.cpp")],
    "emperor_s_dais": [("sfd-207-221", "src/cards/battlefields/0521_emperor_s_dais.cpp")],
    "en_garde": [("ogn-046-298", "src/cards/spells/0046_en_garde.cpp")],
    "ferrous_forerunner": [("sfd-021-221", "src/cards/units/0344_ferrous_forerunner.cpp")],
    "fiora_peerless": [("sfd-110-221", "src/cards/units/0432_fiora_peerless.cpp")],
    "first_mate": [("ogn-132-298", "src/cards/units/0132_first_mate.cpp")],
    "fury_rune": [("ogn-007-298", "src/cards/runes/0007_fury_rune.cpp")],
    "grim_apothecary": [("unl-021-219", "src/cards/units/0583_grim_apothecary.cpp")],
    "hush": [],   # no engine class (Radiance set)
    "inferna": [("unl-002-219", "src/cards/units/0560_inferna.cpp")],
    "irresistible_faefolk": [("unl-112-219", "src/cards/units/0674_irresistible_faefolk.cpp")],
    "kai_sa_survivor": [("ogn-039-298", "src/cards/units/0039_kai_sa_survivor.cpp")],
    "kinkou_initiate": [("unl-097-219", "src/cards/units/0659_kinkou_initiate.cpp")],
    "lonely_poro": [("sfd-036-221", "src/cards/units/0359_lonely_poro.cpp"),
                    ("unl-221-219", "src/cards/units/0777_lonely_poro.cpp")],
    "loyal_poro": [("unl-156-219", "src/cards/units/0718_loyal_poro.cpp")],
    "machine_evangel": [("ogn-239-298", "src/cards/units/0239_machine_evangel.cpp")],
    "master_yi_tempered": [("unl-113-219", "src/cards/units/0675_master_yi_tempered.cpp")],
    "master_yi_wuju_bladesman": [("ogs-019-024", "src/cards/legends/0317_wuju_bladesman_starter.cpp")],
    "moonlight_affliction": [("unl-066-219", "src/cards/spells/0628_moonlight_affliction.cpp")],
    "nidalee_cat_form": [("unl-114-219", "src/cards/units/0676_nidalee_cat_form.cpp")],
    "not_so_fast": [("sfd-045-221", "src/cards/spells/0368_not_so_fast.cpp")],
    "noxian_demolitionist": [("ven-080-166", "src/cards/units/0867_noxian_demolitionist.cpp")],
    "noxus_hopeful": [("ogn-012-298", "src/cards/units/0012_noxus_hopeful.cpp")],
    "onslaught": [("ven-081-166", "src/cards/spells/0868_onslaught.cpp")],
    "pakaa_cub": [("ogn-135-298", "src/cards/units/0135_pakaa_cub.cpp")],
    "pit_rookie": [("ogn-136-298", "src/cards/units/0136_pit_rookie.cpp")],
    "primal_strength": [("ogn-154-298", "src/cards/spells/0154_primal_strength.cpp")],
    "punch_first": [("sfd-097-221", "src/cards/spells/0419_punch_first.cpp")],
    "pyke_dockside_butcher": [("unl-028-219", "src/cards/units/0590_pyke_dockside_butcher.cpp")],
    "rampage": [("ven-083-166", "src/cards/spells/0870_rampage.cpp")],
    "rengar_pridestalker": [("unl-183-219", "src/cards/legends/0744_pridestalker.cpp")],
    "rengar_trophy_hunter": [("unl-120-219", "src/cards/units/0682_rengar_trophy_hunter.cpp")],
    "ruin_runner": [("sfd-105-221", "src/cards/units/0427_ruin_runner.cpp")],
    "sabotage": [("ogn-156-298", "src/cards/spells/0156_sabotage.cpp")],
    "scuttle_crab": [("unl-053-219", "src/cards/units/0615_scuttle_crab.cpp")],
    "seat_of_power": [("sfd-217-221", "src/cards/battlefields/0531_seat_of_power.cpp")],
    "star_spring": [("unl-215-219", "src/cards/battlefields/0771_star_spring.cpp")],
    "thrill_of_the_hunt": [("unl-184-219", "src/cards/spells/0745_thrill_of_the_hunt.cpp")],
    "zhonya_s_hourglass": [("ogn-077-298", "src/cards/gear/0077_zhonya_s_hourglass.cpp")],
}

KW_MAP = {"Accelerate": "accelerate", "Action": "action", "Ambush": "ambush",
          "Assault": "assault", "Deathknell": "deathknell", "Deflect": "deflect",
          "Ganking": "ganking", "Hidden": "hidden", "Hunt": "hunt", "Legion": "legion",
          "Level": "level", "Reaction": "reaction", "Flow": "flow",
          # B2 v2 (2026-11-21): keywords of the Azir / Diana lists
          "Equip": "equip", "Temporary": "temporary", "Tank": "tank", "Shield": "shield", "Vision": "vision",
          "Weaponmaster": "weaponmaster", "Quick-Draw": "quick_draw", "QuickDraw": "quick_draw", "Repeat": "repeat",
          "Empower": "empower", "Deathknell2": "deathknell"}
TYPE_MAP = {"Unit": "unit", "Spell": "spell", "Gear": "gear", "Rune": "rune",
            "Battlefield": "battlefield", "Legend": "legend"}

CPP_NOT_CALLS = {"if", "for", "while", "switch", "return", "static_cast", "sizeof",
                 "RIFT_TRACE", "catch", "def", "override", "const"}


def read_lines(rel):
    with open(os.path.join(REPO, rel), encoding="utf-8") as f:
        return f.read().split("\n")


def cite(rel, a, b=None):
    return f"{rel}:{a}" if b is None or b == a else f"{rel}:{a}-{b}"


def parse_carddef(rel):
    """Stats + keywords from the class's CardDef, each with file:line."""
    L = read_lines(rel)
    out = {"card_type": None, "super_type": None, "domains": [], "energy_cost": 0,
           "power_cost": 0, "might": None, "keywords": [], "assault_value": None,
           "def_id": None, "ability_text": None, "_src": {}}
    in_def = False
    for i, line in enumerate(L, 1):
        if "const CardDef def_" in line:
            in_def = True
        if not in_def:
            continue
        c = cite(rel, i)
        m = re.search(r"d\.card_type = CardType::(\w+)", line)
        if m:
            out["card_type"] = TYPE_MAP[m.group(1)]; out["_src"]["type"] = c
        m = re.search(r"d\.super_type = SuperType::(\w+)", line)
        if m:
            out["super_type"] = m.group(1).lower(); out["_src"]["supertype"] = c
        m = re.search(r"d\.domains = \{([^}]*)\}", line)
        if m:
            out["domains"] = sorted(x.strip().split("::")[1].lower()
                                    for x in m.group(1).split(",") if x.strip())
            out["_src"]["domains"] = c
        for fld in ("energy_cost", "power_cost", "might", "assault_value"):
            m = re.search(r"d\.%s = (-?\d+);" % fld, line)
            if m:
                out[fld] = int(m.group(1)); out["_src"][fld] = c
        m = re.search(r"d\.keywords\.set\(Keyword::(\w+)\)", line)
        if m:
            out["keywords"].append((KW_MAP[m.group(1)], c))
        m = re.search(r'd\.def_id = R"RB\(([^)]*)\)RB"', line)
        if m:
            out["def_id"] = m.group(1)
        if 'd.ability_text = R"RB(' in line:
            j, buf = i, line.split('R"RB(', 1)[1]
            while ')RB"' not in buf:
                buf += "\n" + L[j]; j += 1
            out["ability_text"] = buf.split(')RB"')[0]
            out["_src"]["ability_text"] = cite(rel, i, j)
    return out


def class_body_lines(rel):
    """(line_no, code) of the class body before the CardDef, comments and
    string literals blanked (so trace labels cannot leak names)."""
    L = read_lines(rel)
    res = []
    for i, line in enumerate(L, 1):
        if "const CardDef def_" in line:
            break
        code = re.sub(r"//.*", "", line)
        code = re.sub(r'"(\\.|[^"\\])*"', '""', code)
        res.append((i, code))
    return res


def scan_calls(rel):
    body = class_body_lines(rel)
    text = "\n".join(c for _, c in body)
    local = set(re.findall(r"auto\s+(\w+)\s*=\s*\[", text))
    # names of methods/functions this class defines (signature lines)
    for _, c in body:
        if "override" in c or re.match(r"\s*static\s", c):
            m = re.search(r"(\w+)\s*\(", c.split("override")[0].replace("static", ""))
            if m:
                local.add(m.group(1))
    calls = []
    in_trace = False
    for i, c in body:
        if "RIFT_TRACE" in c:
            in_trace = not c.rstrip().endswith(";")
            continue
        if in_trace:
            in_trace = not c.rstrip().endswith(";")
            continue
        if "override" in c:   # a definition line; its calls are after '{'
            c = c.split("{", 1)[1] if "{" in c else ""
        for m in re.finditer(r"ctx\.executor\.(\w+)\s*\(", c):
            calls.append({"kind": "executor", "name": m.group(1), "line": i})
        for m in re.finditer(r"(?<![\w\.>:])(\w+)\s*\(", c):
            n = m.group(1)
            if n in local or n in CPP_NOT_CALLS or n[0].isupper() and n not in ("Card",):
                continue
            calls.append({"kind": "free", "name": n, "line": i})
    hooks = []
    for i, c in body:
        if "override" in c:
            m = re.search(r"(\w+)\s*\(", c.split("override")[0])
            if m:
                hooks.append({"name": m.group(1), "line": i})
    trig = [{"name": m.group(1), "line": i} for i, c in body
            for m in re.finditer(r"TriggerType::(\w+)", c)]
    treq = [{"name": m.group(1), "line": i} for i, c in body
            for m in re.finditer(r"\.(must_be_\w+|count|optional|max_might)\s*=", c)]
    return {"calls": calls, "hooks": hooks, "triggers": trig, "target_reqs": treq}


if __name__ == "__main__":
    import collections
    agg = collections.Counter()
    for key, cls in CLASSES.items():
        for _, rel in cls:
            s = scan_calls(rel)
            for c in s["calls"]:
                agg[(c["kind"], c["name"])] += 1
            for h in s["hooks"]:
                agg[("hook", h["name"])] += 1
            for t in s["triggers"]:
                agg[("trigger", t["name"])] += 1
            for t in s["target_reqs"]:
                agg[("treq", t["name"])] += 1
    for k, v in sorted(agg.items()):
        print(k, v)
