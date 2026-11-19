"""Path B2 build: mechanics graphs from the engine card CODE -> B's fixed encoder.

Usage:  python extract.py        (paths relative to this file)

Writes (all inside B2_code/)
  vocab.json                 vocabulary map (vocab_src.py) + coverage of every call the 51 classes make
  cards/<key>.json           graph per card key, with "_src" citations on every node
                             (+ "_classes" when a card has several engine classes)
  out/vectors.json           SPEC output contract (keys, printing:, rename:, synthetic:)
  out/paths/<key>.json       explicit (unhashed) path features
  out/text_vs_engine.json    where card text (and path B) differ from the engine code
  out/selfcheck.json         distances / sanity (not the official eval)
"""
import copy
import importlib.util
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import scan  # noqa: E402
import vocab_src as V  # noqa: E402
import graphs_code as GC  # noqa: E402

T0 = time.time()

# ---- path B's encoder, imported unchanged (B and B2 differ only in source)
sys.path.insert(0, os.path.join(ROOT, "B_graph"))
_spec = importlib.util.spec_from_file_location("b_build", os.path.join(ROOT, "B_graph", "build.py"))
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)
GS = B.GS   # path B's text graphs (used for Hush and for the B-vs-B2 diff only)


def strip(x):
    """Drop citation / meta keys ("_..."): what the encoder sees."""
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if not k.startswith("_")}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def vec(graph):
    return B.vectorize(strip(graph))


# ---------------------------------------------------------------- engine keyword expansions

def kwsrc(k):
    return "; ".join(V.KEYWORDS[k]["src"])


def hidden_mode():
    return {"kind": "play_mode", "mode": "from_facedown", "timing": "reaction",
            "setup_cost": {"power": 1, "power_domain": "any", "_src": V.GE + ":5326-5332, 5775-5779"},
            "cost": {"energy": 0, "power": 0, "_src": V.GE + ":5429-5436"},
            "where": {"zone": "battlefield", "filters": [{"f": "controlled_by_you", "v": True}],
                      "_src": V.GE + ":5781-5783"},
            # gate 1: playable only on a turn after it was hidden (hidden_on_turn)
            "condition": [{"c": "hidden_this_turn", "v": True, "neg": True,
                           "_src": V.GE + ":5395, 6127; src/core/game_object.h:75"}],
            "_src": kwsrc("hidden")}


def ambush_mode(enemy_cite=None):
    if enemy_cite:
        filters = [{"f": "any_of", "v": [{"f": "has_friendly_units", "v": True},
                                         {"f": "has_enemy_units", "v": True}], "_src": enemy_cite}]
    else:
        filters = [{"f": "has_friendly_units", "v": True}]
    return {"kind": "play_mode", "mode": "ambush", "timing": "reaction",
            "where": {"zone": "battlefield", "filters": filters, "_src": V.PP + ":17-21"},
            "_src": kwsrc("ambush")}


def accelerate():
    return {"kind": "additional_cost",
            "cost": {"energy": 1, "power": 1, "power_domain": "card", "optional": True,
                     "additional": True, "_src": V.GE + ":2380-2382"},
            "effects": [{"verb": "EnterReady", "target": GC.SELF(V.GE + ":4885"), "_src": V.GE + ":4885"}],
            "_src": kwsrc("accelerate")}


def assault(n, cite):
    return {"kind": "static", "condition": [{"c": "is_attacker", "v": True, "_src": V.GO + ":466"}],
            "effects": [{"verb": "GiveMight", "target": GC.SELF(V.GO + ":467"),
                         "amount": {"n": n, "_src": cite}, "duration": "while", "_src": V.GO + ":466-467"}],
            "_src": kwsrc("assault")}


def flow_mode(cite, energy, power):
    c = {"energy": energy, "_src": cite}
    if power:
        c["power"] = power
        c["power_domain"] = "card"
    return {"kind": "play_mode", "mode": "from_trash", "timing": "same", "cost": c,
            "effects": [{"verb": "Banish", "target": GC.SELF(V.GE + ":8388"), "after": "resolve",
                         "_src": V.GE + ":8388"}],
            "_src": cite + "; " + kwsrc("flow")}


def rune_abilities():
    s = kwsrc("rune")
    return [
        {"kind": "activated", "timing": "reaction", "cost": {"exhaust": True, "_src": V.GE + ":9525-9530"},
         "effects": [{"verb": "AddEnergy", "target": GC.YOU(), "amount": {"n": 1}, "_src": V.GE + ":9525-9530"}],
         "_src": s},
        {"kind": "activated", "timing": "reaction", "cost": {"recycle_self": True, "_src": V.GE + ":9109"},
         "effects": [{"verb": "AddPower", "target": GC.YOU(), "amount": {"n": 1}, "domain": "card",
                      "_src": V.GE + ":9109"}],
         "_src": s},
    ]

# ---------------------------------------------------------------- one class -> graph

def stats_from_def(d):
    t = d["card_type"]
    S = d["_src"]
    s = {"type": t, "supertype": d["super_type"], "domains": d["domains"]}
    src = {"type": S.get("type"), "supertype": S.get("supertype", "default SuperType::None"),
           "domains": S.get("domains", "default {}")}
    if t in ("unit", "spell", "gear"):
        s["energy"] = d["energy_cost"]
        s["power"] = d["power_cost"]
        src["energy"] = S.get("energy_cost", "default 0")
        src["power"] = S.get("power_cost", "default 0")
        if s["power"]:
            s["power_by_domain"] = {dm: s["power"] for dm in (d["domains"] or ["any"])}
            src["power_by_domain"] = src["power"] + " x " + src["domains"] + "; " + V.GE + ":9109"
    else:
        s["energy"] = None
        s["power"] = None
    s["might"] = (d["might"] or 0) if t == "unit" else None
    if t == "unit":
        src["might"] = S.get("might", "default 0")
    s["_src"] = src
    return s


def build_class(key, def_id, rel):
    d = scan.parse_carddef(rel)
    assert d["def_id"] == def_id, (key, def_id, d["def_id"])
    cg = GC.BUILDERS[(key, def_id)]()
    kws = []
    kwv = cg.get("kw_values", {})
    for k, cite in d["keywords"]:
        e = {"kw": k, "_src": cite}
        if k == "assault":
            e["value"] = d["assault_value"]; e["_src"] += "; value " + d["_src"]["assault_value"]
        if k in kwv:
            e["value"] = kwv[k][0]; e["_src"] += "; value " + kwv[k][1]
        kws.append(e)
    names = {k for k, _ in d["keywords"]}
    abil = []
    if "hidden" in names:
        abil.append(hidden_mode())
    if "ambush" in names:
        abil.append(ambush_mode(cg.get("ambush_enemy")))
    if "accelerate" in names:
        abil.append(accelerate())
    if "assault" in names:
        abil.append(assault(d["assault_value"], d["_src"]["assault_value"]))
    if "flow" in names:
        fc, fe, fp = cg["flow"]
        abil.append(flow_mode(fc, fe, fp))
    if d["card_type"] == "rune":
        abil += rune_abilities()
    timing = "reaction" if "reaction" in names else "action" if "action" in names else "main"
    for a in copy.deepcopy(cg["abilities"]):
        if a.get("timing") == "@timing":
            a["timing"] = timing
        abil.append(a)
    g = {"stats": stats_from_def(d), "keywords": kws, "abilities": abil,
         "_meta": {"def_id": def_id, "class_file": rel, "source": "engine_code",
                   "flags": cg.get("flags", [])}}
    return g, d


def hush_graph(printing):
    src = GS.G["hush"]
    gs_line = graphs_src_line("hush")
    g = {"stats": B.stats_from_printing(printing), "keywords": copy.deepcopy(src["keywords"]),
         "abilities": copy.deepcopy(src["abilities"]),
         "_meta": {"def_id": printing["id"], "class_file": None, "source": "TEXT (no engine class)",
                   "flags": ["TEXT-DERIVED: no engine class exists for this card (Radiance set); "
                             "graph = path B's text graph, stats from cards.json"]}}
    g["stats"]["_src"] = "research/card_repr/cards.json printing " + printing["id"]
    for a in g["abilities"]:
        a["_src"] = "research/card_repr/B_graph/graphs_src.py:%d (text)" % gs_line
    return g


_GS_LINES = None


def graphs_src_line(key):
    global _GS_LINES
    if _GS_LINES is None:
        _GS_LINES = open(os.path.join(ROOT, "B_graph", "graphs_src.py"), encoding="utf-8").read().split("\n")
    for i, l in enumerate(_GS_LINES, 1):
        if f'G["{key}"]' in l:
            return i
    if key.endswith("_rune"):
        for i, l in enumerate(_GS_LINES, 1):
            if 'for _r in ("body_rune"' in l:
                return i
    return None

# ---------------------------------------------------------------- synthetic edits (on B2 graphs)

def _spell(g):
    return [a for a in g["abilities"] if a["kind"] == "spell"][0]


def _eff(g, verb):
    for a in g["abilities"]:
        for e in a.get("effects", []):
            if e["verb"] == verb:
                return e
    raise KeyError(verb)


def syn_edit(base, sid):
    g = copy.deepcopy(base)
    if sid == "punch_first__amount4":
        _eff(g, "GiveMight")["amount"]["n"] = 4
    elif sid == "punch_first__energy3":
        g["stats"]["energy"] = 3
    elif sid == "punch_first__power0":
        g["stats"]["power"] = 0
        g["stats"].pop("power_by_domain", None)
    elif sid == "punch_first__reaction":
        _spell(g)["timing"] = "reaction"
        g["keywords"] = [dict(k, kw="reaction") if k["kw"] == "action" else k for k in g["keywords"]]
    elif sid == "discipline__draw2":
        _eff(g, "Draw")["amount"]["n"] = 2
    elif sid == "discipline__enemy":
        _eff(g, "GiveMight")["target"]["side"] = "enemy"
    elif sid == "lonely_poro__draw2":
        _eff(g, "Draw")["amount"]["n"] = 2
    elif sid == "defy__cost2":
        f = [x for x in _eff(g, "Counter")["target"]["filters"] if x["f"] == "energy_cost_le"][0]
        f["v"] = 2
    elif sid == "sabotage__unit":
        _eff(g, "Recycle")["target"]["filters"] = [{"f": "type", "v": "unit"}]
    elif sid == "pit_rookie__might3":
        g["stats"]["might"] = 3
    else:
        raise KeyError(sid)
    g["_meta"] = dict(g["_meta"], synthetic=sid)
    return g

# ---------------------------------------------------------------- score-family variants (smoke/pairs)
# Engine classes: C_engine/card_probe.cpp ScoreSpellVar / PunchKillVar / PunchScoreTargetVar /
# ScoreUnitVar (and PunchFirstVar(-5)), registered in smoke/data/src/decision_export.cpp (1631-1642).
# Each keeps the base card's stats/keywords and replaces the base effect. New schema pieces:
#   verb Score {target: player selector, amount}; a player selector may carry relative_to: <bind>
#   (side is then judged from the bound object's controller: friendly = its controller,
#   enemy = its controller's opponent). Conditions: you_control_units, you_control_other_units
#   (neg = "no ..."), opponent_hand_size_ge, friendly_unit_at_battlefield, opponent_score_gt_yours.
SCORE_VARIANTS = {   # sid -> base key
    "punch_first__score1": "punch_first", "punch_first__score_nounits": "punch_first",
    "punch_first__score_bfunit": "punch_first", "punch_first__score_behind": "punch_first",
    "punch_first__ctrl_scores": "punch_first", "punch_first__opp_of_ctrl_scores": "punch_first",
    "punch_first__kill": "punch_first", "punch_first__minus5": "punch_first",
    "pit_rookie__score1": "pit_rookie", "pit_rookie__score_alone": "pit_rookie",
    "sabotage__score1": "sabotage", "sabotage__score_opphand4": "sabotage",
}
_SRC = "C_engine/card_probe.cpp (play-changing variants)"


def _score(cond=None, side="friendly", rel=None):
    tgt = {"type": "player", "side": side, "zone": "player", "count": 1}
    if rel:
        tgt["relative_to"] = rel
    e = {"verb": "Score", "target": tgt, "amount": {"n": 1}, "_src": _SRC}
    if cond:
        e["condition"] = cond
    return e


def score_edit(base, sid):
    """Graph edit of the base card for a score-family variant (works on B and B2 graphs)."""
    g = copy.deepcopy(base)
    ab = [a for a in g["abilities"] if a["kind"] in ("spell", "triggered")][0]
    unit_any = {"type": "unit", "side": "any", "zone": "board", "count": 1}
    if sid in ("punch_first__score1", "sabotage__score1", "pit_rookie__score1"):
        ab["effects"] = [_score()]
    elif sid == "punch_first__score_nounits":
        ab["effects"] = [_score([{"c": "you_control_units", "v": True, "neg": True}])]
    elif sid == "pit_rookie__score_alone":
        ab["effects"] = [_score([{"c": "you_control_other_units", "v": True, "neg": True}])]
    elif sid == "punch_first__score_bfunit":
        ab["effects"] = [_score([{"c": "friendly_unit_at_battlefield", "v": True}])]
    elif sid == "punch_first__score_behind":
        ab["effects"] = [_score([{"c": "opponent_score_gt_yours", "v": True}])]
    elif sid == "sabotage__score_opphand4":
        ab["effects"] = [_score([{"c": "opponent_hand_size_ge", "v": 4}])]
    elif sid in ("punch_first__ctrl_scores", "punch_first__opp_of_ctrl_scores"):
        side = "friendly" if sid == "punch_first__ctrl_scores" else "enemy"   # the ONLY difference
        ab["effects"] = [{"verb": "ChooseTarget", "target": dict(unit_any, bind="A"), "_src": _SRC},
                         _score(side=side, rel="A")]
    elif sid == "punch_first__kill":
        ab["effects"] = [{"verb": "Kill", "target": dict(unit_any), "_src": _SRC}]
    elif sid == "punch_first__minus5":
        e = [x for x in ab["effects"] if x["verb"] == "GiveMight"][0]
        e["amount"]["n"] = -5
    else:
        raise KeyError(sid)
    if "_meta" in g:
        g["_meta"] = dict(g["_meta"], synthetic=sid)
    return g

# ---------------------------------------------------------------- vocab + coverage

def coverage():
    rows, per_card = [], {}
    tally = {"verb": 0, "structure": 0, "plumbing": 0, "hook": 0, "unmapped": 0}
    names = {"verb": set(), "structure": set(), "plumbing": set(), "hook": set(), "unmapped": set()}
    for key, cls in scan.CLASSES.items():
        for def_id, rel in cls:
            s = scan.scan_calls(rel)
            c_rows = []
            for c in s["calls"]:
                tab = V.EXECUTOR if c["kind"] == "executor" else V.HELPERS
                ent = tab.get(c["name"]) or (V.HOOKS.get(c["name"]) and {"role": "hook"})
                role = ent["role"] if ent else "unmapped"
                tally[role] += 1
                names[role].add(c["name"])
                c_rows.append({"call": c["name"], "kind": c["kind"], "line": f"{rel}:{c['line']}",
                               "role": role, "verb": ent.get("verb") if ent else None})
            for h in s["hooks"]:
                role = "hook" if h["name"] in V.HOOKS else "unmapped"
                tally[role] += 1
                names[role].add(h["name"])
                c_rows.append({"call": h["name"], "kind": "override", "line": f"{rel}:{h['line']}",
                               "role": role})
            per_card[f"{key}@{def_id}"] = c_rows
            rows += c_rows
    trig_used = sorted({t["name"] for k, cls in scan.CLASSES.items() for _, r in cls
                        for t in scan.scan_calls(r)["triggers"]})
    treq_used = sorted({t["name"] for k, cls in scan.CLASSES.items() for _, r in cls
                        for t in scan.scan_calls(r)["target_reqs"]})
    return {"tally_calls": tally, "distinct_names": {k: sorted(v) for k, v in names.items()},
            "trigger_types_used": trig_used, "target_requirement_fields_used": treq_used,
            "per_class": per_card}


def verbs_of(graph):
    out = set()
    def w(x):
        if isinstance(x, dict):
            if "verb" in x and isinstance(x["verb"], str):
                out.add(x["verb"])
            for v in x.values():
                w(v)
        elif isinstance(x, list):
            for v in x:
                w(v)
    w(strip(graph))
    return out


def verb_consistency(graph, rel):
    """Verbs implied by the class's executor/helper calls vs verbs in the class graph."""
    implied = set()
    for c in scan.scan_calls(rel)["calls"]:
        tab = V.EXECUTOR if c["kind"] == "executor" else V.HELPERS
        ent = tab.get(c["name"])
        if ent and ent["role"] == "verb":
            vv = ent["verb"] if isinstance(ent["verb"], list) else [ent["verb"]]
            implied |= {x.split("{")[0] for x in vv}
    have = verbs_of(graph)
    engine_kw_verbs = {"EnterReady", "AddEnergy", "AddPower", "Banish", "GiveMight"}
    # graph verbs not from a call: ChooseTarget (structure), Reveal/LookAt (events), keyword
    # expansions, Counter via helper, CantBeChosen / ReduceCost / GiveKeyword (hooks / auras)
    return {"implied_by_calls": sorted(implied), "in_graph": sorted(have),
            "call_verbs_missing_from_graph": sorted(implied - have),
            "graph_verbs_without_a_call": sorted(have - implied - engine_kw_verbs)}

# ---------------------------------------------------------------- text vs engine

DISAGREEMENTS = [
    {"key": "zhonya_s_hourglass", "kind": "mechanics", "severity": "major",
     "text": "The next time a friendly unit would die, kill this instead. Recall that unit exhausted.",
     "engine": "applyReplacement: killObject(self) then healExhaustRecall(unit) -> heals, exhausts and "
               "recalls; replacementAppliesTo has no 'next time' clause (once only because the gear dies)",
     "engine_cite": ["src/cards/gear/0077_zhonya_s_hourglass.cpp:17-41", "src/engine/effect_executor.cpp:1467-1472"],
     "path_B": "follows the text: no Heal, trigger.once=next_time",
     "B2": "Kill self, Heal subject, Recall subject exhausted; no once"},
    {"key": "rengar_trophy_hunter", "kind": "mechanics", "severity": "major",
     "text": "I can be played to a battlefield where there are enemy units (even if you don't have units there).",
     "engine": "only ambushToEnemyBattlefields(): widens the [Ambush] location; the default "
               "(non-Ambush) play locations are unchanged (extraPlayLocations not overridden)",
     "engine_cite": ["src/cards/units/0682_rengar_trophy_hunter.cpp:18-23", "src/engine/play_permissions.cpp:17-38"],
     "path_B": "static AllowPlayTo self where has_enemy_units (any play)",
     "B2": "Ambush play_mode where any_of[has_friendly_units, has_enemy_units]; no AllowPlayTo"},
    {"key": "rampage", "kind": "mechanics", "severity": "minor",
     "text": "They deal damage equal to their Mights to each other.",
     "engine": "dealDamage(enemy, friendly_might, ctx.source) and dealDamage(friendly, enemy_might, "
               "ctx.source): the SPELL is the damage source, not the units",
     "engine_cite": ["src/cards/spells/0870_rampage.cpp:74-80"],
     "path_B": "source = ref A / ref B (the units)",
     "B2": "source = self (the spell)"},
    {"key": "darius_trifarian", "kind": "mechanics", "severity": "minor",
     "text": "When you play your second card in a turn, ...",
     "engine": "triggerCondition: a token is not a card (returns false); ordinal of the played card == 2",
     "engine_cite": ["src/cards/units/0027_darius_trifarian.cpp:30-37"],
     "path_B": "condition nth_card_this_turn=2 only",
     "B2": "nth_card_this_turn=2 and is_token(subject) neg"},
    {"key": "alpha_strike", "kind": "legality", "severity": "minor",
     "text": "Choose a friendly unit. It deals damage equal to its Might split among enemy units at battlefields.",
     "engine": "hasLegalTargets also requires a choosable enemy unit at a battlefield (play gate)",
     "engine_cite": ["src/cards/spells/0751_alpha_strike.cpp:33-40"],
     "path_B": "not represented",
     "B2": "spell.play_requires = [enemy unit at battlefield] (schema extension)"},
    {"key": "seat_of_power", "kind": "mechanics", "severity": "none in 1v1",
     "text": "draw 1 for each other battlefield you or allies control",
     "engine": "counts battlefields whose controller is the ability's controller (no allies in 1v1)",
     "engine_cite": ["src/cards/battlefields/0531_seat_of_power.cpp:21-30"],
     "path_B": "per other_battlefield_controlled (same)", "B2": "same as B"},
    {"key": "emperor_s_dais", "kind": "wording / token identity", "severity": "minor",
     "text": "... you may pay [1] and return a unit you control here to its owner's hand. If you do, play a "
             "2 [M] Sand Soldier unit token here.",
     "engine": "pay [1] + return a unit is the COST (costDueNow) 'to play' the token; token created with tag "
               "'Shurima' (name 'Sand Soldier')",
     "engine_cite": ["src/cards/battlefields/0521_emperor_s_dais.cpp:36-54", "src/cards/battlefields/0521_emperor_s_dais.cpp:70"],
     "path_B": "optional cost (same structure); token.tag = sand_soldier (the name)",
     "B2": "same structure; token.tag = shurima (code tag) and token.name = sand_soldier (gate 1)"},
    {"key": "fiora_peerless", "kind": "interpretation", "severity": "minor",
     "text": "double my Might this combat",
     "engine": "adds current_might at resolution as a combat-long bonus (snapshot; a later Might change is "
               "not doubled); one on one = exactly 1 attacker and 1 defender at my battlefield",
     "engine_cite": ["src/cards/units/0432_fiora_peerless.cpp:25-62"],
     "path_B": "GiveMight self amount might_of:self mult 1, combat (same)",
     "B2": "same as B (snapshot not expressible)"},
    {"key": "thrill_of_the_hunt", "kind": "edge case", "severity": "minor",
     "text": "Banish a friendly unit, then its owner plays it to any battlefield, ignoring its cost.",
     "engine": "a token is banished and not played back (CR 183.1); 'any battlefield' = those its owner may "
               "play units to; with none it stays banished",
     "engine_cite": ["src/cards/spells/0745_thrill_of_the_hunt.cpp:41-72"],
     "path_B": "no token condition", "B2": "PlayIgnoringCost condition is_token(A) neg"},
    {"key": "defy", "kind": "interpretation", "severity": "minor",
     "text": "Counter a spell that costs no more than [4] and no more than [A].",
     "engine": "compares the spell's PRINTED energy_cost/power_cost from the card table (cost modifiers ignored)",
     "engine_cite": ["src/cards/spells/0045_defy.cpp:27-31"],
     "path_B": "energy_cost_le 4, power_cost_le 1 (same)", "B2": "same as B"},
    {"key": "master_yi_wuju_bladesman", "kind": "interpretation", "severity": "minor",
     "text": "While a friendly unit defends alone, it gets +2 [M].",
     "engine": "exactly one friendly unit with the Defender designation anywhere on the board",
     "engine_cite": ["src/cards/legends/0317_wuju_bladesman_starter.cpp:27-39"],
     "path_B": "defends_alone (same)", "B2": "same as B"},
    {"key": "hush", "kind": "no engine class", "severity": "n/a",
     "text": "Counter a spell with Power cost [A][A] or more.",
     "engine": "no class (Radiance set not implemented)", "engine_cite": [],
     "path_B": "text graph", "B2": "copied from path B, flagged TEXT-DERIVED"},
    {"key": "back_off", "kind": "cosmetic", "severity": "none",
     "text": "[Hidden] (Hide now for [A] to react with later for [0].)",
     "engine": "engine def text drops '[0]' in the reminder (also pakaa_cub, pyke_dockside_butcher); "
               "the engine plays from facedown ignoring base cost (GE:5429-5436)",
     "engine_cite": ["src/cards/spells/0604_back_off.cpp:68"], "path_B": "n/a", "B2": "n/a"},
]


def text_vs_engine(cj, b2_graphs, defs):
    """Hand list (DISAGREEMENTS) + automatic: engine def text vs cards.json text, and the
    explicit path diff between B's text graph and B2's code graph for every card."""
    by_key = {c["key"]: c for c in cj["cards"]}
    auto = {}
    for key, g in b2_graphs.items():
        bg = {"stats": g["stats"], "keywords": GS.G[key]["keywords"], "abilities": GS.G[key]["abilities"]}
        pb = set(full_paths(strip(bg)))
        p2 = set(full_paths(strip(g)))
        prs = by_key[key]["printings"]
        card_text = B.norm_text(prs[0]["ability_text"]) if prs else None
        dtexts = {did: B.norm_text(d["ability_text"]) for did, d in defs.get(key, {}).items()}
        auto[key] = {
            "B_graph_cite": "research/card_repr/B_graph/graphs_src.py:%s" % graphs_src_line(key),
            "B2_cite": "research/card_repr/B2_code/cards/%s.json" % key,
            "paths_only_in_B_text_graph": sorted(pb - p2),
            "paths_only_in_B2_code_graph": sorted(p2 - pb),
            "graphs_identical": pb == p2,
            "engine_def_text_equals_card_text": {did: (t == card_text) if card_text is not None else None
                                                 for did, t in dtexts.items()},
        }
    return {"disagreements": DISAGREEMENTS, "per_card_auto": auto}


def full_paths(graph):
    raw = []
    B.walk(graph, [B.ROOT], raw)
    return ["/".join(s) + ("=%g" % v if v != 1.0 else "") for s, v in raw]

# ---------------------------------------------------------------- main

def main():
    cj = json.load(open(os.path.join(ROOT, "cards.json"), encoding="utf-8"))
    os.makedirs(os.path.join(HERE, "cards"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "out", "paths"), exist_ok=True)
    vecs, graphs, defs, problems, notes = {}, {}, {}, [], []
    class_graphs = {}
    consistency = {}

    for c in cj["cards"]:
        key, name = c["key"], c["name"]
        cls = scan.CLASSES[key]
        if not cls:   # Hush
            g = hush_graph(c["printings"][0])
            graphs[key] = g
            class_graphs[key] = {}
        else:
            class_graphs[key] = {}
            defs[key] = {}
            for def_id, rel in cls:
                g, d = build_class(key, def_id, rel)
                class_graphs[key][def_id] = g
                defs[key][def_id] = d
                consistency[f"{key}@{def_id}"] = verb_consistency(g, rel)
            graphs[key] = class_graphs[key][cls[0][0]]
        # name leak check on what the encoder sees
        s = json.dumps(strip(graphs[key])).lower()
        if name.lower() in s:
            problems.append(f"name '{name}' leaks into {key}")
        for w in re.findall(r"[a-z']{6,}", name.lower()):
            if w in s and w not in ("survivor",):
                problems.append(f"name word '{w}' appears in {key}")
        vecs[key] = vec(graphs[key])
        # printings: own class if it has one, else the card's class (flagged)
        own = {did for did, _ in cls}
        for p in c["printings"]:
            pid = p["id"]
            if pid in own:
                vecs[f"printing:{pid}"] = vec(class_graphs[key][pid])
            else:
                vecs[f"printing:{pid}"] = vec(graphs[key])
                notes.append(f"printing:{pid} ({key}) has no engine class of its own: uses "
                             f"{'the text graph' if not cls else cls[0][1]}")
        for did in own - {p["id"] for p in c["printings"]}:   # legends: classes not in cards.json
            vecs[f"printing:{did}"] = vec(class_graphs[key][did])
            notes.append(f"printing:{did} ({key}) emitted from its engine class (not listed in cards.json)")
        renamed = json.loads(json.dumps(graphs[key]).replace(name, "Zorblax Quintessa"))
        vecs[f"rename:{key}"] = vec(renamed)
        out = dict(graphs[key])
        out["_meta"] = dict(out["_meta"], confidence=GC.CONFIDENCE.get(key, ("high", "direct executor "
                            "calls with literal arguments"))[0],
                            confidence_why=GC.CONFIDENCE.get(key, ("high", "direct executor calls with "
                            "literal arguments"))[1])
        if len(class_graphs.get(key, {})) > 1:
            out["_classes"] = class_graphs[key]
        with open(os.path.join(HERE, "cards", f"{key}.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        with open(os.path.join(HERE, "out", "paths", f"{key}.json"), "w", encoding="utf-8") as f:
            json.dump(dict(sorted(B.graph_paths(strip(graphs[key])).items())), f, indent=0)

    # gate 1 (B1): standalone token-unit graphs, keyed as in the rbenv vocab (token:<name>)
    for tkey, tg in GC.TOKENS.items():
        vecs[tkey] = vec(tg)
        fn = tkey.replace(":", "__")
        with open(os.path.join(HERE, "cards", f"{fn}.json"), "w", encoding="utf-8") as f:
            json.dump(tg, f, indent=1)
        with open(os.path.join(HERE, "out", "paths", f"{fn}.json"), "w", encoding="utf-8") as f:
            json.dump(dict(sorted(B.graph_paths(strip(tg)).items())), f, indent=0)

    syn_graphs = {}
    for s in cj["t2_minimal_pairs_synthetic"]:
        syn_graphs[s["id"]] = syn_edit(graphs[s["base"]], s["id"])
        vecs[f"synthetic:{s['id']}"] = vec(syn_graphs[s["id"]])

    # score-family variants: B2 from the B2 base graph; B from path B's base graph (written into
    # B_graph/out/vectors.json as well, same edit, so B and B2 differ only in their base graphs)
    bpath = os.path.join(ROOT, "B_graph", "out", "vectors.json")
    bout = json.load(open(bpath, encoding="utf-8"))
    for sid, base in SCORE_VARIANTS.items():
        syn_graphs[sid] = score_edit(graphs[base], sid)
        vecs[f"synthetic:{sid}"] = vec(syn_graphs[sid])
        bg = json.load(open(os.path.join(ROOT, "B_graph", "cards", f"{base}.json"), encoding="utf-8"))
        bout["vectors"][f"synthetic:{sid}"] = B.vectorize(strip(score_edit(bg, sid)))
        with open(os.path.join(HERE, "cards", f"synthetic__{sid}.json"), "w", encoding="utf-8") as f:
            json.dump(syn_graphs[sid], f, indent=1)
    with open(bpath, "w", encoding="utf-8") as f:
        json.dump(bout, f)

    with open(os.path.join(HERE, "out", "vectors.json"), "w", encoding="utf-8") as f:
        json.dump({"dim": B.DIM, "vectors": vecs}, f)

    # ---- vocab.json (map + coverage)
    cov = coverage()
    vocab = {"executor_methods": V.EXECUTOR, "card_helpers_and_card_methods": V.HELPERS,
             "card_hooks": V.HOOKS, "target_requirements": V.TARGET_REQUIREMENTS,
             "trigger_types": V.TRIGGERS, "card_def": V.CARDDEF, "keywords": V.KEYWORDS,
             "schema_extensions": {
                 "verb Heal": "healExhaustRecall heals (Zhonya's Hourglass)",
                 "ability.play_requires": "list of selectors that must be choosable for the card to be "
                                          "playable, beyond its effect targets (hasLegalTargets; Alpha Strike)",
                 "condition is_token {v: subject|<bind>}": "generalises B's unit_is_token: Darius (played "
                     "card), Star Spring (played unit), Thrill of the Hunt (banished unit)",
                 "token.tag from code": "createToken tags argument (Emperor's Dais also keeps token.name, gate 1)",
                 "gate 1 additions": {
                     "token:<name> graphs": "standalone token-unit graphs (graphs_code.TOKENS): stats.supertype=token, "
                         "is_token, tag, name, enters_exhausted, no cost; vectors keyed token:mech, token:sand_soldier",
                     "play_mode.condition hidden_this_turn neg": "Hidden: not playable the turn it was hidden "
                         "(hidden_on_turn, game_engine.cpp:6127)",
                     "condition has_buff {v: target} neg": "Buff places nothing on a unit that already has one",
                     "filter cant_be_moved_by_enemy / at_self_location": "Irresistible Faefolk's legal targets",
                     "effect.dest_filters": "location filters on an effect's destination (Thrill: blocks_unit_play, "
                         "owner units_play_base_only)",
                     "effect.as_new_object": "the object returns as a new object (no damage, buffs, statuses)",
                     "effect.dest = owner_hand": "Bounce returns to the OWNER's hand (not the controller's)",
                     "ReduceCost target zone=hand": "a self cost reduction applies while the card is in hand"},
                 "_src / _meta / _classes": "citations and provenance; stripped before encoding"},
             "coverage": {k: v for k, v in cov.items() if k != "per_class"},
             "verb_consistency": consistency}
    with open(os.path.join(HERE, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=1)
    with open(os.path.join(HERE, "out", "coverage_per_class.json"), "w", encoding="utf-8") as f:
        json.dump(cov["per_class"], f, indent=1)

    # ---- text vs engine
    tve = text_vs_engine(cj, graphs, defs)
    with open(os.path.join(HERE, "out", "text_vs_engine.json"), "w", encoding="utf-8") as f:
        json.dump(tve, f, indent=1)

    # ---- self check
    import numpy as np
    keys = [c["key"] for c in cj["cards"]]
    M = np.array([vecs[k] for k in keys])

    def d(a, b):
        return float(np.linalg.norm(np.array(vecs[a]) - np.array(vecs[b])))

    def cosd(a, b):
        x, y = np.array(vecs[a]), np.array(vecs[b])
        return float(1 - x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    t1 = {}
    for c in cj["cards"]:
        for p in c["printings"]:
            t1[f"printing:{p['id']}"] = d(c["key"], f"printing:{p['id']}")
        t1[f"rename:{c['key']}"] = d(c["key"], f"rename:{c['key']}")
    pairs = {}
    for a, b, why in cj["t2_minimal_pairs_real"]:
        dist = np.linalg.norm(M - np.array(vecs[a]), axis=1)
        order = [keys[i] for i in np.argsort(dist) if keys[i] != a]
        pairs[f"{a}|{b}"] = {"dist": d(a, b), "rank_of_partner": order.index(b) + 1, "nearest": order[:3]}
    syn = {s["id"]: d(s["base"], f"synthetic:{s['id']}") for s in cj["t2_minimal_pairs_synthetic"]}
    bvec = json.load(open(os.path.join(ROOT, "B_graph", "out", "vectors.json")))["vectors"]
    b_vs_b2 = {}
    for k in keys:
        x, y = np.array(bvec[k]), np.array(vecs[k])
        b_vs_b2[k] = round(float(1 - x @ y / (np.linalg.norm(x) * np.linalg.norm(y))), 4)
    report = {"max_t1_distance": max(t1.values()), "t1": t1, "real_pairs": pairs, "synthetic_dist": syn,
              "identical_card_vectors": [(keys[i], keys[j]) for i in range(len(keys))
                                         for j in range(i + 1, len(keys)) if np.allclose(M[i], M[j])],
              "cosine_B_vs_B2_per_card": dict(sorted(b_vs_b2.items(), key=lambda kv: -kv[1])),
              "cards_identical_to_B": sorted(k for k, v in b_vs_b2.items() if v < 1e-9),
              "problems": problems, "notes": notes, "n_vectors": len(vecs),
              "coverage_tally": cov["tally_calls"], "unmapped": cov["distinct_names"]["unmapped"],
              "build_seconds": round(time.time() - T0, 2)}
    with open(os.path.join(HERE, "out", "selfcheck.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k not in ("t1",)}, indent=1))


if __name__ == "__main__":
    main()
