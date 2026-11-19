"""Hand-authored mechanics graphs (path B) for the 51 cards in ../cards.json.

Source of truth for keywords + abilities. Stats (type, supertype, domains,
energy, power, might) are NOT written here: build.py derives them from each
printing's data in cards.json (or LEGEND_STATS for the two legends), so every
printing is rebuilt from its own fields. Names, printings, tags of the card
itself and flavour never appear. Schema: schema.md.

Authored by reading the ability text in cards.json, cross-checked against the
engine card class (src/cards/<type>/NNNN_<name>.cpp). Disagreements are listed
in ENGINE_NOTES and in out/report.md.
"""

# ---------------------------------------------------------------- helpers

def sel(type="unit", side="any", zone="board", count=1, filters=None,
        bind=None, optional=False, chooser=None):
    """Target selector."""
    s = {"type": type, "side": side, "zone": zone, "count": count}
    if filters:
        s["filters"] = filters
    if bind:
        s["bind"] = bind          # local label so later effects can refer to it
    if optional:
        s["optional"] = True
    if chooser:
        s["chooser"] = chooser    # default: the ability's controller
    return s


SELF = {"type": "self", "side": "friendly", "zone": "board", "count": 1}
YOU = {"type": "player", "side": "friendly", "zone": "player", "count": 1}
OPP = {"type": "player", "side": "enemy", "zone": "player", "count": 1}


def ref(name):
    """Refer to an object bound earlier in the same ability."""
    return {"type": "ref", "ref": name}


def F(f, v=True, neg=False):
    d = {"f": f, "v": v}
    if neg:
        d["neg"] = True
    return d


def amt(n=None, ref_=None, per=None, mult=None):
    a = {}
    if n is not None:
        a["n"] = n
    if ref_:
        a["ref"] = ref_
    if per:
        a["per"] = per
    if mult is not None:
        a["mult"] = mult
    return a


def eff(verb, target=None, amount=None, duration=None, optional=False,
        condition=None, **extra):
    e = {"verb": verb}
    if target is not None:
        e["target"] = target
    if amount is not None:
        e["amount"] = amount
    if duration:
        e["duration"] = duration
    if optional:
        e["optional"] = True
    if condition:
        e["condition"] = condition
    e.update(extra)
    return e


def C(c, v=True, neg=False):
    """Condition."""
    d = {"c": c, "v": v}
    if neg:
        d["neg"] = True
    return d


def spell(timing, effects, condition=None):
    a = {"kind": "spell", "timing": timing, "effects": effects}
    if condition:
        a["condition"] = condition
    return a


def trig(event, effects, subject=None, condition=None, once=None, cost=None):
    t = {"event": event}
    if subject:
        t["subject"] = subject
    if condition:
        t["condition"] = condition
    if once:
        t["once"] = once
    a = {"kind": "triggered", "trigger": t, "effects": effects}
    if cost:
        a["cost"] = cost
    return a


def static(effects, condition=None):
    a = {"kind": "static", "effects": effects}
    if condition:
        a["condition"] = condition
    return a


def cost(energy=0, power=0, power_domain=None, exhaust=False, optional=False,
         additional=False, other=None, recycle_self=False):
    c = {}
    if energy:
        c["energy"] = energy
    if power:
        c["power"] = power
        c["power_domain"] = power_domain or "card"
    if exhaust:
        c["exhaust"] = True
    if recycle_self:
        c["recycle_self"] = True
    if optional:
        c["optional"] = True
    if additional:
        c["additional"] = True
    if other:
        c["other"] = other
    return c


# keyword expansions (keyword list keeps the keyword; these add its mechanics)

def KW(name, value=None):
    k = {"kw": name}
    if value is not None:
        k["value"] = value
    return k


HIDDEN_MODE = {   # [Hidden]: hide now for [A], play later from facedown for [0] as a Reaction
    "kind": "play_mode", "mode": "from_facedown", "timing": "reaction",
    "setup_cost": cost(power=1, power_domain="any"),
    "cost": {"energy": 0, "power": 0},
    "where": {"zone": "battlefield", "filters": [F("controlled_by_you")]},
}
AMBUSH_MODE = {   # [Ambush]: play as a Reaction to a battlefield where you have units
    "kind": "play_mode", "mode": "ambush", "timing": "reaction",
    "where": {"zone": "battlefield", "filters": [F("has_friendly_units")]},
}


def FLOW_MODE(energy, power=0):
    """[Flow] cost: may play from trash for this cost, then banish it."""
    return {"kind": "play_mode", "mode": "from_trash", "timing": "same",
            "cost": cost(energy=energy, power=power),
            "effects": [eff("Banish", SELF, after="resolve")]}


def DEATHKNELL(effects, condition=None):
    return trig("die", effects, subject="self", condition=condition)


RUNE_ABILITIES = [
    {"kind": "activated", "timing": "reaction", "cost": cost(exhaust=True),
     "effects": [eff("AddEnergy", YOU, amt(1))]},
    {"kind": "activated", "timing": "reaction", "cost": cost(recycle_self=True),
     "effects": [eff("AddPower", YOU, amt(1), domain="card")]},
]

TURN = "turn"

# ---------------------------------------------------------------- graphs
# key -> {"keywords": [...], "abilities": [...]}

G = {}

G["alpha_strike"] = {"keywords": [KW("action")], "abilities": [spell("action", [
    eff("ChooseTarget", sel("unit", "friendly", bind="A")),
    eff("DealDamage", sel("unit", "enemy", "battlefield", count="split", bind="B"),
        amt(ref_="might_of:A"), source=ref("A"), split=True),
    eff("GainXP", YOU, amt(1, per="unit_killed_by_prev")),
])]}

G["back_off"] = {"keywords": [KW("hidden"), KW("action")], "abilities": [
    HIDDEN_MODE,
    spell("action", [
        eff("Stun", sel("unit", "any"), duration=TURN),
        eff("Draw", YOU, amt(1), condition=[C("played_from_hand")]),
    ])]}

for _r in ("body_rune", "calm_rune", "fury_rune"):
    G[_r] = {"keywords": [], "abilities": RUNE_ABILITIES}

G["brittle_steel"] = {"keywords": [KW("flow")], "abilities": [
    spell("main", [eff("Kill", sel("gear", "any"))]),
    FLOW_MODE(4, 1),
]}

G["charm"] = {"keywords": [], "abilities": [spell("main", [
    eff("Move", sel("unit", "enemy"), dest="base_or_other_battlefield"),
])]}

G["darius_trifarian"] = {"keywords": [], "abilities": [
    trig("play_card", [
        eff("GiveMight", SELF, amt(2), TURN),
        eff("Ready", SELF),
    ], subject="you", condition=[C("nth_card_this_turn", 2)]),
]}

G["decisive_strike"] = {"keywords": [KW("action")], "abilities": [spell("action", [
    eff("GiveMight", sel("unit", "friendly", count="all"), amt(2), TURN),
])]}

G["decree_of_focus"] = {"keywords": [KW("reaction")], "abilities": [spell("reaction", [
    eff("GiveMight", sel("unit", "friendly", filters=[F("any_of", [
        F("in_combat_with", {"side": "enemy", "type": "unit", "domain": "fury"}),
        F("chosen_by", {"side": "enemy", "type": "spell", "domain": "fury"}),
    ])]), amt(4), TURN),
])]}

G["decree_of_strength"] = {"keywords": [], "abilities": [spell("main", [
    eff("Reveal", sel("card", "enemy", "hand", count="all")),
    eff("Recycle", sel("card", "enemy", "hand", filters=[F("domain", "mind")])),
])]}

G["defy"] = {"keywords": [KW("reaction")], "abilities": [spell("reaction", [
    eff("Counter", sel("spell", "any", "chain",
                       filters=[F("energy_cost_le", 4), F("power_cost_le", 1)])),
])]}

G["disarming_rake"] = {"keywords": [], "abilities": [
    trig("play", [eff("Kill", sel("gear", "any"), optional=True)], subject="self"),
]}

G["discipline"] = {"keywords": [KW("reaction")], "abilities": [spell("reaction", [
    eff("GiveMight", sel("unit", "any"), amt(2), TURN),
    eff("Draw", YOU, amt(1)),
])]}

G["emperor_s_dais"] = {"keywords": [], "abilities": [
    trig("conquer", [
        eff("CreateToken", sel("unit", "friendly", "here"), amt(1),
            token={"type": "unit", "might": 2, "tag": "sand_soldier"}),
    ], subject="here",
        cost=cost(energy=1, optional=True, other=[
            eff("Bounce", sel("unit", "friendly", "here"))])),
]}

G["en_garde"] = {"keywords": [KW("reaction")], "abilities": [spell("reaction", [
    eff("GiveMight", sel("unit", "friendly", bind="A"), amt(1), TURN),
    eff("GiveMight", ref("A"), amt(1), TURN,
        condition=[C("only_friendly_unit_at_location", "A")]),
])]}

G["ferrous_forerunner"] = {"keywords": [KW("deathknell")], "abilities": [
    DEATHKNELL([eff("CreateToken", sel("unit", "friendly", "base"), amt(2),
                    token={"type": "unit", "might": 3, "tag": "mech"})]),
]}

G["fiora_peerless"] = {"keywords": [], "abilities": [
    trig("attack_or_defend", [
        eff("GiveMight", SELF, amt(ref_="might_of:self", mult=1), "combat"),
    ], subject="self", condition=[C("one_on_one")]),
]}

G["first_mate"] = {"keywords": [], "abilities": [
    trig("play", [eff("Ready", sel("unit", "any", filters=[F("other")]))], subject="self"),
]}

G["grim_apothecary"] = {"keywords": [KW("ambush")], "abilities": [
    AMBUSH_MODE,
    trig("play", [eff("Bounce", sel("unit", "friendly", "battlefield"), optional=True)],
         subject="self"),
]}

G["hush"] = {"keywords": [KW("reaction")], "abilities": [spell("reaction", [
    eff("Counter", sel("spell", "any", "chain", filters=[F("power_cost_ge", 2)])),
])]}

G["inferna"] = {"keywords": [KW("ambush"), KW("assault", 2)], "abilities": [
    AMBUSH_MODE,
    static([eff("GiveMight", SELF, amt(2), "while")], condition=[C("is_attacker")]),
]}

G["irresistible_faefolk"] = {"keywords": [], "abilities": [
    trig("move_to_battlefield", [
        eff("Move", sel("unit", "enemy"), optional=True, dest="self_location"),
    ], subject="self"),
]}

G["kai_sa_survivor"] = {"keywords": [KW("accelerate")], "abilities": [
    {"kind": "additional_cost", "cost": cost(energy=1, power=1, optional=True, additional=True),
     "effects": [eff("EnterReady", SELF)]},
    trig("conquer", [eff("Draw", YOU, amt(1))], subject="self"),
]}

G["kinkou_initiate"] = {"keywords": [], "abilities": [
    trig("play", [eff("Draw", YOU, amt(1),
                      condition=[C("other_friendly_units_total_might_ge", 5)])],
         subject="self"),
]}

G["lonely_poro"] = {"keywords": [KW("deathknell")], "abilities": [
    DEATHKNELL([eff("Draw", YOU, amt(1), condition=[C("died_alone")])]),
]}

G["loyal_poro"] = {"keywords": [KW("deathknell")], "abilities": [
    DEATHKNELL([eff("Draw", YOU, amt(1), condition=[C("died_alone", neg=True)])]),
]}

G["machine_evangel"] = {"keywords": [KW("deathknell")], "abilities": [
    DEATHKNELL([eff("CreateToken", sel("unit", "friendly", "base"), amt(3),
                    token={"type": "unit", "might": 1, "tag": "recruit"})]),
]}

G["master_yi_tempered"] = {"keywords": [KW("hunt", 2), KW("level", 6)], "abilities": [
    trig("conquer_or_hold", [eff("GainXP", YOU, amt(2))], subject="self"),
    static([eff("GiveKeyword", SELF, keyword="deflect", duration="while"),
            eff("GiveKeyword", SELF, keyword="ganking", duration="while")],
           condition=[C("xp_ge", 6)]),
]}

G["master_yi_wuju_bladesman"] = {"keywords": [], "abilities": [
    static([eff("GiveMight", sel("unit", "friendly", count="all",
                                 filters=[F("defends_alone")]), amt(2), "while")]),
]}

G["moonlight_affliction"] = {"keywords": [KW("reaction")], "abilities": [spell("reaction", [
    eff("GiveMight", sel("unit", "any"), amt(-10), TURN),
])]}

G["nidalee_cat_form"] = {"keywords": [KW("ambush")], "abilities": [
    AMBUSH_MODE,
    trig("win_combat", [eff("Draw", YOU, amt(1))], subject="self"),
]}

G["not_so_fast"] = {"keywords": [KW("reaction")], "abilities": [spell("reaction", [
    eff("Counter", sel(["spell", "ability"], "enemy", "chain", filters=[
        F("chooses", {"side": "friendly", "type": ["unit", "gear"]})])),
])]}

G["noxian_demolitionist"] = {"keywords": [], "abilities": [
    trig("conquer", [eff("Kill", sel("gear", "any", filters=[
        F("energy_cost_le", "might_of:self")]), optional=True)], subject="self"),
]}

G["noxus_hopeful"] = {"keywords": [KW("legion")], "abilities": [
    static([eff("ReduceCost", SELF, amt(2), resource="energy")],
           condition=[C("cards_played_this_turn_ge", 1)]),
]}

G["onslaught"] = {"keywords": [KW("flow")], "abilities": [
    spell("main", [eff("GiveMight", sel("unit", "any"), amt(6), TURN)]),
    FLOW_MODE(4),
]}

G["pakaa_cub"] = {"keywords": [KW("hidden")], "abilities": [HIDDEN_MODE]}

G["pit_rookie"] = {"keywords": [], "abilities": [
    trig("play", [eff("Buff", sel("unit", "friendly", filters=[F("other")]), amt(1))],
         subject="self"),
]}

G["primal_strength"] = {"keywords": [KW("action")], "abilities": [spell("action", [
    eff("GiveMight", sel("unit", "any"), amt(7), TURN),
])]}

G["punch_first"] = {"keywords": [KW("action")], "abilities": [spell("action", [
    eff("GiveMight", sel("unit", "any"), amt(5), TURN),
])]}

G["pyke_dockside_butcher"] = {"keywords": [KW("hidden"), KW("ganking")], "abilities": [
    HIDDEN_MODE,
    {"kind": "additional_cost", "cost": cost(power=1, power_domain="fury", optional=True,
                                             additional=True), "effects": []},
    trig("play", [eff("Ready", SELF), eff("GiveMight", SELF, amt(2), TURN)],
         subject="self", condition=[C("paid_additional_cost")]),
]}

G["rampage"] = {"keywords": [], "abilities": [
    {"kind": "additional_cost", "cost": cost(power=1, power_domain="body", optional=True,
                                             additional=True), "effects": []},
    spell("main", [
        eff("ChooseTarget", sel("unit", "friendly", bind="A")),
        eff("ChooseTarget", sel("unit", "enemy", bind="B")),
        eff("GiveMight", ref("A"), amt(2), TURN, condition=[C("paid_additional_cost")]),
        eff("DealDamage", ref("B"), amt(ref_="might_of:A"), source=ref("A")),
        eff("DealDamage", ref("A"), amt(ref_="might_of:B"), source=ref("B")),
    ]),
]}

G["rengar_pridestalker"] = {"keywords": [], "abilities": [
    trig("play_unit", [eff("GiveMight", sel("unit", "any"), amt(1), TURN)], subject="you"),
]}

# Erratum text (errata/unleased.txt; CR 822.1.d): "I can [Ambush] to a battlefield where
# there are enemy units, even if you don't have units there." -> widens Ambush's location.
G["rengar_trophy_hunter"] = {"keywords": [KW("ambush")], "abilities": [
    {"kind": "play_mode", "mode": "ambush", "timing": "reaction",
     "where": {"zone": "battlefield",
               "filters": [{"f": "any_of", "v": [F("has_friendly_units"), F("has_enemy_units")]}]}},
]}

G["ruin_runner"] = {"keywords": [], "abilities": [
    static([eff("CantBeChosen", SELF, by={"side": "enemy", "type": ["spell", "ability"]})]),
]}

G["sabotage"] = {"keywords": [], "abilities": [spell("main", [
    eff("Reveal", sel("card", "enemy", "hand", count="all")),
    eff("Recycle", sel("card", "enemy", "hand", filters=[F("type", "unit", neg=True)])),
])]}

G["scuttle_crab"] = {"keywords": [KW("deathknell")], "abilities": [
    trig("play", [eff("Draw", YOU, amt(1))], subject="self"),
    DEATHKNELL([
        eff("Reveal", sel("card", "enemy", "hand", count="all")),
        eff("LookAt", sel("card", "enemy", "facedown", count="all"), duration=TURN),
        eff("GainXP", YOU, amt(1)),
    ]),
]}

G["seat_of_power"] = {"keywords": [], "abilities": [
    trig("conquer", [eff("Draw", YOU, amt(1, per="other_battlefield_controlled"))],
         subject="here"),
]}

G["star_spring"] = {"keywords": [], "abilities": [
    trig("play_unit", [
        eff("Move", sel("unit", "friendly", "here", filters=[F("other")]), optional=True,
            dest="base"),
    ], subject="any_player_here", condition=[C("unit_is_token", neg=True)],
        once="per_player_per_turn"),
]}

G["thrill_of_the_hunt"] = {"keywords": [KW("reaction")], "abilities": [spell("reaction", [
    eff("Banish", sel("unit", "friendly", bind="A")),
    eff("PlayIgnoringCost", ref("A"), chooser="owner", dest="any_battlefield"),
])]}

G["zhonya_s_hourglass"] = {"keywords": [KW("hidden")], "abilities": [
    HIDDEN_MODE,
    # Erratum text: "If a friendly unit would die, kill this instead. Heal that unit,
    # exhaust it, and recall it."
    {"kind": "replacement", "trigger": {"event": "would_die", "subject": "friendly_unit"},
     "effects": [eff("Kill", SELF), eff("Heal", ref("subject")),
                 eff("Recall", ref("subject"), exhausted=True)]},
]}

# Legends are not in card_index.json; stats from the engine CardDef.
LEGEND_STATS = {
    "rengar_pridestalker": {"type": "legend", "supertype": None,
                            "domains": ["fury", "body"], "energy": None, "power": None,
                            "might": None},
    "master_yi_wuju_bladesman": {"type": "legend", "supertype": None,
                                 "domains": ["calm", "body"], "energy": None, "power": None,
                                 "might": None},
}

# Text vs engine disagreements found while authoring (also in report.md).
ENGINE_NOTES = {
    "emperor_s_dais": "card_index text 'pay [1] and return a unit ... If you do, play' vs engine "
                      "def text '... to play'; engine treats pay [1] + return as the cost of the "
                      "token. Same mechanics; graph models pay+return as an optional cost.",
    "zhonya_s_hourglass": "card_index: 'The next time a friendly unit would die, kill this "
                          "instead. Recall that unit exhausted.' Engine def text and code: "
                          "'If a friendly unit would die ... Heal that unit, exhaust it, and "
                          "recall it' (adds Heal). Graph follows card_index (no Heal).",
    "rengar_trophy_hunter": "Text: 'I can be played to a battlefield where there are enemy "
                            "units'. Engine only relaxes the Ambush location gate "
                            "(ambushToEnemyBattlefields) and its def text says 'I can [Ambush] "
                            "to ...'. card_index holds pre-erratum text; graph now follows the erratum "
                            "(CR 822.1.d): Ambush location widened to enemy-occupied battlefields.",
    "hush": "No engine card class exists (not implemented). Graph from text only.",
    "seat_of_power": "Text 'you or allies control'; engine counts only battlefields the "
                     "controller controls (1v1, no allies). Graph: other_battlefield_controlled.",
    "back_off": "Engine def text drops the '[0]' in the Hidden reminder (cosmetic). Also "
                "pakaa_cub, pyke_dockside_butcher.",
    "first_mate": "'ready another unit' - engine allows any controller's unit (not only "
                  "friendly). Graph side=any, matches.",
    "darius_trifarian": "Engine checks play ordinal == 2 and excludes tokens; the graph "
                        "condition nth_card_this_turn=2 does not express the token exclusion.",
    "alpha_strike": "Engine requires an enemy unit at a battlefield to be choosable to play "
                    "it (legality), not stated in text. Not represented.",
    "master_yi_wuju_bladesman": "Not in card_index; engine def text used. Engine applies +2 "
                                "only when exactly one friendly defender exists.",
    "rengar_pridestalker": "Not in card_index; engine def text used. Engine: any unit on "
                           "board, chosen as trigger finalizes.",
}
