"""B2: mechanics graphs extracted from the engine card classes (not the card text).

One function per engine CLASS (Lonely Poro has two). Each returns the abilities
the class body implements, plus keyword values only the class knows (Hunt /
Level). Stats, keywords and the engine-implemented keyword expansions (Hidden,
Ambush, Accelerate, Assault, Flow, runes) are NOT written here: extract.py reads
them from the class's CardDef and applies vocab_src.KEYWORDS.

Every dict carries "_src" = "file:line[-line]" (or a dict field -> cite) for the
code it came from. Keys starting with "_" are stripped before vectorising, so
citations never reach the vector (and cannot leak names).

Plumbing is not represented: chain resume / suspend / resume_point code,
RIFT_TRACE, finalize markers, fallback target picking.
"""

# ---------------------------------------------------------------- helpers (B's, with _src)

def sel(type="unit", side="any", zone="board", count=1, filters=None, bind=None,
        optional=False, chooser=None, src=None):
    s = {"type": type, "side": side, "zone": zone, "count": count}
    if filters:
        s["filters"] = filters
    if bind:
        s["bind"] = bind
    if optional:
        s["optional"] = True
    if chooser:
        s["chooser"] = chooser
    if src:
        s["_src"] = src
    return s


def SELF(src=None):
    d = {"type": "self", "side": "friendly", "zone": "board", "count": 1}
    if src:
        d["_src"] = src
    return d


def YOU(src=None):
    d = {"type": "player", "side": "friendly", "zone": "player", "count": 1}
    if src:
        d["_src"] = src
    return d


def ref(name, src=None):
    d = {"type": "ref", "ref": name}
    if src:
        d["_src"] = src
    return d


def F(f, v=True, neg=False, src=None):
    d = {"f": f, "v": v}
    if neg:
        d["neg"] = True
    if src:
        d["_src"] = src
    return d


def C(c, v=True, neg=False, src=None):
    d = {"c": c, "v": v}
    if neg:
        d["neg"] = True
    if src:
        d["_src"] = src
    return d


def amt(n=None, ref_=None, per=None, mult=None, src=None):
    a = {}
    if n is not None:
        a["n"] = n
    if ref_:
        a["ref"] = ref_
    if per:
        a["per"] = per
    if mult is not None:
        a["mult"] = mult
    if src:
        a["_src"] = src
    return a


def eff(verb, target=None, amount=None, duration=None, optional=False, condition=None,
        src=None, **extra):
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
    if src:
        e["_src"] = src
    return e


def spell(effects, src, condition=None, play_requires=None):
    # timing is filled in by extract.py from the CardDef keywords (Action / Reaction)
    a = {"kind": "spell", "timing": "@timing", "effects": effects, "_src": src}
    if condition:
        a["condition"] = condition
    if play_requires:
        a["play_requires"] = play_requires
    return a


def trig(event, subject, effects, src, condition=None, once=None, cost=None):
    t = {"event": event, "subject": subject, "_src": src}
    if condition:
        t["condition"] = condition
    if once:
        t["once"] = once
    a = {"kind": "triggered", "trigger": t, "effects": effects, "_src": src}
    if cost:
        a["cost"] = cost
    return a


def static(effects, src, condition=None):
    a = {"kind": "static", "effects": effects, "_src": src}
    if condition:
        a["condition"] = condition
    return a


def cost(energy=0, power=0, power_domain=None, exhaust=False, optional=False,
         additional=False, other=None, recycle_self=False, src=None):
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
    if src:
        c["_src"] = src
    return c


TURN = "turn"


def at(rel):
    """Citation factory for one class file."""
    def c(a, b=None):
        return f"{rel}:{a}" if b is None else f"{rel}:{a}-{b}"
    return c


ES = "src/engine/effect_executor.cpp"
CH = "src/cards/card_helpers.h"
TM = "src/engine/trigger_manager.cpp"
GE = "src/engine/game_engine.cpp"

# ---------------------------------------------------------------- classes
# class graph = {"abilities": [...], "kw_values": {kw: (value, cite)}, "flags": [...],
#                "ambush_enemy": cite?}

K = {}   # (card key, def_id) -> class graph


def _alpha_strike():
    c = at("src/cards/spells/0751_alpha_strike.cpp")
    return {"abilities": [spell([
        eff("ChooseTarget", sel("unit", "friendly", bind="A", src=c(20, 22)), src=c(51, 53)),
        eff("DealDamage", sel("unit", "enemy", "battlefield", count="split", bind="B", src=c(72, 75)),
            amt(ref_="might_of:A", src=c(67)), source=ref("A", src=c(76)), split=True,
            src=c(76) + "; " + CH + ":990-1003"),
        eff("GainXP", YOU(c(96)), amt(1, per="unit_killed_by_prev", src=c(80, 88)), src=c(93, 96)),
    ], src=c(41), play_requires=[sel("unit", "enemy", "battlefield", src=c(35, 39))])],
        "flags": ["play_requires (legality gate: an enemy unit at a battlefield must be choosable) "
                  "is a B2 schema extension; not in the text"]}


def _back_off():
    c = at("src/cards/spells/0604_back_off.cpp")
    return {"abilities": [spell([
        eff("Stun", sel("unit", "any", src=c(18, 19)), duration=TURN, src=c(39) + "; " + ES + ":1732"),
        eff("Draw", YOU(c(49)), amt(1, src=c(49)),
            condition=[C("played_from_hand", src=c(46, 48))], src=c(49)),
    ], src=c(24))]}


def _rune():
    return {"abilities": []}   # RuneCard has an empty body; abilities come from the engine


def _brittle_steel():
    c = at("src/cards/spells/0790_brittle_steel.cpp")
    return {"abilities": [spell([
        eff("Kill", sel("gear", "any", src=c(19, 20)), src=c(27, 28)),
    ], src=c(23))], "flow": (c(30), 4, 1)}


def _charm():
    c = at("src/cards/spells/0043_charm.cpp")
    return {"abilities": [spell([
        eff("Move", sel("unit", "enemy", src=c(25, 27)), dest="base_or_other_battlefield",
            src=c(49) + "; dest " + c(40, 42) + ", " + CH + ":1520-1529"),
    ], src=c(30))]}


def _darius():
    c = at("src/cards/units/0027_darius_trifarian.cpp")
    return {"abilities": [trig("play_card", "you", [
        eff("GiveMight", SELF(c(40)), amt(2, src=c(40)), TURN, src=c(40)),
        eff("Ready", SELF(c(41)), src=c(41)),
    ], src=c(20, 25), condition=[C("nth_card_this_turn", 2, src=c(36)),
                                 C("is_token", "subject", neg=True, src=c(35))])],
        "flags": ["condition is_token(subject) neg: the engine excludes tokens; the text does not say"]}


def _decisive_strike():
    c = at("src/cards/spells/0322_decisive_strike.cpp")
    return {"abilities": [spell([
        eff("GiveMight", sel("unit", "friendly", count="all", src=c(21, 23)), amt(2, src=c(24)),
            TURN, src=c(24)),
    ], src=c(19))]}


def _decree_of_focus():
    c = at("src/cards/spells/0827_decree_of_focus.cpp")
    return {"abilities": [spell([
        eff("GiveMight", sel("unit", "friendly", filters=[F("any_of", [
            F("in_combat_with", {"side": "enemy", "type": "unit", "domain": "fury"}, src=c(29, 36)),
            F("chosen_by", {"side": "enemy", "type": "spell", "domain": "fury"}, src=c(37, 48)),
        ], src=c(28, 49))], src=c(50, 62)), amt(4, src=c(68)), TURN, src=c(68)),
    ], src=c(64))]}


def _decree_of_strength():
    c = at("src/cards/spells/0872_decree_of_strength.cpp")
    return {"abilities": [spell([
        eff("Reveal", sel("card", "enemy", "hand", count="all", src=c(21, 22)), src=c(30, 35)),
        eff("Recycle", sel("card", "enemy", "hand", filters=[F("domain", "mind", src=c(37, 38))],
                           src=c(41)), src=c(43, 47)),
    ], src=c(20))]}


def _defy():
    c = at("src/cards/spells/0045_defy.cpp")
    return {"abilities": [spell([
        eff("Counter", sel("spell", "any", "chain", filters=[F("energy_cost_le", 4, src=c(30)),
                                                              F("power_cost_le", 1, src=c(30))],
                           src=c(27, 31) + "; " + c(42, 44)), src=c(47) + "; " + CH + ":395"),
    ], src=c(40))]}


def _disarming_rake():
    c = at("src/cards/units/0355_disarming_rake.cpp")
    return {"abilities": [trig("play", "self", [
        eff("Kill", sel("gear", "any", src=c(20, 28)), optional=True,
            src=c(39) + "; may " + c(31)),
    ], src=c(18))]}


def _discipline():
    c = at("src/cards/spells/0058_discipline.cpp")
    return {"abilities": [spell([
        eff("GiveMight", sel("unit", "any", src=c(18, 19)), amt(2, src=c(40)), TURN, src=c(40)),
        eff("Draw", YOU(c(43)), amt(1, src=c(43)), src=c(42, 43)),
    ], src=c(22))]}


def _emperor_s_dais():
    c = at("src/cards/battlefields/0521_emperor_s_dais.cpp")
    return {"abilities": [trig("conquer", "here", [
        eff("CreateToken", sel("unit", "friendly", "here", src=c(51)), amt(1, src=c(53)),
            token={"type": "unit", "might": 2, "tag": "shurima", "name": "sand_soldier",
                   "_src": c(52, 54)}, src=c(53, 54)),
    ], src=c(19) + "; here " + c(23), cost=cost(energy=1, optional=True, other=[
        eff("Bounce", sel("unit", "friendly", "here", src=c(26, 34)), dest="owner_hand",
            src=c(43, 46) + "; owner's hand " + ES + ":1166-1215")],
        src=c(37, 46))),
    ], "flags": ["token tag from code is 'Shurima', token name 'Sand Soldier' (gate 1: name kept as "
                 "token.name; path B used tag 'sand_soldier' from the token name in the text)"]}


def _en_garde():
    c = at("src/cards/spells/0046_en_garde.cpp")
    return {"abilities": [spell([
        eff("GiveMight", sel("unit", "friendly", bind="A", src=c(17, 19)), amt(1, src=c(27)), TURN,
            src=c(27)),
        eff("GiveMight", ref("A", src=c(42)), amt(1, src=c(42)), TURN,
            condition=[C("only_friendly_unit_at_location", "A", src=c(32, 41))], src=c(41, 43)),
    ], src=c(21))]}


def _ferrous_forerunner():
    c = at("src/cards/units/0344_ferrous_forerunner.cpp")
    return {"abilities": [trig("die", "self", [
        eff("CreateToken", sel("unit", "friendly", "base", src=c(21)), amt(2, src=c(22, 25)),
            token={"type": "unit", "might": 3, "tag": "mech", "_src": c(22, 23)}, src=c(22, 25)),
    ], src=c(18))]}


def _fiora():
    c = at("src/cards/units/0432_fiora_peerless.cpp")
    return {"abilities": [trig("attack_or_defend", "self", [
        eff("GiveMight", SELF(c(45)), amt(ref_="might_of:self", mult=1, src=c(46)), "combat",
            src=c(46) + "; aura " + c(50, 62)),
    ], src=c(20), condition=[C("one_on_one", src=c(25, 37))])],
        "flags": ["'double my Might' is a snapshot: +current Might at resolution, held for the "
                  "combat (not re-doubled if Might changes later)"]}


def _first_mate():
    c = at("src/cards/units/0132_first_mate.cpp")
    return {"abilities": [trig("play", "self", [
        eff("Ready", sel("unit", "any", filters=[F("other", src=c(24))], src=c(23, 25)), src=c(30)),
    ], src=c(21))]}


def _grim_apothecary():
    c = at("src/cards/units/0583_grim_apothecary.cpp")
    return {"abilities": [trig("play", "self", [
        eff("Bounce", sel("unit", "friendly", "battlefield", src=c(23, 26)), optional=True,
            dest="owner_hand", src=c(36) + "; may " + c(27, 29) + "; owner's hand " + ES + ":1166-1215"),
    ], src=c(19))]}


def _irresistible_faefolk():
    c = at("src/cards/units/0674_irresistible_faefolk.cpp")
    return {"abilities": [trig("move_to_battlefield", "self", [
        eff("Move", sel("unit", "enemy", filters=[F("at_self_location", neg=True, src=c(33)),
                                                  F("cant_be_moved_by_enemy", neg=True, src=c(34))],
                        src=c(29, 35)), optional=True, dest="self_location",
            src=c(45) + "; may " + c(36, 38)),
    ], src=c(19))]}


def _kai_sa():
    c = at("src/cards/units/0039_kai_sa_survivor.cpp")
    return {"abilities": [trig("conquer", "self", [eff("Draw", YOU(c(19)), amt(1, src=c(19)), src=c(19))],
                               src=c(17))]}


def _kinkou():
    c = at("src/cards/units/0659_kinkou_initiate.cpp")
    return {"abilities": [trig("play", "self", [
        eff("Draw", YOU(c(27)), amt(1, src=c(27)),
            condition=[C("other_friendly_units_total_might_ge", 5, src=c(19, 26))], src=c(26, 27)),
    ], src=c(17))]}


def _lonely_poro_359():
    c = at("src/cards/units/0359_lonely_poro.cpp")
    return {"abilities": [trig("die", "self", [
        eff("Draw", YOU(c(40)), amt(1, src=c(40)),
            condition=[C("died_alone", src=c(34, 39) + "; " + CH + ":88")], src=c(40)),
    ], src=c(20))]}


def _lonely_poro_777():
    c = at("src/cards/units/0777_lonely_poro.cpp")
    return {"abilities": [trig("die", "self", [
        eff("Draw", YOU(c(33)), amt(1, src=c(33)),
            condition=[C("died_alone", src=c(27, 31) + "; " + CH + ":88")], src=c(33)),
    ], src=c(18))]}


def _loyal_poro():
    c = at("src/cards/units/0718_loyal_poro.cpp")
    return {"abilities": [trig("die", "self", [
        eff("Draw", YOU(c(28)), amt(1, src=c(28)),
            condition=[C("died_alone", neg=True, src=c(25, 26))], src=c(28)),
    ], src=c(18))]}


def _machine_evangel():
    c = at("src/cards/units/0239_machine_evangel.cpp")
    return {"abilities": [trig("die", "self", [
        eff("CreateToken", sel("unit", "friendly", "base", src=c(20)), amt(3, src=c(21)),
            token={"type": "unit", "might": 1, "tag": "recruit", "_src": c(22, 23)}, src=c(21, 23)),
    ], src=c(17))]}


def _master_yi_tempered():
    c = at("src/cards/units/0675_master_yi_tempered.cpp")
    return {"abilities": [
        trig("conquer_or_hold", "self", [eff("GainXP", YOU(c(22)), amt(2, src=c(22)), src=c(22))],
             src=c(20)),
        static([eff("GiveKeyword", SELF(c(29, 30)), keyword="deflect", duration="while", src=c(31, 34)),
                eff("GiveKeyword", SELF(c(29, 30)), keyword="ganking", duration="while", src=c(35, 38))],
               src=c(26), condition=[C("xp_ge", 6, src=c(27))]),
    ], "kw_values": {"hunt": (2, c(22)), "level": (6, c(27))}}


def _wuju():
    c = at("src/cards/legends/0317_wuju_bladesman_starter.cpp")
    return {"abilities": [static([
        eff("GiveMight", sel("unit", "friendly", count="all", filters=[F("defends_alone", src=c(28, 34))],
                             src=c(28, 34)), amt(2, src=c(38)), "while", src=c(35, 39)),
    ], src=c(17))]}


def _moonlight():
    c = at("src/cards/spells/0628_moonlight_affliction.cpp")
    return {"abilities": [spell([
        eff("GiveMight", sel("unit", "any", src=c(21, 22)), amt(-10, src=c(19)), TURN, src=c(18, 19)),
    ], src=c(17))]}


def _nidalee():
    c = at("src/cards/units/0676_nidalee_cat_form.cpp")
    return {"abilities": [trig("win_combat", "self", [eff("Draw", YOU(c(19)), amt(1, src=c(19)), src=c(19))],
                               src=c(17))]}


def _not_so_fast():
    c = at("src/cards/spells/0368_not_so_fast.cpp")
    return {"abilities": [spell([
        eff("Counter", sel(["spell", "ability"], "enemy", "chain", filters=[
            F("chooses", {"side": "friendly", "type": ["unit", "gear"]}, src=c(34, 43))],
            src=c(30, 44) + "; " + c(51, 55)), src=c(58)),
    ], src=c(50))]}


def _noxian_demolitionist():
    c = at("src/cards/units/0867_noxian_demolitionist.cpp")
    return {"abilities": [trig("conquer", "self", [
        eff("Kill", sel("gear", "any", filters=[F("energy_cost_le", "might_of:self", src=c(25, 36))],
                        src=c(24, 40)), optional=True, src=c(47) + "; may " + c(41, 42)),
    ], src=c(21))]}


def _noxus_hopeful():
    c = at("src/cards/units/0012_noxus_hopeful.cpp")
    tgt = dict(SELF(c(18)), zone="hand")   # selfCostReduction: the cost of this card as it is played from hand
    return {"abilities": [static([eff("ReduceCost", tgt, amt(2, src=c(19)), resource="energy",
                                      src=c(18, 19) + "; " + GE + ":9436")],
                                 src=c(18), condition=[C("cards_played_this_turn_ge", 1, src=c(19))])]}


def _onslaught():
    c = at("src/cards/spells/0868_onslaught.cpp")
    return {"abilities": [spell([
        eff("GiveMight", sel("unit", "any", src=c(19, 20)), amt(6, src=c(27)), TURN, src=c(27)),
    ], src=c(23))], "flow": (c(29), 4, 0)}


def _pakaa():
    return {"abilities": []}


def _pit_rookie():
    c = at("src/cards/units/0136_pit_rookie.cpp")
    return {"abilities": [trig("play", "self", [
        eff("Buff", sel("unit", "friendly", filters=[F("other", src=c(26))], src=c(24, 30)),
            amt(1, src=ES + ":1321-1345"),
            condition=[C("has_buff", "target", neg=True, src=ES + ":1328-1342")], src=c(35)),
    ], src=c(22))]}


def _primal_strength():
    c = at("src/cards/spells/0154_primal_strength.cpp")
    return {"abilities": [spell([
        eff("GiveMight", sel("unit", "any", src=c(21, 22)), amt(7, src=c(19)), TURN, src=c(18, 19)),
    ], src=c(17))]}


def _punch_first():
    c = at("src/cards/spells/0419_punch_first.cpp")
    return {"abilities": [spell([
        eff("GiveMight", sel("unit", "any", src=c(18, 19)), amt(5, src=c(26)), TURN, src=c(26)),
    ], src=c(22))]}


def _pyke():
    c = at("src/cards/units/0590_pyke_dockside_butcher.cpp")
    return {"abilities": [
        {"kind": "additional_cost", "cost": cost(power=1, power_domain="fury", optional=True,
                                                  additional=True, src=c(17, 20)),
         "effects": [], "_src": c(17, 20)},
        trig("play", "self", [eff("Ready", SELF(c(39)), src=c(39)),
                              eff("GiveMight", SELF(c(40)), amt(2, src=c(40)), TURN, src=c(40))],
             src=c(21), condition=[C("paid_additional_cost", src=c(26, 32))]),
    ]}


def _rampage():
    c = at("src/cards/spells/0870_rampage.cpp")
    return {"abilities": [
        {"kind": "additional_cost", "cost": cost(power=1, power_domain="body", optional=True,
                                                  additional=True, src=c(29, 36)),
         "effects": [], "_src": c(29, 36)},
        spell([
            eff("ChooseTarget", sel("unit", "friendly", bind="A", src=c(41, 45)), src=c(52, 54)),
            eff("ChooseTarget", sel("unit", "enemy", bind="B", src=c(41, 45)), src=c(52, 54)),
            eff("GiveMight", ref("A", src=c(70)), amt(2, src=c(70)), TURN,
                condition=[C("paid_additional_cost", src=c(61, 67))], src=c(70)),
            eff("DealDamage", ref("B", src=c(76)), amt(ref_="might_of:A", src=c(74)),
                source=SELF(c(76)), src=c(76)),
            eff("DealDamage", ref("A", src=c(77)), amt(ref_="might_of:B", src=c(75)),
                source=SELF(c(77)), src=c(77)),
        ], src=c(50)),
    ], "flags": ["damage source is the spell (ctx.source), not the two units as the text says"]}


def _pridestalker():
    c = at("src/cards/legends/0744_pridestalker.cpp")
    return {"abilities": [trig("play_unit", "you", [
        eff("GiveMight", sel("unit", "any", src=c(24, 25)), amt(1, src=c(30)), TURN, src=c(30)),
    ], src=c(22))]}


def _rengar_trophy_hunter():
    c = at("src/cards/units/0682_rengar_trophy_hunter.cpp")
    return {"abilities": [], "ambush_enemy": c(23) + "; src/engine/play_permissions.cpp:17-21"}


def _ruin_runner():
    c = at("src/cards/units/0427_ruin_runner.cpp")
    return {"abilities": [static([eff("CantBeChosen", SELF(c(18)),
                                      by={"side": "enemy", "type": ["spell", "ability"], "_src": c(18)},
                                      src=c(18) + "; " + GE + ":8084-8085; " + CH + ":712-717")],
                                 src=c(18))]}


def _sabotage():
    c = at("src/cards/spells/0156_sabotage.cpp")
    return {"abilities": [spell([
        eff("Reveal", sel("card", "enemy", "hand", count="all", src=c(20, 21)), src=c(31, 41)),
        eff("Recycle", sel("card", "enemy", "hand", filters=[F("type", "unit", neg=True, src=c(51))],
                           src=c(47, 65)), src=c(78, 88)),
    ], src=c(18))]}


def _scuttle_crab():
    c = at("src/cards/units/0615_scuttle_crab.cpp")
    return {"abilities": [
        trig("play", "self", [eff("Draw", YOU(c(23)), amt(1, src=c(23)), src=c(22, 23))], src=c(18, 22)),
        trig("die", "self", [
            eff("Reveal", sel("card", "enemy", "hand", count="all", src=c(30, 41)), src=c(41, 42)),
            eff("LookAt", sel("card", "enemy", "facedown", count="all", src=c(48, 52)), duration=TURN,
                src=c(46, 52)),
            eff("GainXP", YOU(c(53)), amt(1, src=c(53)), src=c(53)),
        ], src=c(18, 27)),
    ]}


def _seat_of_power():
    c = at("src/cards/battlefields/0531_seat_of_power.cpp")
    return {"abilities": [trig("conquer", "here", [
        eff("Draw", YOU(c(29)), amt(1, per="other_battlefield_controlled", src=c(22, 27)), src=c(28, 29)),
    ], src=c(18))]}


def _star_spring():
    c = at("src/cards/battlefields/0771_star_spring.cpp")
    return {"abilities": [trig("play_unit", "any_player_here", [
        eff("Move", sel("unit", "friendly", "here", filters=[F("other", src=c(97))], src=c(93, 102)),
            optional=True, dest="base", src=c(109) + "; may " + c(103, 104)),
    ], src=c(27) + "; " + TM + ":574-580, 282", condition=[C("is_token", "subject", neg=True, src=c(57))],
        once="per_player_per_turn")],
        "_once_src": c(32, 58)}


def _thrill():
    c = at("src/cards/spells/0745_thrill_of_the_hunt.cpp")
    return {"abilities": [spell([
        eff("Banish", sel("unit", "friendly", bind="A", src=c(19, 20)), src=c(40, 82)),
        eff("PlayIgnoringCost", ref("A", src=c(84)), chooser="owner", dest="any_battlefield",
            dest_filters=[F("blocks_unit_play", neg=True, src=c(53, 56) + "; " + CH + ":1366-1369"),
                          F("units_play_base_only", "owner", neg=True, src=CH + ":1366-1369")],
            as_new_object=True,
            condition=[C("is_token", "A", neg=True, src=c(42, 46))],
            src=c(84) + "; chooser " + c(77, 78) + "; dest " + c(53, 56) + "; new object (reset) " + c(36, 40)),
    ], src=c(22))], "flags": ["a token is banished and not played back (code-explicit, CR 183.1)"]}


def _zhonya():
    c = at("src/cards/gear/0077_zhonya_s_hourglass.cpp")
    return {"abilities": [{
        "kind": "replacement",
        "trigger": {"event": "would_die", "subject": "friendly_unit", "_src": c(17, 28)},
        "effects": [eff("Kill", SELF(c(36)), src=c(36)),
                    eff("Heal", ref("subject", src=c(39)), src=c(39) + "; " + ES + ":1469"),
                    eff("Recall", ref("subject", src=c(39)), exhausted=True,
                        src=c(39) + "; " + ES + ":1470-1471")],
        "_src": c(29, 41)}],
        "flags": ["engine heals the unit (text does not)", "no 'next time' clause in the engine: "
                  "'If a friendly unit would die' (the gear is killed, so it is once in practice)"]}


# ---------------------------------------------------------------- token units (gate 1, B1)
# Standalone graphs for unit tokens, from the createToken call that makes them
# (EffectExecutor::createToken, effect_executor.cpp:2376-2410: super_type Token, is_token,
# name, base Might, tags, keywords; a unit enters exhausted unless enter_ready).

def _token(name, might, tag, call_src):
    cite = call_src + "; " + ES + ":2386-2413"
    return {"stats": {"type": "unit", "supertype": "token", "domains": [], "energy": None, "power": None,
                      "might": might, "tag": tag, "name": name, "is_token": True,
                      "enters_exhausted": True,
                      "_src": {"all": cite, "supertype/is_token": ES + ":2391-2392",
                               "enters_exhausted": ES + ":2412-2413 (enter_ready=false)"}},
            "keywords": [], "abilities": [],
            "_meta": {"def_id": None, "class_file": None, "source": "engine_code (createToken call)",
                      "flags": ["token unit: no card class; graph from the creating createToken call"]}}


TOKENS = {   # vocab key -> graph
    "token:mech": _token("mech", 3, "mech", "src/cards/units/0344_ferrous_forerunner.cpp:20-25"),
    "token:sand_soldier": _token("sand_soldier", 2, "shurima",
                                 "src/cards/battlefields/0521_emperor_s_dais.cpp:50-54"),
}


BUILDERS = {
    ("alpha_strike", "unl-192-219"): _alpha_strike,
    ("back_off", "unl-042-219"): _back_off,
    ("body_rune", "ogn-126-298"): _rune,
    ("brittle_steel", "ven-003-166"): _brittle_steel,
    ("calm_rune", "ogn-042-298"): _rune,
    ("charm", "ogn-043-298"): _charm,
    ("darius_trifarian", "ogn-027-298"): _darius,
    ("decisive_strike", "ogs-024-024"): _decisive_strike,
    ("decree_of_focus", "ven-040-166"): _decree_of_focus,
    ("decree_of_strength", "ven-085-166"): _decree_of_strength,
    ("defy", "ogn-045-298"): _defy,
    ("disarming_rake", "sfd-032-221"): _disarming_rake,
    ("discipline", "ogn-058-298"): _discipline,
    ("emperor_s_dais", "sfd-207-221"): _emperor_s_dais,
    ("en_garde", "ogn-046-298"): _en_garde,
    ("ferrous_forerunner", "sfd-021-221"): _ferrous_forerunner,
    ("fiora_peerless", "sfd-110-221"): _fiora,
    ("first_mate", "ogn-132-298"): _first_mate,
    ("fury_rune", "ogn-007-298"): _rune,
    ("grim_apothecary", "unl-021-219"): _grim_apothecary,
    ("inferna", "unl-002-219"): lambda: {"abilities": []},
    ("irresistible_faefolk", "unl-112-219"): _irresistible_faefolk,
    ("kai_sa_survivor", "ogn-039-298"): _kai_sa,
    ("kinkou_initiate", "unl-097-219"): _kinkou,
    ("lonely_poro", "sfd-036-221"): _lonely_poro_359,
    ("lonely_poro", "unl-221-219"): _lonely_poro_777,
    ("loyal_poro", "unl-156-219"): _loyal_poro,
    ("machine_evangel", "ogn-239-298"): _machine_evangel,
    ("master_yi_tempered", "unl-113-219"): _master_yi_tempered,
    ("master_yi_wuju_bladesman", "ogs-019-024"): _wuju,
    ("moonlight_affliction", "unl-066-219"): _moonlight,
    ("nidalee_cat_form", "unl-114-219"): _nidalee,
    ("not_so_fast", "sfd-045-221"): _not_so_fast,
    ("noxian_demolitionist", "ven-080-166"): _noxian_demolitionist,
    ("noxus_hopeful", "ogn-012-298"): _noxus_hopeful,
    ("onslaught", "ven-081-166"): _onslaught,
    ("pakaa_cub", "ogn-135-298"): _pakaa,
    ("pit_rookie", "ogn-136-298"): _pit_rookie,
    ("primal_strength", "ogn-154-298"): _primal_strength,
    ("punch_first", "sfd-097-221"): _punch_first,
    ("pyke_dockside_butcher", "unl-028-219"): _pyke,
    ("rampage", "ven-083-166"): _rampage,
    ("rengar_pridestalker", "unl-183-219"): _pridestalker,
    ("rengar_trophy_hunter", "unl-120-219"): _rengar_trophy_hunter,
    ("ruin_runner", "sfd-105-221"): _ruin_runner,
    ("sabotage", "ogn-156-298"): _sabotage,
    ("scuttle_crab", "unl-053-219"): _scuttle_crab,
    ("seat_of_power", "sfd-217-221"): _seat_of_power,
    ("star_spring", "unl-215-219"): _star_spring,
    ("thrill_of_the_hunt", "unl-184-219"): _thrill,
    ("zhonya_s_hourglass", "ogn-077-298"): _zhonya,
}

# Per-class extraction confidence (high: every effect is a direct executor call with
# literal arguments; medium: a condition or selector had to be named from a predicate;
# low: text-derived or a mechanic the schema cannot express exactly).
CONFIDENCE = {
    "alpha_strike": ("medium", "split damage + reflexive per-kill XP + play gate; split division not expressible"),
    "darius_trifarian": ("medium", "play-ordinal condition named from playOrdinalThisTurn"),
    "decree_of_focus": ("medium", "two predicates (in combat with / chosen by enemy Fury) named by hand"),
    "fiora_peerless": ("medium", "snapshot 'double' via aura counter; one_on_one named from a count"),
    "not_so_fast": ("medium", "'chooses' predicate named by hand"),
    "star_spring": ("medium", "first-time-per-player bookkeeping read as once=per_player_per_turn"),
    "zhonya_s_hourglass": ("medium", "replacement; healExhaustRecall split into Heal + Recall exhausted"),
    "emperor_s_dais": ("medium", "cost-to-effect structure under costDueNow"),
    "rampage": ("medium", "pair targets + paid flag + simultaneous damage"),
    "thrill_of_the_hunt": ("medium", "owner chooses location; token branch"),
    "scuttle_crab": ("medium", "reveal/look via events, not executor calls"),
    "sabotage": ("medium", "reveal/choice via events and requestChoice"),
    "decree_of_strength": ("medium", "reveal via events"),
    "hush": ("low", "no engine class: text-derived (path B graph)"),
}
