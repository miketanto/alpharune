"""B2 v2: Azir list, part A (11 cards). Graphs from the engine classes (src/cards/...), in the style
of graphs_code.py. Stats, printed keywords and the engine-expanded keywords (Hidden, Accelerate,
Action / Reaction timing) come from the CardDef via extract.build_class."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import graphs_code as GC  # noqa: E402
from graphs_code import sel, SELF, YOU, ref, F, C, amt, eff, spell, trig, static, cost, at, ES, CH, TURN  # noqa: E402,F401


def _soldier(src):
    # the 2-Might Shurima unit token these cards make (same token as graphs_code._emperor_s_dais)
    return {"type": "unit", "might": 2, "tag": "shurima", "name": "sand_soldier", "_src": src}


def _emperor_of_the_sands():
    c = at("src/cards/legends/0512_emperor_of_the_sands.cpp")
    return {"abilities": [
        static([eff("GiveKeyword", sel("unit", "friendly", count="all",
                                       filters=[F("token_name", "sand_soldier", src=c(35, 37))],
                                       src=c(35, 37)),
                    keyword="weaponmaster", duration="while", src=c(38, 41))], src=c(19, 30)),
        {"kind": "activated", "timing": "main",
         "cost": cost(energy=1, exhaust=True, src=c(45, 48)),
         "condition": [C("equipment_played_this_turn_ge", 1, src=c(50, 52))],
         "effects": [eff("CreateToken", sel("unit", "friendly", "base", src=c(54)), amt(1, src=c(56)),
                         token=_soldier(c(56, 57)), src=c(56, 57))],
         "_src": c(45, 59) + "; timing " + GC.GE + ":6664-6671 (no is_action/is_reaction -> main)"},
    ], "flags": ["new term: token_name (filter; the aura matches the token's name, the class says its "
                 "tag Shurima is not used)",
                 "new term: equipment_played_this_turn_ge (condition; 'use only if')",
                 "keyword weaponmaster is granted, not printed"]}


def _azir_sovereign():
    c = at("src/cards/units/0497_azir_sovereign.cpp")
    return {"abilities": [trig("attack", "self", [
        eff("Move", sel("unit", "friendly", count="n",
                        filters=[F("is_token", src=c(34)), F("other", src=c(32)),
                                 F("at_self_location", neg=True, src=c(35))],
                        optional=True, src=c(31, 40) + "; any number " + c(53, 55)),
            optional=True, dest="self_location",
            src=c(63) + "; may " + c(47, 51) + "; dest " + c(28, 30)),
    ], src=c(22))],
        "flags": ["new term: attack (trigger event; TriggerType::WhenIAttack, attack only)",
                  "new term: is_token (selector filter)",
                  "no effect if the unit is not at a battlefield (my_bf empty)"]}


def _ashe_focused():
    c = at("src/cards/units/0731_ashe_focused.cpp")
    return {"abilities": [trig("play", "self", [
        eff("Reveal", sel("card", "enemy", "hand", count="all", src=c(32, 38)), src=c(37, 49)),
        eff("Banish", sel("card", "enemy", "hand", bind="A", src=c(55, 59)), src=c(62)),
        eff("Bounce", ref("A", src=c(80, 89)), dest="owner_hand",
            after={"event": "hold", "subject": "enemy", "once": "next_time", "_src": c(66, 73)},
            src=c(79, 96)),
    ], src=c(26))],
        "flags": ["new term: hold (event of a delayed effect: the opponent's next hold)",
                  "the delayed return outlives the unit (DelayedAbility kUntilFired, source-independent)",
                  "return is from banishment to the owner's hand (deck::putInHand, not an executor "
                  "call); written as Bounce dest owner_hand"]}


def _trifarian_war_camp():
    c = at("src/cards/battlefields/0289_trifarian_war_camp.cpp")
    return {"abilities": [static([
        eff("GiveMight", sel("unit", "any", "here", count="all", src=c(26, 29)), amt(1, src=c(33)),
            "while", src=c(31, 34)),
    ], src=c(19, 37))]}


def _hall_of_legends():
    c = at("src/cards/battlefields/0524_hall_of_legends.cpp")
    return {"abilities": [trig("conquer", "here", [
        eff("Ready", sel("legend", "friendly", "legend", src=c(41, 42)), src=c(43)),
    ], src=c(19), cost=cost(energy=1, optional=True, src=c(34, 39)))],
        "flags": ["new term: legend (selector type and zone: your legend)",
                  "the [1] is offered even when the legend is already ready (class comment)"]}


def _guards():
    c = at("src/cards/spells/0475_guards.cpp")
    return {"abilities": [spell([
        eff("CreateToken", sel("unit", "friendly", "base_or_controlled_battlefield", bind="T",
                               src=c(31, 48)),
            amt(1, src=c(59)), token=_soldier(c(59, 61)), src=c(59, 61)),
        eff("Ready", ref("T", src=c(79, 80)), optional=True,
            cost=cost(power=1, power_domain="order", optional=True, src=c(94, 100)),
            src=c(105) + "; reflexive " + c(79, 80) + ", " + c(85, 107)),
    ], src=c(27))],
        "flags": ["new term: base_or_controlled_battlefield (zone: where a unit may be played; "
                  "only the hidden battlefield when played from facedown, CR 811.1.d.5)",
                  "effect-level cost: the [C] of the reflexive 'then you may pay to ready it'"]}


def _arise():
    c = at("src/cards/spells/0513_arise.cpp")
    return {"abilities": [spell([
        eff("CreateToken", sel("unit", "friendly", "base_or_controlled_battlefield", bind="T",
                               src=c(44, 55)),
            amt(1, per="equipment_controlled", src=c(37, 42)), token=_soldier(c(48, 50)),
            src=c(46, 55)),
        eff("Ready", sel("unit", "friendly", count=2, optional=True,
                         filters=[F("bound", "T", src=c(59, 61))], src=c(67, 76)),
            src=c(79, 80) + "; reflexive " + c(61) + ", " + c(65, 81)),
    ], src=c(31))],
        "flags": ["new term: equipment_controlled (amount per: friendly gear tagged Equipment, "
                  "counted once at resolution)",
                  "new term: bound (filter: one of the objects bound earlier, here the tokens made)",
                  "new term: base_or_controlled_battlefield (zone; each token's location chosen "
                  "one by one)"]}


def _decree_of_unity():
    c = at("src/cards/spells/0918_decree_of_unity.cpp")
    return {"abilities": [spell([
        eff("Kill", sel(["unit", "gear"], "enemy", filters=[F("domain", "chaos", src=c(31, 32))],
                        src=c(20, 37)), src=c(45) + "; " + ES + ":556"),
    ], src=c(38))]}


def _salvage():
    c = at("src/cards/spells/0224_salvage.cpp")
    return {"abilities": [spell([
        eff("Kill", sel("gear", "any", optional=True, src=c(24, 37)), optional=True, src=c(19, 20)),
        eff("Draw", YOU(c(21)), amt(1, src=c(21)), src=c(21)),
    ], src=c(18))]}


def _hidden_blade():
    c = at("src/cards/spells/0213_hidden_blade.cpp")
    return {"abilities": [spell([
        eff("Kill", sel("unit", "any", "battlefield", bind="A", src=c(18, 21)), src=c(42)),
        eff("Draw", {"type": "player", "side": "any", "zone": "player", "count": 1,
                     "filters": [F("controller_of", "A", src=c(48) + "; " + CH + ":76")],
                     "_src": c(48)},
            amt(2, src=c(53)), src=c(49, 54)),
    ], src=c(29))],
        "flags": ["new term: controller_of (player filter: who controlled the killed unit as it "
                  "died; nobody draws if the death was replaced by a banish)",
                  "the card's own name contains the keyword word 'hidden' (keyword expansion, "
                  "unavoidable name-check hit)"]}


def _deathgrip():
    c = at("src/cards/spells/0484_deathgrip.cpp")
    return {"abilities": [spell([
        eff("ChooseTarget", sel("unit", "friendly", bind="A", src=c(20, 23)), src=c(42)),
        eff("ChooseTarget", sel("unit", "friendly", filters=[F("other", src=c(35))], bind="B",
                                optional=True, src=c(29, 38)), src=c(43)),
        eff("Kill", ref("A", src=c(47)), src=c(47)),
        eff("GiveMight", ref("B", src=c(50)), amt(ref_="might_of:A", src=c(46)), TURN,
            condition=[C("died", "A", src=c(48, 49))], src=c(50)),
        eff("Draw", YOU(c(52)), amt(1, src=c(52)), src=c(52)),
    ], src=c(39))],
        "flags": ["new term: died (condition: the bound unit actually left the board)",
                  "Might is read before the kill (last-known); playable with a single friendly unit "
                  "(then no pump)"]}


BUILDERS = {
    ("emperor_of_the_sands", "sfd-197-221"): _emperor_of_the_sands,
    ("azir_sovereign", "sfd-177-221"): _azir_sovereign,
    ("ashe_focused", "unl-169-219"): _ashe_focused,
    ("trifarian_war_camp", "ogn-294-298"): _trifarian_war_camp,
    ("hall_of_legends", "sfd-210-221"): _hall_of_legends,
    ("guards", "sfd-154-221"): _guards,
    ("arise", "sfd-198-221"): _arise,
    ("decree_of_unity", "ven-131-166"): _decree_of_unity,
    ("salvage", "ogn-224-298"): _salvage,
    ("hidden_blade", "ogn-213-298"): _hidden_blade,
    ("deathgrip", "sfd-163-221"): _deathgrip,
}
