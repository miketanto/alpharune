"""B2 v2: Diana list, part C (legend, champions, battlefields and support units).

Graphs from the engine classes (src/cards/...), same helpers and conventions as graphs_code.py.
Predict (EffectExecutor::predict, effect_executor.cpp:2755) is written as what it does: a private
LookAt of the top card of the Main Deck, then an optional Recycle of it (CR 437.1).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import graphs_code as GC  # noqa: E402
from graphs_code import sel, SELF, YOU, ref, F, C, amt, eff, trig, cost, at, ES, TURN  # noqa: E402

import scan  # noqa: E402

# scan.KW_MAP (CardDef keyword -> schema kw) has no entry for Vendetta's [Empower]; registered here
# so parse_carddef accepts tail_cloaked_matriarch (keyword only: its mechanics are in the builder).

TM = "src/engine/trigger_manager.cpp"


def _scorn():
    c = at("src/cards/legends/0786_scorn_of_the_moon.cpp")
    return {"abilities": [{
        "kind": "activated", "timing": "reaction",
        "cost": cost(exhaust=True, src=c(22)),
        "effects": [eff("AddEnergy", YOU(c(26)), amt(1, src=c(26)), restriction="showdown_only",
                        src=c(25, 26))],
        "_src": c(17, 28) + "; reaction " + c(21) + "; add (no chain) " + c(18, 20)}],
        "flags": ["new term: restriction=showdown_only (energy goes to RunePool::showdown_energy)"]}


def _diana():
    c = at("src/cards/units/0641_diana_lunari.cpp")
    top = lambda s, **kw: sel("card", "friendly", "deck_top", src=s, **kw)
    return {"abilities": [trig("showdown_begin", "here", [
        eff("LookAt", top(c(40)), src=c(40) + "; " + ES + ":2755-2790 (predict: private look)"),
        eff("Recycle", top(c(40)), optional=True, src=c(40) + "; " + ES + ":2796-2800 (predict: may recycle)"),
        eff("Reveal", top(c(48), bind="R"), src=c(45, 58)),
        eff("Draw", YOU(c(60)), amt(1, src=c(60)),
            condition=[C("revealed_card_type", "spell", src=c(59))], src=c(59, 60)),
    ], src=c(21) + "; " + TM + ":2070-2081",
        cost=cost(energy=1, optional=True, src=c(27, 36)))],
        "flags": ["new term: event showdown_begin (subject here = the unit's battlefield)",
                  "new term: zone deck_top", "new term: condition revealed_card_type",
                  "Predict expressed as LookAt + optional Recycle of the top card"]}


def _ravenbloom():
    c = at("src/cards/units/0103_ravenbloom_student.cpp")
    return {"abilities": [trig("play_spell", "you", [
        eff("GiveMight", SELF(c(26)), amt(1, src=c(26)), TURN, src=c(25, 26)),
    ], src=c(23))], "flags": ["new term: event play_spell"]}


def _merchant():
    c = at("src/cards/units/0185_traveling_merchant.cpp")
    return {"abilities": [trig("move", "self", [
        eff("Discard", sel("card", "friendly", "hand", src=c(20)), amt(1, src=c(20)), src=c(20)),
        eff("Draw", YOU(c(21)), amt(1, src=c(21)), src=c(21)),
    ], src=c(23))], "flags": ["new term: event move", "new term: verb Discard"]}


def _tideturner():
    c = at("src/cards/units/0199_tideturner.cpp")
    return {"abilities": [trig("play", "self", [
        eff("ChooseTarget", sel("unit", "friendly", filters=[F("other", src=c(32)),
                                                             F("at_self_location", neg=True, src=c(33))],
                                bind="A", src=c(27, 38)), optional=True,
            src=c(42) + "; may " + c(39, 40)),
        eff("Move", SELF(c(47)), dest="location_of:A", src=c(45, 47)),
        eff("Move", ref("A", src=c(48)), dest="self_location", src=c(45, 48)),
    ], src=c(23))], "flags": ["new term: dest location_of:A (swap: self to A's location, A to self's "
                              "original location)"]}


def _hwei():
    c = at("src/cards/units/0642_hwei_brooding_painter.cpp")
    return {"abilities": [trig("move", "self", [
        eff("Draw", YOU(c(39)), amt(1, src=c(39)), src=c(39)),
        eff("Discard", sel("card", "friendly", "hand", bind="D", src=c(24, 34)), amt(1, src=c(48, 55)),
            src=c(42, 55)),
        eff("Draw", YOU(c(113)), amt(1, src=c(113)),
            condition=[C("discarded_card_type", "spell", src=c(62, 64))], src=c(112, 113)),
        eff("Ready", sel("rune", "any", count=2, optional=True, filters=[F("exhausted", src=c(88))],
                         src=c(85, 94)),
            condition=[C("discarded_card_type", "gear", src=c(62, 64))], src=c(78, 106)),
        eff("GiveMight", SELF(c(119)), amt(3, src=c(119)), TURN,
            condition=[C("discarded_card_type", "unit", src=c(62, 64))], src=c(115, 119)),
    ], src=c(18) + "; reflexive " + c(62, 64) + ", " + c(74, 123))],
        "flags": ["new term: event move", "new term: verb Discard", "new term: condition discarded_card_type",
                  "new term: selector type rune, filter exhausted",
                  "the type-based part is a reflexive trigger (own chain item) folded into the ability"]}


def _vex():
    c = at("src/cards/units/0712_vex_apathetic.cpp")
    return {"abilities": [trig("play_unit", "opponent", [
        eff("Stun", ref("subject", src=c(34, 35)), duration=TURN, src=c(41) + "; " + ES + ":1732"),
        eff("CantMove", ref("subject", src=c(42)), duration=TURN, src=c(42)),
    ], src=c(24) + "; " + TM + ":589-593",
        condition=[C("self_at_battlefield", src=c(29, 32))])],
        "kw_values": {"deflect": (1, c(63))},
        "flags": ["new term: trigger subject opponent", "new term: condition self_at_battlefield",
                  "new term: verb CantMove (cant_move_this_turn: no Standard Move, no effect move by "
                  "its controller)"]}


def _fizz():
    c = at("src/cards/units/0461_fizz_trickster.cpp")
    return {"abilities": [trig("play", "self", [
        eff("PlayIgnoringCost", sel("spell", "friendly", "trash",
                                    filters=[F("energy_cost_le", 3, src=c(34))], bind="A", src=c(27, 39)),
            optional=True, resource="energy", src=c(53) + "; may " + c(40, 45) + "; energy only " + c(24)),
        eff("Recycle", ref("A", src=c(52)), after="leave_chain", src=c(52, 62)),
    ], src=c(19))],
        "flags": ["new term: zone trash", "PlayIgnoringCost resource=energy: only the Energy cost is "
                  "ignored, the Power cost is paid (playPayingCost with ignore_energy)",
                  "new term: after=leave_chain"]}


def _matriarch():
    c = at("src/cards/units/0891_tail_cloaked_matriarch.cpp")
    return {"abilities": [
        {"kind": "activated", "timing": "main",
         "cost": cost(energy=2, power=1, power_domain="chaos", src=c(21)),
         "condition": [C("empowered", neg=True, src=c(21) + "; src/cards/card.cpp:1272")],
         "effects": [eff("Empower", SELF(c(21)), src=c(21))], "_src": c(20, 21)},
        trig("become_empowered", "self", [
            eff("PlayIgnoringCost", sel("unit", "friendly", "trash",
                                        filters=[F("energy_cost_le", 3, src=c(33)),
                                                 F("power_cost_le", 1, src=c(33))], src=c(26, 36)),
                optional=True, dest="base", src=c(47, 48) + "; may " + c(38, 42)),
        ], src=c(25)),
    ], "flags": ["new term: verb Empower, condition empowered, event become_empowered",
                 "new term: zone trash",
                 "[Empower] cost is class-declared (empowerSpec) and not expanded by extract; "
                 "written here as an activated ability"]}


def _peak():
    c = at("src/cards/battlefields/0284_targon_s_peak.cpp")
    return {"abilities": [trig("conquer", "here", [
        eff("Ready", sel("rune", "any", count=2, optional=True, filters=[F("exhausted", src=c(55))],
                         src=c(52, 64)), after="end_of_turn",
            src=c(73) + "; delayed " + c(32, 38)),
    ], src=c(23, 29))],
        "flags": ["new term: after=end_of_turn (delayed ability, runes chosen when it fires)",
                  "new term: selector type rune, filter exhausted"]}


def _hall():
    c = at("src/cards/battlefields/0761_abandoned_hall.cpp")
    return {"abilities": [trig("play_spell", "any_player", [
        eff("GiveMight", sel("unit", "friendly", "here", src=c(27, 36)), amt(1, src=c(46)), TURN,
            optional=True, src=c(46) + "; may " + c(37, 41)),
    ], src=c(19) + "; " + TM + ":846-857")],
        "flags": ["new term: event play_spell, subject any_player (the player who played the spell "
                  "controls the ability; 'friendly' is relative to them)"]}


BUILDERS = {
    ("scorn_of_the_moon", "unl-234-219"): _scorn,
    ("diana_lunari", "unl-079-219"): _diana,
    ("ravenbloom_student", "ogn-103-298"): _ravenbloom,
    ("traveling_merchant", "ogn-185-298"): _merchant,
    ("tideturner", "ogn-199-298"): _tideturner,
    ("hwei_brooding_painter", "unl-080-219"): _hwei,
    ("vex_apathetic", "unl-150-219"): _vex,
    ("fizz_trickster", "sfd-140-221"): _fizz,
    ("tail_cloaked_matriarch", "ven-104-166"): _matriarch,
    ("targon_s_peak", "ogn-289-298"): _peak,
    ("abandoned_hall", "unl-205-219"): _hall,
}
