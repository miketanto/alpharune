"""B2 v2: spells batch D (twelve spell classes). Graphs from the engine card classes, same helpers and
conventions as graphs_code.py; stats, keywords and keyword expansions (Hidden, Flow) come from the
CardDef via extract.build_class."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graphs_code import (sel, SELF, YOU, ref, F, C, amt, eff, spell, static, cost,  # noqa: E402
                         at, TURN, ES, CH, GE)
import scan  # noqa: E402

# scan.KW_MAP has no entry for Keyword::Repeat (Hard Bargain prints it): parse_carddef raises KeyError.
# Registered here, additively, so the CardDef parses; belongs in scan.KW_MAP.

S = "src/cards/spells/"


def _moonfall():
    c = at(S + "0755_moonfall.cpp")
    return {"abilities": [spell([
        eff("ChooseTarget", sel("battlefield", "any", "battlefield",
                                filters=[F("has_friendly_units", src=c(30, 39))], bind="L",
                                src=c(30, 43)), src=c(53, 56)),
        eff("Move", sel("unit", "enemy", filters=[F("at_chosen_battlefield", neg=True, src=c(62, 63))],
                        src=c(59, 65) + "; " + CH + ":712-717"),
            optional=True, dest="chosen_battlefield",
            src=c(83) + "; may " + c(70, 79) + "; " + ES + ":1697"),
        eff("GiveMight", sel("unit", "enemy", count="all", filters=[F("at_chosen_battlefield", src=c(90, 91))],
                             src=c(87, 92)), amt(-2, src=c(93)), TURN, src=c(93)),
    ], src=c(45))],
        "flags": ["new term: selector type 'battlefield' (a chosen location, bound as L)",
                  "new term: filter at_chosen_battlefield; dest chosen_battlefield (the bound L)",
                  "the -2 applies to enemy units at L whether or not a unit was moved (" + c(66) + ")"]}


def _gust():
    c = at(S + "0169_gust.cpp")
    return {"abilities": [spell([
        eff("Bounce", sel("unit", "any", "battlefield", filters=[F("might_le", 3, src=c(38))],
                          src=c(18, 21) + "; " + c(32, 44)),
            dest="owner_hand", src=c(29) + "; owner's hand " + ES + ":1166-1215"),
    ], src=c(25))],
        "flags": ["might_le reads current Might (obj.current_might), not printed Might"]}


def _ride_the_wind():
    c = at(S + "0173_ride_the_wind.cpp")
    return {"abilities": [spell([
        eff("Move", sel("unit", "friendly", bind="A", src=c(28, 36)), dest="base_or_other_battlefield",
            src=c(52) + "; dest " + c(45) + ", " + CH + ":1520-1529"),
        eff("Ready", ref("A", src=c(53)), src=c(53)),
    ], src=c(33))]}


def _hard_bargain():
    c = at(S + "0457_hard_bargain.cpp")
    return {"abilities": [
        {"kind": "additional_cost", "cost": cost(energy=2, optional=True, additional=True, src=c(19, 21)),
         "effects": [eff("Repeat", SELF(c(19, 21)), src=c(18, 21) + "; " + GE + ":4383-4387")],
         "_src": c(18, 21)},
        spell([
            eff("Counter", sel("spell", "any", "chain", src=c(31, 39)),
                condition=[C("target_controller_pays_energy", 2, neg=True, src=c(56, 93))],
                src=c(59) + "; " + c(93) + "; " + CH + ":395"),
        ], src=c(37)),
    ], "flags": ["new term: verb Repeat (Repeat [2]: optional additional cost, the spell's effect is "
                 "executed again with its own choices; engine-charged, CR 820)",
                 "new term: condition target_controller_pays_energy (the countered spell's controller may "
                 "pay [2] to prevent it; auto-countered when it cannot pay)"]}


def _eclipse():
    c = at(S + "0625_eclipse.cpp")
    return {"abilities": [spell([
        eff("GiveMight", sel("unit", "any", src=c(26, 28)), amt(-4, src=c(21)), TURN, src=c(20, 21)),
        eff("LookAt", sel("card", "friendly", "deck_top", 1, bind="P", src=c(24)),
            src=c(24) + "; " + ES + ":2755-2790"),
        eff("Recycle", ref("P", src=c(24)), optional=True, src=c(24) + "; " + ES + ":2797-2830"),
    ], src=c(19))],
        "flags": ["Predict (executor.predict(you, 1)) written as LookAt top card of your deck + may "
                  "Recycle it; new term: zone deck_top",
                  "Predict happens even if the unit target became illegal (" + c(22, 23) + ")"]}


def _star_crossed():
    c = at(S + "0690_star_crossed.cpp")
    return {"abilities": [spell([
        eff("Bounce", sel("unit", "friendly", src=c(41, 47)), dest="owner_hand",
            src=c(71, 73) + "; owner's hand " + ES + ":1166-1215"),
        eff("Bounce", sel("unit", "enemy", src=c(48, 58)), dest="owner_hand",
            src=c(74, 76) + "; owner's hand " + ES + ":1166-1215"),
    ], src=c(40))]}


def _acceptable_losses():
    c = at(S + "0179_acceptable_losses.cpp")
    return {"abilities": [spell([
        eff("Kill", sel("gear", "friendly", src=c(25, 34)), src=c(46, 49)),
        eff("Kill", sel("gear", "enemy", chooser="controller", src=c(30) + "; " + c(37, 38)),
            src=c(46, 49)),
    ], src=c(24))],
        "flags": ["each player chooses one of their own gear (not a target: Deflect / can't-be-chosen do "
                  "not apply); new value chooser=controller for the opponent's pick",
                  "both gear die at the same time (DeathBatch, " + c(47) + ")"]}


def _switcheroo():
    c = at(S + "0466_switcheroo.cpp")
    return {"abilities": [spell([
        eff("ChooseTarget", sel("unit", "any", "battlefield", bind="A", src=c(57, 60)), src=c(71)),
        eff("ChooseTarget", sel("unit", "any", "battlefield",
                                filters=[F("other", src=c(68)), F("at_location_of", "A", src=c(64, 68))],
                                bind="B", src=c(61, 70)), src=c(71)),
        eff("GiveMight", ref("A", src=c(83)),
            {"ref": "might_of:B", "minus": "might_of:A", "_src": c(80, 83)}, TURN, src=c(83)),
        eff("GiveMight", ref("B", src=c(84)),
            {"ref": "might_of:A", "minus": "might_of:B", "_src": c(80, 84)}, TURN, src=c(84)),
    ], src=c(54))],
        "flags": ["swap = GiveMight by the difference of the two current Mights (snapshot at resolution); "
                  "new term: amount field minus",
                  "new term: filter at_location_of (same battlefield as the bound unit)",
                  "played from facedown both units must be at its hidden battlefield (" + c(46, 53) +
                  ", " + c(60) + "); not represented"]}


def _turn_to_dust():
    c = at(S + "0632_turn_to_dust.cpp")
    return {"abilities": [spell([
        eff("GiveKeyword", sel("gear", "any", src=c(30, 32)), keyword="temporary", src=c(26, 27)),
    ], src=c(22))],
        "flags": ["permanent grant (no duration): the gear keeps Temporary until its controller's "
                  "Beginning Phase kills it (engine sweep, " + c(18, 21) + "); Temporary's kill is not expanded"]}


def _decree_of_insight():
    c = at(S + "0848_decree_of_insight.cpp")
    tgt = dict(SELF(c(24)), zone="hand")   # the card while its cost is paid
    return {"abilities": [
        static([eff("IgnoreCost", tgt, keyword="deflect", src=c(18, 24))], src=c(24)),
        spell([
            eff("GiveMight", sel("unit", "enemy", filters=[F("domain", "body", src=c(35))], src=c(28, 37)),
                amt(-5, src=c(43)), TURN, src=c(43)),
        ], src=c(39)),
    ], "flags": ["new term: verb IgnoreCost (keyword deflect: Deflect's additional cost is not paid "
                 "when this spell chooses a unit)"]}


def _singularity():
    c = at(S + "0105_singularity.cpp")
    return {"abilities": [spell([
        eff("DealDamage", sel("unit", "any", count=2, optional=True, src=c(19, 41)), amt(6, src=c(44)),
            source=SELF(c(44)), src=c(43, 44) + "; " + ES + ":68"),
    ], src=c(42))]}


def _up_from_the_deep():
    c = at(S + "0887_up_from_the_deep.cpp")
    return {"abilities": [spell([
        eff("CreateToken", sel("unit", "friendly", "board", src=c(29, 30) + "; " + CH + ":1378-1406"),
            amt(2, src=c(30)), token={"type": "unit", "might": 1, "tag": "bilgewater", "_src": c(25, 27)},
            dest="base_or_controlled_battlefield",
            src=c(24, 30) + "; " + CH + ":1466-1496"),
    ], src=c(22))], "flow": (c(33), 3, 0),
        "flags": ["new term: dest base_or_controlled_battlefield (each token's location chosen by you as it "
                  "is played: your base or a battlefield you control)",
                  "token unit (1 Might, tag bilgewater) has no standalone token graph here"]}


BUILDERS = {
    ("moonfall", "unl-198-219"): _moonfall,
    ("gust", "ogn-169-298"): _gust,
    ("ride_the_wind", "ogn-173-298"): _ride_the_wind,
    ("hard_bargain", "sfd-136-221"): _hard_bargain,
    ("eclipse", "unl-063-219"): _eclipse,
    ("star_crossed", "unl-128-219"): _star_crossed,
    ("acceptable_losses", "ogn-179-298"): _acceptable_losses,
    ("switcheroo", "sfd-145-221"): _switcheroo,
    ("turn_to_dust", "unl-070-219"): _turn_to_dust,
    ("decree_of_insight", "ven-061-166"): _decree_of_insight,
    ("singularity", "ogn-105-298"): _singularity,
    ("up_from_the_deep", "ven-100-166"): _up_from_the_deep,
}
