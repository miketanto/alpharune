"""B2 v2: Kennen list, part E (legend, champion, units, gear, spells, battlefields; 2026-11-22).

Graphs from the engine classes (src/cards/...), same helpers and conventions as graphs_code.py. Stats,
keywords and keyword expansions (Hidden, Flow, Deathknell, Equip) come from the CardDef via
extract.build_class.

[Burn N] (EffectExecutor::burn, CR 441) is written as the new verb Burn: the top N cards of your Main
Deck go to your trash (burning out when it runs out). Minefield's "put the top 2 into your trash" is
the same call in the engine (minefield.cpp:21-25), so it is Burn too.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graphs_code import sel, SELF, YOU, ref, F, C, amt, eff, spell, trig, static, cost, at, ES, CH, GE, TM, TURN  # noqa: E402

BURN = "new term: verb Burn (top N of your Main Deck to your trash; EffectExecutor::burn, CR 441)"
TRASH = "new term: zone trash"


def _heart_of_the_tempest():
    c = at("src/cards/legends/0942_heart_of_the_tempest.cpp")
    return {"abilities": [
        trig("play_not_from_hand", "you", [eff("Empower", SELF(c(21)), src=c(21))], src=c(18, 19)),
        {"kind": "activated", "timing": "action",
         "cost": cost(exhaust=True, other=[eff("Disempower", SELF(c(26)), src=c(26))], src=c(26)),
         "condition": [C("empowered", src=c(26) + " (disempower_self is a cost: only while empowered)")],
         "effects": [eff("GiveKeyword", sel("unit", "any", src=c(27, 35)), keyword="assault", value=2,
                         duration=TURN, src=c(42))],
         "_src": c(23, 31)},
    ], "flags": ["new term: event play_not_from_hand (subject you: a card you play from anywhere but "
                 "your hand: trash, facedown, Champion Zone, by an effect)",
                 "new term: verb Disempower (as a cost: ActivationCost::disempower_self)",
                 "GiveKeyword assault value 2 for the turn (giveTemporaryKeyword)"]}


def _kennen_storm_of_shuriken():
    c = at("src/cards/units/0900_kennen_storm_of_shuriken.cpp")
    return {"abilities": [
        trig("play", "self", [eff("Burn", YOU(c(30)), amt(2, src=c(30)), src=c(30))], src=c(19, 24)),
        trig("conquer", "self", [
            eff("GiveKeyword", sel("spell", "friendly", "trash", src=c(37, 44)), keyword="flow",
                value="own_cost", duration=TURN, src=c(50, 51) + "; " + GE + " (Flow generator reads "
                "__flow_granted_turn)"),
        ], src=c(20, 24)),
    ], "flags": [BURN, TRASH,
                 "GiveKeyword flow value own_cost: the spell may be played from the trash this turn for "
                 "its printed cost, then banished"]}


def _rhasa_the_sunderer():
    c = at("src/cards/units/0195_rhasa_the_sunderer.cpp")
    tgt = dict(SELF(c(18)), zone="hand")   # selfCostReduction: the cost of this card as it is played
    return {"abilities": [static([eff("ReduceCost", tgt, amt(1, per="card_in_your_trash", src=c(19)),
                                      resource="energy", src=c(17, 19))], src=c(17))],
            "flags": ["new term: amount per card_in_your_trash"]}


def _seal_of_discord():
    c = at("src/cards/gear/0204_seal_of_discord.cpp")
    return {"abilities": [{
        "kind": "activated", "timing": "reaction",
        "cost": cost(exhaust=True, src=c(22)),
        "effects": [eff("AddPower", YOU(c(27)), amt(1, src=c(27)), domain="chaos", src=c(25, 27))],
        "_src": c(17, 29) + "; add (no chain) " + c(18, 20)}]}


def _shadow_order_disciple():
    c = at("src/cards/units/0882_shadow_order_disciple.cpp")
    return {"abilities": [trig("move", "self", [
        eff("GiveMight", SELF(c(38)), amt(1, src=c(38)), TURN, src=c(38)),
    ], src=c(20, 21), cost=cost(optional=True, other=[eff("Burn", YOU(c(34)), amt(1, src=c(34)), src=c(34))],
                                src=c(31, 35)))],
        "flags": ["new term: event move", BURN + " (here as the trigger's optional cost)"]}


def _lightning_rush():
    c = at("src/cards/spells/0943_lightning_rush.cpp")
    top3 = lambda s, **kw: sel("card", "friendly", "deck_top", 3, src=s, **kw)
    return {"abilities": [spell([
        eff("LookAt", top3(c(29, 38), bind="L"), src=c(34, 38)),
        eff("Draw", sel("card", "friendly", "deck_top", filters=[F("in", "L", src=c(50, 51))], optional=True,
                        src=c(49, 55)), amt(1, src=c(77, 78)), src=c(61, 78)),
        eff("Mill", ref("L", src=c(79, 88)), dest="trash", src=c(79, 88)),
    ], src=c(23))], "flow": (c(90, 91), 2, 1),
        "flags": ["draws a chosen card among the 3 looked at (drawHeldCard: counts as a draw); the rest "
                  "go from the deck to the trash (written as Mill: not a Burn, per the code comment)",
                  "new term: verb Mill, filter in (bound set)",
                  "Flow's Power is of ANY Domain (flow(2, 1, Fury, any_domain=true)); flow_mode writes "
                  "power_domain card"]}


def _last_rites():
    c = at("src/cards/gear/0471_last_rites.cpp")
    bearer = sel("unit", "friendly", filters=[F("equipped_with_self", src=c(56))], src=c(56))
    return {"abilities": [
        {"kind": "activated", "timing": "main",
         "cost": cost(power=1, power_domain="chaos",
                      other=[eff("Recycle", sel("card", "friendly", "trash", 2, src=c(28, 45)), src=c(32, 48))],
                      src=c(24, 31)),
         "effects": [eff("Attach", sel("unit", "friendly", src=GE + ":5745-5757"), src=ES + ":1617-1640")],
         "_src": c(19, 50)},
        static([eff("GiveMight", bearer, amt(2, src=c(104)), "while", src=c(104) + "; " + ES + ":1636")],
               src=c(104)),
        trig("conquer_or_hold", "equipped_unit", [
            eff("PlayPayingCost", sel("unit", "friendly", "trash", src=c(59, 73)), optional=True,
                dest="base_or_controlled_battlefield", src=c(80, 87) + "; may " + c(69, 70)),
        ], src=c(52, 57) + "; " + TM + ":223-245"),
    ], "flags": ["new term: Attach (verb), equipped_with_self (filter)", TRASH,
                 "new term: verb PlayPayingCost (play a card from a zone, its cost paid)",
                 "Equip cost: [P] Chaos and recycle 2 cards from your trash (Recycle as cost.other)"]}


def _tornado_warrior():
    c = at("src/cards/units/0886_tornado_warrior.cpp")
    return {"abilities": [trig("play", "self", [
        eff("Empower", sel("permanent", "any", "here",
                           filters=[F("empowered", neg=True, src=c(45))], bind="A", src=c(33, 50)),
            optional=True, src=c(69) + "; may " + c(63, 65)),
        eff("Disempower", ref("A", src=c(58)), after="end_of_turn", src=c(52, 61) + "; " + c(70)),
    ], src=c(24, 32), condition=[C("played_from_facedown", src=c(29, 31))])],
        "flags": ["new term: condition played_from_facedown", "new term: verb Disempower",
                  "new term: selector type permanent (a unit or a gear)", "after=end_of_turn (delayed)"]}


def _carrion_dredger():
    c = at("src/cards/units/0715_carrion_dredger.cpp")
    return {"abilities": [trig("die", "self", [
        eff("CreateToken", sel("unit", "friendly", "base", src=c(23)), amt(1, src=c(21)),
            token={"type": "unit", "might": 1, "tag": "bird", "keywords": ["deflect"], "_src": c(20, 22)},
            src=c(20, 24)),
    ], src=c(17, 18))], "flags": ["token keyword deflect (token.keywords)"]}


def _ezreal_prodigy():
    c = at("src/cards/units/0470_ezreal_prodigy.cpp")
    return {"abilities": [
        trig("play", "self", [
            eff("Discard", sel("card", "friendly", "hand", src=c(37, 47)), amt(1, src=c(54)), src=c(52, 55)),
            eff("Draw", YOU(c(56)), amt(2, src=c(56)), src=c(56)),
        ], src=c(23, 25)),
        static([eff("ReduceCost", sel("card", "friendly", "hand", src=c(17, 19)), amt(1, src=c(21)),
                    resource="optional_additional_cost", src=c(20, 22))], src=c(17, 22),
               condition=[C("self_on_board", src=c(21))]),
    ], "flags": ["new term: verb Discard", "new term: ReduceCost resource optional_additional_cost "
                 "(Accelerate, Repeat, 'you may pay as an additional cost': [1] or [A] less)",
                 "new term: condition self_on_board"]}


def _zaun_warrens():
    c = at("src/cards/battlefields/0293_zaun_warrens.cpp")
    return {"abilities": [trig("conquer", "here", [
        eff("Discard", sel("card", "friendly", "hand", src=c(20)), amt(1, src=c(20)), src=c(20) + "; " + CH),
        eff("Draw", YOU(c(21)), amt(1, src=c(21)), src=c(21)),
    ], src=c(23))], "flags": ["new term: verb Discard"]}


def _minefield():
    c = at("src/cards/battlefields/0526_minefield.cpp")
    return {"abilities": [trig("conquer", "here", [eff("Burn", YOU(c(25)), amt(2, src=c(25)), src=c(21, 25))],
                               src=c(19))], "flags": [BURN]}


def _shadow_temple():
    c = at("src/cards/battlefields/0952_shadow_temple.cpp")
    return {"abilities": [trig("hold", "here", [eff("Burn", YOU(c(21)), amt(3, src=c(21)), src=c(21))],
                               src=c(18, 19))], "flags": [BURN, "event hold (subject here)"]}


def _ravenbloom_prefect():
    c = at("src/cards/units/0889_ravenbloom_prefect.cpp")
    return {"abilities": [trig("play_gear", "opponent", [
        eff("Banish", ref("subject", src=c(24, 25)), src=c(45, 53)),
    ], src=c(20, 22), cost=cost(optional=True, other=[eff("Banish", SELF(c(42)), src=c(41, 42))],
                                src=c(38, 43)))],
        "flags": ["new term: event play_gear (subject opponent)",
                  "the played gear is banished even while still on the chain (its permanent item removed, "
                  + c(47, 52) + ")"]}


def _lacerate():
    c = at("src/cards/spells/0914_lacerate.cpp")
    return {"abilities": [spell([
        eff("ChooseTarget", sel("unit", "any", bind="A", src=c(25, 32)), src=c(32)),
        eff("Disempower", ref("A", src=c(35)), condition=[C("empowered", "A", src=c(34))], src=c(34, 36)),
        eff("Kill", ref("A", src=c(42)), condition=[C("might_le", 3, src=c(41))], src=c(39, 42)),
    ], src=c(29))], "flow": (c(44, 45), 4, 2),
        "flags": ["new term: verb Disempower", "the Might check reads Might after the disempower "
                  "(continuous effects recomputed, " + c(36, 40) + ")"]}


BUILDERS = {
    ("heart_of_the_tempest", "ven-155-166"): _heart_of_the_tempest,
    ("kennen_storm_of_shuriken", "ven-113-166"): _kennen_storm_of_shuriken,
    ("rhasa_the_sunderer", "ogn-195-298"): _rhasa_the_sunderer,
    ("seal_of_discord", "ogn-204-298"): _seal_of_discord,
    ("shadow_order_disciple", "ven-095-166"): _shadow_order_disciple,
    ("lightning_rush", "ven-156-166"): _lightning_rush,
    ("last_rites", "sfd-150-221"): _last_rites,
    ("tornado_warrior", "ven-099-166"): _tornado_warrior,
    ("carrion_dredger", "unl-153-219"): _carrion_dredger,
    ("ezreal_prodigy", "sfd-149-221"): _ezreal_prodigy,
    ("zaun_warrens", "ogn-298-298"): _zaun_warrens,
    ("minefield", "sfd-212-221"): _minefield,
    ("shadow_temple", "ven-165-166"): _shadow_temple,
    ("ravenbloom_prefect", "ven-102-166"): _ravenbloom_prefect,
    ("lacerate", "ven-127-166"): _lacerate,
}
