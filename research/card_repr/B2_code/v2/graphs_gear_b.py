"""B2 v2: seven gear (six Equipment + one token-making gear) and four spells.

Equipment (CR 716-725, 818) as the engine implements it:
  - [Equip] is an activated ability, main phase only (GameEngine::generateMainPhaseActions offers it,
    game_engine.cpp:5735-5760), paid with the gear's declared Card::equipCost (SimpleEquipGear:
    equip_base.h:24-27 = power 1 of the given domain, energy 0), put on the chain and resolved by
    EffectExecutor::attachGear (effect_executor.cpp:1617-1640). It is offered only while the gear is
    unattached (card.cpp:1318-1322). New verb: Attach (target = the friendly unit).
  - On attach the unit gets the gear's CardDef might_bonus (effect_executor.cpp:1636): a static
    GiveMight on the equipped unit, duration while.
  - What the gear's effect text gives the bearer: equippedKeywords (game_engine.cpp:7985-8000),
    equippedTriggerType / onEquippedTrigger (trigger_manager.cpp:223-245), or a class
    applyPassiveAura that reads gear.attached_to.
  The equipped unit is a selector with the new filter equipped_with_self.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graphs_code import sel, SELF, YOU, F, C, amt, eff, spell, trig, static, cost, at, ES, GE, TM  # noqa: E402
import scan  # noqa: E402

# scan.KW_MAP (CardDef keyword parse) predates gear: register the two printed keywords these cards
# carry so build_class can read their CardDefs. Idempotent; no expansion is attached to either.

EQB = "src/cards/gear/equip_base.h"
CARDCPP = "src/cards/card.cpp"
EQUIP_FLAGS = ["new term: Attach (verb: attach this gear to the target unit; EffectExecutor::attachGear)",
               "new term: equipped_with_self (filter: the unit this gear is attached to)"]


def BEARER(src, zone="board", filters=None):
    return sel("unit", "friendly", zone, filters=[F("equipped_with_self", src=src)] + (filters or []),
               src=src)


def equip(domain, cost_src):
    """The [Equip] activated ability: pay power 1 (domain) -> Attach this to a unit you control."""
    return {"kind": "activated", "timing": "main",
            "cost": cost(power=1, power_domain=domain, src=cost_src),
            "effects": [eff("Attach", sel("unit", "friendly", src=GE + ":5745-5757; " + CARDCPP + ":1316-1317"),
                            src=ES + ":1617-1640")],
            "_src": GE + ":2673-2710, 5735-5760 (main phase only); unattached only " + CARDCPP + ":1318-1322"}


def bonus(n, src):
    """The CardDef might_bonus, applied to the unit as the gear attaches."""
    return static([eff("GiveMight", BEARER(ES + ":1636"), amt(n, src=src), "while",
                       src=src + "; " + ES + ":1636")], src=src)


def _doran_s_shield():
    c = at("src/cards/gear/0356_doran_s_shield.cpp")
    return {"abilities": [
        equip("calm", c(17) + "; " + EQB + ":24-27"),
        bonus(1, c(35)),
        static([eff("GiveKeyword", BEARER(GE + ":7980-7984"), keyword="tank", duration="while",
                    src=c(19) + "; " + GE + ":7993-8000")], src=c(19)),
    ], "flags": EQUIP_FLAGS}


def _brutalizer():
    c = at("src/cards/gear/0365_brutalizer.cpp")
    return {"abilities": [
        equip("calm", c(19, 22)),
        bonus(1, c(68)),
        static([eff("GiveMight", BEARER(c(41, 49)), amt(2, src=c(50)), "while", src=c(50))],
               src=c(29), condition=[C("attached_this_turn", src=c(42, 46))]),
    ], "flags": EQUIP_FLAGS + [
        "new term: attached_this_turn (condition: this gear became attached to its current unit this "
        "turn, by Equip or any other effect; tracked per bearer change)"]}


def _eye_of_the_herald():
    c = at("src/cards/gear/0474_eye_of_the_herald.cpp")
    return {"abilities": [
        equip("order", c(17) + "; " + EQB + ":24-27"),
        trig("move", "equipped_unit", [
            eff("CreateToken", sel("unit", "friendly", "here", src=c(25, 26)), amt(1, src=c(28)),
                token={"type": "unit", "might": 1, "tag": "recruit", "_src": c(28, 29)}, src=c(28, 29)),
        ], src=c(19, 21) + "; " + TM + ":223-245"),
    ], "flags": EQUIP_FLAGS + [
        "new term: trigger event 'move' (TriggerType::WhenIMove, any move, not only to a battlefield)",
        "new term: trigger subject 'equipped_unit' (the trigger fires on the unit this gear is attached to)",
        "token zone 'here' = the equipped unit's location; nothing if it has left the board (code)"]}


def _b_f_sword():
    c = at("src/cards/gear/0482_b_f_sword.cpp")
    return {"abilities": [
        equip("order", c(17) + "; " + EQB + ":24-27"),
        bonus(3, c(34)),
    ], "flags": EQUIP_FLAGS}


def _soul_sword():
    c = at("src/cards/gear/0601_soul_sword.cpp")
    return {"abilities": [
        equip("calm", c(18, 21)),
        bonus(1, c(56)),
        static([eff("GiveMight", BEARER(c(32, 36)), amt(1, src=c(38)), "while", src=c(37, 38))],
               src=c(22), condition=[C("xp_ge", 3, src=c(25))]),
    ], "flags": EQUIP_FLAGS + ["text '[Level 3]' is not a CardDef keyword: only the xp_ge 3 condition "
                               "from applyPassiveAura is represented"]}


def _hand_hammer():
    c = at("src/cards/gear/0814_hand_hammer.cpp")
    return {"abilities": [
        equip("calm", c(18) + "; " + EQB + ":24-27"),
        bonus(1, c(47)),
        static([eff("GiveMight", BEARER(c(25, 27), zone="battlefield",
                                        filters=[F("other_friendly_units_here_eq", 1, src=c(28))]),
                    amt(2, src=c(29)), "while", src=c(29))], src=c(23, 31)),
    ], "flags": EQUIP_FLAGS + [
        "new term: other_friendly_units_here_eq (filter: exactly N other friendly units at the unit's location)"]}


def _sprite_fountain():
    c = at("src/cards/gear/0640_sprite_fountain.cpp")

    def tok(s):
        return eff("CreateToken", sel("unit", "friendly", "base", src=c(29)), amt(1, src=c(27)),
                   token={"type": "unit", "might": 3, "tag": "fae", "keywords": ["temporary"],
                          "enter_ready": True, "_src": c(26, 30)}, src=s)
    return {"abilities": [
        trig("play", "self", [tok(c(27, 30))], src=c(19, 20)),
        trig("die", "self", [tok(c(27, 30))], src=c(19, 20)),
    ], "flags": ["[Temporary] on the gear (kill at the start of its controller's Beginning Phase, "
                 "trigger_manager.cpp:520-545) is engine-implemented and kept as keyword only: "
                 "extract.build_class has no expansion for it",
                 "token extras keywords/enter_ready (createToken arg5 / arg7=true) are B2 schema extensions"]}


def _stupefy():
    c = at("src/cards/spells/0095_stupefy.cpp")
    return {"abilities": [spell([
        eff("GiveMight", sel("unit", "any", src=c(24, 26)), amt(-1, src=c(20)), "turn", minimum=1,
            src=c(19, 20) + "; minimum " + ES + ":1245-1254"),
        eff("Draw", YOU(c(22)), amt(1, src=c(22)), src=c(22)),
    ], src=c(18))], "flags": ["new term: minimum (GiveMight extra: Might not reduced below N)"]}


def _smoke_screen():
    c = at("src/cards/spells/0093_smoke_screen.cpp")
    return {"abilities": [spell([
        eff("GiveMight", sel("unit", "any", src=c(21, 23)), amt(-4, src=c(19)), "turn", minimum=1,
            src=c(18, 19) + "; minimum " + ES + ":1245-1254"),
    ], src=c(17))], "flags": ["new term: minimum (GiveMight extra: Might not reduced below N)"]}


def _rebuke():
    c = at("src/cards/spells/0172_rebuke.cpp")
    return {"abilities": [spell([
        eff("Bounce", sel("unit", "any", "battlefield", src=c(18, 21)), dest="owner_hand",
            src=c(32) + "; owner's hand " + ES + ":1166-1215"),
    ], src=c(23))]}


def _flash():
    c = at("src/cards/spells/0309_flash.cpp")
    return {"abilities": [spell([
        eff("Move", sel("unit", "friendly", count=2, optional=True,
                        filters=[F("at_base", neg=True, src=c(22, 25) + "; " + c(40, 44))],
                        src=c(66, 75) + " (up to 2, none allowed " + c(70, 72) + ")"),
            dest="base", src=c(84, 87)),
    ], src=c(32))], "flags": ["new term: at_base (filter: the unit is in its controller's base)"]}


BUILDERS = {
    ("doran_s_shield", "sfd-033-221"): _doran_s_shield,
    ("brutalizer", "sfd-042-221"): _brutalizer,
    ("eye_of_the_herald", "sfd-153-221"): _eye_of_the_herald,
    ("b_f_sword", "sfd-161-221"): _b_f_sword,
    ("soul_sword", "unl-039-219"): _soul_sword,
    ("hand_hammer", "ven-027-166"): _hand_hammer,
    ("sprite_fountain", "unl-078-219"): _sprite_fountain,
    ("stupefy", "ogn-095-298"): _stupefy,
    ("smoke_screen", "ogn-093-298"): _smoke_screen,
    ("rebuke", "ogn-172-298"): _rebuke,
    ("flash", "ogs-011-024"): _flash,
}
