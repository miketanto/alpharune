"""B2 vocabulary: engine code -> path-B schema. Written once, used for all cards.

Each entry: role (verb | structure | plumbing | hook), what it maps to, which
argument carries the amount / target, and where it is declared or defined.
`role`:
  verb       an EffectExecutor method / helper that changes game state -> schema verb
  structure  a call that shapes an ability (selector, condition, cost, dest, chooser,
             optional) without being an effect of its own
  plumbing   chain resume / finalize / choice bookkeeping, lookups, formatting:
             ignored (PHASE2: "Ignore plumbing")
  hook       a Card virtual the class overrides (what kind of ability it is)
"""

EX = "src/engine/effect_executor.cpp"
EXH = "src/engine/effect_executor.h"
CH = "src/cards/card_helpers.h"
CARD = "src/cards/card.h"
CARDCPP = "src/cards/card.cpp"
GE = "src/engine/game_engine.cpp"
TM = "src/engine/trigger_manager.cpp"
PP = "src/engine/play_permissions.cpp"
GO = "src/core/game_object.h"
ET = "src/effects/effect_types.h"
DB = "src/core/card_db.h"

EXECUTOR = {
    "drawCards": {"role": "verb", "verb": "Draw", "target": "arg0 player -> {type: player}",
                  "amount": "arg1 count", "src": EX + ":654"},
    "giveTemporaryMight": {"role": "verb", "verb": "GiveMight", "duration": "turn",
                           "target": "arg0", "amount": "arg1 (signed)",
                           "note": "arg2 minimum unused by these cards; Might clamps at 0 (recomputeMight)",
                           "src": EX + ":1222"},
    "killObject": {"role": "verb", "verb": "Kill", "target": "arg0", "src": EX + ":556"},
    "readyObject": {"role": "verb", "verb": "Ready", "target": "arg0", "src": EX + ":1351"},
    "bounceToHand": {"role": "verb", "verb": "Bounce", "target": "arg0", "src": EX + ":1166"},
    "createToken": {"role": "verb", "verb": "CreateToken",
                    "target": "arg6 location -> selector zone (base / here)",
                    "amount": "number of calls (loop count)",
                    "extras": "token.type=arg1, token.might=arg3, token.tag=arg4 (lower-cased); "
                              "arg2 token NAME dropped (names never enter the graph); "
                              "arg7 enter_ready=false is the default -> not represented",
                    "src": EXH + ":612"},
    "dealDamage": {"role": "verb", "verb": "DealDamage", "target": "arg0",
                   "amount": "arg1", "extras": "source = arg2", "src": EX + ":68"},
    "stunUnitBy": {"role": "verb", "verb": "Stun", "duration": "turn", "target": "arg0",
                   "note": "arg1 = who stunned (attribution only)", "src": EX + ":1732"},
    "recycleCards": {"role": "verb", "verb": "Recycle", "target": "arg1 cards",
                     "note": "arg0 = whose deck; arg2 = who sees which card (knowledge, ignored)",
                     "src": EX + ":1842"},
    "buffUnit": {"role": "verb", "verb": "Buff", "target": "arg0",
                 "amount": "constant 1 (one permanent +1 buff, none if it already has one)",
                 "src": EX + ":1321-1345"},
    "moveToBattlefield": {"role": "verb", "verb": "Move", "target": "arg0",
                          "extras": "dest = arg1 battlefield", "src": EX + ":1697"},
    "moveToBase": {"role": "verb", "verb": "Move", "target": "arg0", "extras": "dest = base",
                   "src": EX + ":1409"},
    "banishObject": {"role": "verb", "verb": "Banish", "target": "arg0", "src": EX + ":1977"},
    "playIgnoringCost": {"role": "verb", "verb": "PlayIgnoringCost", "target": "arg1 card",
                         "extras": "chooser/player = arg0, dest = arg2 location",
                         "src": EX + ":3150"},
    "healExhaustRecall": {"role": "verb", "verb": ["Heal", "Recall{exhausted:true}"],
                          "target": "arg0",
                          "note": "healObject + exhaustObject + recallToBase (EX:1467-1472); "
                                  "Heal is a B2 schema extension",
                          "src": EX + ":1467-1472"},
    "unattachGear": {"role": "plumbing", "why": "part of Kill on a gear (leaving the board sheds "
                     "attachments anyway); no separate effect", "src": EX + ":1382"},
    "killLethallyDamagedTogether": {"role": "structure",
                                    "maps_to": "rules-inherent kill of lethally damaged units after a "
                                               "DealDamage; its return value feeds amount.per="
                                               "unit_killed_by_prev of the next effect",
                                    "src": EX + ":413"},
    "queueReflexiveTrigger": {"role": "structure",
                              "maps_to": "'for each X, do' -> amount.per on the reflexive effect",
                              "src": EXH + ":749"},
    "recalculateContinuousEffects": {"role": "plumbing", "why": "aura recompute",
                                     "src": EXH + ":577"},
    "requestChoice": {"role": "plumbing", "why": "agent choice slot", "src": EXH},
    "takeChoice": {"role": "plumbing", "why": "agent choice slot", "src": EXH},
    "cardDB": {"role": "structure", "maps_to": "read of a PRINTED cost -> filter energy_cost_le / "
               "power_cost_le on the selector", "src": EXH},
}

HELPERS = {
    "gainXp": {"role": "verb", "verb": "GainXP", "target": "arg1 player", "amount": "arg2",
               "src": CH + ":1868"},
    "splitDamageAmong": {"role": "verb", "verb": "DealDamage",
                         "target": "arg4 legal units, count=split", "amount": "arg2 total",
                         "extras": "source = arg3; split = true", "src": CH + ":990"},
    "counterChosenItem": {"role": "verb", "verb": "Counter", "target": "arg1 chain item",
                          "src": CH + ":395"},
    "moveTo": {"role": "verb", "verb": "Move", "target": "arg1", "extras": "dest = arg2",
               "src": CH + ":1535"},
    "pickMoveDest": {"role": "structure", "maps_to": "dest = base_or_other_battlefield "
                     "(moveDests: base unless there, every other battlefield)", "src": CH + ":1647"},
    "moveDests": {"role": "structure", "maps_to": "dest = base_or_other_battlefield", "src": CH + ":1520"},
    "chooseableUnits": {"role": "structure", "maps_to": "selector {type: unit, zone: board}; "
                        "the predicate gives side / filters; units with 'can't be chosen by enemy' "
                        "are dropped (that is the CantBeChosen rule, not a filter)",
                        "src": CH + ":712"},
    "enumerateLegalTargets": {"role": "structure", "maps_to": "selector from getTargetRequirements "
                              "(or the class's override)", "src": CARD},
    "chainItemChoices": {"role": "structure", "maps_to": "selector {zone: chain}; predicate -> filters",
                         "src": CH + ":365"},
    "anyChainItem": {"role": "structure", "maps_to": "play gate = a chain item matching the selector "
                     "exists (same as the Counter target; not a separate field)", "src": CH + ":381"},
    "pickChainItem": {"role": "plumbing", "why": "asks for the chain-item target", "src": CARD + ":611"},
    "pickTarget": {"role": "plumbing", "why": "asks for one target (count 1)", "src": CARD + ":522"},
    "pickTargetPair": {"role": "plumbing", "why": "asks for two targets", "src": CARD + ":519"},
    "pickUnseenCard": {"role": "plumbing", "why": "pick from a hidden zone (hand)", "src": CARD + ":541"},
    "lockedPlayTarget": {"role": "plumbing", "why": "target fixed at play", "src": CH + ":1444"},
    "chosenTarget": {"role": "plumbing", "why": "reads a target slot", "src": CARD},
    "pickValueAs": {"role": "structure", "maps_to": "chooser = arg1 (e.g. owner)", "src": CH + ":1926"},
    "confirmOptional": {"role": "structure", "maps_to": "optional: true ('you may')",
                        "src": CARDCPP + ":277"},
    "costDueNow": {"role": "structure", "maps_to": "what follows is a cost (ability.cost)",
                   "src": CARDCPP + ":56"},
    "payEnergy": {"role": "structure", "maps_to": "cost.energy = arg2", "src": CH + ":130"},
    "canPayEnergy": {"role": "structure", "maps_to": "cost.energy gate", "src": CH + ":122"},
    "asItLeft": {"role": "structure", "maps_to": "condition on last-known information "
                 "(died_alone: other_friendly_units_here empty)", "src": CH + ":88"},
    "playOrdinalThisTurn": {"role": "structure", "maps_to": "condition nth_card_this_turn",
                            "src": CH + ":504"},
    "battlefieldOfCard": {"role": "structure", "maps_to": "zone 'here' for a battlefield card",
                          "src": CH + ":192"},
    "battlefieldTakesUnitsFrom": {"role": "structure", "maps_to": "dest restricted to battlefields "
                                  "the player may play units to (rules restriction; dest kept as "
                                  "any_battlefield)", "src": CH + ":1243"},
    "moveToBaseBlocked": {"role": "structure", "maps_to": "rules restriction on a Move to base "
                          "(not represented: applies to every move)", "src": CH + ":1202"},
    "auraMight": {"role": "verb", "verb": "GiveMight", "duration": "while (continuous)",
                  "target": "arg0", "amount": "arg2", "src": CH + ":608"},
    "flow": {"role": "structure", "maps_to": "play_mode from_trash, cost energy=arg0, power=arg1 "
             "(power_domain arg2), then Banish self", "src": CH + ":559"},
    "opponent": {"role": "structure", "maps_to": "side enemy (the one opponent in 1v1)",
                 "src": "src/core/types.h"},
    "friendlyUnitChoice": {"role": "plumbing", "why": "finalize-pass affordability gate", "src": CH + ":139"},
    "mayChooseAtFinalization": {"role": "plumbing", "why": "finalize-pass affordability gate", "src": CARD},
    "finalizingChoices": {"role": "plumbing", "why": "end-of-choices marker", "src": CARDCPP + ":44"},
    "cardDefId": {"role": "plumbing", "why": "finds the card's OWN instances for an aura "
                  "(self-reference, not a check for another card)", "src": CARD},
    "playerIndex": {"role": "plumbing", "why": "counter key", "src": "src/core/types.h"},
    "replacementApplies": {"role": "hook", "maps_to": "replacement condition", "src": CARD},
    "replacementAppliesTo": {"role": "hook", "maps_to": "replacement condition", "src": CARD},
    "conditionsHold": {"role": "hook", "maps_to": "replacement condition (class-local)", "src": "class"},
    "triggerCondition": {"role": "hook", "maps_to": "trigger.condition (intervening if, CR 383.2.a.1)",
                         "src": CARD + ":690"},
    "onReflexiveTrigger": {"role": "hook", "maps_to": "reflexive effect body", "src": CARD},
}

HOOKS = {
    "def": {"maps_to": "stats + keywords (CardDef)", "src": DB + ":19"},
    "onResolve": {"maps_to": "ability kind spell", "src": CARD + ":80"},
    "onTrigger": {"maps_to": "ability kind triggered", "src": CARD + ":279"},
    "triggerType": {"maps_to": "trigger.event/subject", "src": CARD + ":251"},
    "triggerTypes": {"maps_to": "trigger.event/subject (several -> union)", "src": CARD + ":258"},
    "applyPassiveAura": {"maps_to": "ability kind static, duration while", "src": CARD + ":856"},
    "hasReplacementEffect": {"maps_to": "ability kind replacement", "src": CARD + ":887"},
    "applyReplacement": {"maps_to": "replacement effects", "src": CARD + ":895"},
    "getTargetRequirements": {"maps_to": "selector (TargetRequirements)", "src": CARD + ":657"},
    "hasLegalTargets": {"maps_to": "play legality gate; becomes ability.play_requires only where it "
                        "requires more than the effect targets (B2 extension)", "src": CARD + ":732"},
    "optionalAdditionalCost": {"maps_to": "ability kind additional_cost (optional)", "src": CARD + ":1067"},
    "selfCostReduction": {"maps_to": "static ReduceCost self", "src": CARD + ":1287"},
    "canBeChosenByEnemy": {"maps_to": "static CantBeChosen self by enemy spells/abilities",
                           "src": CARD + ":1355"},
    "ambushToEnemyBattlefields": {"maps_to": "widens the Ambush play_mode 'where' to "
                                  "battlefields with enemy units", "src": CARD + ":833"},
    "flowCost": {"maps_to": "play_mode from_trash", "src": CARD + ":196"},
    "choicesAtFinalization": {"maps_to": "plumbing (when choices are asked)", "src": CARD + ":338"},
    "needsPlayTimeTarget": {"maps_to": "plumbing (one Play intent, target as prompt)", "src": CARD},
    "needsPlayTimeTargetPair": {"maps_to": "plumbing", "src": CARD},
    "hasReflexiveTrigger": {"maps_to": "plumbing (reflexive trigger present)", "src": CARD},
}

TARGET_REQUIREMENTS = {
    "_src": ET + ":352-361",
    "count": "selector.count",
    "must_be_unit": "selector.type = unit",
    "must_be_gear": "selector.type = gear",
    "must_be_friendly": "selector.side = friendly",
    "must_be_enemy": "selector.side = enemy",
    "(neither friendly nor enemy)": "selector.side = any",
    "must_be_at_battlefield": "selector.zone = battlefield (default board)",
    "max_might": "filter might_le (unused by these cards)",
    "optional": "selector.optional (unused by these cards)",
}

TRIGGERS = {   # TriggerType (src/effects/effect_types.h) -> {event, subject}
    "WhenYouPlayMe": {"event": "play", "subject": "self", "src": ET + ":60"},
    "WhenIDie": {"event": "die", "subject": "self", "src": ET + ":71",
                 "note": "with Keyword::Deathknell: repeated by Deathknell doublers (TM:1258)"},
    "WhenIConquer": {"event": "conquer", "subject": "self", "src": ET + ":74"},
    "WhenYouConquerHere": {"event": "conquer", "subject": "here", "src": ET + ":79"},
    "WhenYouPlayAUnit": {"event": "play_unit", "subject": "you", "src": ET + ":99",
                         "note": "on a battlefield card it fires for the player who played the "
                                 "unit there (TM:574-580, 282): subject any_player_here"},
    "WhenYouPlayAUnit+WhenYouPlayASpell+WhenYouPlayAGear+WhenYouPlayMe":
        {"event": "play_card", "subject": "you", "src": ET + ":60,98-100"},
    "WhenIWinCombat": {"event": "win_combat", "subject": "self", "src": ET + ":68"},
    "WhenIMoveToFB": {"event": "move_to_battlefield", "subject": "self", "src": ET + ":72"},
    "WhenIAttackOrDefend": {"event": "attack_or_defend", "subject": "self", "src": ET + ":67"},
    "WhenIConquerOrHold": {"event": "conquer_or_hold", "subject": "self", "src": ET + ":76"},
}

CARDDEF = {
    "_src": DB + ":19-48",
    "card_type": "stats.type", "super_type": "stats.supertype (none -> null)",
    "domains": "stats.domains (sorted)", "energy_cost": "stats.energy (units/spells/gear)",
    "power_cost": "stats.power; power_by_domain = {each card domain: power} (Power is paid "
                  "with runes of the card's Domains, GE:9109)",
    "might": "stats.might (units; absent -> 0)",
    "keywords": "keywords[] (+ engine expansion, KEYWORDS)",
    "assault_value": "keyword assault value",
    "ability_text": "NOT used for the graph; only for text_vs_engine.json",
    "dropped": ["id", "def_id", "name", "set_code", "set_name", "public_code", "collector_number",
                "artist", "tags", "rarity", "image_url"],
}

KEYWORDS = {   # how each keyword the 51 cards print is implemented, and how B2 represents it
    "hidden": {"impl": "engine", "src": [GE + ":5775-5779 (hide for [A], canPayHideCost GE:5326)",
                                          GE + ":5405-5436 (play from facedown ignoring base cost)",
                                          GE + ":9304 (hidden card has Reaction timing)"],
               "b2": "keyword + play_mode from_facedown (timing reaction, setup_cost power 1 any, "
                     "cost 0, where = battlefield you control)"},
    "ambush": {"impl": "engine", "src": [PP + ":17-21 (ambushLocation)", PP + ":32",
                                         GE + ":9305 (Reaction timing)"],
               "b2": "keyword + play_mode ambush (timing reaction, where = battlefield with "
                     "friendly units; + enemy units if the class overrides ambushToEnemyBattlefields)"},
    "accelerate": {"impl": "engine", "src": [GE + ":2302-2318 (accelerateAvailable)",
                                             GE + ":2380-2382 ([1] + one Power of the card's Domain)",
                                             GE + ":4885 (enters ready)"],
                   "b2": "keyword + additional_cost optional energy 1 power 1 -> EnterReady self"},
    "assault": {"impl": "engine", "src": [GO + ":466-467 (current_might += assault_value while Attacker)"],
                "b2": "keyword(value=CardDef.assault_value) + static GiveMight self +value while is_attacker"},
    "flow": {"impl": "class flowCost() + engine grant", "src": [GE + ":8382-8392 (trash replay grant, banish_after)"],
             "b2": "keyword + play_mode from_trash cost from flow(...) -> Banish self"},
    "deathknell": {"impl": "class (TriggerType::WhenIDie) + engine doubler", "src": [TM + ":1252-1263"],
                   "b2": "keyword + the class's WhenIDie trigger (event die, subject self)"},
    "hunt": {"impl": "class only (no engine handler: Keyword::Hunt is read nowhere outside types.cpp)",
             "src": ["src/core/types.cpp:140"],
             "b2": "keyword(value = the class's gainXp amount) + the class's WhenIConquerOrHold trigger"},
    "level": {"impl": "class only (applyPassiveAura checks xp)", "src": ["src/core/types.cpp:142"],
              "b2": "keyword(value = the class's xp threshold) + static condition xp_ge"},
    "legion": {"impl": "class only (selfCostReduction, GE:9436)", "src": [GE + ":9435-9436"],
               "b2": "keyword + static ReduceCost condition cards_played_this_turn_ge 1"},
    "ganking": {"impl": "engine (move generator)", "src": [GE + ":5563-5567"],
                "b2": "keyword only (rules flag)"},
    "deflect": {"impl": "engine (choice cost)", "src": [EX + ":722-732"],
                "b2": "keyword only (rules flag); granted by Level 6 via GiveKeyword"},
    "action": {"impl": "engine (timing)", "src": [GE + ":6235-6240"], "b2": "keyword + spell timing action"},
    "reaction": {"impl": "engine (timing)", "src": [GE + ":6236", "src/engine/chain_manager.cpp:678"],
                 "b2": "keyword + spell timing reaction"},
    "rune": {"impl": "engine (card type Rune, class body empty)",
             "src": [GE + ":9109 (recycle a rune of matching domain for 1 Power)",
                     GE + ":9525-9530 (exhaust a ready rune for 1 Energy)"],
             "b2": "two activated abilities: exhaust -> AddEnergy 1; recycle self -> AddPower 1 (card domain)"},
}
