# Generates research/card_repr/audit_gate0/facts.json (gate 0 observation audit, Rengar deck).
import json, collections, sys

OUT = sys.argv[1]
E = "research/card_repr/smoke/data/src/decision_export.cpp"
GE = "src/engine/game_engine.cpp"

# Observation field shorthands (rbenv layout = exporter layout, SCHEMA.txt / rbenv README)
F = {
 "ctrl": "tok[c][2] controller", "owner": "tok[c][3] owner", "zone": "tok[c][4] zone",
 "loc": "tok[c][5] loc", "exh": "tok[c][6] exhausted", "stun": "tok[c][7] stunned",
 "dmg": "tok[c][8] damage", "might": "tok[c][9] might (current, incl. Assault while attacker)",
 "buffs": "tok[c][10] buffs", "tmp": "tok[c][11] temp_might", "istok": "tok[c][12] is_token",
 "fd": "tok[c][13] facedown", "combat": "tok[c][14] combat", "key": "tok[*][1] card key -> R(card)",
 "bfctrl": "tok[b][3] controller", "bfcont": "tok[b][4..5] contested/contested_by",
 "bfcombat": "tok[b][7..8] combat/attacker", "bffd": "tok[b][10..11] facedown_self/opp",
 "bfscored": "tok[b][12..13] scored_this_turn",
 "kitem": "tok[k] chain item (key, controller, is_ability...)", "kchosen": "tok[k][11..15] chosen object tokens",
 "g69": "g[69] cards played this turn self (g[70] opp)", "pool": "g[19..38] rune pools", "runes": "g[39..62] runes on board by domain ready/exhausted",
 "phase": "g[1] phase, g[2] is turn player", "timing": "g[63] chain size, g[64] showdown, g[66] priority, g[67] focus",
 "pk": "g[68] prompt kind + psrc", "fin": "g[71] asked at finalization",
 "act_dst": "act[30] dst", "act_mv": "act[12..15] mv", "act_v": "act[31..32] v", "act_ch": "act[8..11] ch",
 "handopp": "g[10] opp hand size", "trash": "g[13..14] trash sizes", "deck": "g[11..12] main deck sizes",
}

recs = []
def r(card, key, kind, fact, src, cls, field, b2, gap="", diff=""):
    d = dict(card=card, key=key, kind=kind, fact=fact, source=src, mapping=cls, field_or_derivation=field, b2=b2)
    if diff: d["difficulty"] = diff
    if gap: d["gap"] = gap
    recs.append(d)

# ---------------- Legend
c, k, s = "Rengar, Pridestalker (legend)", "rengar_pridestalker", "src/cards/legends/0744_pridestalker.cpp"
r(c,k,"rules","Triggers when its controller plays a unit (any play incl. effect plays, e.g. Thrill replay)", s+":22; src/engine/trigger_manager.cpp:596","OBSERVABLE",F["kitem"]+" (key rengar_pridestalker, is_ability) once triggered; legend itself: tok[c] zone=7","covered (trigger play_unit/you)")
r(c,k,"rules","Target pool: any unit on the board, either side", s+":24-25","OBSERVABLE",F["ctrl"]+", "+F["loc"]+"; unit type via R(key)","covered (side any, zone board)")
r(c,k,"rules","Effect: +1 Might this turn (temporary)", s+":30","OBSERVABLE",F["tmp"]+", "+F["might"],"covered (duration turn)")
r(c,k,"rules","Choice made as the trigger is finalized; the triggering (just-played) unit is the chain item's triggering_subject", s+":21,27-28","MISSING","chain token has no triggering_subject; prompt timing itself is g[71]","n/a (state)","NEW-chain-subject")
r(c,k,"decision","Which unit will fight this turn (attacker/defender, at contested bf) gets the most from +1",s+":30","OBSERVABLE",F["combat"]+", "+F["loc"]+", "+F["bfcombat"]+", "+F["bfcont"],"n/a")

# ---------------- Champion
c, k, s = "Rengar, Trophy Hunter (champion)", "rengar_trophy_hunter", "src/cards/units/0682_rengar_trophy_hunter.cpp"
r(c,k,"rules","Ambush location: a bf where I have units OR where enemy units are", s+":23; src/engine/play_permissions.cpp:17-21,31-32","DERIVABLE",F["ctrl"]+" + "+F["loc"]+" per slot; also act[30] dst of each offered play","covered (any_of friendly/enemy units)","","easy")
r(c,k,"rules","[Ambush] grants Reaction timing (playable at chain priority / in showdown)", "src/engine/play_permissions.cpp:27-32","OBSERVABLE",F["timing"]+", "+F["pk"],"covered (play_mode ambush, timing reaction)")
r(c,k,"rules","Played from the champion zone (chosen champion) as well as hand", E+":283-285","OBSERVABLE",F["zone"]+"=6; act[36] ps=3","n/a")
r(c,k,"decision","Effective cost now (5E+1 Body power; no in-deck modifier) and affordability", s+":39-40","DERIVABLE",F["pool"]+" + "+F["runes"]+" + printed cost from R(key)","covered (stats)","A6","medium")
r(c,k,"decision","Ambush into a combat it can swing: total enemy vs own Might at that bf", s+":41","DERIVABLE","sum "+F["might"]+" by "+F["loc"]+"/"+F["ctrl"]+"; "+F["bfcombat"],"n/a","A7","easy")

# ---------------- Main deck
c, k, s = "Noxus Hopeful", "noxus_hopeful", "src/cards/units/0012_noxus_hopeful.cpp"
r(c,k,"rules","[Legion] cost reduction: costs 2 less if its controller has played >=1 card this turn", s+":18-19; "+GE+":9435","OBSERVABLE",F["g69"],"partial: condition cards_played_this_turn_ge 1 correct, but ReduceCost target zone='board' (it applies in hand)","")
r(c,k,"decision","Effective cost now (4 or 2) and whether it is affordable", s+":19,35","DERIVABLE","R(key) energy 4 - 2*[g[69]>=1], vs "+F["pool"]+"+"+F["runes"],"partial (see above)","A6","medium")
r(c,k,"decision","Sequencing: play a cheap card first to turn Legion on", s+":19","DERIVABLE","own hand tokens (zone=2) + their R costs + "+F["g69"],"n/a","A6","medium")
r(c,k,"rules","Play location: base or a battlefield I control", "src/engine/play_permissions.cpp:37-41","OBSERVABLE",F["bfctrl"]+"; act[30] dst","n/a (generic rule)")

c, k, s = "Pit Rookie", "pit_rookie", "src/cards/units/0136_pit_rookie.cpp"
r(c,k,"rules","Target: another friendly unit on the board", s+":24-30","OBSERVABLE",F["ctrl"]+", "+F["loc"],"covered (friendly, other)")
r(c,k,"rules","Buff does nothing to a unit that already has a buff (CR 427.1.c)", s+":18-20,35","OBSERVABLE",F["buffs"],"partial: graph gives amount n=1 unconditionally, no 'only if unbuffed'")
r(c,k,"rules","Choice made at finalization of the trigger", s+":21,33","OBSERVABLE",F["fin"],"n/a")
r(c,k,"decision","Is there an unbuffed friendly unit worth +1 (else the trigger is wasted)", s+":35","DERIVABLE","any own board token with buffs==0","n/a","","easy")

c, k, s = "Sabotage", "sabotage", "src/cards/spells/0156_sabotage.cpp"
r(c,k,"rules","Opponent hand non-empty (else nothing happens)", s+":24","OBSERVABLE",F["handopp"],"covered")
r(c,k,"rules","Choice set: the NON-unit cards of the revealed hand", s+":47-57","OBSERVABLE","appended card tokens via Encoder::ref() for act[8..11] ch, only during that ChooseObject prompt ("+E+":359-368)","covered (filter type!=unit)")
r(c,k,"rules","Revealed UNIT cards of the opponent's hand (revealed to all, CR 425.1)", s+":31-41","MISSING","never tokenized: only choice objects are ref()'d","covered (Reveal all)","A8")
r(c,k,"decision","Opponent's hand contents after the reveal (later turns)", s+":34-41","MISSING","no known-opponent-hand tokens; own hand only ("+E+":275)","n/a","A8")
r(c,k,"decision","The recycled card is at the bottom of the opponent's deck, known to everyone", s+":88","MISSING","no deck-knowledge field","n/a","A9")
r(c,k,"decision","Probability the opponent holds a non-unit card (decklist minus opp trash/played)", s+":48-51","MISSING","opp trash contents absent (sizes g[14] only)","n/a","A1")
r(c,k,"rules","Timing: own main phase, empty chain, no showdown (no Action/Reaction keyword)", s+":104-109","OBSERVABLE",F["phase"]+", "+F["timing"],"covered (timing main)")
r(c,k,"decision","Effective cost 1E+1 Body power, affordable now", s+":106-107","DERIVABLE","R cost + "+F["pool"]+"+"+F["runes"],"covered (stats)","A6","medium")

c, k, s = "Punch First", "punch_first", "src/cards/spells/0419_punch_first.cpp"
r(c,k,"rules","Target: any unit on the board", s+":19,23","OBSERVABLE",F["loc"]+", "+F["ctrl"],"covered (side any)")
r(c,k,"rules","Effect +5 Might this turn", s+":26","OBSERVABLE",F["tmp"]+", "+F["might"],"covered")
r(c,k,"rules","[Action] timing: own turn main phase or any showdown", s+":43","OBSERVABLE",F["phase"]+", "+F["timing"],"covered (timing action)")
r(c,k,"decision","Target is (about to be) in combat; +5 flips the combat or survives damage", s+":26","DERIVABLE",F["combat"]+", "+F["might"]+", "+F["dmg"]+", per-bf sums","n/a","","medium")
r(c,k,"decision","Effective cost 1E+2 Body power, affordable now", s+":41-42","DERIVABLE","R cost + "+F["pool"]+"+"+F["runes"],"covered","A6","medium")

c, k, s = "Inferna", "inferna", "src/cards/units/0560_inferna.cpp"
r(c,k,"rules","[Assault 2]: +2 Might while an attacker (applied at combat designation)", s+":34,36; "+GE+":7052-7059","OBSERVABLE",F["might"]+" once "+F["combat"]+"=1; printed value via R(key)","covered (static is_attacker GiveMight 2)")
r(c,k,"rules","Effective Assault value (temporary grants/losses, temp_assault_value)", "src/engine/effect_executor.cpp:1305-1311; "+GE+":1496-1500","MISSING","no keyword/value fields on tokens (no in-deck source; opponent cards only)","n/a","A3")
r(c,k,"rules","[Ambush] location: a bf where I have units", "src/engine/play_permissions.cpp:17-21","DERIVABLE",F["ctrl"]+"+"+F["loc"],"covered","","easy")
r(c,k,"decision","Assault matters only if it will attack: will it be an attacker at a contested bf", s+":38","DERIVABLE",F["bfctrl"]+", enemy tokens at loc, own move options act[12..15]/[30]","n/a","A7","medium")

c, k, s = "Grim Apothecary", "grim_apothecary", "src/cards/units/0583_grim_apothecary.cpp"
r(c,k,"rules","May return a friendly unit AT A BATTLEFIELD (incl. itself) to its owner's hand", s+":21-36","OBSERVABLE",F["ctrl"]+", "+F["loc"]+">=1","covered (friendly, zone battlefield, optional)")
r(c,k,"rules","Returns to OWNER's hand (a stolen unit goes back to the opponent)", s+":36","OBSERVABLE",F["owner"],"partial (graph has no owner/controller distinction)")
r(c,k,"rules","[Ambush] location: bf with my units", "src/engine/play_permissions.cpp:17-21","DERIVABLE",F["ctrl"]+"+"+F["loc"],"covered","","easy")
r(c,k,"decision","Bounce value: damage marked, spent play trigger worth replaying, unit about to die in combat", s+":36","DERIVABLE",F["dmg"]+", "+F["might"]+", "+F["combat"]+", R(key) of candidate","n/a","","medium")

c, k, s = "Kinkou Initiate", "kinkou_initiate", "src/cards/units/0659_kinkou_initiate.cpp"
r(c,k,"rules","Draw 1 if my OTHER units on the board have total current Might >= 5 (incl. temp Might)", s+":19-26","DERIVABLE","sum "+F["might"]+" over own tokens with loc>=0 and unit type (R(key); tokens: is_token)","covered (other_friendly_units_total_might_ge 5)","","easy")
r(c,k,"decision","Sequencing: pump/play units before Kinkou to reach 5", s+":26","DERIVABLE","same sum + hand tokens' R Might","n/a","","medium")

c, k, s = "Irresistible Faefolk", "irresistible_faefolk", "src/cards/units/0674_irresistible_faefolk.cpp"
r(c,k,"rules","Triggers when it moves to a battlefield (not when played there)", s+":19","OBSERVABLE","act[0] t=StandardMove with act[12..15] mv + act[30] dst","covered (move_to_battlefield/self)")
r(c,k,"rules","Candidates: enemy units not already at its bf", s+":30-35","OBSERVABLE",F["ctrl"]+", "+F["loc"],"covered (enemy, dest self_location)")
r(c,k,"rules","Excludes units that can't be moved by enemy effects (cant_be_moved_by_enemy)", s+":34; src/core/game_object.h:378","MISSING","no status flag on tokens (only inferable from targets offered at the prompt)","missing filter","NEW-status-flags")
r(c,k,"decision","Result of the pull: combat at the destination (who attacks, total Mights), or emptying an enemy-held bf", s+":45","DERIVABLE","dst slot -> tok[b] by slot, sum Mights per loc","n/a","A7","medium")

c, k, s = "First Mate", "first_mate", "src/cards/units/0132_first_mate.cpp"
r(c,k,"rules","Ready another unit (any unit, either side)", s+":23-30","OBSERVABLE",F["exh"]+", "+F["loc"],"covered (side any, other)")
r(c,k,"decision","An exhausted friendly unit that can then move/attack this turn", s+":30","DERIVABLE",F["exh"]+", "+F["phase"]+", bf state","n/a","","easy")

c, k, s = "Thrill of the Hunt", "thrill_of_the_hunt", "src/cards/spells/0745_thrill_of_the_hunt.cpp"
r(c,k,"rules","Target: a friendly unit on the board", s+":20,23","OBSERVABLE",F["ctrl"]+", "+F["loc"],"covered")
r(c,k,"rules","A token target is banished and gone", s+":42-46","OBSERVABLE",F["istok"],"covered (condition is_token neg)")
r(c,k,"rules","Location options: battlefields the OWNER may play units to (not blocks_unit_play)", s+":52-62","DERIVABLE","all tok[b]; blocks_unit_play only via R(bf key) (no such bf in scope)","partial (dest any_battlefield, no filter)","","easy")
r(c,k,"rules","Location prompt answer is v = index into that FILTERED bf list, asked to the owner; no dst/bf on the action", s+":77-81; src/cards/card_helpers.h:1926-1952; "+E+":439-442","DERIVABLE","act[32] v == bf order index; equals slot order only when no bf is filtered","n/a","NEW-action-ref (C4)","medium")
r(c,k,"rules","Unit comes back as a new object: damage, buffs, temp Might, stun, combat role cleared", s+":36-39","OBSERVABLE",F["dmg"]+", "+F["buffs"]+", "+F["tmp"]+", "+F["stun"]+", "+F["combat"],"partial (PlayIgnoringCost; reset not encoded)")
r(c,k,"rules","The replay is a play: counts as a card played (Darius, Legion), re-fires play triggers, Pridestalker, Star Spring; may pay Accelerate", "src/engine/effect_executor.cpp:3179-3183,3224","DERIVABLE","R(key) of target + "+F["g69"],"partial","","medium")
r(c,k,"rules","[Reaction] timing", s+":106","OBSERVABLE",F["timing"],"covered")
r(c,k,"decision","Save a unit targeted by an enemy chain item / lethal combat", s+":40","OBSERVABLE",F["kitem"]+", "+F["kchosen"]+", "+F["dmg"]+", "+F["might"],"n/a")
r(c,k,"decision","Move a unit from base into a contested bf (combat consequence)", s+":84","DERIVABLE","per-slot Might sums, "+F["bfctrl"],"n/a","A7","medium")

c, k, s = "Rampage", "rampage", "src/cards/spells/0870_rampage.cpp"
r(c,k,"rules","Optional additional cost [O] (1 Body power)", s+":29-35","OBSERVABLE",F["pk"]+" (AdditionalCost 15), "+F["pool"],"covered")
r(c,k,"rules","Paid flag (__rampage_paid card_counter) read at resolution: +2 Might to the friendly unit", s+":28,61-70","MISSING","no per-object memory; after the cost decision nobody (incl. the opponent responding) sees whether it was paid","covered (paid_additional_cost)","A4")
r(c,k,"rules","Needs >=1 friendly and >=1 enemy unit on the board", s+":46-48","OBSERVABLE",F["ctrl"]+", "+F["loc"],"covered")
r(c,k,"rules","Pair chosen at finalization; on the chain the chosen units are known", s+":18-23,52-54","OBSERVABLE",F["kchosen"],"covered (bind A/B)")
r(c,k,"rules","Damage = current Might of each (after +2), simultaneous; lethal if damage+marked >= Might", s+":73-79","DERIVABLE",F["might"]+", "+F["dmg"],"covered (might_of:A/B)","","easy")
r(c,k,"decision","Opponent's response to Rampage on the chain (needs to know whether +2 applies)", s+":70","MISSING","paid flag absent","n/a","A4")
r(c,k,"decision","Effective cost 3E (+[O]) affordable now", s+":95","DERIVABLE","R cost + "+F["pool"]+"+"+F["runes"],"covered","A6","medium")

c, k, s = "Pakaa Cub", "pakaa_cub", "src/cards/units/0135_pakaa_cub.cpp"
r(c,k,"rules","Hide: pay [A] at a bf I control with a free facedown slot", GE+":5768-5793","OBSERVABLE",F["bfctrl"]+", "+F["bffd"]+", "+F["pool"]+"; act t=HideCard bf","covered (setup_cost power any, controlled_by_you)")
r(c,k,"rules","Facedown slot capacity (facedown_max_occupancy, default 1)", "src/core/game_state.h:464; "+GE+":5785","MISSING","not in tok[b]; constant 1 in scope","n/a","NEW-bf-static","")
r(c,k,"rules","A hidden card may be played only on a LATER turn than it was hidden (hidden_on_turn)", "src/core/game_object.h:75; "+GE+":5395,6127","MISSING","no per-object memory; inferable only when its play action is listed","missing: graph lacks the next-turn restriction","A4")
r(c,k,"rules","Playable from facedown only while I control that bf; lost when control is lost", GE+":6115-6121","OBSERVABLE",F["bfctrl"]+", "+F["fd"],"covered (controlled_by_you)")
r(c,k,"decision","Hidden matters at a bf I control that may be attacked; opponent facedown cards are unknown (key 'hidden')", E+":255-257","OBSERVABLE",F["bffd"]+", "+F["bfcont"]+"; key 'hidden' for opp facedown","n/a")

c, k, s = "Ferrous Forerunner", "ferrous_forerunner", "src/cards/units/0344_ferrous_forerunner.cpp"
r(c,k,"rules","[Deathknell]: on death, two 3-Might Mech unit tokens in my base", s+":18-25","OBSERVABLE","R(key) before; after: tokens key token:mech, "+F["might"]+", "+F["istok"],"covered (CreateToken x2 might 3 tag mech)")
r(c,k,"decision","Trading it away is cheap (death gives 6 Might of tokens)", s+":45","OBSERVABLE","R(key)","covered")
r(c,k,"decision","Effective cost 6E+1 Fury power", s+":41-42","DERIVABLE","R + pools","covered","A6","medium")

c, k, s = "Kai'Sa, Survivor", "kai_sa_survivor", "src/cards/units/0039_kai_sa_survivor.cpp"
r(c,k,"rules","[Accelerate]: optional +[1][Fury] to enter ready", s+":40; "+GE+":2302-2310","OBSERVABLE",F["pk"]+" (Accelerate 13), "+F["pool"],"covered (additional_cost energy1 power1 card-domain, EnterReady)")
r(c,k,"rules","When I conquer (gain control of a bf): draw 1", s+":17-19","DERIVABLE","moving to a bf with no enemy units / winning combat: per-slot tokens + "+F["bfctrl"],"covered (conquer/self)","A7","medium")
r(c,k,"decision","Accelerate only pays if it can move and conquer this turn", s+":41","DERIVABLE",F["phase"]+", bf states, pool","n/a","A6","medium")

c, k, s = "Nidalee, Cat Form", "nidalee_cat_form", "src/cards/units/0676_nidalee_cat_form.cpp"
r(c,k,"rules","When I win a combat (remain after combat): draw 1", s+":17-19","DERIVABLE",F["combat"]+", per-bf Might sums, "+F["dmg"],"covered (win_combat/self)","","medium")
r(c,k,"rules","[Ambush] location: bf with my units", "src/engine/play_permissions.cpp:17-21","DERIVABLE",F["ctrl"]+"+"+F["loc"],"covered","","easy")
r(c,k,"decision","Effective cost 3E+1 Body", s+":37-38","DERIVABLE","R + pools","covered","A6","medium")

c, k, s = "Darius, Trifarian", "darius_trifarian", "src/cards/units/0027_darius_trifarian.cpp"
r(c,k,"rules","Triggers on my SECOND card of the turn (incl. himself), by the play ordinal stamped on the played card", s+":30-37; "+GE+":3764","OBSERVABLE",F["g69"]+" (next play is 2nd iff g69==1)","covered (nth_card_this_turn 2)")
r(c,k,"rules","Ordinal of a spell already on the chain (trigger fires at its resolution, after responses)", s+":26-29,36","MISSING","the __announce_play stamp is not on tok[k]/tok[c]; reconstructable only from g69 history","n/a","A4")
r(c,k,"rules","Tokens are not cards (excluded)", s+":35","OBSERVABLE",F["istok"],"covered (flagged)")
r(c,k,"rules","Effect: +2 Might this turn and ready me (only if on the board)", s+":39-41","OBSERVABLE",F["tmp"]+", "+F["exh"],"covered")
r(c,k,"decision","Hold a card so the 2nd play happens with Darius on board; Thrill replay counts", s+":30-37","DERIVABLE","hand tokens + "+F["g69"],"n/a","","medium")

c, k, s = "Pyke, Dockside Butcher", "pyke_dockside_butcher", "src/cards/units/0590_pyke_dockside_butcher.cpp"
r(c,k,"rules","Optional additional cost [R] (1 Fury power)", s+":17-19","OBSERVABLE",F["pk"]+" (AdditionalCost), "+F["pool"],"covered")
r(c,k,"rules","Paid flag (__pyke_db_paid) is the trigger condition", s+":26-31","DERIVABLE","trigger exists on the chain only if paid: tok[k] key pyke_dockside_butcher is_ability","covered (paid_additional_cost)","A4","easy")
r(c,k,"rules","Effect: ready me, +2 Might this turn", s+":38-40","OBSERVABLE",F["exh"]+", "+F["tmp"],"covered")
r(c,k,"rules","[Ganking]: can move bf->bf", GE+":5556-5568","OBSERVABLE",F["act_mv"]+" + "+F["act_dst"],"covered (keyword only)")
r(c,k,"rules","[Hidden]: next-turn restriction (hidden_on_turn)", GE+":6127","MISSING","no per-object memory","missing (as Pakaa Cub)","A4")
r(c,k,"decision","Ganking move consequences: contest another bf, leave own bf undefended", GE+":5563","DERIVABLE","dst -> tok[b] slot, per-slot sums","n/a","A7","medium")

# ---------------- Battlefields
c, k, s = "Emperor's Dais (bf)", "emperor_s_dais", "src/cards/battlefields/0521_emperor_s_dais.cpp"
r(c,k,"rules","Trigger: when you conquer here", s+":19","OBSERVABLE","tok[b] key emperor_s_dais + "+F["bfctrl"]+"; tok[k] when triggered","covered (conquer/here)")
r(c,k,"rules","Optional cost: [1] energy and return a unit I control here", s+":36-47","OBSERVABLE",F["pool"]+", "+F["runes"]+", own tokens at this slot","covered (cost energy 1 + Bounce here)")
r(c,k,"rules","Effect: 2-Might Sand Soldier unit token here, tag Shurima", s+":53-54","OBSERVABLE","token key token:sand_soldier, "+F["might"]+", "+F["istok"],"partial: tag shurima kept, token name dropped (flagged)")
r(c,k,"decision","Which unit to return: damaged/exhausted, one whose play trigger is worth replaying", s+":43","DERIVABLE",F["dmg"]+", "+F["exh"]+", R(key)","n/a","","medium")

c, k, s = "Seat of Power (bf)", "seat_of_power", "src/cards/battlefields/0531_seat_of_power.cpp"
r(c,k,"rules","On conquer here: draw 1 per OTHER bf I control", s+":21-30","DERIVABLE","count tok[b] with controller=1, slot != this","covered (per other_battlefield_controlled)","","easy")
r(c,k,"decision","Conquer order: take Seat after the others", s+":28","DERIVABLE",F["bfctrl"],"n/a","","medium")

c, k, s = "Star Spring (bf)", "star_spring", "src/cards/battlefields/0771_star_spring.cpp"
r(c,k,"rules","Trigger: a player plays a non-token unit HERE", s+":27,54-57; src/engine/trigger_manager.cpp:575-579","OBSERVABLE",F["act_dst"]+", "+F["istok"],"covered (play_unit any_player_here, non-token)")
r(c,k,"rules","Only the first such play per player per turn (card_counters __starspring_turn/unit/play_p*)", s+":28-53,86-91","MISSING","no per-object memory on tok[b]","covered (once per_player_per_turn)","A4")
r(c,k,"rules","Second play inside one resolution is not first: pending item for another subject", s+":66-71","MISSING","tok[k] has no triggering_subject","n/a","NEW-chain-subject")
r(c,k,"rules","Candidates: my other units here that can move to base (moveToBaseBlocked)", s+":93-101","OBSERVABLE",F["ctrl"]+", "+F["loc"]+" (blocked flag MISSING, rare)","covered (friendly here other, dest base)","")
r(c,k,"decision","Retreat a damaged unit / one losing combat; trigger is still available this turn?", s+":103","MISSING","availability = the missing first-time stamp","n/a","A4")

# ---------------- Runes
for nm, kk, ss in [("Body Rune", "body_rune", "src/cards/runes/0126_body_rune.cpp"), ("Fury Rune", "fury_rune", "src/cards/runes/0007_fury_rune.cpp")]:
    r(nm,kk,"rules","Exhaust: +1 energy; recycle: +1 power of its domain", ss+":28-29 (RuneCard base)","OBSERVABLE",F["runes"]+", "+F["pool"]+" (runes are not tokens)","covered")
    r(nm,kk,"decision","Domain-specific power needs: [O] Body (Rampage, Sabotage, Punch First, Nidalee, Rengar), [R] Fury (Pyke, Kai'Sa Accelerate, Ferrous, Darius)", ss,"OBSERVABLE","g[20..25] power by domain, g[39..62] ready/exhausted by domain","n/a","A10")
    r(nm,kk,"decision","Remaining rune deck composition (domain of next channel)", ss,"DERIVABLE","g[15] rune deck size + board counts vs the known 8/4 split (recycled runes return)","n/a","A10","medium")

# ---------------- Tokens
for nm, kk, ss, m, tag in [("Mech token (Ferrous Forerunner)", "token:mech", "src/cards/units/0344_ferrous_forerunner.cpp:22-25", 3, "Mech"),
                          ("Sand Soldier token (Emperor's Dais)", "token:sand_soldier", "src/cards/battlefields/0521_emperor_s_dais.cpp:53-54", 2, "Shurima")]:
    r(nm,kk,"rules","Identity / printed stats (%d Might unit, no keywords)" % m, ss,"MISSING","vocab has the key (rbenv vocab.json) but no cards.json / B2 graph -> R is OOV/zero; live might in "+F["might"],"missing (no graph)","B1")
    r(nm,kk,"rules","Tag '%s' (selectors could read it)" % tag, ss,"MISSING","tags only via R(card); none for tokens","missing","B1/B5")
    r(nm,kk,"rules","Is a token: not a card (Darius), not 'non-token' (Star Spring), banished = gone (Thrill)", ss,"OBSERVABLE",F["istok"],"n/a")

# ---------------- Cross-card / global facts
c, k = "(global, all cards)", "*"
r(c,k,"decision","Own remaining main deck composition (decklist minus hand/board/trash)", E+":384","MISSING","sizes only g[11]; own trash contents absent","n/a","A9")
r(c,k,"decision","Opponent cards already used (closed-decklist inference)", E+":384,412-413","MISSING","trash sizes only g[13..14]; Rengar deck has no card that reads the trash","n/a","A1")
r(c,k,"rules","Banishment contents", E+":386","MISSING (unused)","sizes only g[17..18]; no in-scope card reads banishment; Thrill banishes and replays at once","n/a","A2")
r(c,k,"rules","Effective keywords of every object (granted/lost)", E+":258-262","MISSING","no keyword fields; in-scope cards only have printed keywords (R suffices) unless the opponent grants/removes","partial (printed only)","A3")
r(c,k,"decision","Effective cost / affordable now for each hand card", E+":405-413","DERIVABLE","R cost + Legion(g69) + Accelerate + pools; legality only via act list","n/a","A6","medium")
r(c,k,"decision","Consequence of each move/play location (contest, combat as attacker, conquer, leave bf undefended)", E+":439-440","DERIVABLE","act dst -> tok[b] slot join + per-slot Might sums","n/a","A7","medium")

json.dump(recs, open(OUT, "w"), indent=1, ensure_ascii=False)
cnt = collections.Counter((x["mapping"]) for x in recs)
print(len(recs), dict(cnt))
miss = collections.defaultdict(list)
for x in recs:
    if x["mapping"] == "MISSING": miss[x.get("gap","?")].append((x["card"], x["kind"]))
for g, v in sorted(miss.items()):
    print(g, len(v), sorted(set(c for c,_ in v)), collections.Counter(kd for _,kd in v))
