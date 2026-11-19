# Gold label review sheet (draft from provisional labels)

How to read: each card lists its text (reminder text stripped), then its labels. 
**Marked ⚠ = provisional** (derived by scanning the engine C++ or the text; these are the ones to check). Unmarked labels come straight from the card data.
Binary labels are shown only when 1 (the card has it); everything not listed is 0.

51 cards · 57 labels · notes: {"hush": "no engine class found for printings ['rad-041-167']; engine labels text-derived"}

## Alpha Strike  `alpha_strike`
> [Action] Choose a friendly unit. It deals damage equal to its Might split among enemy units at battlefields. Then for each unit this kills, do this: Gain 1 XP.

- **numbers:** energy_cost=3, power_cost=1, might=None, n_domains=2, n_effects=3⚠
- **kind:** card_type=spell, super_type=signature, timing=action, target_side=both⚠
- **has:** domain_body, domain_calm, eff_damage⚠, eff_kill⚠, eff_xp⚠, kw_action

## Back Off  `back_off`
> [Hidden] [Action] [Stun] a unit. If you played this from your hand, draw 1.

- **numbers:** energy_cost=3, power_cost=0, might=None, n_domains=1, n_effects=2⚠
- **kind:** card_type=spell, super_type=none, timing=action, target_side=any⚠
- **has:** domain_calm, eff_draw⚠, eff_stun⚠, has_additional_cost⚠, kw_action, kw_hidden, kw_stun, needs_play_target

## Body Rune  `body_rune`
> (no text)

- **numbers:** energy_cost=None, power_cost=None, might=None, n_domains=1, n_effects=0⚠
- **kind:** card_type=rune, super_type=none, timing=not_played, target_side=none⚠
- **has:** domain_body

## Brittle Steel  `brittle_steel`
> Kill a gear. [Flow] [4][R]

- **numbers:** energy_cost=2, power_cost=1, might=None, n_domains=1, n_effects=1⚠
- **kind:** card_type=spell, super_type=none, timing=main_only, target_side=any⚠
- **has:** domain_fury, eff_kill⚠, has_additional_cost⚠, has_optional_additional_cost, kw_flow, needs_play_target

## Calm Rune  `calm_rune`
> (no text)

- **numbers:** energy_cost=None, power_cost=None, might=None, n_domains=1, n_effects=0⚠
- **kind:** card_type=rune, super_type=none, timing=not_played, target_side=none⚠
- **has:** domain_calm

## Charm  `charm`
> Move an enemy unit.

- **numbers:** energy_cost=1, power_cost=1, might=None, n_domains=1, n_effects=1⚠
- **kind:** card_type=spell, super_type=none, timing=main_only, target_side=enemy⚠
- **has:** domain_calm, eff_move⚠, needs_play_target

## Darius, Trifarian  `darius_trifarian`
> When you play your second card in a turn, give me +2 [M] this turn and ready me.

- **numbers:** energy_cost=5, power_cost=1, might=5, n_domains=1, n_effects=2⚠
- **kind:** card_type=unit, super_type=champion, timing=main_only, target_side=friendly⚠
- **has:** domain_fury, eff_might_change⚠, eff_ready⚠, has_trigger, trig_WhenYouPlayMe

## Decisive Strike  `decisive_strike`  _(partner card)_
> [Action] Give friendly units +2 [M] this turn.

- **numbers:** energy_cost=5, power_cost=1, might=None, n_domains=2, n_effects=1⚠
- **kind:** card_type=spell, super_type=signature, timing=action, target_side=friendly⚠
- **has:** domain_body, domain_order, eff_might_change⚠, kw_action

## Decree of Focus  `decree_of_focus`
> [Reaction] Choose a friendly unit that's in combat with an enemy Fury unit or that's being chosen by an enemy Fury spell. Give it +4 [M] this turn.

- **numbers:** energy_cost=1, power_cost=0, might=None, n_domains=1, n_effects=1⚠
- **kind:** card_type=spell, super_type=none, timing=reaction, target_side=both⚠
- **has:** domain_calm, eff_might_change⚠, kw_reaction, needs_play_target

## Decree of Strength  `decree_of_strength`
> Choose an opponent. They reveal their hand and you choose a Mind card from it. They recycle that card.

- **numbers:** energy_cost=1, power_cost=0, might=None, n_domains=1, n_effects=1⚠
- **kind:** card_type=spell, super_type=none, timing=main_only, target_side=enemy⚠
- **has:** domain_body, eff_recycle⚠

## Defy  `defy`
> [Reaction] Counter a spell that costs no more than [4] and no more than [A].

- **numbers:** energy_cost=1, power_cost=1, might=None, n_domains=1, n_effects=1⚠
- **kind:** card_type=spell, super_type=none, timing=reaction, target_side=any⚠
- **has:** domain_calm, eff_counter⚠, kw_reaction

## Disarming Rake  `disarming_rake`
> When you play me, you may kill a gear.

- **numbers:** energy_cost=3, power_cost=1, might=2, n_domains=1, n_effects=1⚠
- **kind:** card_type=unit, super_type=none, timing=main_only, target_side=friendly⚠
- **has:** domain_calm, eff_kill⚠, has_trigger, trig_WhenYouPlayMe

## Discipline  `discipline`
> [Reaction] Give a unit +2 [M] this turn. Draw 1.

- **numbers:** energy_cost=2, power_cost=0, might=None, n_domains=1, n_effects=2⚠
- **kind:** card_type=spell, super_type=none, timing=reaction, target_side=any⚠
- **has:** domain_calm, eff_draw⚠, eff_might_change⚠, kw_reaction, needs_play_target

## Emperor's Dais  `emperor_s_dais`
> When you conquer here, you may pay [1] and return a unit you control here to its owner's hand. If you do, play a 2 [M] Sand Soldier unit token here.

- **numbers:** energy_cost=None, power_cost=None, might=None, n_domains=0, n_effects=2⚠
- **kind:** card_type=battlefield, super_type=none, timing=not_played, target_side=friendly⚠
- **has:** eff_bounce⚠, eff_token⚠, has_trigger, trig_WhenYouConquerHere

## En Garde  `en_garde`
> [Reaction] Give a friendly unit +1 [M] this turn, then an additional +1 [M] this turn if it is the only unit you control there.

- **numbers:** energy_cost=1, power_cost=0, might=None, n_domains=1, n_effects=1⚠
- **kind:** card_type=spell, super_type=none, timing=reaction, target_side=friendly⚠
- **has:** domain_calm, eff_might_change⚠, kw_reaction

## Ferrous Forerunner  `ferrous_forerunner`
> [Deathknell] — Play two 3 [M] Mech unit tokens to your base.

- **numbers:** energy_cost=6, power_cost=1, might=6, n_domains=1, n_effects=1⚠
- **kind:** card_type=unit, super_type=none, timing=main_only, target_side=none⚠
- **has:** domain_fury, eff_token⚠, has_trigger, kw_deathknell, trig_WhenIDie

## Fiora, Peerless  `fiora_peerless`
> When I attack or defend one on one, double my Might this combat.

- **numbers:** energy_cost=3, power_cost=1, might=3, n_domains=1, n_effects=1⚠
- **kind:** card_type=unit, super_type=champion, timing=main_only, target_side=friendly⚠
- **has:** domain_body, eff_might_change⚠, has_passive_aura, has_trigger

## First Mate  `first_mate`
> When you play me, ready another unit.

- **numbers:** energy_cost=3, power_cost=0, might=3, n_domains=1, n_effects=1⚠
- **kind:** card_type=unit, super_type=none, timing=main_only, target_side=friendly⚠
- **has:** domain_body, eff_ready⚠, has_trigger, trig_WhenYouPlayMe

## Fury Rune  `fury_rune`
> (no text)

- **numbers:** energy_cost=None, power_cost=None, might=None, n_domains=1, n_effects=0⚠
- **kind:** card_type=rune, super_type=none, timing=not_played, target_side=none⚠
- **has:** domain_fury

## Grim Apothecary  `grim_apothecary`
> [Ambush] When you play me, you may return a friendly unit at a battlefield to its owner's hand.

- **numbers:** energy_cost=3, power_cost=0, might=3, n_domains=1, n_effects=1⚠
- **kind:** card_type=unit, super_type=none, timing=reaction, target_side=friendly⚠
- **has:** domain_fury, eff_bounce⚠, has_trigger, kw_ambush, trig_WhenYouPlayMe

## Hush  `hush`  _(partner card)_
> [Reaction] Counter a spell with Power cost [A][A] or more.

- **numbers:** energy_cost=1, power_cost=2, might=None, n_domains=0, n_effects=1⚠
- **kind:** card_type=spell, super_type=none, timing=reaction, target_side=any⚠
- **has:** eff_counter⚠, kw_reaction

## Inferna  `inferna`
> [Ambush] [Assault 2]

- **numbers:** energy_cost=2, power_cost=0, might=1, n_domains=1, n_effects=0⚠
- **kind:** card_type=unit, super_type=none, timing=reaction, target_side=none⚠
- **has:** domain_fury, kw_ambush, kw_assault

## Irresistible Faefolk  `irresistible_faefolk`
> When I move to a battlefield, you may move an enemy unit to that battlefield.

- **numbers:** energy_cost=2, power_cost=0, might=1, n_domains=1, n_effects=1⚠
- **kind:** card_type=unit, super_type=none, timing=main_only, target_side=enemy⚠
- **has:** domain_body, eff_move⚠, has_trigger

## Kai'Sa, Survivor  `kai_sa_survivor`
> [Accelerate] When I conquer, draw 1.

- **numbers:** energy_cost=4, power_cost=0, might=4, n_domains=1, n_effects=1⚠
- **kind:** card_type=unit, super_type=champion, timing=main_only, target_side=none⚠
- **has:** domain_fury, eff_draw⚠, has_additional_cost⚠, has_trigger, kw_accelerate, trig_WhenIConquer

## Kinkou Initiate  `kinkou_initiate`
> When you play me, draw 1 if your other units have total Might 5 or more.

- **numbers:** energy_cost=3, power_cost=0, might=3, n_domains=1, n_effects=1⚠
- **kind:** card_type=unit, super_type=none, timing=main_only, target_side=friendly⚠
- **has:** domain_body, eff_draw⚠, has_trigger, trig_WhenYouPlayMe

## Lonely Poro  `lonely_poro`
> [Deathknell] — If I died alone, draw 1.

- **numbers:** energy_cost=2, power_cost=0, might=2, n_domains=1, n_effects=1⚠
- **kind:** card_type=unit, super_type=none, timing=main_only, target_side=none⚠
- **has:** domain_calm, eff_draw⚠, has_trigger, kw_deathknell, trig_WhenIDie

## Loyal Poro  `loyal_poro`  _(partner card)_
> [Deathknell][>] If I didn't die alone, draw 1.

- **numbers:** energy_cost=3, power_cost=0, might=3, n_domains=1, n_effects=1⚠
- **kind:** card_type=unit, super_type=none, timing=main_only, target_side=none⚠
- **has:** domain_order, eff_draw⚠, has_trigger, kw_deathknell, trig_WhenIDie

## Machine Evangel  `machine_evangel`  _(partner card)_
> [Deathknell] — Play three 1 [M] Recruit unit tokens into your base.

- **numbers:** energy_cost=5, power_cost=1, might=4, n_domains=1, n_effects=1⚠
- **kind:** card_type=unit, super_type=none, timing=main_only, target_side=none⚠
- **has:** domain_order, eff_token⚠, has_trigger, kw_deathknell, trig_WhenIDie

## Master Yi, Tempered  `master_yi_tempered`
> [Hunt 2] [Level 6][>] I have [Deflect] and [Ganking].

- **numbers:** energy_cost=4, power_cost=0, might=4, n_domains=1, n_effects=1⚠
- **kind:** card_type=unit, super_type=champion, timing=main_only, target_side=none⚠
- **has:** domain_body, eff_xp⚠, has_passive_aura, has_trigger, kw_deflect, kw_ganking, kw_hunt, kw_level

## Master Yi, Wuju Bladesman  `master_yi_wuju_bladesman`
> not in cards/card_index.json; take text/stats from the engine card class (src/cards/legends/)

- **numbers:** energy_cost=None, power_cost=None, might=None, n_domains=2, n_effects=1⚠
- **kind:** card_type=legend, super_type=none, timing=not_played, target_side=friendly⚠
- **has:** domain_body, domain_calm, eff_might_change⚠, has_passive_aura

## Moonlight Affliction  `moonlight_affliction`  _(partner card)_
> [Reaction] Give a unit -10 [M] this turn.

- **numbers:** energy_cost=7, power_cost=0, might=None, n_domains=1, n_effects=1⚠
- **kind:** card_type=spell, super_type=none, timing=reaction, target_side=any⚠
- **has:** domain_mind, eff_might_change⚠, kw_reaction

## Nidalee, Cat Form  `nidalee_cat_form`
> [Ambush] When I win a combat, draw 1.

- **numbers:** energy_cost=3, power_cost=1, might=4, n_domains=1, n_effects=1⚠
- **kind:** card_type=unit, super_type=champion, timing=reaction, target_side=none⚠
- **has:** domain_body, eff_draw⚠, has_trigger, kw_ambush

## Not So Fast  `not_so_fast`
> [Reaction] Counter an enemy spell or ability that chooses a friendly unit or gear.

- **numbers:** energy_cost=2, power_cost=1, might=None, n_domains=1, n_effects=1⚠
- **kind:** card_type=spell, super_type=none, timing=reaction, target_side=both⚠
- **has:** domain_calm, eff_counter⚠, kw_reaction

## Noxian Demolitionist  `noxian_demolitionist`
> When I conquer, you may kill a gear with Energy cost no more than my Might.

- **numbers:** energy_cost=2, power_cost=0, might=1, n_domains=1, n_effects=1⚠
- **kind:** card_type=unit, super_type=none, timing=main_only, target_side=friendly⚠
- **has:** domain_body, eff_kill⚠, has_trigger, trig_WhenIConquer

## Noxus Hopeful  `noxus_hopeful`
> [Legion] — I cost [2] less.

- **numbers:** energy_cost=4, power_cost=0, might=4, n_domains=1, n_effects=0⚠
- **kind:** card_type=unit, super_type=none, timing=main_only, target_side=none⚠
- **has:** domain_fury, has_cost_reduction, kw_legion

## Onslaught  `onslaught`
> Give a unit +6 [M] this turn. [Flow] [4]

- **numbers:** energy_cost=4, power_cost=0, might=None, n_domains=1, n_effects=1⚠
- **kind:** card_type=spell, super_type=none, timing=main_only, target_side=any⚠
- **has:** domain_body, eff_might_change⚠, has_additional_cost⚠, has_optional_additional_cost, kw_flow, needs_play_target

## Pakaa Cub  `pakaa_cub`
> [Hidden]

- **numbers:** energy_cost=3, power_cost=0, might=3, n_domains=1, n_effects=0⚠
- **kind:** card_type=unit, super_type=none, timing=main_only, target_side=none⚠
- **has:** domain_body, has_additional_cost⚠, kw_hidden

## Pit Rookie  `pit_rookie`
> When you play me, buff another friendly unit.

- **numbers:** energy_cost=2, power_cost=0, might=2, n_domains=1, n_effects=1⚠
- **kind:** card_type=unit, super_type=none, timing=main_only, target_side=friendly⚠
- **has:** domain_body, eff_buff⚠, has_trigger, trig_WhenYouPlayMe

## Primal Strength  `primal_strength`  _(partner card)_
> [Action] Give a unit +7 [M] this turn.

- **numbers:** energy_cost=4, power_cost=1, might=None, n_domains=1, n_effects=1⚠
- **kind:** card_type=spell, super_type=none, timing=action, target_side=any⚠
- **has:** domain_body, eff_might_change⚠, kw_action

## Punch First  `punch_first`
> [Action] Give a unit +5 [M] this turn.

- **numbers:** energy_cost=1, power_cost=2, might=None, n_domains=1, n_effects=1⚠
- **kind:** card_type=spell, super_type=none, timing=action, target_side=any⚠
- **has:** domain_body, eff_might_change⚠, kw_action, needs_play_target

## Pyke, Dockside Butcher  `pyke_dockside_butcher`
> [Hidden] [Ganking] You may pay [R] as an additional cost to play me. When you play me, if you paid the additional cost, ready me and give me +2 [M] this turn.

- **numbers:** energy_cost=3, power_cost=0, might=2, n_domains=1, n_effects=2⚠
- **kind:** card_type=unit, super_type=champion, timing=main_only, target_side=friendly⚠
- **has:** domain_fury, eff_might_change⚠, eff_ready⚠, has_additional_cost⚠, has_optional_additional_cost, has_trigger, kw_ganking, kw_hidden, trig_WhenYouPlayMe

## Rampage  `rampage`
> As you play this, you may pay [O] as an additional cost. Choose a friendly unit and an enemy unit. If you paid the additional cost, give the friendly unit +2 [M] this turn. They deal damage equal to their Mights to each other.

- **numbers:** energy_cost=3, power_cost=0, might=None, n_domains=1, n_effects=3⚠
- **kind:** card_type=spell, super_type=none, timing=main_only, target_side=both⚠
- **has:** domain_body, eff_damage⚠, eff_kill⚠, eff_might_change⚠, has_additional_cost⚠, has_optional_additional_cost, needs_play_target

## Rengar, Pridestalker  `rengar_pridestalker`
> not in cards/card_index.json; take text/stats from the engine card class (src/cards/legends/)

- **numbers:** energy_cost=None, power_cost=None, might=None, n_domains=2, n_effects=1⚠
- **kind:** card_type=legend, super_type=none, timing=not_played, target_side=any⚠
- **has:** domain_body, domain_fury, eff_might_change⚠, has_trigger

## Rengar, Trophy Hunter  `rengar_trophy_hunter`
> [Ambush] I can be played to a battlefield where there are enemy units.

- **numbers:** energy_cost=5, power_cost=1, might=6, n_domains=1, n_effects=0⚠
- **kind:** card_type=unit, super_type=champion, timing=reaction, target_side=enemy⚠
- **has:** domain_body, kw_ambush

## Ruin Runner  `ruin_runner`
> I can't be chosen by enemy spells and abilities.

- **numbers:** energy_cost=6, power_cost=0, might=5, n_domains=1, n_effects=0⚠
- **kind:** card_type=unit, super_type=none, timing=main_only, target_side=enemy⚠
- **has:** domain_body

## Sabotage  `sabotage`
> Choose an opponent. They reveal their hand. Choose a non-unit card from it, and recycle that card.

- **numbers:** energy_cost=1, power_cost=1, might=None, n_domains=1, n_effects=1⚠
- **kind:** card_type=spell, super_type=none, timing=main_only, target_side=enemy⚠
- **has:** domain_body, eff_recycle⚠

## Scuttle Crab  `scuttle_crab`
>  When you play me, draw 1. [Deathknell][>] Choose an opponent. They reveal their hand. You can look at their facedown cards this turn. Gain 1 XP.

- **numbers:** energy_cost=2, power_cost=0, might=0, n_domains=1, n_effects=2⚠
- **kind:** card_type=unit, super_type=none, timing=main_only, target_side=both⚠
- **has:** domain_calm, eff_draw⚠, eff_xp⚠, has_trigger, kw_deathknell, trig_WhenIDie, trig_WhenYouPlayMe

## Seat of Power  `seat_of_power`
> When you conquer here, draw 1 for each other battlefield you or allies control..

- **numbers:** energy_cost=None, power_cost=None, might=None, n_domains=0, n_effects=1⚠
- **kind:** card_type=battlefield, super_type=none, timing=not_played, target_side=none⚠
- **has:** eff_draw⚠, has_trigger, trig_WhenYouConquerHere

## Star Spring  `star_spring`
> The first time a player plays a non-token unit here each turn, they may move another unit they control here to its base.

- **numbers:** energy_cost=None, power_cost=None, might=None, n_domains=0, n_effects=1⚠
- **kind:** card_type=battlefield, super_type=none, timing=not_played, target_side=friendly⚠
- **has:** eff_move⚠, has_trigger

## Thrill of the Hunt  `thrill_of_the_hunt`
> [Reaction] Banish a friendly unit, then its owner plays it to any battlefield, ignoring its cost.

- **numbers:** energy_cost=2, power_cost=1, might=None, n_domains=2, n_effects=2⚠
- **kind:** card_type=spell, super_type=signature, timing=reaction, target_side=friendly⚠
- **has:** domain_body, domain_fury, eff_banish⚠, eff_play_free⚠, kw_reaction, needs_play_target

## Zhonya's Hourglass  `zhonya_s_hourglass`
> [Hidden] The next time a friendly unit would die, kill this instead. Recall that unit exhausted.

- **numbers:** energy_cost=2, power_cost=0, might=None, n_domains=1, n_effects=2⚠
- **kind:** card_type=gear, super_type=none, timing=main_only, target_side=friendly⚠
- **has:** domain_calm, eff_kill⚠, eff_recall⚠, has_additional_cost⚠, has_replacement, kw_hidden
