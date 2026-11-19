# Path B schema: card mechanics graph

One JSON object per card (`cards/<key>.json`). There is no name, printing id, set, rarity,
art, or the card's own tags (tags are flavour/synergy labels and are not used by any of
the 51 cards' own text, so they are left out). Token tags created by an effect *are* kept
(`token.tag`), because they are part of what the card does.

```
card
├── stats
│   ├── type        unit | spell | gear | rune | battlefield | legend
│   ├── supertype   null | champion | signature
│   ├── domains     [fury | calm | mind | body | order | chaos]   (sorted)
│   ├── energy      int   (unit/spell/gear; null otherwise)
│   ├── power       int   (unit/spell/gear; 0 if none)
│   ├── power_by_domain  {domain: n}   (the card's domains; "any" if the card has none)
│   └── might       int   (units; 0 if printed 0/blank)
├── keywords        [{kw, value?}]      e.g. {kw: assault, value: 2}, {kw: hunt, value: 2}
└── abilities       [ability]
```

## ability

| field | values |
|---|---|
| `kind` | `spell` (what a spell does on resolve), `triggered`, `static`, `activated`, `replacement`, `play_mode` (an alternative way to play the card: Hidden, Ambush, Flow), `additional_cost` (optional/mandatory extra cost paid as the card is played: Accelerate, "you may pay [R] as an additional cost") |
| `timing` | `main` (main-phase only, no keyword) / `action` / `reaction`; on `play_mode`: the timing that mode grants (`same` = the card's own) |
| `trigger` | `{event, subject, condition?, once?}`; events: `play`, `play_card`, `play_unit`, `die`, `conquer`, `conquer_or_hold`, `win_combat`, `move_to_battlefield`, `attack_or_defend`, `would_die` (replacement). subject: `self`, `you`, `here` (battlefield cards), `any_player_here`, `friendly_unit`. once: `per_player_per_turn`, `next_time` |
| `condition` | list of `{c, v, neg?}` (intervening-if / static "while"): `nth_card_this_turn`, `one_on_one`, `died_alone`, `other_friendly_units_total_might_ge`, `paid_additional_cost`, `played_from_hand`, `only_friendly_unit_at_location`, `xp_ge`, `cards_played_this_turn_ge`, `is_attacker`, `unit_is_token` |
| `cost` | `{energy, power, power_domain (card/any/<domain>), exhaust, recycle_self, optional, additional, other: [effect]}` (other = non-resource costs such as "return a unit you control here") |
| `setup_cost` | Hidden's hide cost (`power 1, any`) |
| `where` | play location for a play_mode / permission: `{zone, filters}` |
| `effects` | ordered list of effect |

## effect

| field | values |
|---|---|
| `verb` | closed vocabulary. From `src/effects/effect_types.h` `EffectType`: `DealDamage, Draw, Kill, GiveMight, GiveKeyword, Buff, Ready, Bounce, Move, Stun, Recycle, Counter, Banish, PlayIgnoringCost, ChooseTarget, GainXP`. Added (no EffectType fits): `CreateToken, Reveal, LookAt, Recall, EnterReady, ReduceCost, AllowPlayTo, CantBeChosen, AddEnergy, AddPower` |
| `target` | selector (below), or `{type: ref, ref: A}` for an object bound earlier in the same ability (`ref: subject` = the trigger's subject) |
| `amount` | `{n}` (constant) or `{ref: "might_of:A"}` (a quantity) with optional `per` (`unit_killed_by_prev`, `other_battlefield_controlled`) and `mult` |
| `duration` | `turn` / `combat` / `while` (static, while the condition holds); absent = permanent / instantaneous |
| `optional` | true for "you may" |
| `condition` | as above, scoped to this effect |
| extras | `dest` (move/play destination: `base`, `self_location`, `base_or_other_battlefield`, `any_battlefield`), `source` (who deals damage), `split` (damage divided), `token {type, might, tag}`, `keyword`, `resource`, `by {side, type}`, `domain`, `chooser` (`owner`), `exhausted`, `after` |

## selector

`{type, side, zone, count, filters?, bind?, optional?, chooser?}`

- `type`: unit, gear, spell, ability, card, player, self (or a list: `["spell","ability"]`)
- `side`: friendly / enemy / any
- `zone`: board, battlefield, base, here, hand, chain, facedown, player
- `count`: 1, n, `all`, `split` (any number, damage divided)
- `filters`: list of `{f, v, neg?}`: `other`, `domain`, `type` (with `neg` = "non-unit"), `energy_cost_le`, `power_cost_le`, `power_cost_ge`, `defends_alone`, `any_of [...]`, `in_combat_with {...}`, `chosen_by {...}`, `chooses {...}`, `has_friendly_units`, `has_enemy_units`, `controlled_by_you`

## Keywords expanded

Keywords stay in `keywords` (with their numeric value) **and** their mechanics are spelled out
as abilities, so two cards that get the same mechanics in different words meet:

| keyword | expansion |
|---|---|
| Hidden | `play_mode from_facedown`, timing reaction, setup_cost power 1 any, cost 0, where = battlefield you control |
| Ambush | `play_mode ambush`, timing reaction, where = battlefield with friendly units |
| Flow [N][D] | `play_mode from_trash`, cost energy N (+power), then Banish self |
| Accelerate | `additional_cost` optional energy 1 + power 1 (card domain) → `EnterReady self` |
| Deathknell | `triggered`, event `die`, subject self |
| Assault N | `static` GiveMight self +N while `is_attacker` |
| Hunt N | `triggered conquer_or_hold` → GainXP N |
| Level N | static condition `xp_ge N` |
| Legion | condition `cards_played_this_turn_ge 1` |
| Action / Reaction | spell `timing` |
| Ganking, Deflect | keyword only (pure static rules flags) |
| Runes | two `activated` abilities: exhaust → AddEnergy 1; recycle self → AddPower 1 (card domain) |

## Example (a 1-energy, 2-power Body Action spell: "Give a unit +5 [M] this turn.")

```json
{"stats": {"type": "spell", "supertype": null, "domains": ["body"], "energy": 1, "power": 2,
           "power_by_domain": {"body": 2}, "might": null},
 "keywords": [{"kw": "action"}],
 "abilities": [{"kind": "spell", "timing": "action", "effects": [
   {"verb": "GiveMight", "target": {"type": "unit", "side": "any", "zone": "board", "count": 1},
    "amount": {"n": 5}, "duration": "turn"}]}]}
```

## Vector encoding (fixed, non-learned)

1. **Paths.** Walk the graph from the root. Every dict that names itself (`kind`, `verb`, `f`,
   `c`, `kw`) becomes a typed path segment, e.g. `kind:spell@action`, `verb:GiveMight`,
   `f:energy_cost_le`; the node itself emits a presence count of 1. Other keys become segments
   (`target`, `amount`, `cost`, `trigger`, ...). The structural list keys `abilities`/`effects`
   are dropped (the labels carry them). List order is ignored (bag of paths).
2. **Leaves.** A string leaf → path `.../key=value` with value 1. A number → path `.../key`
   with the number as value **and** `.../key=value` with value 1. Numbers in `stats` (energy,
   power, power per domain, Might) are raw; numbers inside abilities are `sign(v)·log2(1+|v|)`
   so a −10 does not dominate the structure. `true` → `.../key` = 1; `null`/`false` omitted.
3. **Multi-granularity.** Each root-to-leaf path is also emitted at every shorter suffix:
   `kind:spell@action/verb:GiveMight/amount/n`, `verb:GiveMight/amount/n`, `amount/n`, `n`.
   Long paths keep context (which ability, which effect); short ones let the same field
   change (e.g. any amount) point the same way across different cards.
4. **Feature hashing.** Path string → MD5 → slot `= first 4 bytes mod 8192`, sign = bit 0 of
   byte 5; values summed into an 8192-dim vector. No normalisation. The vocabulary is open:
   a new verb or filter needs no code change and never changes existing vectors.

`out/paths/<key>.json` lists each card's explicit (unhashed) path features.

## Structured block (gate 2 fix, 2026-11-19)

`vectorize(graph)` = the 8192-dim hash above (`hash_vectorize`, unchanged) **++ a structured block**
(`struct_block.py`), so `dim` = 8192 + 3019 = 11211 for both B and B2. Pre-block vectors are kept in
`out/pre_struct/`. The block's vocabularies are frozen in `struct_vocab.json`, derived from the B2
graphs (cards, engine classes, token units, synthetic variants) by `struct_vocab_build.py` plus a few
schema-documented values; anything else goes to the field's `<other>` coordinate.

| part | fields (one-hot unless noted) |
|---|---|
| card header | type, supertype, domains (multi), keywords (multi); numeric: energy, power, Might, keyword values (Assault/Hunt/Level/Shield/Deflect N) |
| ability slot ×3 (graph order) | present, kind, timing, trigger event, trigger subject, once, trigger conditions (condition×polarity, multi), ability conditions (multi), play mode, where zone, where filters (multi), may (optional cost), cost flags (multi); numeric: cost energy, cost power; play_requires |
| effect slot ×4 per ability (graph order) | verb, ref (none / choice0..2 / subject), target side, type, zone, count, binds (which choice index this effect's selector binds), relative_to (ref, side, type), target filters (filter×polarity, multi), chooser, duration, dest, may, effect conditions (multi), amount kind (n / might_of:choiceK / might_of:self), amount per, source (self / choiceK), keyword; numeric: amount (clipped ±10), token Might |
| slot signatures | per slot (card header, each ability header, each effect slot) a dense unit-norm 64-dim code seeded by the MD5 of the slot's whole content: any change inside a slot moves its code by ~√2 |

- **Binding:** ChooseTarget effects whose only job is to bind an object are folded out of the effect
  grid; every reference to a bound object (`ref`, `relative_to`, `might_of:X`, `source`) is replaced by
  the bound selector's side and type plus its choice index (order of binding inside the ability).
- **Order:** slot position = effect / ability order.
- Every categorical field of a present slot emits a value (`none` when absent), so a change is a flip.
- Weights: one per group times `ALPHA` (`struct_block.WEIGHTS`), chosen by `gate2/tune_struct.py` and
  `gate2/tune_sigcap.py`; see `gate2/REPORT_v2.txt`. `python struct_block.py` prints the layout.
