# Archetypes of the 41 deck lists in alpharune/decks (2026-11-20)

**How the labels were assigned.**
- **Web labels:** taken from the riftbound.gg tier list and guides, eneba, Cardsrealm and Comicbook. They describe tournament lists, not necessarily ours.
- **Stats labels:** computed from our lists, using the card table exported by `riftbound --dump-registry`.
  - avg cost is energy + power, champion included, with no cost reductions applied.
  - cheap means cost ≤ 2; big means cost ≥ 5.
  - The removal, draw and token counts are regex counts over card text, so they are approximate.

| deck file | legend | units/spells/gear | avg cost | cheap/big | web label | stats label | **use** |
|---|---|---|---|---|---|---|---|
| lillia | Lillia | 8/23/9 | 2.30 | 29/3 | aggro | low-curve spells | **aggro (spell tempo)** |
| rell | *Lucian* (no Rell legend) | 25/0/15 | 2.33 | 27/0 | — | low-curve units + equipment | **aggro (equip)** |
| yasuo_546 | Yasuo | 22/12/6 | 2.48 | 31/3 | control (stuns) | lowest unit curve | **aggro** (this list) |
| azir | Azir | 1/22/17 | 2.52 | 24/3 | control-oriented | spells/gear that make soldiers | **midrange (tokens)** |
| reksai | Rek'Sai | 22/18/0 | 2.67 | 21/2 | — | cheap, 11 token cards | **aggro (go-wide)** |
| yasuao | Yasuo | 17/22/1 | 2.75 | 24/6 | control (stuns) | cheap spells | **tempo** |
| yi_rebuild | Master Yi | 18/20/2 | 2.77 | 24/5 | tempo / midrange | cheap, spell-heavy | **tempo** |
| leona | Leona | 13/16/11 | 2.80 | 23/7 | midrange stabiliser | removal 13 | **midrange** |
| ambessa | Ambessa | 20/19/1 | 2.90 | 14/2 | board deck | flat curve | **midrange** |
| diana | Diana | 17/21/2 | 2.90 | 18/3 | — | draw 10, removal 10 | **midrange** |
| vex | Vex | 20/16/4 | 2.95 | 21/5 | holding / control | — | **control (stuns)** |
| ezreal | Ezreal | 12/20/8 | 3.00 | 18/3 | control | removal 25 | **control** |
| irelia | Irelia | 17/16/7 | 3.00 | 21/4 | — | balanced, draw 9 | **tempo** |
| sivir | Sivir | 22/15/3 | 3.00 | 13/1 | miracle | unit-heavy | **midrange** |
| ahri | Ahri | 18/17/5 | 3.10 | 20/9 | tempo / combo | draw 14 | **tempo (value)** |
| poppy | Poppy | 21/16/3 | 3.10 | 16/5 | holding | — | **midrange (holding)** |
| rengar | Rengar | 30/10/0 | 3.15 | 12/3 | aggro | unit-heavy, mid curve | **aggro** (Ambush/Accelerate; the stats don't show it) |
| kennen | Kennen | 20/15/5 | 3.17 | 19/3 | tempo | — | **tempo** |
| vi | Vi | 22/12/6 | 3.20 | 16/6 | conquest | draw 13 | **midrange (aggressive)** |
| khazix | Kha'Zix | 24/16/0 | 3.23 | 11/4 | — | unit-heavy, low draw | **midrange (beatdown)** |
| missfortune | Miss Fortune | 20/17/3 | 3.25 | 15/7 | aggro (burn) | mid curve | **aggro / midrange** |
| lucian | Lucian | 20/9/11 | 3.27 | 11/4 | midrange | — | **midrange** |
| daisy | *Ivern* (40 units) | 40/0/0 | 3.30 | 18/4 | — | units only | **midrange (holding)** |
| ivern | Ivern | 26/14/0 | 3.30 | 17/7 | holding | — | **midrange (holding)** |
| mel | Mel | 17/23/0 | 3.33 | 22/6 | midrange value | draw 13 | **midrange** |
| ornn | Ornn | 15/12/13 | 3.33 | 21/12 | — | gear, removal 14, top end | **midrange-ramp (gear)** |
| akali | Akali | 19/15/6 | 3.35 | 18/7 | value | draw 14, removal 13 | **midrange** |
| draven | Draven | 23/14/3 | 3.38 | 12/6 | midrange / miracle | removal 15 | **midrange** |
| pyke | Pyke | 24/15/1 | 3.60 | 15/4 | — | unit-heavy, removal 11 | **midrange** |
| darius | Darius | 24/8/8 | 3.73 | 15/10 | aggro (Overwhelm) | top-heavy | **midrange (aggressive finisher)** |
| leblanc | LeBlanc | 24/15/1 | 3.75 | 13/12 | combo | removal 16, big 12 | **midrange-control** |
| rumble | Rumble | 20/19/1 | 3.75 | 15/12 | — | big 12 | **midrange (top-heavy)** |
| viktor | Viktor | 16/22/2 | 3.75 | 14/12 | control | removal 17 | **control** |
| teemo | Teemo | 19/18/3 | 3.80 | 9/7 | tempo (traps) | removal 18 | **control (traps)** |
| zed | Zed | 25/9/6 | 3.80 | 11/7 | — | unit-heavy | **midrange** |
| kaisa | Kai'Sa | 20/20/0 | 4.20 | 13/11 | aggro or control | high curve, draw 11 | **control** |
| renata | Renata | 20/17/3 | 4.33 | 8/13 | — | removal 14, tokens 15 | **control** |
| nasus | Nasus | 19/21/0 | 4.40 | 20/15 | — | draw 17, unit Might 4.8 | **ramp** |
| lux | Lux | 13/21/6 | 4.62 | 13/16 | control | big 16 | **control-ramp** |
| jayce | Jayce | 4/25/11 | 5.05 | 15/16 | ramp | 4 units at Might 9 | **ramp** |
| fiora | Fiora | 15/17/8 | 5.33 | 15/17 | midrange (gear) | highest curve, unit Might 6.5 | **ramp** (this list) |

## Archetype buckets (the "use" column)

- **Aggro:** Rengar, Lillia, Rek'Sai, Yasuo (yasuo_546), Lucian (rell.txt), Miss Fortune.
- **Tempo:** Master Yi, Yasuo (yasuao), Irelia, Kennen, Ahri.
- **Midrange:** Azir, Diana, Leona, Ambessa, Sivir, Vi, Kha'Zix, Lucian, Mel, Akali, Draven, Pyke, Darius, Rumble, Zed, Ornn, LeBlanc.
- **Holding:** Ivern, Ivern (daisy.txt), Poppy.
- **Control:** Kai'Sa, Ezreal, Viktor, Vex, Teemo, Renata, Lux.
- **Ramp:** Nasus, Jayce, Fiora.

## Caveats

- The labels in this table don't account for keywords: Rengar's Ambush and Accelerate, and Darius's Overwhelm, play faster than their curves suggest.
- `rell.txt` is a Lucian list and `daisy.txt` is an all-unit Ivern list. There is no Rell legend.
- Our Fiora and Yasuo lists differ from the web descriptions, so these labels follow the list.
