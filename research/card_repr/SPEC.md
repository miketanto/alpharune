# Card representation, phase 1: A vs B vs C on tests T1–T3

Goal: a representation of a card built only from what the card **does** under the
rules (never its name, id or printing), compared three ways, before any network is
trained. Scope: the 45 cards of `decks/rengar.txt` + `decks/yi_rebuild.txt`
(with sideboards) plus 6 partner cards used only as minimal-pair partners — 51
cards, listed in `cards.json`.

## The three paths

| Path | Built from | Directory |
|---|---|---|
| A | card text (name replaced by `<SELF>`) + stats line, embedded by a local text model | `A_text/` |
| B | a structured mechanics graph per card (abilities → timing, trigger, condition, cost, effect, target selector, amount, duration; keywords; stats), vectorised by a **fixed, non-learned** encoding | `B_graph/` |
| C | what the engine does: play / trigger / activate the card in fixed probe positions and record state deltas + legality | `C_engine/` |

No trained network in phase 1: comparison is by distances and linear probes only.

## Rules every path follows

- **Printing never matters.** All printings of a card are the same card. A path
  must produce one vector per card key *by construction*; it also emits one vector
  per printing so T1 can confirm they coincide.
- **Name never matters.** Self-references to the card's name become `<SELF>` (A),
  are not represented (B), or cannot appear (C).
- **Costs are first-class.** Energy cost, power cost and power domain(s), plus
  additional / optional costs (e.g. Accelerate, Ambush, activation costs) must be
  in every representation, not only effects. Might for units, too.
- **Timing is first-class:** Action / Reaction / main-phase only, where it can be played.
- Only local compute. No external APIs, no downloads beyond what is already in
  the local HuggingFace cache (`bge-small-en-v1.5`, `paraphrase-mpnet-base-v2`).

## Output contract (`<path dir>/out/`)

- `vectors.json`: `{"dim": N, "vectors": {"<key>": [...], "printing:<printing id>": [...],
  "rename:<key>": [...], "synthetic:<id>": [...]}}` — every card key, every printing,
  a renamed copy of every card, and every synthetic variant in `cards.json`.
  A path that cannot produce some entries omits them and lists them in the report.
- `report.md`: method, design choices, what it could not represent, and **cost**:
  wall time to build, compute, manual effort, and the marginal cost of adding one
  new card (and one new set).

## Tests (scored by `eval/`, identically for all paths)

- **T1 same mechanics → same vector:** printings and renames of a card at distance
  ≈ 0; the partner ranks first among all cards.
- **T2 one detail different → different vector:** each real and synthetic pair in
  `cards.json` above the T1 noise floor; the difference vector aligns with other
  changes of the same field (e.g. amount changes resemble each other).
- **T3 mechanics recoverable:** leave-one-card-out linear probes predict mechanics
  labels from each vector. Provisional labels come from engine/static facts
  (`eval/labels.json`); a hand-checked gold sheet replaces them later.
