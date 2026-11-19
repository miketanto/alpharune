# Path A: text embedding: build report

 _Written by the orchestrator from the builder agent's returned report: the workflow harness does not let subagents write report files._

## Method

I built path A in research/card_repr/A_text/. The script build.py turns each card into two strings, embeds them with a local text model, and writes one vector per card. It produced all 171 entries in the output contract. The one thing missing is out/report.md: the harness refused to let me write it, so the report content is in this summary and the missing/cost fields instead.

Inputs, per card:
- **Rules text:** ability_text + effect_text. The full name and the part before the comma become <SELF>. The em dash and the [>] icon both become ':'. Icons are spelled out ([1][R] becomes '1 energy and 1 fury power'; R/G/O/B/Y/P letters read from card_index.json).
- **Reminder text:** every parenthetical is stripped, and a fixed glossary sentence is added for each printed keyword (14 keywords) and 4 rules terms (buff, recall, alone, win a combat).
- **Stats line:** type, supertype, domains, energy cost, power cost with its domain(s), Might, timing, keywords, and additional costs. Timing is Action / Reaction / main phase only, plus the Ambush and Hidden routes, or 'not played from hand' for legends, battlefields and runes. Additional costs are parsed for Accelerate, Hidden, Flow, 'pay X as an additional cost' and Legion-style 'cost X less'. Tags are left out because champion tags are names.
- **The two missing legends** were transcribed from their engine CardDef files: Pridestalker unl-183-219 and Wuju Bladesman ogs-019-024. Their printing: entries use those ids.

Why reminder text is stripped (decided by reasoning, before looking at any distances):
- In both T1 cases, Lonely Poro and Rengar, Trophy Hunter, the printings differ only in reminder text and the dash vs [>] separator. The SPEC requires printing-invariance by construction.
- Stripping alone would lose costs that only appear in reminders, such as Accelerate's [1][R] and Hidden's [A]. The glossary and the stats line put them back.
- build.py asserts that every printing and every rename gives byte-identical inputs, so their vectors are identical.
- For comparison, the keep-reminders variant puts the two Lonely Poro printings 0.037 apart, which is more than most one-detail synthetic changes.

Layout: the stats line and the rules text are embedded separately. Each is L2-normalised, they are concatenated, and the result is scaled by 1/sqrt(2). In a single string the cost and amount numbers barely register: +5 vs +4 Might and Draw 1 vs 2 each move the vector about 0.0003. Segmented, those become 0.0067 and 0.0037.

Primary model is bge-small-en-v1.5 (CLS pooling, out/vectors.json, dim 768). paraphrase-mpnet-base-v2 (mean pooling, dim 1536) is in out/vectors_paraphrase-mpnet-base-v2.json. I chose bge because a paraphrase model is trained to make close wordings coincide, which works against T2; T1 is already handled by construction. bge is also smaller and faster.

Local fixes needed to run: transformers 4.23 cannot read model.safetensors without the safetensors package, and torch 2.0 cannot convert to or from numpy 2. build.py therefore reads the safetensors file itself with torch.frombuffer and returns vectors through .tolist(). A cat/kitten/stock-market similarity check confirms the weights load correctly.

My own distance checks (bge, strip, segmented; not eval/, which was empty):
- Printings and renames are at exactly 0.
- Median distance to the nearest other card is 0.068.
- Synthetic changes range from 0.0005 (pit_rookie Might 3) to 0.081 (Action changed to Reaction).
- Real pairs range from 0.010 to 0.050.

All of these come from build.py and are in out/diagnostics.json for all 8 variants (2 models x 2 reminder modes x 2 layouts).

## Could not represent / limits

- Numbers are weakly represented: Might, cost and amount changes move the vector very little (pit_rookie__might3 0.0005, punch_first__power0 0.0020, defy__cost2 0.0019). There is no guarantee that amount changes point the same way as each other (T2 alignment). This is the main weakness of text embeddings.
- Legends Rengar, Pridestalker and Master Yi, Wuju Bladesman have one printing each, taken from the engine CardDef (unl-183-219, ogs-019-024); cards.json lists no printings for them.
- Keywords granted conditionally (Master Yi, Tempered's Level 6 Deflect/Ganking) are listed in the stats keyword field as if printed; the rules text keeps the condition.
- For two-domain cards the power-cost domain is written as 'X (a or b)', because the data does not say which domain pays it.
- Tags are dropped (they carry champion names), so tribal facts such as Mech and Poro are lost; no card in scope depends on them.
- Scuttle Crab's '0-Might units can conquer and hold' reminder is a general rules reminder; it is stripped and not covered by the glossary.
- The glossary and icon map are hand-written; a keyword not in them appears only as its bare word.

## Cost

- **Wall time:** about 35 min for the agent session (reading the spec and data, writing build.py, two fixes, the run).
- **Compute:** a full build takes about 70 s on CPU (16 cores, no GPU) for all 8 variants. The primary model alone (bge, strip, segmented) takes about 4 s for 171 entries plus 0.7 s to load; mpnet takes about 11 s. Memory stays under 1 GB. No downloads, no APIs.
- **Manual effort (one-off):** the keyword glossary (14 keywords + 4 rules terms), the icon map, the stats template, the additional-cost regexes, transcribing the 2 legends, and coding the 10 synthetic edits. About 1 to 1.5 hours for a human.
- **Marginal cost, one new card:** no manual work if its keywords are already in the glossary; add the printing and rerun (about 20 ms of embedding). A card that is not in card_index.json needs about 2 min of transcription.
- **Marginal cost, one new set:** about 5 to 15 min per new keyword or rules term (one glossary sentence, plus a cost regex if the keyword has a cost), and one map entry per new icon. Rerunning a full set of about 220 cards takes under a minute on CPU.
- **Determinism:** fully deterministic, with no learned parameters.

## Files

- `A_text\build.py`
- `A_text\out\vectors.json`
- `A_text\out\vectors_paraphrase-mpnet-base-v2.json`
- `A_text\out\inputs_strip.json`
- `A_text\out\inputs_keep.json`
- `A_text\out\diagnostics.json`
