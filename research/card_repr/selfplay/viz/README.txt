viz: annotated HTML replays, self-play network vs ISMCTS-100 (Rengar mirror)
==========================================================================

UPDATE 2026-11-19 (schema v4): this folder now holds the v4 tool. The v2 tool (old network, old replays)
was moved, unchanged apart from relative paths, to v2/; replays/ is still the v2 output.
  * rbenv_viz.cpp = the working-tree selfplay/rbenv/rbenv.cpp (schema v4, the gate-4 playout code included
    and unused) + the same viz patch (ReplayWriter, trace, [NET] / [TEACHER] annotations, rbenv_viz_* C API).
    decision_export.cpp / card_probe.cpp / vocab.json: copies of smoke/data/src/decision_export.cpp (v4),
    C_engine/card_probe.cpp and rbenv/vocab.json. model.py: learner/model.py (VOCAB_JSON -> ./vocab.json).
    rbenv_viz.py: rbenv/rbenv.py + the viz bindings.
  * build_viz.sh builds ~/riftbound/viz_build/librbenv_viz4.so (nice 10, one compile; the v2 library
    librbenv_viz.so stays). rbenv's own build (build_pic) is not touched.
  * play_viz.py: default network runs/gate4_B2/latest.pt. Net options come from the checkpoint as in
    gate4/po.py; --greedy plays the top choice. run_games.sh plays GREEDY, nice 10, 3 at a time, into replays_v4/.
  * transcript.py writes a readable .txt next to each replay (replays/ and replays_v4/).
  * New games: replays_v4/ (6/6 won). Old vs new comparison: COMPARE.txt.
  * Check: in all 6 v4 games, every NET / TEACHER "*" line matches the decision panel's "==>" line (685 of 685).
  * The engine writes GAME_OVER and the final scoring lines after the last snapshot, so they are not in the
    HTML; the transcripts take the result from summary.tsv.

----- original (v2) notes; paths now under v2/ -----
~/riftbound/viz_build (WSL).

Network: runs/gate2v2_sanity_B2_ask/latest.pt (B2, --ask-head, schema v2), loaded with the
HEAD (25d8dd6) copy of model.py. It picks moves by SAMPLING from its policy, as in the 86% eval
against ISMCTS-25 (verify_eval.py / act_rows); torch is seeded with the game seed.
Opponent: ismcts:sims=100, run inside C++ (agent seed = engine_seed*2+seat, as in rbenv). Autopay is on.

Open replays/game_seed<N>.html in a browser and step through it with the arrow keys
(this is the engine's ReplayWriter: board, decision and trace panels).
  * Network decisions: the trace panel starts with "[NET] value v (win ~p%)", then the top 3
    options by probability. The chosen option is marked "*" and is added when it is not in
    the top 3 ("sampled"). The value also fills the replay's win-probability slot.
  * Teacher decisions: "[TEACHER ismcts:sims=100] root value", then the top 3 options by
    visits with their Q. Caution: ISMCTS scores leaves with the score-difference heuristic,
    so its "win ~%" sits near 50% for most of a game and is not a calibrated probability.
  * Check: in all 6 games, all 701 "*" lines (NET and TEACHER) match the "==>" chosen line
    in the decision panel.
  * This is how ReplayWriter already works, not something added here: the engine's own
    "DECISION #n" / "CHOSE" trace lines for a decision appear on the snapshot before it.
    The decision panel and the [NET]/[TEACHER] lines are on the right snapshot.
  * Both hands are shown (StateRenderer show_hand).

Games (6 played, all shown below; summary.tsv has the raw rows)
------------------------------------------------------------------------------------------------
seed  net seat  winner           final (P1-P2)  turns  decisions (net)  file
7001  P1        ISMCTS-100 (P2)  4-8            12     225 (38)         replays/game_seed7001.html
7002  P1        network (P1)     8-6            15     357 (66)         replays/game_seed7002.html
7003  P1        network (P1)     8-1            11     236 (42)         replays/game_seed7003.html
7101  P2        network (P2)     7-8            15     375 (78)         replays/game_seed7101.html
7102  P2        network (P2)     2-8            11     251 (52)         replays/game_seed7102.html
7103  P2        network (P2)     5-8            16     457 (95)         replays/game_seed7103.html
Network 5/6: 2/3 as seat 1 and 3/3 as seat 2. That is a tiny sample, not an evaluation.
Recommended viewing: 7001 (the loss), 7101 (closest game), 7003 (one-sided win).

Notes from the annotations (python3 analyze.py lists value drops and low-probability picks)
- 7001 (LOSS): The network led 4-3 after turn 7, then lost.
  * Turn 7: it sampled "Play Pit Rookie -> Base" at 2.4% (Emperor's Dais was 91%). It also
    sent two 1-Might Irresistible Faefolk to Emperor's Dais.
  * Turn 8: ISMCTS moved Inferna in, killed both Faefolk and conquered (4-5).
  * Turn 9: the network attacked Star Spring piecemeal. Pit Rookie went alone: adding First
    Mate was 72%, but it sampled "Move no more units" at 28%. Pit Rookie died, then First
    Mate attacked alone and traded with Noxus Hopeful. Its value fell from +0.10 to -0.50 to
    -0.73.
  * From there ISMCTS held and conquered to 8. Its root value rose to +0.3/+0.4 only once it
    was ahead.
  * The bad moves here are mostly low-probability samples (2-28%). With argmax play, the
    turn-7 and turn-9 mistakes would not have been made.
- 7002 (win 8-6): The network led from turn 5 and its value was +0.8 or more from turn 7.
  Rengar went to the opponent's Emperor's Dais at turn 7 (a 9.5% sample) and it still won.
- 7003 (win 8-1): A clean snowball: 5-1 by turn 7, network value +0.95. The teacher's root
  value fell to -0.78 by turn 10, so both agents agreed it was lost.
- 7101 (win 8-7, as P2): The closest game; scores were level 5-5 and 6-6. At turn 12 ISMCTS
  attacked Emperor's Dais one unit at a time into Ferrous Forerunner and lost Inferna, then
  Grim Apothecary. That is the same piecemeal error the network made in 7001. The network's
  value was +0.7 to +0.9 throughout, even when 6-7 down, which looks over-confident. Its
  turn-13 attack used several sampled choices (12-16%).
- 7102 (win 8-2, as P2): The network was ahead from turn 3, with value +0.96 at turn 8.
  Several sampled unit picks (2-5%) did no harm.
- 7103 (win 8-5, as P2): The longest game (16 turns, 95 network decisions). Value was +0.85
  or more from turn 8. Its low-probability picks are mostly unit-choice prompts.
Patterns across the games:
  * The network's value goes up quickly once it leads on points, and stays high even when
    the score is close (7101).
  * Its worst moves are low-probability samples. Sampling (not argmax) is how it was
    evaluated, so the replays show that play.
  * Mulligan probabilities are flat (about 10-16% per option) and every game's mulligan was
    a 6-11% pick. The mulligan policy has not learned much.

How it was built
- rbenv_viz.cpp: a copy of HEAD rbenv/rbenv.cpp with these additions. For env 0 it attaches
  a ReplayWriter (StateRenderer, on_log trace lines, phase seams, engine->on_decision) and
  writes replays/game_seed<seed>.html when the game ends. It adds ISMCTS lastSearch()
  annotations and the C API calls rbenv_viz_config (dir, opponent seat, labels),
  rbenv_viz_option_text (renderChosenAction of a legal option) and rbenv_viz_note. The
  opponent can play seat 1 (opp_seat=1), which is how the network played P2.
- decision_export.cpp: snapshot of smoke/data/src/decision_export.cpp. That file is
  gitignored, so it has no HEAD version. The copy was taken from the working tree at
  16:05 while the other agent was editing it. Its only change since the commit is additive:
  schema v3 Encoder::optCost and an "oc" JSON field. The HEAD rbenv code never calls them,
  so the v2 observations are unchanged. Its card_probe.cpp include points to card_probe.cpp
  here, which is the HEAD copy.
- model.py: HEAD copy, with VOCAB_JSON pointed at vocab.json here (the HEAD copy).
  rbenv_viz.py: the HEAD rbenv.py, plus the viz arguments, bindings and LIB_PATH.
- build_viz.sh: compiles only rbenv_viz.cpp and links librbenv_viz.so with rbenv's own link
  line. It links against a private copy of the WSL build_pic (~/riftbound/viz_build), so it
  needs no cmake and does not touch rbenv's target or build dir. Viz sources are copied to
  ~/riftbound/alpharune/research/card_repr/selfplay/viz/ (a new folder).
- Usage (WSL):
    bash build_viz.sh
    python3 play_viz.py --seed 7001 --net-seat 1 [--opp ismcts:sims=100] [--greedy]
    bash run_games.sh 7001:1 7002:1 7003:1 7101:2 7102:2 7103:2   # parallel, ~8 s total
    python3 analyze.py
