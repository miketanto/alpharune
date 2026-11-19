Does auto-paying rune costs weaken the ISMCTS teacher?  (win-rate test)
=====================================================================

Answer: no loss can be seen. In the Rengar mirror (the headline after the scope
change) the auto-pay agent won 52.6% of 2340 games against the unchanged
teacher (95% Wilson CI 50.5-54.6%). The data exclude any loss of the auto-pay
side; a true drop of 2.9 points (unpaired) or 2.5 points (paired by seed) would
have been detected with 80% power. Auto-pay also skips 39% of the teacher's
searches in RR (41% less agent time).

Agents
------
teacher  ismcts:sims=100   IsMctsAgent (honest sampled worlds, uct_c 2.0,
                           score-difference leaf heuristic), unchanged.
autopay  autopay:sims=100  the same IsMctsAgent with the same seeds, except that
                           at every PromptKind::RuneForEnergy / RuneForPower
                           decision it returns legal.front() (the engine's
                           default order) and does not call the agent. Every
                           other decision goes to the agent. The search tree
                           still branches on rune choices inside its own
                           simulations; only the root decision is skipped.
Agent seed per seat = engine_seed*2 + seat (as in smoke/data).

Mirror sync. IsMctsAgent's LiveMirror catches up from GameState::action_history
at the next decision it is asked. It replays every token made since then,
including the rune answers autopay gave without asking it, just as it replays
the opponent's decisions. Checks:
 * 8 check games with --check-mirror: an independent LiveMirror run at every
   decision of both seats. 0 decisions where it was not an exact replay
   (mirror_bad 0 of 2,089), with RIFTBOUND_ISMCTS_CHECK_WORLDS=1: no world
   contradicted its player's knowledge.
 * Every game counts multi-option decisions after setup that the agent answered
   without a search (a broken mirror shows up here). It was 0 in all games, for
   both sides.
 * Determinism: the 8 check games played again without the checks gave
   identical choices and the same winners and fingerprints.
 * Replay: every finished game (3622 mixed + 1561 RR-only = 5183, plus the 8
   check games) was replayed from seed + choice indices with no agent (decision_export
   verify). Fingerprint and observation hash were checked at every multi-option
   decision (824,317 checks), plus the winner and the length. 0
   mismatches (logs/verify_*.txt; verify_rr1 also covers rr1b, which was
   then verified again on its own).

Games
-----
Decks: decks/rengar.txt (R) and decks/yi_rebuild.txt (Y), the smoke-data decks.
Seeds are 9,000,000 + k. They are new: smoke/data used 1..~4000, 50001.. and
100000*k.. (up to 1,150,040). Every seed is played twice: autopay as deck A
(gid AP<k>a) and as deck B (AP<k>b). The two games share the deal and shuffles,
which gives a paired comparison.
 * main1, main2 (k 1001..2000, 2001..3000): the 4 matchups cycled by k % 4.
   Stopped part-way when the scope changed to RR only (main2), or when the
   process count was reduced (main1).
 * rr1, rr1b (k 20001..21000; rr1b = what rr1 had not finished): Rengar mirror
   only. Stopped part-way because RAM ran low (see Caveats).
 * Games are not lost to stops: each game line is flushed when it ends. A stop
   loses only the games in progress, so a few seeds have one game of their pair.

Results (results.json; source raw/*.games.jsonl)
------------------------------------------------
HEADLINE: Rengar vs Rengar, all RR games (rr batches + RR games of main batches)
  games 2340, errors 0, no-winner games 0
  autopay win rate 0.5256   Wilson95 [0.5054, 0.5458]
    autopay as deck A (seat 1): 1180 games, 0.5212 [0.4927, 0.5496]
    autopay as deck B (seat 2): 1160 games, 0.5302 [0.5014, 0.5587]
  paired by seed (1155 complete pairs; autopay won 0/1/2 of the pair:
    189/720/246): mean 0.5247, normal95 [0.507, 0.542]
  smallest detectable drop (two-sided 5%, 80% power): 2.9 points unpaired,
    2.5 points paired; 95% half-width at p=0.5: 2.0 points
  RR-only batches alone: 1561 games, 0.5279 [0.5031, 0.5525]
  RR games of the mixed batches: 779 games, 0.5212 [0.4861, 0.5561]

Speed in RR, per game and side (mean):
                 seconds in decide   searches   rune prompts   multi-option
  autopay        2.09                53.0       37.5 (all autopaid)   87.0
  teacher        3.54                86.3       37.3 (33.8 searched)  86.3
  -> autopay does 39% fewer searches and spends 41% less agent time. Mean game
     5.65 s wall (median 5.38), 13.2 turns. About 0.04 s per search.
  The teacher picked something other than legal.front() at 72% of the rune
  prompts it searched. This matches the offline finding: most rune options are
  identical runes, so the pick is tie-break noise.

SECONDARY: mixed batches, 4 matchups (3622 games, errors 0, no-winner 0)
  overall 0.5182 [0.5019, 0.5345]; paired (1790 pairs) 0.517 [0.504, 0.530]
  by matchup:  RR 779 0.521 [0.486, 0.556]   YY 992 0.513 [0.482, 0.544]
               RY 919 0.521 [0.489, 0.553]   YR 932 0.518 [0.486, 0.550]
  by autopay seat: seat 1 1822 0.513 [0.490, 0.536]; seat 2 1800 0.523 [0.500, 0.546]
  by matchup and seat (the cross matchups mostly show deck strength:
  Rengar beats Yi about 64-66%):
    RR s1 400 0.505  s2 379 0.538 | YY s1 494 0.498  s2 498 0.528
    RY s1 465 0.656  s2 454 0.383 | YR s1 463 0.393  s2 469 0.642
  speed: autopay 1.88 s / 43.7 searches per game; teacher 3.44 s / 76.2
  (43% fewer searches).

Caveats
-------
* Auto-pay comes out slightly ahead (about +2.5 points, CI just above 0.5). A
  plausible reason: the teacher's 100-sim rune picks are mostly noise between
  identical or near-identical runes, and the engine's default order is a sound
  heuristic. The test was set up to detect a loss. Read the gain as "at least
  as strong", not as a proven improvement.
* Strength is measured against this one teacher (sims=100) on two decks only.
  A stronger search, or decks with more rune-sensitive costs, could differ.
* Seconds were measured under load: 8, then 4, then 3 processes on cores 0-7
  (taskset in WSL), with another agent benchmarking on cores 8-15. Absolute
  seconds are rough. The autopay/teacher ratio comes from the same games and is
  reliable.
* Process count: 8 processes for main1, cut to 4 and then 3 at the
  coordinator's RAM rule, and stopped at 10:52 when Windows free RAM fell below
  1 GB. Each process used about 21 MB RSS; the RAM pressure came from other
  processes (vmmem about 5.6 GB), not from this run.
* The 8 check / 8 rerun games (seeds 9,000,001..4) are not in the results.

Files
-----
src/autopay_match.cpp     the tool. It #includes ../../data/src/decision_export.cpp
                          with main renamed (Seat, replay, verify, fingerprints)
src/autopay_match.cmake   EXCLUDE_FROM_ALL WSL target; src/build.sh builds it
src/make_list.py          game lists (lists/*.txt); [RR|YY|RY|YR] restricts the matchup
src/run_play.sh           P shards of one list, taskset -c 0-7 (--check-mirror passes through)
src/run_main.sh           batches in order, stoppable with ~/apw/STOP; src/stop.sh stops now
src/sync.sh               WSL ~/apw -> raw/, logs/
src/make_verify.py, src/run_verify.sh   replay jobs (jobs/) and their run
src/analyze.py            -> results.json
raw/<batch>__s<k>.games.jsonl  one line per game: winner, term, turns, scores,
                          secs, side_a / side_b counters (dec, multi, searches,
                          secs, rune, rune_multi, autopaid, autopaid_multi,
                          rune_searched, rune_nonfront, unsearched_multi,
                          mirror_checked, mirror_bad), choices, checks
logs/                     per-shard stderr and verify_*.txt
