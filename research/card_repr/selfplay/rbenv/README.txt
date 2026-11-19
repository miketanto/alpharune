rbenv: vectorised Riftbound environment for no-search self-play
================================================================
Plan and Python contract: research/card_repr/SELFPLAY.md. Run under WSL python3.

Files
  rbenv.cpp     C ABI library (rbenv_init/dims/create/observe/step/results/
                counters/destroy). #includes smoke/data/src/decision_export.cpp
                (main renamed) and reuses its Encoder, card keys, registerVariant
                and makeAgent, so observations are the exporter's.
  rbenv.cmake   target `rbenv` (SHARED, EXCLUDE_FROM_ALL), appended to the WSL
                build copy's CMakeLists.txt by build.sh.
  build.sh      copies sources into ~/riftbound/alpharune and builds
                build_pic/librbenv.so. build_pic is a second build dir of the
                WSL copy configured with -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                (the normal build's static engine libs are not PIC) and the
                already-fetched deps (FETCHCONTENT_SOURCE_DIR_*, disconnected:
                no downloads). Cores 8-15, 6 jobs.
  rbenv.py      RBEnv (ctypes + numpy). Library path: LIB_PATH in rbenv.py =
                ~/riftbound/alpharune/build_pic/librbenv.so.
  vocab.json    0 '<pad>', 1 'hidden', 2..52 cards.json keys, 53..74 the 22
                'synthetic:*' variants, then 'token:mech', 'token:sand_soldier',
                'keyword_ability', 'none', 'other' (any other key; counted in
                counters()['unk_key']). V = 80.
  tests/        test_random, test_determinism, test_agreement, test_opponent,
                bench (throughput); run_all.sh runs them all.

Decks: RBEnv(..., deck_a='rengar') resolves a name via the repo decks/
(decks/rengar.txt, as decision_export used) or smoke/data/decks/<name>.txt,
or takes a path. Variant decks need variant=<name> (one variant per process).

Decisions
  Answered in C++: single-option decisions; with autopay=True every
  RuneForEnergy / RuneForPower prompt (legal[0]); with opponent=<spec> every
  seat-2 decision (makeAgent: 'random-player[:pass=P]', 'random',
  'ismcts:sims=N'; agent seed = engine_seed*2+1 as in the exporter).
  Everything else is external (need=True). Decider = prompt.decider, else
  decidingPlayer (as the exporter). seat 1 = deck_a = Player1.
  Engine seed = game_seed & 0x7fffffff (0 -> 1), as the exporter. Env i starts
  with game seed seed+i; each new game takes the next unused seed.
  A game that exceeds 20000 decisions or throws is reported with `error`.

Dims: G 72, T 64, F 17, A 96, K 46, V 80.
  Caps from smoke/data/out (597,994 decisions incl. rune prompts): max 40
  tokens, max 64 actions; lists: tg/ch/mv <= 1, mu <= 2, dm <= 6 pairs.
  Truncation (tokens past T, actions past A, list entries past their cap) is
  counted in counters(); references to dropped tokens become -1. None seen in
  the tests.

glob (N,72): the exporter's "g" exactly (SCHEMA.txt).

tok (N,T,F), tok_mask: the exporter's token list in the same order.
  [0] kind 1 card, 2 battlefield, 3 chain item; [1] vocab index of card key.
  card  [2..15] controller, owner, zone, loc, exhausted, stunned, damage,
        might, buffs, temp_might, is_token, facedown, combat, n_attachments;
        [16] attached_to token (-1)
  bf    [2..14] slot, controller, contested, contested_by, showdown, combat,
        attacker, staged, facedown_self, facedown_opp, scored_self,
        scored_opp, is_token; [15..16] 0
  chain [2..9] controller, pos_from_top, resolving, is_spell, is_ability,
        is_activated, finalized(status), is_equip; [10] source token (-1);
        [11..14] chosen object tokens (-1); [15] number chosen; [16] 0

act (N,A,K), act_mask: the legal options in engine order (index = choice).
  0 t (IntentType)     1 src (-1)       2 abs (-1)      3 ai (-1)
  4..7 tg tokens       8..11 ch         12..15 mv       16..17 mu   (-1 pad)
  18..29 dm pairs (token, damage) x6 (-1 pad)
  30 dst (-1 none; 0 base, 1+i battlefield i)
  31 has v (0/1)       32 v (0 if none) 33 ci chain-item token (-1)
  34 nc vocab index of the named card (0 none)
  35 bf slot (-1)      36 ps play source (0 hand)
  37 alt  38 plc  39 dec (0/1)     40 first (-1 unless t == PlayFirstDecision)
  41..45 counts of tg, ch, mv, mu, dm (before truncation)

Extra observe() keys: psrc (exporter psrc), nlegal (options before truncation).
results(): {env, game_seed, winner, length (all decisions), n_ext, turns, error}.

Results (2026-11-19, WSL, cores 8-15)
  1 random legal actions, 200 Rengar-mirror games: 0 errors (winners 101/99).
  2 determinism: 24 games / 410 steps identical observations and results.
  3 agreement with decision_export JSON (autopay=False, replay of recorded
    choices): base_random__s0 20 Rengar-mirror games, 4018 decisions;
    base_teacher__s0 10 games, 1768 decisions: 0 mismatches (tok, act, g,
    psrc, seat), winners equal.
  4 throughput (uniform random over legal options, autopay on):
      n_envs 64,  1 proc: 152 games/s, 18.6k ext decisions/s
      n_envs 64,  4 proc: 431 games/s, 52.3k decisions/s
      n_envs 256, 1 proc: 130 games/s, 16.3k decisions/s
      n_envs 256, 4 proc: 392 games/s, 50.1k decisions/s
  5 opponent='random-player': 100 games, 0 errors, only seat 1 external.
    'ismcts:sims=25' also ran (8 games, 0 errors).
