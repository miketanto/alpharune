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
  tests/        test_random, test_determinism, test_agreement (v1 data; superseded by
                test_agreement_v2), test_fields (v2 field check), test_opponent,
                bench (throughput), test_multivariant (mixed-variant decks: no errors, no
  unknown keys, determinism); run_all.sh runs the original five.

Decks: RBEnv(..., deck_a='rengar') resolves a name via the repo decks/
(decks/rengar.txt, as decision_export used) or smoke/data/decks/<name>.txt,
or takes a path. Variant decks need variant=<name>, or a list of names
(2026-11-19): all are registered at once (each variant has its own class id and
card key), so a deck may mix variants. One variant set per process.

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

SCHEMA v2 (gate 1, 2026-11-19). rbenv_schema_version() == 2 (RBEnv.schema_version()).
  ALL CHECKPOINTS TRAINED BEFORE 2026-11-19 (schema v1: T 64, F 17, K 46, 3 token
  kinds) ARE INVALID: the input layout changed and the B2 table changed (token
  vectors, 10 fixed cards). They will not load into model.py (shape mismatch), and
  the v1 probe files in learner/probes/*.npz are skipped until re-collected.
  The v1 layout is below the v2 one for reference (v1 = v2 without the additions).

Dims (v2): G 72, T 96, F 24, A 96, K 49, V 80.
  tok kind [0]: 1 card, 2 battlefield, 3 chain item, 4 pile (new).
  card  [2..16] as v1; [17] play ordinal this turn, [18] hidden this turn,
        [19] effective energy cost, [20] effective power cost, [21] affordable now
        (19..21: own hand / champion-zone cards, -1 otherwise), [22] effective
        Assault, [23] effective Shield
  bf    [2..14] as v1; [15] Star Spring stamp used this turn by self, [16] by opp
  chain [2..15] as v1; [16] additional cost paid, [17] source play ordinal,
        [18] triggering subject token (-1)
  pile  [2] owner, [3] zone (4 trash, 5 banishment, 0 own main deck), [4] count;
        one token per (pile, owner, card)
  Known opponent hand cards are card tokens (zone 2, owner 2).
  act   [30] dst also for value-answered Location prompts (real slot); [46] dst
        controller (-1 n/a, 0 none, 1 self, 2 opp), [47] would start a combat as
        attacker, [48] would contest.
  Definitions: smoke/data/SCHEMA.txt "SCHEMA VERSION 2". Max tokens seen in 300
  random Rengar-mirror games: 56 (T 96).
  rbenv_verify / RBEnv.verify(): independent engine-state check of every v2 field
  (tests/test_fields.py).

v1 dims (historical): G 72, T 64, F 17, A 96, K 46, V 80.
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

Results, schema v2 (2026-11-19, gate 1; audit_gate1/REPORT.txt)
  test_random 200 games 0 errors; test_determinism ok (410 steps, 24 games);
  test_multivariant ok (3 decks x 40 games, 0 errors, determinism 315 steps);
  test_fields 300 games: every v2 field equals an independent engine read (0 mismatches);
  test_agreement_v2: rbenv == decision_export v2 JSON on 18 games / 3263 decisions
  (audit_gate1/agree, 12 random-player + 6 ismcts:sims=25 vs random-player).

Results (schema v1, 2026-11-19, WSL, cores 8-15)
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

  6 multi-variant (2026-11-19): 5 variants registered at once, decks
    rengar__exp_train / exp_heldout / exp_heldout_sab1 (cardexp/make_exp_decks.py):
    40 random games each, 0 errors, 0 unknown keys, each deck's three variant keys
    seen; same seed + choices -> identical (315 steps, 24 games). test_determinism
    re-run after the change: ok (410 steps, 24 games).
