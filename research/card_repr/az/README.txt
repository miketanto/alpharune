az: AlphaZero-style training, Azir focus (2026-11-21; PLAYTEST_PLAN.md)
=====================================================================

The search shows the network what to play (search/REPORT.txt M1: 64-simulation PUCT beats the same
network 64%, 200 simulations 71%). The network then learns the search's choices, and the next round of
search starts from that better network.

Prerequisites built for this run (all research code; nothing in src/ changed)
-----------------------------------------------------------------------------
1. Card vocabulary v2 (cards_v2/). The v1 vocabulary covered only the 51 original cards; every other card
   was the one 'other' key (memory: card-vocab-gap).
   - build_v2.py: every card the engine registers (955 -> 941 keys + 10 tokens; vocab 979; v1 indices kept)
     gets a key and a text vector (A_text's pipeline: bge-small, current CardDef text).
   - rbenv.py / model.py: RB_VOCAB=v2 is the default; RB_VOCAB=v1 restores the old map.
   - Unknown-key scan, random games on all 41 decks: 0 unknown keys.
   - Model: card embedding += W_txt(text vector) + E_res(card), both zero at init; a real card without a
     B2 vector starts on the 'other' row. The v1 vocabulary buffers are rebuilt, not loaded.
   - Parity: G2 gives identical outputs under v1 and v2 where v1 had no unknown cards (Rengar mirror,
     Yi vs Rengar: max |d logp| = 0, max |d v| = 0). On other decks the observation itself changes: pile
     tokens per card instead of one 'other' pile.
   - Replay buffers: tok / act stored as int16 (int8 would have clipped card keys above 127).
2. Public action history (selfplay/rbenv/history.h). The last 48 external decisions of both seats are
   encoded per observer as a separate tensor hist[48][16] (fields in history.h). Card identities follow
   strict visibility; a card seen only by one player is resolved from the encoded state, so a sampled
   world's history agrees with the world.
   - Leak test (search/m0.py leak, G2, 40 games x 8 worlds each, azir:azir, azir:diana, diana:rengar):
     about 140,000 worlds, 0 mismatches with the history compared byte for byte; 0 check problems.
   - Model: H_* block (event embedding, 1 self-attention layer, cross-attention from the board tokens,
     output projection zero at init). G2 with and without history: max difference 0 at init.
3. rbsearch.cpp: history inside simulations; AlphaZero mode (seat_mode gen + az: the search's move is
   played, visit-proportional for the first 30 decisions of a game, Dirichlet noise 0.3 / 0.25 at full
   searches, playout cap randomisation p_full 0.5: 64 simulations recorded / 16 not); per-slot matchups.

Files
-----
gen_az.py    generator (both seats search; records full searches) -> buf/c_*.npz
train_az.py  trainer: CE to visit distribution + (v - z)^2; replay ratio --reuse 4; exports
             run/gen_weights.pt every 100 steps, checkpoints run/az_<step>.pt every 1000 steps
run_az.sh    both together (generator cores 0-11, trainer 12-13), AZ_HOURS (default 6)
eval_az.py   greedy head-to-heads piloting Azir vs G2 (mirror, vs Rengar, vs Diana, with pilot-swap
             baselines) and vs the Azir specialist az_main@300; results eval/results.jsonl

Run 1 (started 2026-11-21 17:35, 8 h): init G2 = g2_main@1000; matchups azir:azir x2, azir:rengar,
rengar:azir, azir:diana, diana:azir (168 slots); measured smoke throughput ~6.4k games/h, ~450k
records/h (half the decisions recorded).
Checks before launch:
- An exported copy of G2 evaluated through eval_az.py reproduced G2 exactly (0.500 mirror).
- The trainer ran on smoke data: KL to the visit distribution fell 0.59 -> 0.23 in 50 steps.

Run 1 stopped at 18:51 (trainer step ~1150; checkpoint run/az_001000.pt kept) to switch on B2 v2.
Run 2, B2 v2 (started 2026-11-21 18:53, 8 h, ends ~02:53): RB_B2=v2.
  - Inputs: init run/az_001000.pt, run dir run_b2, same buffer (records store card keys, not embeddings).
  - Checkpoints record 'b2'. Loaders (league/common.load_net, search/rbsearch.load_net) build each net
    with its own B2 file: G2 / S_A (no 'b2' key) stay v1, so the comparison is fair.
  - First steps: the expected dip from the new embeddings of the 48 cards. Fresh KL to the search went
    0.08 -> 0.17 and fresh agreement 0.76 -> 0.64 at step 50.
  - Evaluation queued (chain_eval.sh, AZ_RUN=run_b2): runs when the run ends; writes eval/REPORT.txt.

Run 2 results and continuation (2026-11-22)
-------------------------------------------
- A generator crash (PermissionError renaming a chunk the trainer had opened) was fixed: gen_az.py writes
  tmp_<name> (outside the trainer's c_*.npz glob) and retries the rename. watchdog.sh restarts the
  generator if it dies and logs free memory. Run 2 ended at step 9944 (02:53). run_cont.sh continued it
  (same buffer, resumes from the last checkpoint, new generator seed) to step 15543 (08:25).
- Greedy play regressed (eval_az, 400 games, Azir pilot vs G2): mirror 0.500 (G2) / 0.490 (@1000) /
  0.430 (@4000).
- Search play improved, then plateaued. search_check.sh / search_check2.sh: the Azir seat searches
  (64 sims, net X) against G2 greedy, 400 seat-balanced games, out in ../search/out/az_*.json:
    net        mirror                 vs Rengar
    G2         0.610 [0.561, 0.657]   0.593 [0.544, 0.640]
    @1000      0.675 [0.628, 0.719]
    @4000      0.703 [0.656, 0.745]
    @9944      0.703 [0.656, 0.745]   0.605 [0.556, 0.652]
  Most of the gain comes in the first 1000 steps; none from 4000 to 9944; none against Rengar.
- Diagnostics: KL to the search's visits is about the same on training and fresh records (0.073 vs
  0.07-0.08): the policy fits its targets, so capacity is not the first suspect. Value MSE is 0.59 on
  training and 0.62-0.70 on fresh data, and did not improve. Open questions (next tests): is the
  64-simulation teacher the limit (more sims at eval; search vs own greedy over steps), is G2 greedy a
  saturating yardstick (checkpoint vs checkpoint, both searching), is the net too small (a 4x net on the
  same buffer, held-out loss)?
- Recipe 2 (train_az2.py, not yet run at scale): completed-Q policy target (Gumbel MuZero style),
  value target mixing z with the root value (--zmix), optional KL anchor to the init net (--anchor).
  Smoke-tested on CPU only.
- Replays: replay_9944/ (seed 9101, @9944 + search vs @9944 greedy); watchable in tools/play_web
  (Watch a replay).
