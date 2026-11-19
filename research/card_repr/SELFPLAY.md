# No-search self-play loop: what exists, what's missing, cost (2026-11-19)

Goal: test the card representation and train for strength in one experiment.
- Rengar mirror only.
- Reward is win/loss only.
- No search at play time, PPO-style (as in arXiv 2303.05197, Hearthstone).
- Card input conditions: B2 vs id vs random.
- Card checks with known right answers, plus held-out variants (score-based pairs from `smoke/pairs/`).

## What exists

| Piece | Where | Usable as is? |
|---|---|---|
| Pull-driven engine API: `beginGame()` / `currentStep()` / `applyChoice(idx)` → `StepResult{kind, perspective, legal, prompt}` | `src/engine/game_engine.h:45-191` | Yes. This is the env step. |
| Observation exporter: tokens (card key + live state), legal actions with token refs, 72 global ints | `smoke/data/src/decision_export.cpp` (~lines 226-626) | Logic yes, format no: it writes JSON text, and about 3/4 of random-player time was spent writing it |
| Test-only variant cards (`--variant`), including the score pairs | `C_engine/card_probe.cpp`, `decision_export.cpp` | Yes |
| ISMCTS teacher as an opponent seat | `decision_export.cpp` (includes `agents/mcts_agent.cpp`) | Yes, for evaluation games |
| Network (entity tokens, transformer, pointer policy, value) | `smoke/train/train.py` | Yes, after a one-line CUDA fix (line 108) |
| CUDA torch | WSL `python3` (torch 2.7.1+cu126) | Yes. The whole loop can run in WSL. |
| Python bindings for the engine | none. pybind11 is not installed (only pybind11_json); Python.h is present in WSL | Missing. Use a C ABI shared library + ctypes, which needs no downloads |

## What has to be built (research code only, no `src/` edits)

1. **`rbenv`: a vectorised env as a shared library** (WSL target, EXCLUDE_FROM_ALL), about 1–1.5 days.
   - One handle holds N games.
   - C functions: `create(n, deckA, deckB, seed, variant, autopay)`, `observe(...)`, `step(actions[])`, `results(...)`.
   - `observe` fills int32 buffers directly instead of JSON: global[72], tokens[N,T,F] with card indices, actions[N,A,K] with token refs, and masks.
   - In C++, it auto-answers rune payment (auto-pay, shown not weaker) and single-option decisions.
   - Optional opponent seat run in C++ (`random-player`, `ismcts:sims=N`), for evaluation.
   - Tests:
     - the same seed and actions replay identically;
     - observations match `decision_export`'s, by hash;
     - throughput benchmark.
2. **Python actor** (WSL, GPU), about 0.5 day.
   - A few processes, each stepping 256–1024 games at once.
   - One batched forward pass per step; sample actions; store trajectories (obs, action, log-prob, value, seat).
3. **Learner**, about 1 day.
   - PPO with a value baseline (GAE; γ = 1; reward ±1 at game end).
   - Opponent pool of past checkpoints (latest plus a sample of older ones), so it doesn't overfit to its own latest self.
   - Card input conditions: B2, id, random, with 2–3 seeds each.
4. **Evaluation**, about 0.5–1 day. Every K updates:
   - win rate vs random-player, vs ISMCTS at 25 and 100 sims, and vs earlier checkpoints;
   - **card checks** (rule-defined right answers):
     - does "score 1 if behind" get played when behind and held otherwise?
     - does "controller scores" target a friendly unit and "opponent of controller scores" an enemy unit?
   - **held-out variants**: zero-shot use, then games needed to learn to use them.

Total: about **3–4 agent-days** to a working loop, then compute.

## Python interface contract (fixed before the parallel build; both agents code against it)

Lives in `research/card_repr/selfplay/rbenv/rbenv.py`. It runs under WSL `python3` (numpy 2.2.6, torch 2.7.1+cu126).

```python
class RBEnv:
    def __init__(self, n_envs, deck_a, deck_b, seed, variant=None, autopay=True, opponent=None):
        """n_envs games run at once; a finished game auto-resets with the next seed.
        opponent=None: both seats are external (self-play). Otherwise an agent spec
        ('random-player', 'ismcts:sims=25'): seat 2 is played in C++ and only seat 1
        is external. In self-play, which network plays which seat is the caller's
        business: it gets `seat` per env."""
    dims: dict  # G (globals, 72), T (max tokens), F (token fields), A (max actions),
                # K (action fields), V (card-vocab size); vocab in rbenv/vocab.json
                # (index 0 = pad, 1 = 'hidden', then card keys incl. 'synthetic:*', 'token:*')
    def observe(self) -> dict:
        """numpy arrays, decider's view (as in smoke/data/SCHEMA.txt):
        glob   int32 (N,G)
        tok    int32 (N,T,F)   tok[...,0]=kind (1 card, 2 battlefield, 3 chain item),
                               tok[...,1]=card index into vocab, rest = situational fields
        tok_mask bool (N,T)
        act    int32 (N,A,K)   act[...,0]=intent type, token refs are indices into tok (-1 none)
        act_mask bool (N,A)    legal options
        need   bool (N,)       True where an external decision is pending
        seat   int32 (N,)      deciding seat (1 or 2) for envs with need
        pk     int32 (N,)      prompt kind (PromptKind int)"""
    def step(self, choices):  # int array (N,): index into act for envs with need, ignored otherwise
    def results(self) -> list:  # games finished since the last call:
                                # dicts {env, game_seed, winner (1|2|0), length, error}
```

## Expected speed (from `smoke/speed/` measurements; estimates, not measured end to end)
- Engine alone: about 3 ms per game. With auto-pay, about 85 searched decisions per game.
- GPU inference: about 37k positions/s at batch 256, so the GPU becomes the limit at roughly 400 games/s (about 1.5M games/h).
- Realistic target after Python overhead: **50k–200k games/h**.
- Learner: about 24k samples/s, so updating on 10k games (about 1M decisions) takes about 1 minute of GPU time.
- Memory: WSL has about 11 GB. 1,024 concurrent games plus a replay buffer of about 1M decisions (about 1 GB packed) fits.

## Risks
- **PPO on a hidden-information card game** can be unstable or learn slowly. Controls:
  - first show that the id condition beats random-player and then ISMCTS-25;
  - only then compare representations.
- **No-search play is weaker than search** at equal network quality. Beating the ISMCTS-100 teacher may later need the network inside the search (engine work: batched network calls from `IsMctsSearch`).
- **The score variants are artificial.** They test whether the model reads mechanics, not whether it understands real cards.
- **Rengar mirror only:** results may not transfer to other decks. This is accepted for now.
