# Network-guided search (roadmap step 3): scope, 2026-11-20

**Why:** PPO plus league training plateaued. Elo rose from 981 to about 1,390 by update 1,400, then stayed flat to update 2,080 (slope 0 ± 16 Elo per 100 updates). The exploiter never beat the main agent more than 43% of the time. Search is the standard way to get policy improvement beyond what the raw network finds.

## Recommendation

**Determinized PUCT ("AlphaZero + ISMCTS"), written in C++ inside rbenv, with the network evaluated in Python on the GPU in batches.**

**Prove it at eval only first.** Then use it as a teacher: Gumbel root selection, and a cross-entropy loss pulling the policy towards the search's improved policy, inside the existing PPO plus league setup.

**Not ReBeL / Student of Games.** Public belief states over an opponent hand drawn from a ~40-card legend pool, plus CFR on subgames, are too costly for one desktop. Revisit only if humans exploit the agent's handling of hidden information.

## What exists (and what is missing)

| Piece | Where | Status |
|---|---|---|
| ISMCTS, single-observer keys, one world per simulation, `SearchSummary` (visits and values per option) | src/agents/mcts_agent.cpp:476-677, mcts_agent.h:85-99 | Reuse. **It is plain UCT with no priors** (`selectUct` 648-667); the score-difference leaf is replaced by the net value. |
| Honest world sampling | `WorldSampler`, src/agents/determinizer.h:82-115; `makeWorld` mcts_agent.cpp:721-746 | Reuse as is. |
| Cloning | `RiftboundState::Clone` (lazy), riftbound_state.cpp:415-434 | About 60-80 µs to clone and take the first step. **An edited world cannot be cloned at a prompt** (169-177), so each simulation re-walks from the root and nodes store no states, as ISMCTS already does. |
| Encoder for any (state, decider) | decision_export.cpp:287-289 | Missing: (1) a const engine accessor on `RiftboundState` (a small src/ change, so a PR); (2) a per-simulation copy of `MoveTrack`; (3) splitting `encode` off the shared env buffers and counters so it is safe across threads. |
| rbenv snapshot / edit (gates 4-5) | rbenv.cpp:891-1029 | Main-phase decisions only; not enough for a search. `rbenv_advise` replays the whole game on every call, so it is fine for diagnostics only. |

## Design

1. **API (no callbacks, no Python GIL problems):** `search_begin(envs, sims)` → `search_leaves()` (glob/tok/act for the leaves awaiting evaluation) → `search_feed(logp, v)` → `search_result()` (visits and π′).
   - A C++ thread pool runs the environments.
   - Each environment keeps 2-4 leaves in flight using virtual loss, giving GPU batches of 512-1024.
2. **At a leaf:** encode the sampled world from the decider's point of view, expand using the net prior, and back up the net value (sign-flipped for the opponent).
   - Inside simulations, rune payments are auto-paid and single-option decisions are auto-taken, as in training.
3. **Budget:** every external decision with 2 or more options is searched. Main-phase decisions get the full budget, prompts about a quarter. Optionally skip a decision when the net's top option has probability above 0.9.
4. **Training (later):**
   - Gumbel AlphaZero at the root (sequential halving, completed-Q π′) at 16-32 simulations.
   - Step 1: PPO acts with the raw net, plus a cross-entropy loss towards π′ on a sample of decisions.
   - Step 2: full Expert Iteration, where the learner acts by search, against league opponents that play their raw nets.

## Leak and bias guards

- **Test:** for every sampled world, `encode(world, root decider)` must be byte-identical to `encode(truth, root decider)`.
- **Strategy fusion at opponent nodes:** compare the opponent's prior from its own view against sampling opponent moves from the net.
- **Opponent deck model = whole legend pool:** honest, but it blurs the guess at the opponent's hand. Add an optional "public decklist" setting; this also feeds the open- vs closed-decklist experiment.
- `world=perfect` must never be used to produce training targets.

## Milestones

| # | Work | Pass test | Effort | Compute |
|---|---|---|---|---|
| M0 | Split `encode`; add the engine accessor; leak test; ms-per-simulation benchmark with a dummy evaluator | Every world encodes identically; ms per simulation measured | ~2 h | minutes |
| M1 | Eval-only PUCT with net prior and value; leaf-queue API; threads | Searched net vs the same net greedy, 400 seat-balanced games with common seeds: **N=64 wins at least 58%**, win rate rises from N=16 to 64 to 200; no regression vs ISMCTS-200 | 4-6 h | 5-15 min per point |
| M2 | Gumbel root; prompt budget; skip confident decisions | Gumbel at N=16 at least matches PUCT at N=64 | 1-2 h | ~30 min |
| M3 | π′ cross-entropy inside PPO plus league | After 200 updates, beats the plateaued main at least 55% (greedy); league Elo +50 | 3-4 h | 1-2 GPU-h |
| M4 | Full Expert Iteration / Gumbel-AZ self-play | Elo above M3; beats M3's main agent with both sides searching | 3-5 h | overnight |

**Stop rule:** if M1 does not pass at N=64 or N=200, the value head is the bottleneck (explained variance 0.68-0.75). In that case, improve the value head (more value-focused training, outcome targets from more games) before building M2+.

## Throughput (estimate; M0 measures it)

- About 0.65 ms per simulation on one thread: building the world 0.15-0.4 ms, walking the tree 0.05-0.15 ms, encoding ~0.05 ms, teardown 0.04 ms.
- About 12k simulations per second on the machine, with 16 threads and the GPU evaluating.
- About 70 searched decisions per seat per game.

| Simulations | Eval (one seat searches) | Self-play (both seats search) |
|---|---|---|
| 16 | ~38k games/h | ~19k games/h |
| 64 | ~9.6k games/h | ~4.8k games/h |
| 200 | ~3.1k games/h | ~1.5k games/h |

For comparison, PPO plays about 133k games/h and ISMCTS-100 about 950 games/h per process.

## Risks

- A noisy value head limits strength at small budgets.
- Building the root at a prompt means replaying from the last main-phase decision, which adds 50-300 µs per simulation.
- WSL thread contention and Windows memory pressure: run one heavy job at a time.
