# play_web — play Riftbound against a self-play network in the browser

A local web UI where a human plays a full game against a trained checkpoint: the AlphaZero networks of
`research/card_repr/az/run_b2` (default: step 9944, Azir) or the older ones in `research/card_repr/selfplay/runs`.
**Opponent's thinking** picks the network alone (argmax of its policy) or the network with the research search
(64 or 200 simulations; a few seconds per answer on CPU). The search gets only what its seat may know and samples your
hidden cards. Every game is saved after each answer to `research/card_repr/human_games` and listed under *Watch a replay*.

The card vocabulary is v2 (every card its own key). `RB_VOCAB=v1` starts the server with the 51-card map the
`g1` / `g2` / `ov_main` networks were trained with; the AZ networks need v2.

## Launch

The server runs under WSL Python (numpy + torch, CPU only) and needs the WSL build copy
(`~/riftbound/alpharune`) with `build_pic` set up by `research/card_repr/selfplay/rbenv/build.sh`.

```bash
# once, and after any change to playweb.cpp (builds only the `playweb` target, 2 jobs, cores 12-15)
bash tools/play_web/build.sh

# then, from the repository root (Windows checkout through /mnt/c, or the WSL copy)
CUDA_VISIBLE_DEVICES= taskset -c 12-15 python3 tools/play_web/server.py --port 8765
```

Open <http://localhost:8765>, pick your deck, the network's deck, the checkpoint and the seat,
and press **Start game**. **Resume current game** reattaches to the game the server holds (one
game at a time; reloading the page loses nothing).

Headless check (a random "human" against the network, full games):

```bash
CUDA_VISIBLE_DEVICES= taskset -c 12-15 python3 tools/play_web/smoke_test.py 2
```

Environment overrides: `RIFTBOUND_WSL_COPY` (build copy, default `~/riftbound/alpharune`),
`PLAYWEB_LIB` (library path), `CARD_REPR` (the research tree with checkpoints and vocab,
default the Windows checkout's `research/card_repr`), `PLAYWEB_TORCH_THREADS` (default 2).

## How to play

* **Board.** The opponent is on top, you are at the bottom, the battlefields are in the middle (each
  split into the opponent's half and yours, with controller / combat / showdown badges). Legends and
  Champion Zones are on the left with both score tracks; runes are the coloured discs (dim and turned
  when exhausted). The right column has the turn and phase, the current question, the chain (top item
  first; hover an item to outline its targets) and the log.
* **Acting.** Cards you can act with glow. Click one for a menu of what it can do (play it to your base
  or a battlefield, move it, activate it). When a card can go to several places, the base and
  battlefields it can go to glow too, and clicking one does it. Buttons in the question box answer
  everything that names no card (End turn, Pass, Yes / No, modes, values). **All options** lists every
  legal answer in words, as a fallback. Space or Enter passes / ends the turn.
* **Targets** glow on the board; click one. **Mulligan:** click up to two hand cards, then confirm.
* **Card zoom:** hover any card (board, hand, chain, battlefield name, an option) for the full card
  and its live Might, damage and keywords.
* **The network's moves** are played back one frame at a time; changed cards flash red and a banner
  says what it did ("Opponent plays Pit Rookie", "moves Kai'Sa → Emperor's Dais").
* **Log:** engine lines you may read, coloured by player. "all lines" also shows prompts and passes.

## Watching replays

The setup screen also lists every recorded search game under the research tree
(`research/card_repr/**/declog_seed*.json`, written by `research/card_repr/search/replay_search.py play`).
**Watch** opens one in the same board view:

* The game is played again in `libplayweb` from its seed and the recorded choice indices (same engine,
  same rbenv start, autopay). A whole game replays in a few tens of milliseconds, so every position is
  rebuilt on demand and jumping anywhere is instant. The decks are taken from the log when it names
  them, otherwise the server finds the pair (`azir`, `rengar`, `diana`, ...) that replays the log to its
  recorded end, with every decider and the winner matching.
* **Controls:** the slider, ⏮ « ‹ ▶ › » ⏭ (start, previous turn, previous decision, autoplay, next
  decision, next turn, end), and the keys ← → (decision), Shift+← → (turn), Page Up / Down (10),
  Home / End, Space (autoplay). Moves made while a frame loads add up.
* **View:** watch as either seat; **both hands** shows the other seat's hand face up. Facedown cards
  stay facedown (each view is still the engine's per-observer view).
* **Decision panel:** who decides (`search N=64` or `network greedy`), its value estimate, the move it
  made (outlined on the board), the search's top three options by visit share, and, where the search
  overrode the network's own top choice, what the network would have played and both Q values.

## Fair play

Everything the browser receives is built for the human's seat only, in C++, from the engine's own
per-observer rules: `ObserverView` (board, chain, public piles, the opponent's hand cards you were
shown), `knowledge::identityVisible` (whether an object's card may be named: facedown cards and the
opponent's hand never are), `describePrompt` (the decision's words and options for its decider only)
and `LogEvent::visibleTo` (a log line is sent only if its audience includes your seat). The network's
decisions are never described to the browser beyond what the log and board show.

## Simplifications

* Rune payments are made automatically for both players (the first legal rune), as in training.
* A decision with a single legal option is applied without asking (as rbenv does) — e.g. "End turn"
  when nothing else can be done.
* A network knows the cards it was trained with: `az/run_b2` the Azir, Rengar and Diana decks; the older ones 51 cards.
* Against search, only the first 96 options of one decision can be chosen (the network's action limit).

## How it is built

| File | Role |
|---|---|
| `playweb.cpp` | C ABI library: one game (`pw_create` / `pw_step` / `pw_status`), the network's observation (`pw_observe`, rbenv layout), the human's view as JSON (`pw_view`). It `#include`s `research/card_repr/selfplay/rbenv/rbenv.cpp`, so the network sees exactly the encoding it was trained on. |
| `playweb.cmake`, `build.sh` | The `playweb` target, appended to the WSL build copy's CMakeLists like rbenv's, built in `build_pic`. |
| `playweb.py` | ctypes wrapper and `NetAgent` (checkpoint loading and greedy choice through `league/common.py`'s `load_net` / `to_dev`, read-only). |
| `server.py` | Stdlib HTTP server: `/api/setup`, `/api/new`, `/api/act`, `/api/state`; after each human answer it plays the network's decisions and returns one frame per decision for the client to animate. Replays: `/api/replays`, `/api/replay/open`, `/api/replay/goto`, `/api/replay/view`. Loopback only. |
| `replay.py` | Recorded search games: finds the decision logs, works out their decks, rebuilds any position and annotates the pending decision from the log. |
| `static/` | The page (`index.html`, `play.css`, `play.js`), no external assets except card art from the URLs in `CardDef::image_url`. |
| `search_test.py` | Headless full games against search (both seats) and the AZ network; the displayed game and the search's copy must encode the same observation at every human decision; saved games must replay. |
| `smoke_test.py` | Headless full games with checks on every human frame. |

No engine `src/` file is changed. The engine's existing `riftbound --agent1 human` play server
(`src/io/play_server.cpp`) was not reused because its AI seats are C++ agents, while the network
runs in Python on the research encoder; its frames also carry no card ids (art), chain or prompt
descriptions.

The layout follows common online TCG client patterns (opponent mirrored on top, central battlefields,
side score tracks, a right-hand log, click-to-act menus and hover zoom); no assets or code were copied
from any other client.
