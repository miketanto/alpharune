# Many legends: generalist or specialists? Research notes and an inductive proof plan (2026-11-20)

**Question.** There are 41 deck lists in `alpharune/decks/`, which gives about 860 matchups counting mirrors. Should we train one generalist, one specialist per legend, or specialists distilled into a generalist? And what small-scale evidence would justify committing compute to all of them?

## 1. What the literature did

| System | Problem shape | What they did | Lesson for us |
|---|---|---|---|
| **AlphaStar** (Vinyals et al., Nature 2019) | 3 races, 9 matchups | One league with **3 main agents, one per race**, 3 main exploiters and 6 league exploiters (two per race). Every agent meets every race in the league. Early demo versions were Protoss mirror only, like our Rengar mirror. | They chose **specialists by own race, generalist by opponent**. The league and exploiters are what kept each main agent robust; the race split was a capacity and credit-assignment choice. |
| **OpenAI Five** (Berner et al. 2019) | 117 heroes | Shipped with **17**. Scaling to 25 heroes reached about 5k MMR but learned too slowly for pro level. Their appendix says extra heroes slow *early* training only slightly. The blog lists untested causes: capacity, matchmaking, new heroes needing time to catch up. | Breadth costs compute, and the cause was **never isolated**. That is the experiment we should run instead of assuming. |
| **JueWu / Honor of Kings** (Ye et al., NeurIPS 2020) | 40 heroes, about 2×10^11 lineups | **Curriculum self-play learning:** (1) train teachers on fixed lineups, (2) **multi-teacher policy distillation** into one student, (3) continue RL on random picks. Also uses MCTS for the draft. | This is the clearest answer to "generalist vs specialist": **specialists first, distilled into a generalist, then fine-tuned on the full pool**. It beat top humans with 40 heroes, where OpenAI stopped at 17. |
| **PopArt** (Hessel et al., AAAI 2019) | Atari-57, one network | Normalises each task's value targets so that no task dominates the gradient. | Our reward is ±1 for every matchup, so scale isn't the issue. **Sampling share and per-matchup advantage normalisation are**: easy or long-game matchups can still dominate the gradient. |
| **VGC-Bench** (Angliss et al., AAMAS 2026) | Pokémon VGC teams | As the team pool grows, the best single-team method gets **weaker and more exploitable on training teams, but generalises better to unseen teams**. Single-team agents beat a pro only in a fixed-team mirror. | Expect this trade-off directly. A generalist must be judged on **per-deck strength against a specialist**, not only on breadth. |
| **Metamon** (Grigsby et al., RLC 2025) | Pokémon singles, any team | Offline RL on 3.5M human games with transformers up to 200M parameters. Generalisation came from **data diversity plus model scale**. | Capacity matters once breadth goes up; this ties into our E3. |
| **Card-text representations** (Cardsformer for Hearthstone; LLM+RL at AAMAS 2025; generalised card representations for MTG, 2024) | Unseen cards | Encoding cards by **text/mechanics** gives a small loss on unseen cards; encoding by ID does not transfer. The MTG study predicts 55% of human picks on fully unseen cards. | This supports our B2 mechanics representation. The **held-out legend** test below measures it in our game. |
| **ByteRL** (LOCM / Hearthstone, 2024 analysis) | Card games | Strong against humans, but **highly exploitable**. | Win rate against fixed anchors is not enough. Use **exploiters per legend** in the proof. |

**Consensus:**
- Nobody went from one deck straight to all decks.
- Systems that scaled used **(a) a league over all matchups, (b) a curriculum from small to large pools, and (c) distillation** when one network struggled.
- The open variables are **capacity** and **matchup sampling**, and those are what we measure.

## 2. What we already know (our base case)

- **Learning on one legend works:** Rengar, 2,080 updates, still improving (E0).
- **Transfer to a second legend is positive:** Rengar weights start Azir at 20–24% against ISMCTS, compared with about 14% for an untrained network on Rengar. The Azir league reached 100% against both ISMCTS anchors by update 200.
- **Forgetting is real:** Azir@300 playing Rengar beats ISMCTS-100 93.5% of the time, but beats Rengar@2080 only **11.3%**. Sequential training overwrites; mixed training is needed.
- **Not yet measured:**
  - a generalist against specialists at equal compute;
  - any cross-deck matchup in training (both leagues were mirrors);
  - a held-out legend.

## 3. The claim to prove inductively

Let k be the number of legends trained together. We want three properties to hold at k and keep holding as k grows. Then the step to 41 is an extrapolation of a measured trend, not a hope.

- **P1. No per-deck loss (non-inferiority).** On each trained legend X, the generalist piloting X performs at least as well as X's specialist, at matched compute.
  - **Pilot-swap test:** play generalist(X) against ref(Y), and specialist(X) against the same ref(Y), across the same opponent decks Y. The difference cancels deck strength.
  - Pass: the difference is at least −3 points, averaged over X.
- **P2. Transfer grows with k.** For a legend H never trained on, measure:
  - (a) zero-shot strength against ISMCTS-100/200 on H's mirror and cross-matchups;
  - (b) the updates needed to reach a fixed bar when H is added, e.g. 50% head-to-head against a from-scratch H specialist that had 300 updates.
  - Pass: (b) **decreases** from k=1 to k=4 to k=8, and (a) increases. This is the direct proof that the network understands cards rather than memorising decks.
- **P3. Bounded marginal cost.** The compute needed to bring legend k+1 to the bar, without regressing legends 1..k by more than 3 points (h2h against frozen references), stays flat or falls as k grows.

If P1–P3 hold at k = 2, 4, 8 with a non-worsening trend, run all 41. If P1 fails while P2 holds, switch to the **JueWu recipe**:
1. specialists per legend cluster;
2. multi-teacher distillation;
3. full-pool league fine-tuning.

That recipe is plan B, because P2 says the representation transfers while P1 says capacity or interference is the bottleneck. If P1 fails, run P1 again with the larger network (E3). Capacity is the OpenAI Five hypothesis nobody tested.

## 4. Experiments (one GPU training at a time; ~3.5 h per 500 league updates now)

**G1. Two legends: Rengar + Azir generalist** (about 4 h of training plus 1 h of evaluation)
- **Training:**
  - start from ov_main@2080;
  - sample matchups uniformly over the 2×2 grid: two mirrors and two cross matchups, both seat orders;
  - PFSP opponents per matchup;
  - league exploiter per legend.
- **Budget:** 500 updates, matching the Azir specialist's 300 plus Rengar's continued 200, which is roughly equal *added* compute.
- **Measurement:** P1 by the pilot swap against az_main@300 and ov_main@2080 + 200; and **forgetting**: Rengar head-to-head against ov_main@2080.
- **Needs:** a `--deck-mix` option in the training env. Today one deck is used for both seats; `RBEnv(n, d1, d2, ...)` already takes two decks.

**G2. Four legends + one held out** (overnight)
- **Legends:** Rengar, Azir, plus two that differ in mechanics from both (pick from the 41 by domain and archetype: one spell-heavy control deck, one go-wide unit deck).
- **Held out:** one more legend, never trained on.
- **Training:** a generalist on all 16 ordered matchups.
- **Specialists:** add short specialists (300 updates each) for the two new legends as the P1 references. These double as the "from scratch" bar for P2(b).
- **Held-out test:** zero-shot evaluation on H, then fine-tune H into the mix and count updates to the bar.

**G3. Eight legends + two held out** (one or two nights)
- **Training:** the same protocol, starting from the G2 generalist.
- **The induction check:** three points (k = 1 from the Rengar-only network on H, k = 4, k = 8) on the P2 curve, plus P1 and P3 at each k.

**G4. Capacity check at the widest k** (ties to E3)
- **When:** if P1 shrinks as k grows.
- **Test:** repeat the last rung with D=128×4 at equal updates.
- **Pass:** the larger network recovers P1.

## 5. Measurement rules for multi-deck work

- **Deck strength is a confound.** Never compare raw win rates across matchups. Use the pilot swap (P1), or paired comparisons on the same matchup.
- **The matchup matrix is the report:** for every trained (X, Y), the generalist's win rate against the references, seat-balanced.
  - Summarise as the mean over X of the pilot-swap difference;
  - also give the worst single legend, since exploitability lives in the worst cell.
- **Exploiters per legend** (AlphaStar, ByteRL lesson): at each rung, train a short main exploiter against the generalist on each legend.
  - Report its best win rate as a robustness number.
  - A generalist that is broad but exploitable fails.
- **Anchors:** ISMCTS saturates on trained legends but is useful for **zero-shot** (P2a), where agents sit around 20–90%.
- **Matchup sampling:**
  - Uniform over matchups to start.
  - Then PFSP-weighted *per matchup*: oversample the cells where the generalist is behind its specialist.
  - Normalise advantages per matchup, as the reward-scale analogue of PopArt.
- **Mulligans and decks:** the mulligan head matters (+10.8 points on Rengar), so P1 must include it. Deck-list knowledge stays closed (open/closed decklist from the requirements memo).

## 6. Order and cost

| Step | Cost | Decision it unlocks |
|---|---|---|
| E2 result (running) | done tonight | whether search teacher is in the recipe for all rungs |
| G1 (k=2) | ~5 h | mixed training beats sequential: forgetting fixed, P1 at k=2 |
| G2 (k=4 + 1 held out) | overnight | first P2 point and P3 slope |
| G3 (k=8 + 2 held out) | 1–2 nights | the inductive trend; go/no-go for 41 |
| G4 (capacity) | overnight, only if P1 shrinks | network size for the full run |

**Go rule for all 41 legends:**
- P1 ≥ −3 points at k = 8;
- P2(b) falling across k = 1, 4, 8;
- P3 not rising.

Otherwise fall back to the JueWu specialist → distill → fine-tune recipe, with the measured k at which P1 broke as the cluster size.

## Sources

- AlphaStar: https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/ ; TStarBot-X league study https://arxiv.org/abs/2011.13729
- OpenAI Five: https://arxiv.org/abs/1912.06680 ; https://openai.com/blog/how-to-train-your-openai-five/
- JueWu full MOBA: https://arxiv.org/abs/2011.12692
- PopArt: https://arxiv.org/abs/1809.04474
- VGC-Bench: https://arxiv.org/abs/2506.10326
- Metamon: https://arxiv.org/abs/2504.04395
- Card representations: https://arxiv.org/html/2407.05879v1 (MTG); https://aamas.csc.liv.ac.uk/Proceedings/aamas2025/pdfs/p2795.pdf (LLM+RL Hearthstone); ByteRL exploitability https://arxiv.org/abs/2404.16689
