Minimal decision network: training and evaluation (smoke test, 2026-11-19)
========================================================================

Pipeline (run from this directory, Windows Anaconda python, torch 2.0 CPU):
  python prep.py        -> packed/teacher.pt, packed/swap.pt   (teacher decisions only; ~20 s)
  python reprs.py       -> packed/reprs.pt, reprs.json         (R(card) per condition)
  python launch.py 3000 -> runs/<cond>__<regime>__s<seed>/     (45 runs, 3 workers x 2 threads)
  python aggregate.py   -> results.json, table_*.csv
  python keyswap.py     -> keyswap.json (supplementary E1 analysis)

Network (train.py, identical for every condition, 148,675 params excl. the card table):
  token = W_card R(card) [256->64, no bias] + per-kind situational state embedding
          + special-key embedding (hidden / token:* / keyword_ability: same rows in every
          condition) + kind embedding; global token from the 72 global ints + prompt kind.
  2-layer pre-norm transformer, d=64, 4 heads, ff=128, no positions.
  value head on the global token (tanh); two pointer policy heads on
  h_a = E_akind + h_src + mean(h_refs) + E_dst + W_v scalars + h_global:
    head_v trained on the teacher visit distribution (CE), head_q trained on
    softmax(q / 0.05) over legal options (searched decisions only).
  loss = CE_vis + CE_q + MSE(value, z). AdamW lr 1e-3, wd 0.01, warmup 200, cosine,
  batch 256, 3000 steps, grad clip 1.0.
  Per-head checkpoint selection on validation loss every 250 steps (policy heads: CE_vis+CE_q,
  value head: MSE) using validation games of the run's own regime only.

R(card): every raw vector is centred on the 51 base cards, scaled to mean norm 1 and sent
through the same fixed Gaussian projection to 256 dims (seed 1234). random = iid N(0,1)
(seed 777); stats = type, champion/signature, domains, energy, power, Might, keywords that
open a line of text (variants changed per their 'field'); Bshuf = B permuted across card keys
by a fixed derangement (seed 4242); id = trainable table initialised from random, unseen key
-> UNK row trained by 2% key dropout. Diagnostics: reprs.json (variant-to-base distances).

Regimes: base = base_teacher train games only (every variant card unseen: E1);
         all  = base + all variant teacher train games (E0, E2).
Seeds 0,1,2: seed sets init (identical across non-id conditions) and batch order.

History: a first launch at 4000 steps was stopped (projected 5 h); a second at 3000 steps
without checkpoint selection was stopped after 3 runs when value overfitting showed
(val MSE 0.58 at step 500 -> 0.83 at 3000). Those runs are in runs_superseded/ and are
not used. See launch_log.txt.
