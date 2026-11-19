"""Supplementary T3 run (NOT the official score): same probes, labels, folds and CIs as
eval.py, but with a different per-fold feature preprocessing, to show how much of a path's
T3 result depends on eval.py's per-dimension z-scoring.

  official (eval.py): z-score every dimension in the training fold (each used dim gets unit
                      variance, so a dimension used by 1-2 cards weighs as much as a dense one)
  global   (this)   : centre each dimension, divide all dims by ONE scalar (sqrt of total
                      variance), i.e. keep the representation's own relative scaling

usage: python t3_sensitivity.py <vectors.json> --name X [--boot 500]
writes results_t3global_<name>.json
"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval as ev


def global_scale_fold(Xtr, Xte):
    mu = Xtr.mean(0)
    s = float(np.sqrt(((Xtr - mu) ** 2).sum(1).mean())) or 1.0
    return (Xtr - mu) / s, (Xte - mu) / s


ap = argparse.ArgumentParser()
ap.add_argument("vectors"); ap.add_argument("--name", required=True)
ap.add_argument("--boot", type=int, default=500); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
ev.standardize_fold = global_scale_fold
cards = json.load(open(ev.CARDS_PATH, encoding="utf8"))
dim, vecs, invalid = ev.load_vectors(a.vectors)
base = [c["key"] for c in cards["cards"] if c["key"] in vecs]
labels = json.load(open(os.path.join(HERE, "labels.json"), encoding="utf8"))
t3 = ev.run_t3(labels, vecs, base, np.random.default_rng(a.seed), a.boot)
out = {"name": a.name, "preprocessing": "centre + single global scale (supplementary)", "boot": a.boot, "t3": t3}
p = os.path.join(HERE, "results_t3global_%s.json" % a.name)
json.dump(ev._clean(out), open(p, "w", encoding="utf8"), indent=1)
s = t3["summary"]
for g in ("non_provisional", "provisional"):
    print(a.name, g, "beat", s[g]["n_beats_baseline"], "/", s[g]["n_labels"], "R2 %.3f bacc %.3f" % (s[g]["mean_r2_numeric"], s[g]["mean_bacc_logistic"]))
