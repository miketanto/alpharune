"""Post-hoc (added after seeing results): restrict the difference metrics of eval2 to pairs whose
true difference is measurable.

Rule (data only, no model output used): a pair counts for the S difference metrics if its true
S difference is at least 4x its sampling noise (diff_snr_S >= 4); for the H metrics if its paired H
difference is nonzero (diff_rel_err_H defined). Costs and timing variants have no measurable delta
difference because costs are excluded from the targets and timing does not change the delta.
Bootstrap over pairs (seed-averaged), seed 0, 2000 resamples. Merged into results.json.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "results.json")
r = json.load(open(P))
rng = np.random.default_rng(0)
CONDS = ["id", "random", "stats", "A", "B", "B2", "shuffled-B"]


def ci(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return None
    bs = v[rng.integers(0, len(v), (2000, len(v)))].mean(1)
    return [float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


out = {"rule": __doc__.strip()}
for blk, fold_of in (("leave_one_pair_out", lambda var: var), ("strict_all_held_out", lambda var: "STRICT_all_11")):
    pf = r["eval2_heldout_cards"]["per_fold"]
    ref = pf[fold_of("primal_strength")]["primal_strength"]["B"] if blk == "leave_one_pair_out" else None
    vars_ = [p["variant"] for p in r["eval2_heldout_cards"]["pairs"]]
    sel_S = [v for v in vars_ if pf[fold_of(v)][v]["B"]["diff_snr_S"] >= 4]
    sel_H = [v for v in vars_ if pf[fold_of(v)][v]["B"].get("diff_rel_err_H") is not None]
    res = {"pairs_S": sel_S, "pairs_H": sel_H}
    per = {}
    for c in CONDS:
        g = lambda v, k: pf[fold_of(v)][v][c][k]
        per[c] = {
            "diff_cos_S": np.array([g(v, "diff_cos_S") for v in sel_S]),
            "diff_rel_err_S": np.array([g(v, "diff_rel_err_S") for v in sel_S]),
            "diff_cos_H": np.array([g(v, "diff_cos_H") for v in sel_H]),
            "diff_rel_err_H": np.array([g(v, "diff_rel_err_H") for v in sel_H]),
            "err_variant_minus_copybase_S": np.array([g(v, "err_variant_S") - g(v, "err_copy_base_pred_S") for v in vars_]),
        }
    for c in CONDS:
        res[c] = {k: ci(v) for k, v in per[c].items()}
        for refc in ("random", "stats", "shuffled-B"):
            if refc != c:
                for k in ("diff_cos_S", "diff_cos_H", "diff_rel_err_H"):
                    res[c][f"{k}_minus_{refc}"] = ci(per[c][k] - per[refc][k])
    out[blk] = res
r["eval2_posthoc_measurable_pairs"] = out
json.dump(r, open(P, "w"), indent=1)
for blk in ("leave_one_pair_out", "strict_all_held_out"):
    print("==", blk, out[blk]["pairs_S"], out[blk]["pairs_H"])
    for c in CONDS:
        d = out[blk][c]
        f = lambda x: "-" if x is None else f"{x[0]:.3f} [{x[1]:.3f},{x[2]:.3f}]"
        print(f"{c:11s} cosS {f(d['diff_cos_S'])} relS {f(d['diff_rel_err_S'])} cosH {f(d['diff_cos_H'])} relH {f(d['diff_rel_err_H'])} cosH-rand {f(d.get('diff_cos_H_minus_random'))} var-copy {f(d['err_variant_minus_copybase_S'])}")
