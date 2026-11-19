"""Print compact tables from results.json (and write tables.txt)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, "results.json")))
out = []


def f(v, nd=3):
    if v is None or v[0] is None:
        return "-"
    return f"{v[0]:.{nd}f} [{v[1]:.{nd}f},{v[2]:.{nd}f}]"


for ev in ("eval1_S_train_to_S_test", "eval3_S_train_to_H", "eval3b_H_to_S_test"):
    e = r[ev]
    out.append(f"== {ev} (n_cards={e['n_cards']})")
    for blk in ("all_fields", "effect_fields"):
        out.append(f"  -- {blk}: mean_err | minus_random | minus_stats")
        for cond, d in e[blk].items():
            out.append(f"  {cond:26s} {f(d['mean_err'])}  {f(d.get('minus_random'))}  {f(d.get('minus_stats'))}")
    if "per_seed_mean_err" in e:
        out.append("  per-seed: " + json.dumps({k: [round(x, 3) for x in v] for k, v in e["per_seed_mean_err"].items()}))

e2 = r["eval2_heldout_cards"]
for blk in ("leave_one_pair_out", "strict_all_held_out"):
    out.append(f"== eval2 {blk}")
    keys = ["err_variant_S", "err_copy_base_pred_S", "err_base_on_S_test", "log_ratio_var_vs_copybase_S",
            "diff_cos_S", "diff_rel_err_S", "diff_cos_H", "diff_rel_err_H", "err_variant_H"]
    for k in keys:
        out.append(f"  -- {k}")
        for cond, d in e2[blk].items():
            out.append(f"  {cond:12s} {f(d.get(k))}   -random {f(d.get(k + '_minus_random'))}  -stats {f(d.get(k + '_minus_stats'))}")

out.append("== eval2 per fold (seed mean): err_var_S / err_copybase_S / diff_cos_S / diff_rel_S / diff_cos_H / snr_S")
for fold, vs in e2["per_fold"].items():
    for var, cd in vs.items():
        out.append(f"  {fold} :: {var}")
        for cond, m in cd.items():
            g = lambda k: ("%.3f" % m[k]) if m.get(k) is not None else "-"
            out.append(f"    {cond:12s} {g('err_variant_S')} {g('err_copy_base_pred_S')} {g('diff_cos_S')} {g('diff_rel_err_S')} {g('diff_cos_H')} snr={g('diff_snr_S')} dgap={g('data_gap_base_vs_variant_S')}")

txt = "\n".join(out)
print(txt)
open(os.path.join(HERE, "tables.txt"), "w").write(txt + "\n")
