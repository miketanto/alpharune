"""Does auto-pay (the engine's default rune order, act[0]) lose value against the
teacher's best rune choice?  Reads the existing teacher decisions only.

Per prompt kind:
  regret_default = q[best] - q[act0]      (what auto-pay gives up, per the teacher)
  regret_random  = q[best] - mean(q)      (a uniformly random choice; noise yardstick)
  regret_teacher = q[best] - q[pi]        (the teacher's own pick)
  spread         = q[best] - q[worst]
q comes from ~100 sims spread almost evenly over the options, so q[best] is the max
of noisy estimates and every regret is biased upward by that noise.  The comparison
that matters is default vs random vs the same numbers on decisions that do matter
(main phase), and the share of total q-spread each kind carries.
"""
import gzip, json, glob, os, sys, collections, statistics
from multiprocessing import Pool

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "out")
RUNE = {"rune for energy", "rune for power"}


def scan(path):
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    pairs = collections.defaultdict(list)   # rune: (kind of default, kind of best) -> regret
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if not d.get("searched") or "q" not in d:
                continue
            q, vis, pk = d["q"], d["vis"], d["pk"]
            if len(q) < 2 or min(vis) == 0:
                continue
            best = max(range(len(q)), key=q.__getitem__)
            a = agg[pk]
            a["n_opt"].append(len(q))
            a["reg_def"].append(q[best] - q[0])
            a["reg_rand"].append(q[best] - sum(q) / len(q))
            a["reg_teach"].append(q[best] - q[d["pi"]])
            a["spread"].append(q[best] - min(q))
            a["def_is_best"].append(1.0 if best == 0 else 0.0)
            a["teach_is_def"].append(1.0 if d["pi"] == 0 else 0.0)
            if pk in RUNE:
                tok = d["tok"]
                def key(i):
                    ch = d["act"][i].get("ch") or []
                    return tok[ch[0]][1] if ch else "?"
                pairs[(pk, key(0), key(best))].append(q[best] - q[0])
    return ({k: dict(v) for k, v in agg.items()}, dict(pairs))


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(DATA, "*teacher*.decisions.jsonl.gz")))
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    pairs = collections.defaultdict(list)
    with Pool(6) as pool:
        for a, p in pool.imap_unordered(scan, files):
            for pk, v in a.items():
                for k, xs in v.items():
                    agg[pk][k].extend(xs)
            for k, xs in p.items():
                pairs[k].extend(xs)

    total_spread = sum(sum(v["spread"]) for v in agg.values())
    rows = []
    for pk, v in agg.items():
        n = len(v["spread"])
        rows.append({
            "pk": pk, "n": n,
            "mean_options": round(statistics.mean(v["n_opt"]), 2),
            "spread_mean": round(statistics.mean(v["spread"]), 4),
            "share_of_total_spread": round(sum(v["spread"]) / total_spread, 4),
            "regret_default_mean": round(statistics.mean(v["reg_def"]), 4),
            "regret_default_p90": round(q(v["reg_def"], 0.9), 4),
            "regret_default_p99": round(q(v["reg_def"], 0.99), 4),
            "frac_regret_default_gt_0.05": round(sum(x > 0.05 for x in v["reg_def"]) / n, 4),
            "frac_regret_default_gt_0.10": round(sum(x > 0.10 for x in v["reg_def"]) / n, 4),
            "regret_random_mean": round(statistics.mean(v["reg_rand"]), 4),
            "regret_teacher_mean": round(statistics.mean(v["reg_teach"]), 4),
            "default_is_best": round(statistics.mean(v["def_is_best"]), 4),
            "teacher_played_default": round(statistics.mean(v["teach_is_def"]), 4),
        })
    rows.sort(key=lambda r: -r["n"])
    pair_rows = sorted(
        ({"pk": k[0], "default": k[1], "best": k[2], "n": len(xs),
          "regret_mean": round(statistics.mean(xs), 4),
          "frac_gt_0.10": round(sum(x > 0.10 for x in xs) / len(xs), 4)}
         for k, xs in pairs.items() if len(xs) >= 200),
        key=lambda r: -r["n"])
    out = {"files": len(files), "by_prompt_kind": rows, "rune_by_default_vs_best_kind": pair_rows}
    here = os.path.dirname(__file__)
    json.dump(out, open(os.path.join(here, "autopay_check.json"), "w"), indent=1)
    cols = ["pk", "n", "mean_options", "spread_mean", "share_of_total_spread",
            "regret_default_mean", "regret_default_p90", "frac_regret_default_gt_0.10",
            "regret_random_mean", "regret_teacher_mean", "default_is_best"]
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r[c]) for c in cols))
    print()
    for r in pair_rows[:20]:
        print(r)
