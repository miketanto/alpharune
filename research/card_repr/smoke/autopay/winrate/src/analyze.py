"""Summarize autopay vs teacher games: raw/main*.games.jsonl -> results.json.
RR games (raw/main* RR + raw/rr*) are the headline; the mixed batches are secondary."""
import glob, json, math, os, sys, collections, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
Z = 1.959964


def wilson(k, n):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def wr(rows):
    n = len(rows)
    w = sum(1 for r in rows if r["ap"] == 1)
    l = sum(1 for r in rows if r["ap"] == 0)
    dr = n - w - l
    return {"games": n, "autopay_wins": w, "autopay_losses": l, "no_winner": dr,
            "win_rate": round(w / n, 4) if n else None,
            "wilson95": wilson(w, n),
            "score_rate_draw_half": round((w + 0.5 * dr) / n, 4) if n else None}


def load(prefixes):
    games, errors = [], []
    for pref in prefixes:
      for f in sorted(glob.glob(os.path.join(ROOT, "raw", pref + "*.games.jsonl"))):
        for line in open(f):
            d = json.loads(line)
            if d["term"].startswith("error"):
                errors.append({"gid": d["gid"], "seed": d["seed"], "term": d["term"]})
                continue
            ap_seat = 1 if d["agent_a"].startswith("autopay") else 2
            mu = ("R" if "rengar" in d["deck_a"] else "Y") + ("R" if "rengar" in d["deck_b"] else "Y")
            ap = 0.5 if d["winner"] == 0 else (1 if d["winner"] == ap_seat else 0)
            apside = d["side_a"] if ap_seat == 1 else d["side_b"]
            tside = d["side_b"] if ap_seat == 1 else d["side_a"]
            games.append({"gid": d["gid"], "seed": d["seed"], "mu": mu, "seat": ap_seat, "ap": ap,
                          "winner": d["winner"], "term": d["term"], "secs": d["secs"], "turns": d["turns"],
                          "aps": apside, "ts": tside, "batch": os.path.basename(f).split("__")[0]})
    return games, errors


def summarize(games, errors, source):
    res = {"source": source, "n_games": len(games), "n_errors": len(errors), "errors": errors[:50]}
    res["overall"] = wr(games)
    res["by_matchup"] = {m: wr([g for g in games if g["mu"] == m]) for m in ["RR", "YY", "RY", "YR"]}
    res["by_autopay_seat"] = {f"seat{s}": wr([g for g in games if g["seat"] == s]) for s in (1, 2)}
    res["by_matchup_and_seat"] = {f"{m}_seat{s}": wr([g for g in games if g["mu"] == m and g["seat"] == s])
                                  for m in ["RR", "YY", "RY", "YR"] for s in (1, 2)}
    # seat-1 (deck A, plays first?) advantage regardless of agent, as a context number
    res["deck_a_win_rate"] = round(sum(1 for g in games if g["winner"] == 1) / len(games), 4) if games else None
    res["no_winner_terms"] = dict(collections.Counter(g["term"] for g in games if g["winner"] == 0))

    # paired by seed: the same deal with autopay in seat 1 and in seat 2
    pairs = collections.defaultdict(dict)
    for g in games:
        pairs[g["seed"]][g["seat"]] = g["ap"]
    full = [v[1] + v[2] for v in pairs.values() if 1 in v and 2 in v]
    if len(full) > 1:
        m = st.mean(full) / 2
        sd = st.stdev(full) / 2
        se = sd / math.sqrt(len(full))
        res["paired"] = {
            "pairs": len(full),
            "pair_outcomes(autopay wins in the pair: 0/1/2)": {str(k): full.count(k) for k in (0, 0.5, 1, 1.5, 2) if full.count(k)},
            "mean_score": round(m, 4), "normal95": [round(m - Z * se, 4), round(m + Z * se, 4)],
            "per_game_sd_equiv": round(sd * math.sqrt(2), 4),
            "note": "pair score = mean of the two games of one seed; same deal, seats swapped",
        }
        # minimum detectable effect (two-sided 5%, power 80%) on the pair design
        res["paired"]["mde_80pct_power"] = round((Z + 0.841621) * se, 4)

    n = len(games)
    res["detectable"] = {
        "half_width_95_at_p0.5_unpaired": round(Z * math.sqrt(0.25 / n), 4) if n else None,
        "mde_unpaired_alpha05_power80": round((Z + 0.841621) * math.sqrt(0.25 / n), 4) if n else None,
        "explanation": "smallest true drop of the autopay win rate below 0.5 that a two-sided 5% test "
                       "detects with 80% power at this n (binomial, p~0.5); the paired number uses the "
                       "observed pair variance",
    }


    def side(rows, key):
        out = {}
        for f in ["secs", "searches", "dec", "multi", "rune", "rune_multi", "autopaid", "autopaid_multi",
                  "rune_searched", "rune_nonfront", "unsearched_multi"]:
            out["mean_" + f] = round(st.mean(r[key][f] for r in rows), 3)
        out["secs_per_search"] = round(sum(r[key]["secs"] for r in rows) / max(1, sum(r[key]["searches"] for r in rows)), 4)
        out["total_unsearched_multi"] = sum(r[key]["unsearched_multi"] for r in rows)
        return out


    if games:
        res["per_side"] = {"autopay": side(games, "aps"), "teacher": side(games, "ts")}
        ap, t = res["per_side"]["autopay"], res["per_side"]["teacher"]
        res["per_side"]["search_reduction"] = round(1 - ap["mean_searches"] / t["mean_searches"], 4)
        res["per_side"]["agent_time_reduction"] = round(1 - ap["mean_secs"] / t["mean_secs"], 4)
        rs = sum(g["ts"]["rune_searched"] for g in games)
        res["per_side"]["teacher_rune_nonfront_share"] = round(sum(g["ts"]["rune_nonfront"] for g in games) / max(1, rs), 4)
        res["game_secs"] = {"mean": round(st.mean(g["secs"] for g in games), 3),
                            "median": round(st.median(g["secs"] for g in games), 3)}
        res["mean_turns"] = round(st.mean(g["turns"] for g in games), 2)

    return res


main_g, main_e = load(["main"])
rr_g, rr_e = load(["rr"])
rr_all = [g for g in main_g if g["mu"] == "RR"] + rr_g
out = {
    "headline_RR": summarize(rr_all, rr_e, "RR games: raw/main*.games.jsonl (matchup RR) + raw/rr*.games.jsonl"),
    "RR_only_batches": summarize(rr_g, rr_e, "raw/rr*.games.jsonl"),
    "RR_from_mixed_batches": summarize([g for g in main_g if g["mu"] == "RR"], [], "raw/main*.games.jsonl, RR only"),
    "secondary_mixed_batches_all_matchups": summarize(main_g, main_e, "raw/main*.games.jsonl (RR, YY, RY, YR cycled)"),
}
json.dump(out, open(os.path.join(ROOT, "results.json"), "w"), indent=1)
for k, r in out.items():
    print(k, r["n_games"], "err", r["n_errors"], r["overall"]["win_rate"], r["overall"]["wilson95"],
          {s: (v["games"], v["win_rate"], v["wilson95"]) for s, v in r["by_autopay_seat"].items()},
          "paired", r.get("paired", {}).get("mean_score"), r.get("paired", {}).get("normal95"),
          "mde", r["detectable"]["mde_unpaired_alpha05_power80"], r.get("paired", {}).get("mde_80pct_power"))
