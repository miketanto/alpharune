"""Shared scorer for card-representation phase 1 (tests T1, T2, T3).

    python eval.py <vectors.json> [--name X] [--labels labels.json] [--boot 2000] [--quiet]

Reads the output contract of SPEC.md ({"dim": N, "vectors": {...}}) and the
contract in ../cards.json; writes eval/results_<name>.json and prints a summary.
numpy/scipy only. Every path is scored with the same code and the same settings;
nothing here is tuned to any path's output.

Distances
  cos  : cosine distance 1 - cos(u, v)   (0 for two zero vectors, 1 for one zero vector)
  l2s  : Euclidean distance after per-dimension standardisation, mean/sd taken over
         the base-card vectors (a dimension with sd 0 is left unscaled)
  Each is also reported normalised by the median pairwise distance among base
  cards ("*_rel"), so numbers are comparable across paths of different dimension.
Ranks are pessimistic: rank = 1 + number of OTHER base cards at distance <= the
partner's distance, so ties count against the representation (a constant vector
ranks last, not first).
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
CARDS_PATH = os.path.join(HERE, "..", "cards.json")
EPS = 1e-9          # "distance is zero" tolerance
NEAR_ZERO_REL = 0.01  # T1 "same vector": < 1% of the median pairwise base distance
TIE_TOL = 1e-12


# ---------------------------------------------------------------- loading
def load_vectors(path):
    raw = json.load(open(path, encoding="utf8"))
    dim = raw.get("dim")
    vecs, invalid = {}, []
    for k, v in (raw.get("vectors") or {}).items():
        try:
            a = np.asarray(v, dtype=float).ravel()
        except Exception:
            invalid.append({"key": k, "why": "not numeric"})
            continue
        if dim is not None and a.size != dim:
            invalid.append({"key": k, "why": "length %d != dim %s" % (a.size, dim)})
            continue
        if not np.all(np.isfinite(a)):
            invalid.append({"key": k, "why": "non-finite values"})
            continue
        vecs[k] = a
    if dim is None and vecs:
        dim = next(iter(vecs.values())).size
    return dim, vecs, invalid


# ---------------------------------------------------------------- distances
class Space:
    def __init__(self, base_mat):
        self.mu = base_mat.mean(0)
        sd = base_mat.std(0)
        self.sd = np.where(sd > 1e-12, sd, 1.0)
        self.base = base_mat
        self.base_s = (base_mat - self.mu) / self.sd
        n = base_mat.shape[0]
        iu = np.triu_indices(n, 1)
        self.med = {}
        for m in ("cos", "l2s"):
            D = self.pairwise(base_mat, m)
            vals = D[iu]
            med = float(np.median(vals)) if vals.size else float("nan")
            self.med[m] = med if med > 0 else float("nan")

    def std(self, v):
        return (v - self.mu) / self.sd

    @staticmethod
    def cos_d(u, v):
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu < 1e-12 and nv < 1e-12:
            return 0.0
        if nu < 1e-12 or nv < 1e-12:
            return 1.0
        return float(max(0.0, 1.0 - np.dot(u, v) / (nu * nv)))

    def d(self, u, v, m):
        if m == "cos":
            return self.cos_d(u, v)
        return float(np.linalg.norm(self.std(u) - self.std(v)))

    def to_all(self, q, m):
        if m == "cos":
            return np.array([self.cos_d(q, b) for b in self.base])
        return np.linalg.norm(self.base_s - self.std(q), axis=1)

    def pairwise(self, M, m):
        return np.array([[self.d(a, b, m) for b in M] for a in M])


def pess_rank(dists, target_idx, exclude=()):
    dt = dists[target_idx]
    others = [i for i in range(len(dists)) if i != target_idx and i not in exclude]
    return 1 + int(sum(dists[i] <= dt + TIE_TOL for i in others))


def rel(x, med):
    return None if (med is None or not np.isfinite(med)) else float(x / med)


def summ(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    a = np.asarray(xs, float)
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "max": float(a.max()), "min": float(a.min())}


# ---------------------------------------------------------------- T1
def run_t1(cards, vecs, sp, base_keys):
    idx = {k: i for i, k in enumerate(base_keys)}
    entries, missing = [], []
    for c in cards["cards"]:
        k = c["key"]
        variants = []
        if len(c["printings"]) > 1:
            variants += [("printing", "printing:" + p["id"]) for p in c["printings"]]
        variants.append(("rename", "rename:" + k))
        for kind, vk in variants:
            if k not in idx or vk not in vecs:
                missing.append(vk if k in idx else "%s (base %s missing)" % (vk, k))
                continue
            q = vecs[vk]
            e = {"key": vk, "base": k, "kind": kind}
            for m in ("cos", "l2s"):
                dists = sp.to_all(q, m)
                e[m] = float(dists[idx[k]])
                e[m + "_rel"] = rel(e[m], sp.med[m])
                e["rank_" + m] = pess_rank(dists, idx[k])
            entries.append(e)
    out = {"entries": entries, "missing": missing}
    s = {}
    for kind in ("printing", "rename", "all"):
        es = [e for e in entries if kind == "all" or e["kind"] == kind]
        d = {"n": len(es)}
        for m in ("cos", "l2s"):
            vals = [e[m] for e in es]
            d[m] = summ(vals)
            d[m + "_rel"] = summ([e[m + "_rel"] for e in es])
            d["frac_rank1_" + m] = (float(np.mean([e["rank_" + m] == 1 for e in es])) if es else None)
            d["frac_near_zero_" + m] = (float(np.mean([(e[m + "_rel"] is not None and e[m + "_rel"] < NEAR_ZERO_REL)
                                                         or e[m] < EPS for e in es])) if es else None)
            d["noise_floor_" + m] = float(max(vals)) if vals else None
        s[kind] = d
    out["summary"] = s
    # the noise floor T2 uses: max T1 distance over printings AND renames
    out["noise_floor"] = {m: s["all"]["noise_floor_" + m] for m in ("cos", "l2s")}
    out["noise_floor_printings_only"] = {m: s["printing"]["noise_floor_" + m] for m in ("cos", "l2s")}
    return out


# ---------------------------------------------------------------- T2
def ratio(d, floor):
    if floor is None:
        return None
    if floor < EPS:
        return "inf" if d > EPS else 0.0
    return float(d / floor)


def above(d, floor):
    if floor is None:
        return None
    return bool(d > max(floor, EPS))


def run_t2(cards, vecs, sp, base_keys, t1, rng, n_perm=2000):
    idx = {k: i for i, k in enumerate(base_keys)}
    floor, pfloor = t1["noise_floor"], t1["noise_floor_printings_only"]
    real, synth, missing = [], [], []

    def pair_entry(qv, partner, exclude, extra):
        e = dict(extra)
        for m in ("cos", "l2s"):
            dists = sp.to_all(qv, m)
            dd = float(dists[idx[partner]])
            e[m] = dd
            e[m + "_rel"] = rel(dd, sp.med[m])
            e[m + "_ratio_to_floor"] = ratio(dd, floor[m])
            e[m + "_above_floor"] = above(dd, floor[m])
            e[m + "_ratio_to_printing_floor"] = ratio(dd, pfloor[m])
            e[m + "_above_printing_floor"] = above(dd, pfloor[m])
            e[m + "_distinct"] = bool(dd > EPS)
            e["rank_partner_" + m] = pess_rank(dists, idx[partner], exclude)
        return e

    for a, b, what in cards["t2_minimal_pairs_real"]:
        if a not in idx or b not in idx:
            missing.append("%s|%s" % (a, b))
            continue
        real.append(pair_entry(vecs[a], b, {idx[a]}, {"a": a, "b": b, "change": what}))
    for s in cards["t2_minimal_pairs_synthetic"]:
        sk = "synthetic:" + s["id"]
        if sk not in vecs or s["base"] not in idx:
            missing.append(sk)
            continue
        synth.append(pair_entry(vecs[sk], s["base"], set(),
                                {"id": s["id"], "base": s["base"], "field": s["field"], "change": s["change"]}))

    # ---- direction test
    diffs = {}
    for s in cards["t2_minimal_pairs_synthetic"]:
        sk = "synthetic:" + s["id"]
        if sk in vecs and s["base"] in vecs:
            diffs[s["id"]] = (s["field"], vecs[sk] - vecs[s["base"]], sp.std(vecs[sk]) - sp.std(vecs[s["base"]]))
    direction = {"raw": {}, "std": {}}
    ids = list(diffs)
    fields = [diffs[i][0] for i in ids]
    for space_i, space in ((1, "raw"), (2, "std")):
        V = [diffs[i][space_i] for i in ids]
        nz = [np.linalg.norm(v) > 1e-12 for v in V]
        C = np.full((len(V), len(V)), np.nan)
        for i in range(len(V)):
            for j in range(len(V)):
                if nz[i] and nz[j]:
                    C[i, j] = float(np.dot(V[i], V[j]) / (np.linalg.norm(V[i]) * np.linalg.norm(V[j])))

        def within_mean(labels):
            vals = [C[i, j] for i in range(len(V)) for j in range(i + 1, len(V))
                    if labels[i] == labels[j] and np.isfinite(C[i, j])]
            return float(np.mean(vals)) if vals else None

        groups = {}
        for f in sorted(set(fields)):
            members = [ids[i] for i in range(len(ids)) if fields[i] == f]
            g = {"members": members, "n": len(members)}
            mi = [i for i in range(len(ids)) if fields[i] == f]
            w = [C[i, j] for a_, i in enumerate(mi) for j in mi[a_ + 1:] if np.isfinite(C[i, j])]
            g["mean_within_cos"] = float(np.mean(w)) if w else None
            x = [C[i, j] for i in mi for j in range(len(ids)) if fields[j] != f and np.isfinite(C[i, j])]
            g["mean_cos_to_other_fields"] = float(np.mean(x)) if x else None
            groups[f] = g
        obs = within_mean(fields)
        perm = []
        if obs is not None:
            for _ in range(n_perm):
                pl = list(rng.permutation(fields))
                pm = within_mean(pl)
                if pm is not None:
                    perm.append(pm)
        # random base-pair difference baseline: cosine between two random card-difference vectors
        rb = []
        nb = len(base_keys)
        if nb >= 4:
            M = sp.base if space == "raw" else sp.base_s
            for _ in range(n_perm):
                i, j, k, l = rng.choice(nb, 4, replace=False)
                u, v = M[i] - M[j], M[k] - M[l]
                if np.linalg.norm(u) > 1e-12 and np.linalg.norm(v) > 1e-12:
                    rb.append(float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))
        direction[space] = {
            "groups": groups,
            "n_zero_difference": int(sum(not z for z in nz)),
            "mean_within_field_cos": obs,
            "shuffled_field_labels_mean": float(np.mean(perm)) if perm else None,
            "shuffled_p_value": (float((1 + sum(p >= obs for p in perm)) / (1 + len(perm)))
                                 if perm else None),
            "random_card_difference_cos_mean": float(np.mean(rb)) if rb else None,
            "random_card_difference_abs_cos_mean": float(np.mean(np.abs(rb))) if rb else None,
            "pairwise_cos_matrix": {"ids": ids, "C": [[None if not np.isfinite(x) else round(float(x), 4)
                                                       for x in row] for row in C]},
        }

    def fr(es, key):
        vals = [e[key] for e in es if e.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    summary = {}
    for name, es in (("real", real), ("synthetic", synth), ("all", real + synth)):
        d = {"n": len(es)}
        for m in ("cos", "l2s"):
            d["frac_distinct_" + m] = fr(es, m + "_distinct")
            d["frac_above_floor_" + m] = fr(es, m + "_above_floor")
            d["frac_above_printing_floor_" + m] = fr(es, m + "_above_printing_floor")
            d[m + "_rel"] = summ([e[m + "_rel"] for e in es])
            nums = [e[m + "_ratio_to_floor"] for e in es if isinstance(e[m + "_ratio_to_floor"], float)]
            d["ratio_to_floor_" + m] = summ(nums)
            d["median_rank_partner_" + m] = (float(np.median([e["rank_partner_" + m] for e in es])) if es else None)
        summary[name] = d
    return {"real": real, "synthetic": synth, "missing": missing, "direction": direction, "summary": summary}


# ---------------------------------------------------------------- T3 probes
def standardize_fold(Xtr, Xte):
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    keep = sd > 1e-12
    Ztr = np.zeros_like(Xtr)
    Zte = np.zeros_like(Xte)
    Ztr[:, keep] = (Xtr[:, keep] - mu[keep]) / sd[keep]
    Zte[:, keep] = (Xte[:, keep] - mu[keep]) / sd[keep]
    return Ztr, Zte


def ridge_fit_predict(Ztr, ytr, Zte):
    """Linear-kernel ridge with alpha from closed-form inner LOO over a fixed grid."""
    n = Ztr.shape[0]
    K = Ztr @ Ztr.T
    ym = ytr.mean()
    yc = ytr - ym
    scale = max(np.trace(K) / n, 1e-12)
    best = None
    for a in scale * np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]):
        Ainv = np.linalg.inv(K + a * np.eye(n))
        H = K @ Ainv
        h = np.clip(np.diag(H), None, 1 - 1e-9)
        # LOO for centred y (intercept effect on leverage ignored; same rule for every path)
        loo = (yc - H @ yc) / (1 - h)
        err = float(np.mean(loo ** 2))
        if best is None or err < best[0]:
            best = (err, a, Ainv)
    _, a, Ainv = best
    if np.allclose(K, 0):
        return np.full(Zte.shape[0], ym), float(a)
    return ym + (Zte @ Ztr.T) @ (Ainv @ yc), float(a)


def logistic_fit_predict(Ztr, ytr, Zte, classes, C=1.0):
    """Multinomial L2 logistic regression (sklearn's C convention: sum CE + ||W||^2/(2C)),
    solved in the span of the training rows (W = Z^T A), so cost does not depend on dim."""
    n = Ztr.shape[0]
    k = len(classes)
    K = Ztr @ Ztr.T
    Y = np.zeros((n, k))
    for i, y in enumerate(ytr):
        Y[i, classes.index(y)] = 1.0
    if np.allclose(K, 0):
        # no usable feature: the fit is the class prior
        prior = Y.mean(0)
        return [classes[int(np.argmax(prior))]] * Zte.shape[0]

    def f(theta):
        A = theta[: n * k].reshape(n, k)
        b = theta[n * k:]
        F = K @ A + b
        F = F - F.max(1, keepdims=True)
        lse = np.log(np.exp(F).sum(1, keepdims=True))
        P = np.exp(F - lse)
        loss = -float(np.sum(Y * (F - lse))) + 0.5 / C * float(np.sum(A * (K @ A)))
        G = P - Y
        gA = K @ G + (K @ A) / C
        gb = G.sum(0)
        return loss, np.concatenate([gA.ravel(), gb])

    res = minimize(f, np.zeros(n * k + k), jac=True, method="L-BFGS-B",
                   options={"maxiter": 500})
    A = res.x[: n * k].reshape(n, k)
    b = res.x[n * k:]
    F = (Zte @ Ztr.T) @ A + b
    return [classes[int(i)] for i in np.argmax(F, 1)]


def centroid_predict(Ztr, ytr, Zte, classes):
    cents = {c: Ztr[[i for i, y in enumerate(ytr) if y == c]].mean(0) for c in classes if c in ytr}
    out = []
    for z in Zte:
        best = min(cents, key=lambda c: (float(np.linalg.norm(z - cents[c])), str(c)))
        out.append(best)
    return out


def balanced_acc(y, p):
    cls = sorted(set(y), key=str)
    rec = [np.mean([p[i] == c for i in range(len(y)) if y[i] == c]) for c in cls]
    return float(np.mean(rec))


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    sst = float(np.sum((y - y.mean()) ** 2))
    if sst < 1e-12:
        return None
    return 1.0 - float(np.sum((y - p) ** 2)) / sst


def boot_ci(metric, n, rng, B):
    vals = []
    for _ in range(B):
        ii = rng.integers(0, n, n)
        v = metric(ii)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if len(vals) < 20:
        return None
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def run_t3(labels_doc, vecs, base_keys, rng, B):
    L = labels_doc["labels"]
    kinds = labels_doc["label_kinds"]
    keys = [k for k in base_keys if k in L]
    missing_labels = [k for k in base_keys if k not in L]
    X_all = np.array([vecs[k] for k in keys])
    res, skipped = {}, {}
    todo = [(l, "numeric") for l in kinds["numeric"]] + \
           [(l, "categorical") for l in kinds["categorical"]] + \
           [(l, "binary") for l in kinds["binary"]]
    for lab, kind in todo:
        rows = [(i, L[k][lab]["value"]) for i, k in enumerate(keys)
                if lab in L[k] and L[k][lab]["value"] is not None]
        prov = any(L[keys[i]][lab]["provisional"] for i, _ in rows)
        srcs = sorted({L[keys[i]][lab]["source"].split(":")[0] for i, _ in rows})
        n = len(rows)
        if n < 8:
            skipped[lab] = "only %d cards have a value" % n
            continue
        ii = [i for i, _ in rows]
        y = [v for _, v in rows]
        X = X_all[ii]
        info = {"kind": kind, "n": n, "provisional": prov, "sources": srcs}
        if kind == "numeric":
            y = np.asarray(y, float)
            if np.std(y) < 1e-12:
                skipped[lab] = "constant"
                continue
            pred, base, alphas = np.zeros(n), np.zeros(n), []
            for t in range(n):
                tr = [j for j in range(n) if j != t]
                Ztr, Zte = standardize_fold(X[tr], X[[t]])
                p, a = ridge_fit_predict(Ztr, y[tr], Zte)
                pred[t], base[t] = p[0], y[tr].mean()
                alphas.append(a)
            info.update({
                "r2": r2(y, pred), "baseline_r2": r2(y, base),
                "mae": float(np.mean(np.abs(y - pred))), "baseline_mae": float(np.mean(np.abs(y - base))),
                "r2_ci95": boot_ci(lambda s: r2(y[s], pred[s]), n, rng, B),
                "r2_minus_baseline_ci95": boot_ci(
                    lambda s: (None if r2(y[s], pred[s]) is None else r2(y[s], pred[s]) - r2(y[s], base[s])),
                    n, rng, B),
                "median_alpha": float(np.median(alphas)),
            })
            ci = info["r2_minus_baseline_ci95"]
            info["beats_baseline"] = bool(ci is not None and ci[0] > 0)
        else:
            y = [str(v) for v in y]
            cls = sorted(set(y))
            counts = {c: y.count(c) for c in cls}
            if len(cls) < 2:
                skipped[lab] = "constant (%s)" % cls
                continue
            if sorted(counts.values())[-2] < 3:
                skipped[lab] = "too few minority examples %s" % counts
                continue
            plog, pcen, pbase = [], [], []
            for t in range(n):
                tr = [j for j in range(n) if j != t]
                ytr = [y[j] for j in tr]
                ctr = sorted(set(ytr))
                Ztr, Zte = standardize_fold(X[tr], X[[t]])
                plog.append(logistic_fit_predict(Ztr, ytr, Zte, ctr)[0])
                pcen.append(centroid_predict(Ztr, ytr, Zte, ctr)[0])
                pbase.append(max(ctr, key=lambda c: (ytr.count(c), c)))
            ya = np.array(y, dtype=object)
            pl, pc, pb = (np.array(v, dtype=object) for v in (plog, pcen, pbase))

            def acc(s, p):
                return float(np.mean(ya[s] == p[s]))

            def bacc(s, p):
                return balanced_acc(list(ya[s]), list(p[s])) if len(set(ya[s])) > 1 else None

            info.update({
                "class_counts": counts,
                "acc_logistic": acc(slice(None), pl), "acc_centroid": acc(slice(None), pc),
                "acc_majority": acc(slice(None), pb),
                "bacc_logistic": bacc(slice(None), pl), "bacc_centroid": bacc(slice(None), pc),
                "bacc_majority": bacc(slice(None), pb),
                "acc_logistic_ci95": boot_ci(lambda s: acc(s, pl), n, rng, B),
                "bacc_logistic_ci95": boot_ci(lambda s: bacc(s, pl), n, rng, B),
                "bacc_logistic_minus_majority_ci95": boot_ci(
                    lambda s: (None if bacc(s, pl) is None else bacc(s, pl) - bacc(s, pb)), n, rng, B),
                "bacc_centroid_ci95": boot_ci(lambda s: bacc(s, pc), n, rng, B),
            })
            ci = info["bacc_logistic_minus_majority_ci95"]
            info["beats_baseline"] = bool(ci is not None and ci[0] > 0)
        res[lab] = info

    def agg(sel):
        num = [v for v in sel if v["kind"] == "numeric" and v["r2"] is not None]
        cat = [v for v in sel if v["kind"] != "numeric"]
        return {
            "n_labels": len(sel),
            "n_beats_baseline": int(sum(v["beats_baseline"] for v in sel)),
            "mean_r2_numeric": float(np.mean([v["r2"] for v in num])) if num else None,
            "mean_baseline_r2_numeric": float(np.mean([v["baseline_r2"] for v in num])) if num else None,
            "mean_bacc_logistic": float(np.mean([v["bacc_logistic"] for v in cat])) if cat else None,
            "mean_bacc_centroid": float(np.mean([v["bacc_centroid"] for v in cat])) if cat else None,
            "mean_bacc_majority": float(np.mean([v["bacc_majority"] for v in cat])) if cat else None,
            "mean_acc_logistic": float(np.mean([v["acc_logistic"] for v in cat])) if cat else None,
            "mean_acc_majority": float(np.mean([v["acc_majority"] for v in cat])) if cat else None,
        }

    vals = list(res.values())
    summary = {"all": agg(vals),
               "non_provisional": agg([v for v in vals if not v["provisional"]]),
               "provisional": agg([v for v in vals if v["provisional"]])}
    return {"n_cards": len(keys), "cards_without_labels": missing_labels, "labels": res,
            "skipped": skipped, "summary": summary}


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("vectors")
    ap.add_argument("--name")
    ap.add_argument("--labels", default=os.path.join(HERE, "labels.json"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    name = a.name or os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(a.vectors)))) or "run"
    rng = np.random.default_rng(a.seed)

    cards = json.load(open(CARDS_PATH, encoding="utf8"))
    dim, vecs, invalid = load_vectors(a.vectors)
    all_keys = [c["key"] for c in cards["cards"]]
    base_keys = [k for k in all_keys if k in vecs]
    expected = set(all_keys)
    expected |= {"printing:" + p["id"] for c in cards["cards"] for p in c["printings"]}
    expected |= {"rename:" + k for k in all_keys}
    expected |= {"synthetic:" + s["id"] for s in cards["t2_minimal_pairs_synthetic"]}
    missing_expected = sorted(expected - set(vecs))
    extra = sorted(set(vecs) - expected)

    result = {"name": name, "vectors_file": os.path.abspath(a.vectors), "dim": dim,
              "n_vectors": len(vecs), "invalid": invalid,
              "missing_expected": missing_expected, "unexpected_keys": extra[:50],
              "n_unexpected_keys": len(extra), "n_base_cards": len(base_keys),
              "settings": {"eps": EPS, "near_zero_rel": NEAR_ZERO_REL, "boot": a.boot, "seed": a.seed,
                           "logistic_C": 1.0, "ridge_alpha_grid": "trace(K)/n * {1e-3..1e2}, inner LOO"}}
    if len(base_keys) < 4:
        result["error"] = "fewer than 4 base-card vectors; nothing scored"
        _write(result, a.out_dir, name)
        print(json.dumps(result, indent=1)[:2000])
        return result
    sp = Space(np.array([vecs[k] for k in base_keys]))
    result["median_pairwise_base"] = sp.med
    result["n_identical_base_pairs_cos"] = int(np.sum(np.triu(sp.pairwise(sp.base, "cos") < EPS, 1)))
    result["t1"] = run_t1(cards, vecs, sp, base_keys)
    result["t2"] = run_t2(cards, vecs, sp, base_keys, result["t1"], rng)
    try:
        labels_doc = json.load(open(a.labels, encoding="utf8"))
        result["t3"] = run_t3(labels_doc, vecs, base_keys, rng, a.boot)
    except FileNotFoundError:
        result["t3"] = {"error": "labels file not found: %s" % a.labels}
    _write(result, a.out_dir, name)
    if not a.quiet:
        print_summary(result)
    return result


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _write(result, out_dir, name):
    p = os.path.join(out_dir, "results_%s.json" % name)
    with open(p, "w", encoding="utf8") as f:
        json.dump(_clean(result), f, indent=1)
    result["_written_to"] = p


def f3(x):
    return "-" if x is None else ("%.3f" % x if isinstance(x, float) else str(x))


def print_summary(r):
    print("== %s  dim=%s  vectors=%d  base cards=%d  missing=%d  invalid=%d"
          % (r["name"], r["dim"], r["n_vectors"], r["n_base_cards"], len(r["missing_expected"]),
             len(r["invalid"])))
    if r["missing_expected"]:
        print("   missing:", ", ".join(r["missing_expected"][:12]),
              "..." if len(r["missing_expected"]) > 12 else "")
    print("   median pairwise base distance: cos %s  l2s %s ; identical base pairs (cos): %d"
          % (f3(r["median_pairwise_base"]["cos"]), f3(r["median_pairwise_base"]["l2s"]),
             r["n_identical_base_pairs_cos"]))
    s = r["t1"]["summary"]
    for kind in ("printing", "rename"):
        d = s[kind]
        if d["n"]:
            print("T1 %-8s n=%-3d  cos max %s (rel %s)  rank1 %s  near0 %s | l2s max %s  rank1 %s"
                  % (kind, d["n"], f3(d["cos"]["max"]), f3(d["cos_rel"]["max"] if d["cos_rel"] else None),
                     f3(d["frac_rank1_cos"]), f3(d["frac_near_zero_cos"]), f3(d["l2s"]["max"]),
                     f3(d["frac_rank1_l2s"])))
    print("   noise floor (max T1): cos %s  l2s %s ; printings only: cos %s"
          % (f3(r["t1"]["noise_floor"]["cos"]), f3(r["t1"]["noise_floor"]["l2s"]),
             f3(r["t1"]["noise_floor_printings_only"]["cos"])))
    for kind in ("real", "synthetic"):
        d = r["t2"]["summary"][kind]
        print("T2 %-9s n=%-3d distinct %s  >floor cos %s l2s %s  >printing-floor cos %s  rel-dist median %s  median partner rank %s"
              % (kind, d["n"], f3(d["frac_distinct_cos"]), f3(d["frac_above_floor_cos"]),
                 f3(d["frac_above_floor_l2s"]), f3(d["frac_above_printing_floor_cos"]),
                 f3(d["cos_rel"]["median"] if d["cos_rel"] else None), f3(d["median_rank_partner_cos"])))
    for sp_ in ("raw", "std"):
        d = r["t2"]["direction"][sp_]
        g = d["groups"].get("amount", {})
        print("T2 direction (%s): within-field cos %s  shuffled %s  p=%s  random-diff |cos| %s  amount-group %s"
              % (sp_, f3(d["mean_within_field_cos"]), f3(d["shuffled_field_labels_mean"]),
                 f3(d["shuffled_p_value"]), f3(d["random_card_difference_abs_cos_mean"]),
                 f3(g.get("mean_within_cos"))))
    t3 = r["t3"]
    if "summary" in t3:
        for part in ("non_provisional", "provisional"):
            d = t3["summary"][part]
            print("T3 %-15s labels=%d beat baseline=%d  R2 %s (base %s)  bacc logit %s centroid %s (majority %s)"
                  % (part, d["n_labels"], d["n_beats_baseline"], f3(d["mean_r2_numeric"]),
                     f3(d["mean_baseline_r2_numeric"]), f3(d["mean_bacc_logistic"]),
                     f3(d["mean_bacc_centroid"]), f3(d["mean_bacc_majority"])))
    else:
        print("T3:", t3.get("error"))
    print("-> wrote", r.get("_written_to"))


if __name__ == "__main__":
    main()
