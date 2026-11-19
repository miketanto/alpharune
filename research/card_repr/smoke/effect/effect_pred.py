"""Effect-prediction smoke test: can a small MLP learn a card's state delta from R(card)
and generalise to a held-out card that differs from a seen card in one detail?

Input:  R(card) (frozen, one learned linear projection) + situation kind (play / trigger).
Target: the card's recorded state delta (fixed field list TARGETS below), z-scaled by the
        per-sample S-train std of each field.
Training objective: full-batch MSE to the per-(card, kind) mean delta of the training pool,
        every (card, kind) group weighted equally. (Equal, up to a constant, to per-sample MSE
        with each group's samples reweighted to the same total weight; the model only sees
        card + kind, so it can only learn the group mean anyway.)

Costs are EXCLUDED from the targets: pool H pays from a prepaid pool (pool.energy,
pool.power_*), pool S pays by exhausting / recycling runes (me.runes_exhausted, me.runes),
so the two pools record costs in non-comparable fields; printed costs are also part of the
stats input, which would make cost targets trivially stats-predictable.

Evaluations
  (1) S-train -> S-test, same cards.
  (2) leave-one-pair-out held-out cards (punch_first/primal_strength + 10 synthetic variants),
      plus a 'strict' fold holding out all 11 test cards together.
  (3) pool transfer S-train -> H (and H -> S-test).

Run:  python effect_pred.py           (writes results.json, summary.csv next to this file)
"""
import collections
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

torch.set_num_threads(4)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
SIT = os.path.join(ROOT, "situations")

SEEDS = [0, 1, 2]
BOOT_SEED = 0
N_BOOT = 2000
STEPS = int(os.environ.get("EFF_STEPS", 2000))
LR = 3e-3
WD = 1e-4
D_PROJ = 64
D_HID = 128
RANDOM_DIM = 256
KINDS = ["play", "trigger"]

# Fixed target fields (all present in both pools' delta vocabularies; costs excluded).
TARGETS = [
    # draw / hand / deck
    "ev_drawn_me", "ev_drawn_opp", "me.hand", "opp.hand", "me.deck", "opp.deck",
    # might
    "me.sum_might", "opp.sum_might", "SELF.might", "SELF.temp_might", "me.buffs", "opp.buffs",
    # damage
    "ev_dmg_to_opp_units", "ev_dmg_to_my_units", "opp.sum_damage", "me.sum_damage",
    # deaths / removal / units / gear
    "ev_died_opp", "ev_died_me", "opp.units", "me.units", "ev_banished", "ev_returned_to_hand",
    "opp.gear", "me.gear",
    # tokens
    "me.tokens", "ev_tokens_me", "ev_token_might_me", "opp.tokens",
    # trash / counter / spells
    "opp.trash", "me.trash", "ev_spells_resolved_opp", "ev_spells_resolved_me",
    # move / ready / stun / exhaust
    "ev_moved_me", "ev_moved_opp", "ev_readied_me", "ev_stun_opp_units", "ev_stun_my_units",
    "me.exhausted_units", "opp.exhausted_units", "opp.stunned", "me.stunned",
    # xp
    "me.xp", "opp.xp",
    # the card's own zone after the action
    "SELF.on_board", "SELF.in_trash", "SELF.in_banish", "SELF.in_hand", "SELF.exhausted",
]
# Bookkeeping fields fixed mostly by card type (played card leaves hand, unit enters board...).
BOOKKEEPING = {"me.hand", "me.trash", "me.units", "me.exhausted_units", "ev_spells_resolved_me",
               "SELF.on_board", "SELF.in_trash", "SELF.in_banish", "SELF.in_hand", "SELF.exhausted"}
EFFECT_IDX = [i for i, f in enumerate(TARGETS) if f not in BOOKKEEPING]
TIDX = {f: i for i, f in enumerate(TARGETS)}

CONDITIONS = ["id", "random", "stats", "A", "B", "B2", "shuffled-B"]
VEC_FILES = {"A": "A_text/out/vectors.json", "B": "B_graph/out/vectors.json",
             "B2": "B2_code/out/vectors.json"}

REAL_PAIR = ("punch_first", "primal_strength")


# ---------------------------------------------------------------- data

def situation_target(beh):
    """Mean over executed actions (all locations, first/last target policy; 'hide' excluded)."""
    ex = [d for k, d in beh.get("exec", {}).items() if not k.startswith("hide")]
    if not ex:
        return None
    v = np.zeros(len(TARGETS))
    for d in ex:
        for f, x in d.items():
            j = TIDX.get(f)
            if j is not None:
                v[j] += x
    return v / len(ex)


def load_pool(name):
    out = json.load(open(os.path.join(SIT, "out", name + ".json")))["cards"]
    hkinds = None
    if name == "H":
        hkinds = [s["kind"] for s in json.load(open(os.path.join(SIT, "H.json")))["situations"]]
        nh = len(hkinds)
    groups = collections.defaultdict(list)
    for card, c in out.items():
        B = c["behaviours"]
        for row, b in c["situations"]:
            beh = B[b]
            kind = hkinds[row % nh] if name == "H" else beh.get("kind")
            if kind not in KINDS:
                continue
            t = situation_target(beh)
            if t is None:
                continue
            groups[(card, kind)].append(t)
    return {k: np.array(v) for k, v in groups.items()}


def stats_vectors():
    """Printed facts from the engine CardDef (pool H 'card' situation), for every card incl. synthetics."""
    out = json.load(open(os.path.join(SIT, "out", "H.json")))["cards"]
    facts = {}
    for card, c in out.items():
        row, b = c["situations"][0]
        assert row % 33 == 0
        facts[card] = c["behaviours"][b].get("legal", {})
    keys = sorted({k for d in facts.values() for k in d})
    return {c: np.array([float(d.get(k, 0)) for k in keys]) for c, d in facts.items()}, keys


def load_vecs(path):
    v = json.load(open(os.path.join(ROOT, path)))["vectors"]
    return {k: np.array(x, dtype=float) for k, x in v.items() if not k.startswith(("printing:", "rename:"))}


# ---------------------------------------------------------------- model

class Net(torch.nn.Module):
    def __init__(self, d_in, n_cards, n_out, use_id):
        super().__init__()
        self.use_id = use_id
        if use_id:
            self.proj = torch.nn.Embedding(n_cards, D_PROJ)
        else:
            self.proj = torch.nn.Linear(d_in, D_PROJ)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(D_PROJ + len(KINDS), D_HID), torch.nn.ReLU(),
            torch.nn.Linear(D_HID, D_HID), torch.nn.ReLU(),
            torch.nn.Linear(D_HID, n_out))

    def forward(self, x, kind_onehot):
        return self.mlp(torch.cat([self.proj(x), kind_onehot], dim=1))


def T(a, dtype=torch.float32):
    return torch.tensor(np.asarray(a).tolist(), dtype=dtype)


def prep_inputs(cond, R, cards, train_cards):
    """Center by the training-card mean and divide by one global RMS norm (fit on training cards)."""
    if cond == "id":
        return None
    M = np.stack([R[c] for c in cards])
    tr = np.array([c in train_cards for c in cards])
    keep = np.abs(M).sum(0) > 0  # drop all-zero columns (no effect on the function class)
    M = M[:, keep]
    mu = M[tr].mean(0)
    Mc = M - mu
    s = math.sqrt((Mc[tr] ** 2).sum(1).mean()) or 1.0
    return Mc / s


def train_predict(cond, R, cards, train_groups, seed):
    """train_groups: list of (card, kind, target z-vector). Returns predict(card, kind) for all cards."""
    train_cards = {g[0] for g in train_groups}
    X = prep_inputs(cond, R, cards, train_cards)
    cidx = {c: i for i, c in enumerate(cards)}
    torch.manual_seed(seed)
    net = Net(0 if X is None else X.shape[1], len(cards), len(TARGETS), cond == "id")
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    ids = [cidx[c] for c, _, _ in train_groups]
    if X is None:
        xin = torch.tensor(ids, dtype=torch.long)
    else:
        xin = T(X[ids])
    kin = T([[1.0 if k == kk else 0.0 for kk in KINDS] for _, k, _ in train_groups])
    y = T(np.stack([t for _, _, t in train_groups]))
    for step in range(STEPS):
        opt.zero_grad()
        loss = ((net(xin, kin) - y) ** 2).mean()
        loss.backward()
        opt.step()
    final = float(loss.item())
    net.eval()
    with torch.no_grad():
        allx = torch.arange(len(cards)) if X is None else T(X)
        preds = {}
        for kk in KINDS:
            ko = T([[1.0 if k == kk else 0.0 for k in KINDS]] * len(cards))
            P = np.array(net(allx, ko).tolist())
            for c in cards:
                preds[(c, kk)] = P[cidx[c]]
    return preds, final


# ---------------------------------------------------------------- stats helpers

def boot_ci(vals, rng, n=N_BOOT):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return [None, None, None]
    m = vals.mean()
    idx = rng.integers(0, len(vals), size=(n, len(vals)))
    bs = vals[idx].mean(1)
    return [float(m), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def gerr(p, t, idx=None):
    d = (p - t) ** 2
    return float(d.mean() if idx is None else d[idx].mean())


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(a @ b / (na * nb))


# ---------------------------------------------------------------- main

def main():
    t0 = time.time()
    cardsj = json.load(open(os.path.join(ROOT, "cards.json")))
    pairs = [(REAL_PAIR[0], REAL_PAIR[1], "real: amount +5 vs +7 and cost")]
    for s in cardsj["t2_minimal_pairs_synthetic"]:
        pairs.append((s["base"], "synthetic:" + s["id"], s["field"]))

    pools = {p: load_pool(p) for p in ("S_train", "S_test", "H")}
    # z-scale: per-sample std on S-train (all groups pooled)
    allS = np.concatenate([v for v in pools["S_train"].values()])
    sd = allS.std(0)
    dead = [TARGETS[i] for i in range(len(TARGETS)) if sd[i] < 1e-9]
    sd[sd < 1e-9] = 1.0
    zp = {p: {k: v / sd for k, v in g.items()} for p, g in pools.items()}
    means = {p: {k: v.mean(0) for k, v in g.items()} for p, g in zp.items()}
    counts = {p: {k: len(v) for k, v in g.items()} for p, g in zp.items()}
    within_var = {p: {k: float(v.var(0).mean()) for k, v in g.items()} for p, g in zp.items()}
    # S all (train + test) for held-out cards
    sall = {}
    for p in ("S_train", "S_test"):
        for k, v in zp[p].items():
            sall.setdefault(k, []).append(v)
    sall = {k: np.concatenate(v) for k, v in sall.items()}
    mean_sall = {k: v.mean(0) for k, v in sall.items()}

    stats, stat_keys = stats_vectors()
    reps = {"stats": stats}
    for c in ("A", "B", "B2"):
        reps[c] = load_vecs(VEC_FILES[c])

    cards = sorted({k[0] for p in pools.values() for k in p})
    for name, R in reps.items():
        miss = [c for c in cards if c not in R]
        assert not miss, (name, miss)

    def rep_for(cond, seed):
        if cond == "id":
            return None
        if cond == "random":
            rng = np.random.default_rng(1000 + seed)
            return {c: rng.standard_normal(RANDOM_DIM) for c in cards}
        if cond == "shuffled-B":
            rng = random.Random(2000 + seed)
            perm = cards[:]
            while True:
                rng.shuffle(perm)
                if all(a != b for a, b in zip(cards, perm)):
                    break
            return {c: reps["B"][p] for c, p in zip(cards, perm)}
        return reps[cond]

    def groups_of(pool, exclude=()):
        return [(c, k, means[pool][(c, k)]) for (c, k) in sorted(means[pool]) if c not in exclude]

    rng_b = np.random.default_rng(BOOT_SEED)
    res = {"config": {
        "seeds": SEEDS, "bootstrap_seed": BOOT_SEED, "n_boot": N_BOOT, "steps": STEPS, "lr": LR,
        "weight_decay": WD, "optimizer": "AdamW full-batch", "arch": f"proj(R)->{D_PROJ} ++ kind onehot({KINDS}) -> {D_HID} ReLU -> {D_HID} ReLU -> {len(TARGETS)}",
        "random_dim": RANDOM_DIM, "random_vector_seed": "1000+seed", "shuffle_seed": "2000+seed (derangement over all cards)",
        "input_preprocessing": "drop all-zero columns; center by training-card mean; divide by one global RMS norm of training cards",
        "targets": TARGETS, "effect_subset": [TARGETS[i] for i in EFFECT_IDX],
        "targets_dead_in_S_train": dead, "target_scale": "per-field per-sample std on S-train",
        "costs": "excluded (H pays from pool.*, S by exhausting/recycling runes: non-comparable; printed costs are in the stats input)",
        "situation_target": "mean over executed actions of the situation (all locations, first/last target policy), hide actions excluded; kinds play+trigger only (S has no 'act' rows, 'inplay'/'held' record no delta)",
        "stats_keys": stat_keys, "torch_threads": 4, "n_cards": len(cards),
        "group_counts": {p: len(means[p]) for p in means},
        "sample_counts": {p: int(sum(counts[p].values())) for p in counts},
    }}
    print("groups", res["config"]["group_counts"], "samples", res["config"]["sample_counts"], "dead", dead, flush=True)

    # ---------------- (1) + (3): train on all S-train
    E1 = {}  # cond -> seed -> card -> list errs
    E3 = {}
    E3e = {}
    E1e = {}
    final_loss = collections.defaultdict(dict)
    full_preds = {}
    for cond in CONDITIONS:
        for seed in SEEDS:
            preds, fl = train_predict(cond, rep_for(cond, seed), cards, groups_of("S_train"), seed)
            final_loss[cond][f"full_S_train/{seed}"] = fl
            full_preds[(cond, seed)] = preds
            for pool, store, storee in (("S_test", E1, E1e), ("H", E3, E3e)):
                for (c, k), t in means[pool].items():
                    if (c, k) not in means["S_train"]:
                        continue  # same cards & kinds as training only
                    store.setdefault(cond, {}).setdefault(c, []).append(gerr(preds[(c, k)], t))
                    storee.setdefault(cond, {}).setdefault(c, []).append(gerr(preds[(c, k)], t, EFFECT_IDX))
        print("full", cond, f"{time.time()-t0:.0f}s", flush=True)

    # oracles / floors for (1) and (3)
    def oracle(pool, idx=None):
        o, f = {}, {}
        glob = {k: np.mean([means["S_train"][g] for g in means["S_train"] if g[1] == k], 0) for k in KINDS}
        for (c, k), t in means[pool].items():
            if (c, k) not in means["S_train"]:
                continue
            o.setdefault(c, []).append(gerr(means["S_train"][(c, k)], t, idx))
            f.setdefault(c, []).append(gerr(glob[k], t, idx))
        return o, f

    def summarize(store, extra):
        cards_ = sorted(next(iter(store.values())).keys())
        out = {}
        per = {cond: np.array([np.mean(store[cond][c]) for c in cards_]) for cond in store}
        for name, d in extra.items():
            per[name] = np.array([np.mean(d[c]) for c in cards_])
        for cond, v in per.items():
            out[cond] = {"mean_err": boot_ci(v, rng_b)}
            for ref in ("random", "id", "stats"):
                if ref in per and ref != cond:
                    out[cond][f"minus_{ref}"] = boot_ci(v - per[ref], rng_b)
        return out, cards_, per

    o1, f1 = oracle("S_test")
    o1e, f1e = oracle("S_test", EFFECT_IDX)
    s1, c1, p1 = summarize(E1, {"oracle_S_train_card_mean": o1, "floor_kind_mean": f1})
    s1e, _, _ = summarize(E1e, {"oracle_S_train_card_mean": o1e, "floor_kind_mean": f1e})
    o3, f3 = oracle("H")
    o3e, f3e = oracle("H", EFFECT_IDX)
    s3, c3, p3 = summarize(E3, {"oracle_S_train_card_mean": o3, "floor_kind_mean": f3})
    s3e, _, _ = summarize(E3e, {"oracle_S_train_card_mean": o3e, "floor_kind_mean": f3e})
    # per-seed spread
    seed_spread = {}
    for cond in CONDITIONS:
        seed_spread[cond] = [float(np.mean([E1[cond][c][i] for c in c1])) for i in range(len(SEEDS))]
    res["eval1_S_train_to_S_test"] = {
        "metric": "group-mean error: mean over target fields of (pred - S-test (card,kind) mean)^2 in z units, averaged over a card's kinds, then over cards; 95% bootstrap CI over cards (seed-averaged)",
        "n_cards": len(c1), "all_fields": s1, "effect_fields": s1e, "per_seed_mean_err": seed_spread,
        "within_group_variance_S_test_mean": float(np.mean([within_var["S_test"][g] for g in means["S_test"] if g in means["S_train"]])),
        "note": "per-sample MSE = group-mean error + within-group (situational) variance, which no card-only model can reduce",
        "per_card": {cond: {c: float(v) for c, v in zip(c1, p1[cond])} for cond in p1},
    }
    res["eval3_S_train_to_H"] = {
        "metric": "as eval1 but targets are pool-H (card,kind) means; same cards and kinds as S-train only",
        "n_cards": len(c3), "all_fields": s3, "effect_fields": s3e,
        "per_card": {cond: {c: float(v) for c, v in zip(c3, p3[cond])} for cond in p3},
    }
    print("eval1/3 done", f"{time.time()-t0:.0f}s", flush=True)

    # ---------------- (3b) H -> S-test
    E3b, E3be = {}, {}
    for cond in CONDITIONS:
        for seed in SEEDS:
            preds, fl = train_predict(cond, rep_for(cond, seed), cards, groups_of("H"), seed)
            final_loss[cond][f"full_H/{seed}"] = fl
            for (c, k), t in means["S_test"].items():
                if (c, k) not in means["H"]:
                    continue
                E3b.setdefault(cond, {}).setdefault(c, []).append(gerr(preds[(c, k)], t))
                E3be.setdefault(cond, {}).setdefault(c, []).append(gerr(preds[(c, k)], t, EFFECT_IDX))
    oh, oh_e = {}, {}
    for (c, k), t in means["S_test"].items():
        if (c, k) in means["H"]:
            oh.setdefault(c, []).append(gerr(means["H"][(c, k)], t))
            oh_e.setdefault(c, []).append(gerr(means["H"][(c, k)], t, EFFECT_IDX))
    s3b, c3b, _ = summarize(E3b, {"oracle_H_card_mean": oh})
    s3be, _, _ = summarize(E3be, {"oracle_H_card_mean": oh_e})
    res["eval3b_H_to_S_test"] = {"n_cards": len(c3b), "all_fields": s3b, "effect_fields": s3be,
                                 "note": "H has 1-10 probe situations per card; trained on H (card,kind) means"}
    print("eval3b done", f"{time.time()-t0:.0f}s", flush=True)

    # ---------------- (2) held-out cards
    def heldout_metrics(preds, base, var):
        m = {}
        kinds = [k for k in KINDS if (var, k) in mean_sall and (base, k) in mean_sall]
        kinds_h = [k for k in KINDS if (var, k) in means["H"] and (base, k) in means["H"]]
        if not kinds:
            return None
        ev, ecb, ebase, edm, dcos, drel, dcos_h, drel_h, evh, ecbh = [], [], [], [], [], [], [], [], [], []
        dnorm, dnoise = [], []
        for k in kinds:
            tv = mean_sall[(var, k)]
            ev.append(gerr(preds[(var, k)], tv))
            ecb.append(gerr(preds[(base, k)], tv))
            edm.append(gerr(mean_sall[(base, k)], tv))
            if (base, k) in means["S_test"]:
                ebase.append(gerr(preds[(base, k)], means["S_test"][(base, k)]))
            td = tv - mean_sall[(base, k)]
            pd = preds[(var, k)] - preds[(base, k)]
            dcos.append(cos(pd, td))
            drel.append(float(((pd - td) ** 2).sum() / max((td ** 2).sum(), 1e-12)))
            dnorm.append(float((td ** 2).sum()))
            dnoise.append(float((sall[(var, k)].var(0) / len(sall[(var, k)])).sum()
                                + (sall[(base, k)].var(0) / len(sall[(base, k)])).sum()))
        for k in kinds_h:
            td = means["H"][(var, k)] - means["H"][(base, k)]
            pd = preds[(var, k)] - preds[(base, k)]
            dcos_h.append(cos(pd, td))
            drel_h.append(float(((pd - td) ** 2).sum() / max((td ** 2).sum(), 1e-12)) if (td ** 2).sum() > 1e-12 else None)
            evh.append(gerr(preds[(var, k)], means["H"][(var, k)]))
            ecbh.append(gerr(preds[(base, k)], means["H"][(var, k)]))
        m["err_variant_S"] = float(np.mean(ev))
        m["err_copy_base_pred_S"] = float(np.mean(ecb))
        m["err_base_on_S_test"] = float(np.mean(ebase)) if ebase else None
        m["data_gap_base_vs_variant_S"] = float(np.mean(edm))
        m["diff_cos_S"] = float(np.mean(dcos))
        m["diff_rel_err_S"] = float(np.mean(drel))
        m["true_diff_sqnorm_S"] = float(np.mean(dnorm))
        m["true_diff_noise_sqnorm_S"] = float(np.mean(dnoise))
        m["diff_snr_S"] = float(np.mean(dnorm) / max(np.mean(dnoise), 1e-12))
        if kinds_h:
            m["err_variant_H"] = float(np.mean(evh))
            m["err_copy_base_pred_H"] = float(np.mean(ecbh))
            m["diff_cos_H"] = float(np.mean(dcos_h))
            rr = [x for x in drel_h if x is not None]
            m["diff_rel_err_H"] = float(np.mean(rr)) if rr else None
            m["true_diff_zero_in_H"] = not rr
        return m

    folds = {}
    for base, var, field in pairs:
        folds[var] = {"base": base, "field": field, "holdout": [var]}
    strict_hold = [v for _, v, _ in pairs]
    folds["STRICT_all_11"] = {"base": None, "field": "all test cards held out together", "holdout": strict_hold}
    base_of = {v: b for b, v, _ in pairs}
    E2 = {}
    for fname, fd in folds.items():
        for cond in CONDITIONS:
            for seed in SEEDS:
                preds, fl = train_predict(cond, rep_for(cond, seed), cards,
                                          groups_of("S_train", exclude=set(fd["holdout"])), seed)
                final_loss[cond][f"fold:{fname}/{seed}"] = fl
                for var in fd["holdout"]:
                    base = base_of[var]
                    if base in fd["holdout"]:
                        continue
                    m = heldout_metrics(preds, base, var)
                    if m is None:
                        continue
                    E2.setdefault(fname, {}).setdefault(var, {}).setdefault(cond, []).append(m)
        print("fold", fname, f"{time.time()-t0:.0f}s", flush=True)

    def agg2(fold_names, var_from_fold):
        """Seed-average per test card, then bootstrap over test cards."""
        out = {}
        keys = ["err_variant_S", "err_copy_base_pred_S", "err_base_on_S_test", "diff_cos_S", "diff_rel_err_S",
                "err_variant_H", "err_copy_base_pred_H", "diff_cos_H", "diff_rel_err_H"]
        per = {}
        for cond in CONDITIONS:
            per[cond] = {}
            for key in keys:
                vals = []
                for fname in fold_names:
                    for var in var_from_fold(fname):
                        ms = E2.get(fname, {}).get(var, {}).get(cond)
                        if not ms:
                            continue
                        xs = [m.get(key) for m in ms if m.get(key) is not None]
                        vals.append(float(np.mean(xs)) if xs else float("nan"))
                per[cond][key] = np.array(vals)
            # ratio: variant error relative to copy-the-base-prediction (log ratio averaged)
            per[cond]["log_ratio_var_vs_copybase_S"] = np.log(per[cond]["err_variant_S"] / per[cond]["err_copy_base_pred_S"])
            per[cond]["var_minus_base_err_S"] = per[cond]["err_variant_S"] - per[cond]["err_base_on_S_test"]
        for cond in CONDITIONS:
            out[cond] = {}
            for key, v in per[cond].items():
                v2 = v[~np.isnan(v)]
                out[cond][key] = boot_ci(v2, rng_b)
            for ref in ("random", "id", "stats", "shuffled-B"):
                if ref == cond:
                    continue
                for key in ("err_variant_S", "diff_cos_S", "diff_rel_err_S", "diff_cos_H"):
                    a, b = per[cond][key], per[ref][key]
                    ok = ~(np.isnan(a) | np.isnan(b))
                    out[cond][f"{key}_minus_{ref}"] = boot_ci(a[ok] - b[ok], rng_b)
        return out

    lopo = [v for _, v, _ in pairs]
    res["eval2_heldout_cards"] = {
        "metrics": {
            "err_variant_S": "group-mean error of the held-out card vs its mean S delta (S-train+S-test, never trained on)",
            "err_copy_base_pred_S": "error if the model's prediction for the BASE card were used for the variant",
            "err_base_on_S_test": "the same model's error on its base card (trained card) vs the base's S-test mean",
            "log_ratio_var_vs_copybase_S": "log(err_variant/err_copy_base_pred); <0 = the card vector moved the prediction toward the variant's truth",
            "diff_cos_S": "cosine(pred(variant)-pred(base), true mean(variant)-mean(base)) over target fields, S (unpaired situations)",
            "diff_rel_err_S": "||pred_diff - true_diff||^2 / ||true_diff||^2; 1 = predicting no difference",
            "diff_cos_H / diff_rel_err_H": "same on pool H, where variant and base are run in identical probe situations (paired)",
        },
        "pairs": [{"base": b, "variant": v, "field": f} for b, v, f in pairs],
        "leave_one_pair_out": agg2(lopo, lambda f: [f]),
        "strict_all_held_out": agg2(["STRICT_all_11"], lambda f: strict_hold),
        "per_fold": {fname: {var: {cond: {k: float(np.mean([m[k] for m in ms if m.get(k) is not None]))
                                          if any(m.get(k) is not None for m in ms) else None
                                          for k in ms[0] if not isinstance(ms[0][k], bool)}
                                   for cond, ms in d.items()} for var, d in vs.items()}
                     for fname, vs in E2.items()},
    }
    res["final_train_loss"] = final_loss
    res["runtime_s"] = time.time() - t0
    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1)

    # summary csv
    with open(os.path.join(HERE, "summary.csv"), "w") as f:
        f.write("eval,condition,metric,mean,ci_lo,ci_hi\n")
        for ev, blk in (("1_S_train_to_S_test", s1), ("1_effect_fields", s1e), ("3_S_train_to_H", s3),
                        ("3_effect_fields", s3e), ("3b_H_to_S_test", s3b)):
            for cond, d in blk.items():
                for k, v in d.items():
                    f.write(f"{ev},{cond},{k},{v[0]},{v[1]},{v[2]}\n")
        for ev, blk in (("2_LOPO", res["eval2_heldout_cards"]["leave_one_pair_out"]),
                        ("2_strict", res["eval2_heldout_cards"]["strict_all_held_out"])):
            for cond, d in blk.items():
                for k, v in d.items():
                    f.write(f"{ev},{cond},{k},{v[0]},{v[1]},{v[2]}\n")
    print("done", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
