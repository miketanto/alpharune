"""Structured feature block (gate 2 fix), concatenated after the 8192-dim path hash.

Shared by path B (build.py) and path B2 (extract.py imports build.py), so both get it.
Layout is documented in schema.md ("Structured block") and printed by `python struct_block.py`.

Design
  - card header: type, supertype, domains, keywords (one-/multi-hot) + numeric energy, power, Might,
    keyword values (Assault / Hunt / Level / Shield / Deflect N ...).
  - a fixed grid of N_AB=3 ability slots x N_EFF=4 effect slots. Slot position = order.
  - ability slot: kind, timing, trigger event / subject / once, play-mode + where, trigger conditions
    and ability conditions (condition x polarity), "may" (optional cost), cost flags + numeric cost.
  - effect slot: verb, target side / type / zone / count, chooser, duration, destination, "may",
    effect conditions (x polarity), target filters (x polarity), amount (kind one-hot + numeric).
  - binding: ChooseTarget effects that only bind an object ("bind": X) are folded out of the grid;
    every reference to a bound object (target ref, relative_to, amount might_of:X, source) is replaced
    by that object's side and type plus its reference index (order of binding inside the ability).
    So "which chosen unit does the +2 hit" is a side/type/index difference in that effect's slot.
  - every categorical field in a present slot always emits a value ("none" when absent), so any change
    of a field is a flip of two coordinates, never a single disappearing one.
  - values outside the frozen vocabulary go to the field's "<other>" coordinate.

Weights: one weight per feature GROUP (WEIGHTS), times a global ALPHA relative to the hash. They were
chosen by gate2/tune_struct.py (see gate2/REPORT_v2.txt). Numbers are raw values times the group weight.
"""
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VOCAB_PATH = os.path.join(HERE, "struct_vocab.json")
N_AB, N_EFF = 3, 4
OTHER = "<other>"

# group of every field (the key is the field name without the ab{i}/e{j} prefix)
FIELD_GROUP = {
    # card header
    "type": "hdr", "supertype": "hdr", "domains": "dom", "keywords": "hdr",
    "energy": "num", "power": "num", "might": "might", "kwval": "kwval",
    # ability
    "present": "kind", "kind": "kind", "timing": "kind", "trig_event": "trig", "trig_subject": "trig",
    "trig_once": "trig", "mode": "kind", "where_zone": "kind", "where_filters": "dec",
    "trig_cond": "cond", "cond": "cond", "may": "dec", "cost_flags": "kind",
    "cost_energy": "num", "cost_power": "num", "play_requires": "kind",
    # effect
    "verb": "verb", "tgt_type": "tgt", "tgt_zone": "tgt", "tgt_count": "tgt", "tgt_side": "side",
    "ref": "bind", "binds": "bind", "rel_side": "side", "rel_type": "tgt", "rel_ref": "bind",
    "filters": "filt", "chooser": "dec", "duration": "dec", "dest": "dec", "eff_may": "dec",
    "eff_cond": "cond", "amt_kind": "amt", "amt_per": "amt", "amount": "amtnum", "source": "bind",
    "keyword": "verb", "token_might": "num",
}
MULTI = {"domains", "keywords", "where_filters", "trig_cond", "cond", "cost_flags", "filters",
         "eff_cond"}
NUMERIC = {"energy", "power", "might", "kwval", "cost_energy", "cost_power", "amount", "token_might"}

# slot signatures: a dense unit-norm code (K dims, seeded by the MD5 of the slot's whole canonical content)
# per slot -- card header, each ability header, each effect slot. Any change inside a slot moves its code
# by ~sqrt(2) (codes of different contents are near-orthogonal), so a one-word edit costs as much as
# replacing the whole effect, while the one-hot fields keep the per-field similarity.
SIG_K = 64
SIG_SLOTS = ["card"] + [f"ab{i}" for i in range(N_AB)] + [f"ab{i}.e{j}" for i in range(N_AB) for j in range(N_EFF)]

# tuned in gate2/tune_struct.py; ALPHA scales the whole block against the hash
ALPHA = 1.09
# gate2/tune_sigcap.py, candidate '0.25_h0.6': the hash keeps 60% of the typical squared distance between
# real cards, the slot-signature codes at most 25%, the one-hot / numeric fields the rest (gate2/REPORT_v2.txt)
WEIGHTS = {"amt": 0.524, "amtnum": 0.924, "bind": 0.46, "cond": 0.714, "dec": 0.53, "dom": 3.17, "filt": 0.89, "hdr": 0.543, "kind": 0.391, "kwval": 0.907, "might": 0.607, "num": 0.391, "side": 0.866, "sig_ab": 2.288, "sig_card": 2.851, "sig_eff": 3.538, "tgt": 0.486, "trig": 1.217, "verb": 0.49}

# schema-documented values added to the graph-derived vocabulary (schema.md), so an edit to a
# documented value has its own coordinate instead of <other>
SCHEMA_EXTRAS = {
    "chooser": ["owner", "controller", "opponent", "you"],
    "tgt_side": ["friendly", "enemy", "any"], "rel_side": ["friendly", "enemy", "any"],
    "duration": ["turn", "combat", "while"],
    "kind": ["spell", "triggered", "static", "activated", "replacement", "play_mode", "additional_cost"],
    "timing": ["main", "action", "reaction", "same"],
    "kwval": ["assault", "hunt", "level", "shield", "deflect"],
    "keywords": ["shield", "deflect", "tank", "temporary"], "domains": ["chaos"],
    "ref": ["choice2"], "binds": ["choice2"],
}


def _side_type(sel):
    t = sel.get("type")
    if isinstance(t, list):
        t = "|".join(t)
    return sel.get("side", "none"), t or "none"


def _cond_items(conds):
    return [f"{c['c']}/{'neg' if c.get('neg') else 'pos'}" for c in (conds or [])]


def _filter_items(filters):
    out = []
    for f in filters or []:
        out.append(f"{f['f']}/{'neg' if f.get('neg') else 'pos'}")
    return out


def features(graph):
    """-> list of (field_path, field_name, value, x).  Categorical: value=str, x=1 (or weight 1).
    Numeric: value=None, x=float.  field_path is the slot-qualified name (e.g. 'ab1.e0.verb')."""
    out = []
    st = graph.get("stats", {})

    def cat(path, name, v):
        out.append((path, name, "none" if v is None else str(v), 1.0))

    def num(path, name, x):
        out.append((path, name, None, float(x or 0)))

    cat("type", "type", st.get("type"))
    cat("supertype", "supertype", st.get("supertype"))
    for d in st.get("domains") or []:
        cat("domains", "domains", d)
    for k in graph.get("keywords", []):
        cat("keywords", "keywords", k["kw"])
        if "value" in k and isinstance(k["value"], (int, float)):
            num(f"kwval.{k['kw']}", "kwval", k["value"])
    num("energy", "energy", st.get("energy"))
    num("power", "power", st.get("power"))
    num("might", "might", st.get("might"))

    for i, a in enumerate(graph.get("abilities", [])[:N_AB]):
        p = f"ab{i}."
        cat(p + "present", "present", "yes")
        cat(p + "kind", "kind", a.get("kind"))
        cat(p + "timing", "timing", a.get("timing") if isinstance(a.get("timing"), str) else None)
        tr = a.get("trigger") or {}
        cat(p + "trig_event", "trig_event", tr.get("event"))
        cat(p + "trig_subject", "trig_subject", tr.get("subject"))
        cat(p + "trig_once", "trig_once", tr.get("once"))
        for c in _cond_items(tr.get("condition")):
            cat(p + "trig_cond", "trig_cond", c)
        for c in _cond_items(a.get("condition")):
            cat(p + "cond", "cond", c)
        cat(p + "mode", "mode", a.get("mode"))
        wh = a.get("where") or {}
        cat(p + "where_zone", "where_zone", wh.get("zone"))
        for f in _filter_items(wh.get("filters")):
            cat(p + "where_filters", "where_filters", f)
        cost = a.get("cost") or {}
        cat(p + "may", "may", "may" if cost.get("optional") else "must")
        for fl in ("exhaust", "recycle_self", "additional"):
            if cost.get(fl):
                cat(p + "cost_flags", "cost_flags", fl)
        if cost.get("other"):
            cat(p + "cost_flags", "cost_flags", "other")
        if a.get("setup_cost"):
            cat(p + "cost_flags", "cost_flags", "setup_cost")
        num(p + "cost_energy", "cost_energy", cost.get("energy"))
        num(p + "cost_power", "cost_power", cost.get("power"))
        cat(p + "play_requires", "play_requires", "yes" if a.get("play_requires") else None)

        effs = a.get("effects", []) or []
        # binding table: every selector with "bind" -> (index in binding order, side, type)
        binds = {}
        for e in effs:
            t = e.get("target") if isinstance(e.get("target"), dict) else {}
            if t.get("bind") and t["bind"] not in binds:
                binds[t["bind"]] = (len(binds),) + _side_type(t)

        def resolve(ref):
            if ref in binds:
                ix, s, ty = binds[ref]
                return f"choice{ix}", s, ty
            if ref == "subject":
                return "subject", "subject", f"subject:{tr.get('subject')}"
            if ref == "self":
                return "self", "friendly", "self"
            return f"ref:{ref}", "none", "none"

        slots = [e for e in effs
                 if not (e.get("verb") == "ChooseTarget" and isinstance(e.get("target"), dict)
                         and e["target"].get("bind"))]
        for j, e in enumerate(slots[:N_EFF]):
            q = f"{p}e{j}."
            cat(q + "verb", "verb", e.get("verb"))
            t = e.get("target") if isinstance(e.get("target"), dict) else {}
            if t.get("type") == "ref":
                r, s, ty = resolve(t.get("ref"))
                cat(q + "ref", "ref", r)
                cat(q + "tgt_side", "tgt_side", s)
                cat(q + "tgt_type", "tgt_type", ty)
                cat(q + "tgt_zone", "tgt_zone", "bound")
                cat(q + "tgt_count", "tgt_count", "bound")
            else:
                s, ty = _side_type(t) if t else ("none", "none")
                cat(q + "ref", "ref", "none")
                cat(q + "tgt_side", "tgt_side", s)
                cat(q + "tgt_type", "tgt_type", ty)
                cat(q + "tgt_zone", "tgt_zone", t.get("zone"))
                cat(q + "tgt_count", "tgt_count", t.get("count"))
            cat(q + "binds", "binds", f"choice{binds[t['bind']][0]}" if t.get("bind") in binds else None)
            if t.get("relative_to"):
                r, s, ty = resolve(t["relative_to"])
                cat(q + "rel_ref", "rel_ref", r)
                cat(q + "rel_side", "rel_side", s)
                cat(q + "rel_type", "rel_type", ty)
            else:
                cat(q + "rel_ref", "rel_ref", None)
            for f in _filter_items(t.get("filters")):
                cat(q + "filters", "filters", f)
            cat(q + "chooser", "chooser", e.get("chooser") or t.get("chooser"))
            cat(q + "duration", "duration", e.get("duration"))
            cat(q + "dest", "dest", e.get("dest"))
            cat(q + "eff_may", "eff_may", "may" if (e.get("optional") or t.get("optional")) else "must")
            for c in _cond_items(e.get("condition")):
                cat(q + "eff_cond", "eff_cond", c)
            am = e.get("amount") if isinstance(e.get("amount"), dict) else {}
            if "n" in am and isinstance(am["n"], (int, float)):
                cat(q + "amt_kind", "amt_kind", "n")
                num(q + "amount", "amount", max(-10, min(10, am["n"])))
            elif "ref" in am:
                ref = str(am["ref"])
                if ref.startswith("might_of:"):
                    ref = "might_of:" + resolve(ref.split(":", 1)[1])[0]
                cat(q + "amt_kind", "amt_kind", ref)
            else:
                cat(q + "amt_kind", "amt_kind", None)
            cat(q + "amt_per", "amt_per", am.get("per"))
            src = e.get("source") if isinstance(e.get("source"), dict) else None
            if src:
                cat(q + "source", "source", resolve(src["ref"])[0] if src.get("type") == "ref"
                    else src.get("type"))
            else:
                cat(q + "source", "source", None)
            cat(q + "keyword", "keyword", e.get("keyword"))
            tok = e.get("token") if isinstance(e.get("token"), dict) else None
            if tok and isinstance(tok.get("might"), (int, float)):
                num(q + "token_might", "token_might", tok["might"])
    return out


# ---------------------------------------------------------------- vocabulary + layout

def derive_vocab(graphs):
    """Field name -> sorted categorical values seen in `graphs` (+ SCHEMA_EXTRAS); kwval keywords."""
    voc = {}
    for g in graphs:
        for path, name, v, x in features(g):
            if name == "kwval":
                voc.setdefault("kwval", set()).add(path.split(".", 1)[1])
            elif v is not None:
                voc.setdefault(name, set()).add(v)
    for k, vs in SCHEMA_EXTRAS.items():
        voc.setdefault(k, set()).update(vs)
    return {k: sorted(v) for k, v in sorted(voc.items())}


def _slot_fields():
    """Ordered (path_prefix, field_name) for the whole block, in layout order."""
    card = ["type", "supertype", "domains", "keywords", "energy", "power", "might", "kwval"]
    ab = ["present", "kind", "timing", "trig_event", "trig_subject", "trig_once", "trig_cond", "cond",
          "mode", "where_zone", "where_filters", "may", "cost_flags", "cost_energy", "cost_power",
          "play_requires"]
    ef = ["verb", "ref", "tgt_side", "tgt_type", "tgt_zone", "tgt_count", "binds", "rel_ref", "rel_side",
          "rel_type", "filters", "chooser", "duration", "dest", "eff_may", "eff_cond", "amt_kind", "amount",
          "amt_per", "source", "keyword", "token_might"]
    out = [("", f) for f in card]
    for i in range(N_AB):
        out += [(f"ab{i}.", f) for f in ab]
        for j in range(N_EFF):
            out += [(f"ab{i}.e{j}.", f) for f in ef]
    return out


_LAYOUT = None


def layout(vocab=None):
    """-> (index dict {(path, value): idx}, list of (start, end, path, group, values), size)."""
    global _LAYOUT
    if vocab is None and _LAYOUT is not None:
        return _LAYOUT
    voc = vocab or json.load(open(VOCAB_PATH, encoding="utf-8"))["fields"]
    idx, spans, n = {}, [], 0
    for pre, f in _slot_fields():
        start = n
        if f == "kwval":
            vals = voc.get("kwval", [])
            for v in vals + [OTHER]:
                idx[(f"kwval.{v}", None)] = n
                n += 1
        elif f in NUMERIC:
            vals = ["<value>"]
            idx[(pre + f, None)] = n
            n += 1
        else:
            vals = voc.get(f, []) + [OTHER]
            for v in vals:
                idx[(pre + f, v)] = n
                n += 1
        spans.append((start, n, pre + f, FIELD_GROUP[f], vals))
    for sl in SIG_SLOTS:
        start = n
        idx[("sig:" + sl, None)] = n
        n += SIG_K
        spans.append((start, n, "sig:" + sl, _sig_group(sl), [f"<code {SIG_K}>"]))
    res = (idx, spans, n)
    if vocab is None:
        _LAYOUT = res
    return res


def _sig_group(slot):
    return "sig_card" if slot == "card" else ("sig_eff" if ".e" in slot else "sig_ab")


def _slot_of(path):
    parts = path.split(".")
    if parts[0].startswith("ab") and parts[0][2:].isdigit():
        if len(parts) > 2 and parts[1].startswith("e") and parts[1][1:].isdigit():
            return parts[0] + "." + parts[1]
        return parts[0]
    return "card"


def _code(content):
    seed = int.from_bytes(hashlib.md5(content.encode("utf-8")).digest()[:8], "little")
    v = np.random.default_rng(seed).standard_normal(SIG_K)
    return v / np.linalg.norm(v)


def signatures(graph):
    """slot -> canonical content string (only slots that exist in the graph)."""
    cont = {}
    for path, name, v, x in features(graph):
        sl = _slot_of(path)
        if sl not in SIG_SLOTS:
            continue
        rel = path[len(sl) + 1:] if sl != "card" else path
        cont.setdefault(sl, []).append(f"{rel}={v if v is not None else repr(round(x, 6))}")
    return {sl: "|".join(sorted(c)) for sl, c in cont.items()}


def group_vectors(graph):
    """Unweighted block per group: {group: {index: value}} (gate2/tune_struct.py uses this)."""
    idx, spans, n = layout()
    out = {}
    for path, name, v, x in features(graph):
        if name == "kwval":
            key = (path, None) if (path, None) in idx else ("kwval." + OTHER, None)
        elif v is None:
            key = (path, None)
        else:
            key = (path, v) if (path, v) in idx else (path, OTHER)
        if key not in idx:
            continue   # beyond the grid
        g = FIELD_GROUP[name]
        d = out.setdefault(g, {})
        d[idx[key]] = d.get(idx[key], 0.0) + x
    for sl, content in signatures(graph).items():
        d = out.setdefault(_sig_group(sl), {})
        i0 = idx[("sig:" + sl, None)]
        for t, c in enumerate(_code(sl + "#" + content)):
            d[i0 + t] = float(c)
    return out


def size():
    return layout()[2]


def vectorize(graph, weights=None, alpha=None):
    w = WEIGHTS if weights is None else weights
    a = ALPHA if alpha is None else alpha
    v = [0.0] * size()
    for g, d in group_vectors(graph).items():
        for i, x in d.items():
            v[i] += a * w[g] * x
    return v


def describe():
    idx, spans, n = layout()
    lines = [f"structured block: {n} dims (N_AB={N_AB} x N_EFF={N_EFF}), ALPHA={ALPHA}",
             "weights: " + json.dumps(WEIGHTS)]
    for s, e, path, g, vals in spans:
        if path.startswith("ab") and not path.startswith(("ab0.", )):
            continue
        if path.startswith("ab0.e") and not path.startswith("ab0.e0."):
            continue
        lines.append(f"  [{s:4d},{e:4d}) {path:24s} {g:6s} {vals if len(vals) < 40 else vals[:40] + ['...']}")
    lines.append("  (ab1, ab2 and effect slots e1..e3 repeat the ab0 / ab0.e0 layout)")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
