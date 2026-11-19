# Count decisions (base_teacher shards) where each missing fact is live, from the decider's view.
import gzip, json, glob, sys, collections
files = sorted(glob.glob(sys.argv[1] + "/base_teacher__s*.decisions.jsonl.gz"))
C = collections.Counter()
for fn in files:
    last_sab = {}          # (gid, seat) -> seq of a Sabotage choose prompt
    for line in gzip.open(fn, "rt"):
        d = json.loads(line)
        C["decisions"] += 1
        tok, act, g, pk = d["tok"], d["act"], d["g"], d["pk"]
        key = lambda i: tok[i][1] if 0 <= i < len(tok) else None
        cards = [t for t in tok if t[0] == "c"]
        bfs = [t for t in tok if t[0] == "b"]
        items = [t for t in tok if t[0] == "k"]
        # B1: a token unit with no card vector present
        if any(t[1] in ("token:mech", "token:sand_soldier") for t in cards): C["B1 token unit present"] += 1
        if any(t[1] in ("token:mech", "token:sand_soldier") for t in cards) and any("tg" in a or "ch" in a or "mv" in a or "src" in a for a in act):
            refs = set()
            for a in act:
                for k in ("tg", "ch", "mv"):
                    refs.update(a.get(k, []))
                if "src" in a: refs.add(a["src"])
            if any(key(i) in ("token:mech", "token:sand_soldier") for i in refs): C["B1 token unit referenced by an action"] += 1
        # A4 Star Spring: own unit play offered to the Star Spring slot
        ss = [t[2] for t in bfs if t[1] == "star_spring"]
        if ss:
            slot = 1 + ss[0]
            if any(a["t"] in (0, 5, 8) and a.get("dst") == slot and key(a.get("src", -1)) not in (None,) for a in act):
                C["A4 star_spring: unit play to its slot offered"] += 1
        # A4 hidden_on_turn: own facedown card on the board
        if any(t[13] == 1 and t[2] == 1 for t in cards): C["A4 own facedown card (hidden turn unknown)"] += 1
        # A4 Rampage paid flag: Rampage on the chain
        if any(t[1] == "rampage" for t in items): C["A4 rampage on chain"] += 1
        if any(t[1] == "rampage" for t in items) and g[66] == 1 and tok[[i for i, t in enumerate(tok) if t[0] == 'k' and t[1] == 'rampage'][0]][2] == 2:
            C["A4 opp rampage on chain, me to respond"] += 1
        # A8: Sabotage choice prompt; later decisions of same seat in same game
        if pk == "choose" and d["psrc"] >= 0 and key(d["psrc"]) == "sabotage":
            C["A8 sabotage choose prompt"] += 1
            last_sab[(d["gid"], d["seat"])] = d["seq"]
        elif (d["gid"], d["seat"]) in last_sab:
            C["A8 later decisions of a seat that saw a Sabotage reveal"] += 1
        # Thrill location prompt: options vs battlefields
        if pk == "location" and d["psrc"] >= 0 and key(d["psrc"]) == "thrill_of_the_hunt":
            C["Thrill location prompt"] += 1
            if len(act) != len(bfs): C["Thrill location prompt with filtered list (v != slot)"] += 1
        # Legion live: Noxus Hopeful in hand
        if any(t[1] == "noxus_hopeful" and t[4] == 2 for t in cards): C["A6 noxus_hopeful in own hand"] += 1
        # Inferna potential attacker on board
        if any(t[1] == "inferna" and t[2] == 1 and t[5] >= 0 for t in cards): C["A3 own inferna on board"] += 1
        if any(t[1] == "darius_trifarian" and t[2] == 1 and t[5] >= 0 for t in cards) and items: C["A4 darius on board with a chain"] += 1
print(json.dumps(C, indent=1))
