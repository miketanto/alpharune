"""Replay jobs (decision_export verify format) from raw/<prefix>*.games.jsonl.
usage: make_verify.py <prefix> <out>   (games that ended in an error are skipped)"""
import glob, json, sys, os
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw")
n = 0
with open(sys.argv[2], "w", newline="\n") as out:
    for f in sorted(glob.glob(os.path.join(here, sys.argv[1] + "*.games.jsonl"))):
        for line in open(f):
            d = json.loads(line)
            if d["term"].startswith("error"):
                continue
            ch = d["choices"]
            parts = [d["gid"], d["deck_a"], d["deck_b"], str(d["seed"]), str(d["winner"]), str(len(ch))]
            parts += [str(c) for c in ch] + [str(len(d["checks"]))]
            for sq, fp, h in d["checks"]:
                parts += [str(sq), fp, h]
            out.write(" ".join(parts) + "\n")
            n += 1
print(n, "games")
