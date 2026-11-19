"""Game list for autopay vs teacher.  Usage: make_list.py <out> <first_k> <n_seeds> [sims] [RR|YY|RY|YR]
With a matchup given, every seed plays that matchup only (the RR-only batches).
Seed k (seed = 9_000_000 + k) plays matchup k % 4 of (R,R) (Y,Y) (R,Y) (Y,R) twice:
gid AP<k>a with autopay as deck A (seat 1) and AP<k>b with autopay as deck B (seat 2).
Same seed => same deal and shuffles for both games of the pair; agent seeds are
engine_seed*2+seat, so each seat's search RNG is also the same in both games."""
import sys
R, Y = "decks/rengar.txt", "decks/yi_rebuild.txt"
M = [(R, R), (Y, Y), (R, Y), (Y, R)]
out, k0, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
sims = sys.argv[4] if len(sys.argv) > 4 else "100"
only = sys.argv[5] if len(sys.argv) > 5 else None
D = {"R": R, "Y": Y}
T, A = f"ismcts:sims={sims}", f"autopay:sims={sims}"
with open(out, "w", newline="\n") as f:
    for k in range(k0, k0 + n):
        a, b = (D[only[0]], D[only[1]]) if only else M[k % 4]
        s = 9_000_000 + k
        f.write(f"AP{k}a {a} {b} {A} {T} {s} 0\n")
        f.write(f"AP{k}b {a} {b} {T} {A} {s} 0\n")
