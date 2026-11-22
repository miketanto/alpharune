"""Head-to-head evaluation of AlphaZero checkpoints piloting Azir (az/README.txt).

  python3 eval_az.py --ckpt run/az_004000.pt [--ckpt ...] --games 1000 --procs 14 --cores 0-13

Greedy (argmax) both sides, seat-balanced, the same deals for both seat orders (g1/eval_g1.play_pair).
For each candidate X (piloting Azir):
  X(azir) vs G2(azir)          mirror against its starting point
  X(azir) vs G2(rengar)        and the pilot-swap baseline G2(azir) vs G2(rengar)
  X(azir) vs G2(diana)         and the pilot-swap baseline G2(azir) vs G2(diana)
  X(azir) vs S_A(azir)         the Azir specialist (az_main@300)
Pilot swap = X's win rate minus G2's against the same reference: the deck is the same, only the pilot changes.
Results: <out> (jsonl, cached by checkpoint file stamp), table printed.
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CR, 'g1'))
import eval_g1 as E  # noqa: E402

RUNS = os.path.join(CR, 'selfplay', 'runs')
G2 = os.path.join(RUNS, 'g2_main', 'league_01000.pt')
SA = os.path.join(RUNS, 'az_main', 'league_00300.pt')


def stamp(p):
    st = os.stat(p)
    return f'{os.path.abspath(p)}:{st.st_size}:{int(st.st_mtime)}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', action='append', required=True)
    ap.add_argument('--games', type=int, default=1000)
    ap.add_argument('--procs', type=int, default=14)
    ap.add_argument('--cores', default='0-13')
    ap.add_argument('--seed', type=int, default=4_400_000)
    ap.add_argument('--out', default=os.path.join(HERE, 'eval', 'results.jsonl'))
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.cores.split('-'))
    cores = set(range(lo, hi + 1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cache = {}
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line); cache[r['key']] = r
            except Exception:
                pass

    def pair(name, ckA, dA, ckB, dB):
        key = json.dumps([stamp(ckA), dA, stamp(ckB), dB, a.games, a.seed])
        if key in cache:
            return cache[key]
        r = E.play_pair(ckA, dA, ckB, dB, a.games, a.procs, a.seed, cores)
        r.update(key=key, name=name)
        cache[key] = r
        with open(a.out, 'a') as f:
            f.write(json.dumps(r) + '\n')
        print(f'  {name:42s} {r["wr"]:.3f} [{r["ci"][0]:.3f}, {r["ci"][1]:.3f}] n={r["games"]} err={r["errors"]}',
              flush=True)
        return r

    base_r = pair('G2(azir) vs G2(rengar)', G2, 'azir', G2, 'rengar')
    base_d = pair('G2(azir) vs G2(diana)', G2, 'azir', G2, 'diana')
    base_s = pair('G2(azir) vs S_A(azir)', G2, 'azir', SA, 'azir')
    rows = []
    for ck in a.ckpt:
        tag = os.path.relpath(ck, HERE)
        m = pair(f'{tag}(azir) vs G2(azir)', ck, 'azir', G2, 'azir')
        r = pair(f'{tag}(azir) vs G2(rengar)', ck, 'azir', G2, 'rengar')
        d = pair(f'{tag}(azir) vs G2(diana)', ck, 'azir', G2, 'diana')
        s = pair(f'{tag}(azir) vs S_A(azir)', ck, 'azir', SA, 'azir')
        rows.append((tag, m, r, d, s))
    print('\ncandidate | vs G2 mirror | vs G2(rengar) [swap] | vs G2(diana) [swap] | vs S_A mirror [G2: %.3f]'
          % base_s['wr'])
    for tag, m, r, d, s in rows:
        print(f'{tag} | {m["wr"]:.3f} | {r["wr"]:.3f} [{r["wr"] - base_r["wr"]:+.3f}] | '
              f'{d["wr"]:.3f} [{d["wr"] - base_d["wr"]:+.3f}] | {s["wr"]:.3f}')


if __name__ == '__main__':
    main()
