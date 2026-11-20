"""Reads the annotated replays (replays/game_seed*.html) and prints, per game: the network's and
the teacher's value per turn, the network's largest value drops, and its low-probability
(sampled) choices, to help find notable plays.  python3 analyze.py [replays/game_seedN.html ...]"""
import glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def snaps(path):
    s = open(path, encoding='utf-8').read()
    out = []
    for m in re.finditer(r'turn: (".*?"),\n\s*board: .*?\n\s*decision: (".*?"),\n\s*trace: (\[.*?\]),\n', s, re.S):
        try:
            out.append(dict(turn=json.loads(m.group(1)), decision=json.loads(m.group(2)),
                            trace=json.loads(m.group(3))))
        except Exception:
            pass
    return out


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, '..', 'replays', 'game_seed*.html')))
    for f in files:
        ss = snaps(f)
        print('=' * 100)
        print(os.path.basename(f), len(ss), 'snapshots')
        prev = None
        per_turn = {}
        for k, sn in enumerate(ss):
            tm = re.match(r'Turn (\d+)', sn['turn'])
            t = int(tm.group(1)) if tm else -1
            score = re.search(r'Score: (.*)$', sn['turn'])
            for i, l in enumerate(sn['trace']):
                if l.startswith('[NET] value'):
                    v = float(l.split()[2])
                    opts = [x for x in sn['trace'][i + 1:i + 6] if x.startswith('[NET]   ')]
                    ch = [x for x in opts if x.startswith('[NET]   * ')]
                    pv = per_turn.setdefault(t, {})
                    pv.setdefault('net', v)
                    pv['score'] = score.group(1) if score else ''
                    if prev is not None and v - prev < -0.3:
                        print(f'  T{t} NET value drop {prev:+.2f} -> {v:+.2f}  [{sn["turn"]}]')
                        print('     prev decision chose:', last_choice)
                    if ch and float(ch[0].split()[2].rstrip('%')) < 15:
                        print(f'  T{t} low-prob pick: {ch[0][8:]}  (v {v:+.2f})')
                    prev = v
                    last_choice = ch[0][8:] if ch else '?'
                if l.startswith('[TEACHER ') and 'root value' in l:
                    v = float(l.split('root value')[1].split()[0])
                    per_turn.setdefault(t, {}).setdefault('teacher', v)
        print('  turn: net value / teacher root value (first decision of the turn) / score')
        for t in sorted(per_turn):
            p = per_turn[t]
            print(f'   T{t:2d}: net {p.get("net", float("nan")):+.2f}  teacher {p.get("teacher", float("nan")):+.2f}  {p.get("score", "")}')
        if ss:
            print('  end:', [x for x in ss[-1]['trace'] if 'GAME_OVER' in x])


if __name__ == '__main__':
    main()
