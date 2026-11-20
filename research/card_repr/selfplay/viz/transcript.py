"""Readable transcript of an annotated replay (v2 or v4 HTML; same ReplayWriter format).

  python transcript.py replays_v4/game_seed7001.html [...]   -> writes game_seed7001.txt next to each
  python transcript.py                                        -> all of replays/ and replays_v4/

Per turn: the external decisions (network and ISMCTS; forced and autopaid ones are skipped, as are
ISMCTS pass-priority/pass-focus), plays, moves, hides, combats with the units on both sides, kills,
conquers/holds and the running score. Every network decision gets one line: its pick and the top
alternative, with probabilities, and its value. ISMCTS decisions show the pick's visit share.
"""
import glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def snaps(path):
    s = open(path, encoding='utf-8').read()
    hdr = re.search(r'game seed \d+ \(engine seed \d+\) \| P1=([^|]+) \| P2=([^|"<]+)', s)
    out = []
    for m in re.finditer(r'turn: (".*?"),\n\s*board: .*?\n\s*decision: (".*?"),\n\s*trace: (\[.*?\]),\n', s, re.S):
        try:
            out.append(dict(turn=json.loads(m.group(1)), decision=json.loads(m.group(2)),
                            trace=json.loads(m.group(3))))
        except Exception:
            pass
    labels = (hdr.group(1).strip(), hdr.group(2).strip()) if hdr else ('P1', 'P2')
    return labels, out


def clean(t):
    t = re.sub(r' \(obj=\d+\)', '', t)
    t = re.sub(r' #\d+', '', t)
    return t.strip()


def ev(line):
    """'[TRC] [D:012.03] KIND: rest' -> (KIND, rest) or None."""
    m = re.match(r'\[(?:TRC|DBG|INF|WRN)\] \[D:[\d.]+\] (.*)$', line)
    if not m:
        return None
    body = m.group(1)
    k = re.match(r'\s*([A-Z_\-]+): ?(.*)$', body)
    return (k.group(1), k.group(2)) if k else ('', body)


def transcript(path):
    labels, ss = snaps(path)
    seed = re.search(r'seed(\d+)', os.path.basename(path)).group(1)
    net_seat = 1 if labels[0].startswith('network') else 2
    who = {'P1': 'NET' if net_seat == 1 else 'ISMCTS', 'P2': 'NET' if net_seat == 2 else 'ISMCTS'}
    bf = {}
    score = {'P1': 0, 'P2': 0}
    L = []
    L.append(f'game seed {seed}   P1 = {labels[0]}   P2 = {labels[1]}')
    L.append('Legend: "P1[NET]" = a network decision, "P2[ISMCTS]" = a teacher decision. "net:" lines: the pick,')
    L.append('        its probability, the most likely other option, and the value estimate v (-1..+1; win ~ (v+1)/2).')
    L.append('        "~" lines are engine events. BF names: P1 brings the first battlefield, P2 the second.')
    pending_combat = None
    hands = {'P1': [], 'P2': []}
    in_open = None
    end = ''
    for sn in ss:
        d = sn['decision']
        tr = sn['trace']
        # --- the decision taken at this snapshot (it precedes this snapshot's trace lines)
        chosen = [x for x in d.splitlines() if '==>' in x]
        net = [x for x in tr if x.startswith('[NET]')]
        tea = [x for x in tr if x.startswith('[TEACHER')]
        if chosen and (net or tea):
            m = re.search(r'==> (P[12])(?: \(forced\))?: (.*)$', chosen[0])
            p, txt = m.group(1), clean(m.group(2))
            if net:
                v = float(re.search(r'value ([+-][\d.]+)', net[0]).group(1))
                opts = []
                for x in net[1:]:
                    mm = re.match(r'\[NET\]   (\* )?\s*([\d.]+)% : (.*)$', x)
                    if mm:
                        opts.append((bool(mm.group(1)), float(mm.group(2)), clean(mm.group(3))))
                pick = [o for o in opts if o[0]]
                alt = [o for o in opts if not o[0]]
                alt.sort(key=lambda o: -o[1])
                pp = pick[0][1] if pick else float('nan')
                sampled = ' [SAMPLED, not top choice]' if any('sampled' in x for x in net) else ''
                L.append(f'  {p}[NET] {txt}')
                L.append(f'        net: pick {pp:.1f}%' + (f' | alt: {alt[0][2]} {alt[0][1]:.1f}%' if alt else '')
                         + f' | v {v:+.2f}{sampled}')
            else:
                if not re.match(r'Pass (Priority|Focus)$', txt):
                    star = [x for x in tea if x.startswith('[TEACHER]   * ')]
                    vis = ''
                    if star:
                        mm = re.match(r'\[TEACHER\]   \* (\d+)/(\d+) visits', star[0])
                        if mm:
                            vis = f'   (ismcts {mm.group(1)}/{mm.group(2)} visits)'
                    L.append(f'  {p}[ISMCTS] {txt}{vis}')
        # --- engine events
        for line in tr:
            e = ev(line)
            if not e:
                continue
            k, r = e
            if k == 'SETUP':
                m = re.search(r'(P[12]) brings battlefield #\d+ \((.*?)\)', r)
                if m:
                    bf['BF#%d' % (0 if m.group(1) == 'P1' else 1)] = m.group(2)
                    L.append(f'~ {m.group(1)} brings {m.group(2)}')
                continue
            def B(s):
                return re.sub(r'BF#\d', lambda mm: bf.get(mm.group(0), mm.group(0)), s)
            if k == '' and 'Opening Hands' in r:
                in_open = True
                continue
            if k == '' and 'Mulligan Phase' in r:
                in_open = False
                for p in ('P1', 'P2'):
                    L.append(f'~ {p} opening hand: ' + ', '.join(hands[p]))
                continue
            if k == 'DRAW_HAND':
                cur = r.split()[0]
                continue
            if k == 'DREW' and in_open:
                hands[cur].append(re.sub(r' \(id=\d+\)', '', r))
                continue
            if k == 'MULLIGAN':
                L.append('~ ' + re.sub(r' \(id=\d+\)', '', r).replace(' — PRIVATE to P2', '').replace(' — PRIVATE to P1', ''))
                continue
            if k == '' and 'TURN' in r and '════' in r:
                m = re.search(r'TURN (\d+) \((P[12])\)', r)
                L.append('')
                L.append(f'=== TURN {m.group(1)} ({m.group(2)}, {who[m.group(2)]})   score P1 {score["P1"]} - P2 {score["P2"]} ===')
                continue
            if k == 'PLAY':
                m = re.match(r'(.*?) \(id=\d+, (\w+), cost=(\w*)\)', r)
                L.append(f'    ~ play {m.group(1)} ({m.group(2)}, {m.group(3)})' if m else f'    ~ play {r}')
            elif k == 'SPELL':
                L.append(f'    ~ spell {re.sub(r" [(]id=[0-9]+[)]", "", r)}')
            elif k == 'MOVE':
                L.append('    ~ move ' + B(re.sub(r' \(id=\d+\)', '', r)))
            elif k == 'HIDE' and ' by ' in r:
                L.append('    ~ hide ' + B(r))
            elif k == 'COMBAT':
                pending_combat = B(r)
            elif k == 'DAMAGE_STEP':
                m = re.match(r'(BF#\d) ATK=\[(.*?)\] DEF=\[(.*?)\]', r)
                c = pending_combat or ''
                am = re.search(r'attacker=(P\d) defender=(P\d)', c)
                if m:
                    a, dd = (am.group(1), am.group(2)) if am else ('?', '?')
                    L.append(f'    ~ COMBAT at {B(m.group(1))}: {a} attacks [{m.group(2)}]  vs  {dd} defends [{m.group(3)}]')
                else:
                    L.append('    ~ COMBAT ' + B(r))
                pending_combat = None
            elif k == 'KILL':
                L.append('    ~ KILL ' + re.sub(r'id=\d+, ', '', r))
            elif k == 'CONQUER':
                m = re.match(r'(P\d) conquers (BF#\d)', r)
                L.append(f'    ~ {m.group(1)} conquers {B(m.group(2))}' if m else '    ~ ' + B(r))
            elif k == 'SCORE':
                m = re.match(r'(P\d) \+(\d+) via (.*?) -> (\d+)', r)
                if m:
                    score[m.group(1)] = int(m.group(4))
                    L.append(f'    ~ SCORE {m.group(1)} +{m.group(2)} ({B(m.group(3))})   -> P1 {score["P1"]} - P2 {score["P2"]}')
            elif k in ('IRRESISTIBLE', 'SABOTAGE', 'STAR', 'THRILL', 'PYKE', 'ADDITIONAL_COST'):
                L.append(f'    ~ {k.lower()}: ' + B(re.sub(r' \(id=\d+\)', '', r)))
            elif k == 'EFFECT' and 'draws' not in r:
                L.append('    ~ effect: ' + B(re.sub(r' \(id=\d+\)', '', r)))
            elif k == 'GAME_OVER':
                end = r
    L.append('')
    if not end:   # ReplayWriter drops trace lines after the last snapshot: take the result from summary.tsv
        try:
            for row in open(os.path.join(os.path.dirname(path), 'summary.tsv'), encoding='utf-8'):
                c = row.rstrip('\n').split('\t')
                if c[0] == seed:
                    end = (f'winner P{c[1]} ({who["P" + c[1]]}) score {c[2]}-{c[3]}, {c[4]} turns, {c[5]} decisions'
                           f' -- {c[9]}  (from summary.tsv; the final scoring lines are not in the replay)')
        except OSError:
            pass
    L.append('GAME OVER: ' + end)
    return '\n'.join(L) + '\n'


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, 'replays', 'game_seed*.html')) +
                                   glob.glob(os.path.join(HERE, 'replays_v4', 'game_seed*.html')))
    for f in files:
        out = os.path.splitext(f)[0] + '.txt'
        open(out, 'w', encoding='utf-8').write(transcript(f))
        print('wrote', out)


if __name__ == '__main__':
    main()
