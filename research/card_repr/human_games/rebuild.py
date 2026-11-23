"""Rebuild a human game held by tools/play_web from its saved frame (live_state.json).

The server keeps no choice list. The engine is deterministic given (decks, seed, choices) and the network is greedy,
so the network's choices are recomputed and the human's are recovered by search: at each human decision every option is
tried and kept only if the human-visible log it produces is a prefix of (or extended by) the recorded log.
Writes game.json {deck, seed, choices: [[seat, index]], ...}.
"""
import json, os, sys, time
PW = '/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/.claude/worktrees/play-web-ui-e914cf/tools/play_web'
sys.path.insert(0, PW)
import playweb
HERE = os.path.dirname(os.path.abspath(__file__))
DECKS = os.path.join(PW, '..', '..', 'decks')
f = json.load(open(os.path.join(HERE, 'live_state.json')))['frames'][-1]
M = f['meta']
human = M['seat']
rec = [l['msg'] for l in f['log']]
dh, da = (os.path.join(DECKS, M[k] + '.txt') for k in ('deck', 'agent_deck'))
p1, p2 = (dh, da) if human == 1 else (da, dh)
agent = playweb.NetAgent(os.path.join(playweb.SELFPLAY, 'runs', M['checkpoint']))


def replay(ch):
    g = playweb.Game(p1, p2, human, M['seed'])
    for _, c in ch:
        g.step(c)
    return g


def ok(g):
    """Matched log length if this game's human log is a prefix of the record, else -1."""
    log = [l['msg'] for l in g.view(human, 0)['log']]
    if len(log) > len(rec) or log != rec[:len(log)]:
        return -1
    return len(log)


def run_net(g, ch):
    while True:
        st = g.status()
        if st['done'] or not st['need'] or st['seat'] == human:
            return st
        a, v = agent.choose(g.observe(), st['nlegal'])
        g.step(a)
        ch.append([st['seat'], a])


best = [0]
nodes = [0]
t0 = time.time()


def solve(ch):
    """Depth-first with backtracking (iterative deepening is not needed: the log is discriminating)."""
    stack = [list(ch)]
    while stack:
        ch = stack.pop()
        g = replay(ch)
        st = run_net(g, ch)
        n = ok(g)
        nodes[0] += 1
        if n < 0:
            g.close(); continue
        if n > best[0]:
            best[0] = n
            print(f'{time.time()-t0:6.1f}s nodes {nodes[0]} matched {n}/{len(rec)} choices {len(ch)} open {len(stack)}', flush=True)
        if st['done']:
            if n == len(rec):
                g.close(); return ch
            g.close(); continue
        opts = list(range(st['nlegal']))
        g.close()
        cand = []
        for k in opts:
            g = replay(ch)
            g.step(k)
            m = ok(g)
            g.close()
            if m >= 0:
                cand.append((m, k))
        cand.sort()                      # the longest match is tried first (top of the stack)
        for m, k in cand:
            stack.append(ch + [[human, k]])
        if nodes[0] > 20000:
            print('gave up'); return None
    return None


ch = solve([])
if ch is None:
    print('NOT FOUND, best', best[0], 'of', len(rec))
    sys.exit(1)
json.dump(dict(p1=os.path.basename(p1), p2=os.path.basename(p2), human=human, seed=M['seed'], checkpoint=M['checkpoint'],
               choices=ch), open(os.path.join(HERE, f'game_{M["seed"]}.json'), 'w'))
g = replay(ch)
print('REBUILT', len(ch), 'choices', g.status(), 'log equal', [l['msg'] for l in g.view(human, 0)['log']] == rec)
