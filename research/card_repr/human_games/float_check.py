"""Is exhausting / recycling a rune ever an action of its own, and how often does a payment waste a ready rune's Energy?
G2 greedy Azir mirrors. A 'wasted' rune: recycled for Power while ready, with nothing floating (Energy that floating would have kept)."""
import os, re, sys
from multiprocessing import Pool
PW = '/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/.claude/worktrees/play-web-ui-e914cf/tools/play_web'
sys.path.insert(0, PW)
os.environ.update(CUDA_VISIBLE_DEVICES='', PLAYWEB_TORCH_THREADS='1')
CK = '/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/az/run_b2/az_009944.pt'
_A = None


def game(seed):
    global _A
    import playweb
    if _A is None:
        _A = playweb.NetAgent(CK)
    dk = os.path.join(PW, '..', '..', 'decks', 'azir.txt')
    g = playweb.Game(dk, dk, 1, seed)
    n = rune_opt = ready_rec = ex_rec = 0
    kinds = {}
    used_after = 0
    while True:
        st = g.status()
        if st['done'] or not st['need']:
            break
        s = st['seat']
        v = g.view(s, 1 << 40)
        runes = {r['id']: r['exhausted'] for r in v['runes'] if r['ctrl'] == 1}
        allr = {r['id']: (r['exhausted'], r['ctrl']) for r in v['runes']}
        n += 1
        rune_opt += any(set(o['objs']) & set(allr) for o in v['decision']['options'])
        a, _ = _A.choose(g.observe(), st['nlegal'])
        txt = next((o['text'] for o in v['decision']['options'] if o['index'] == a), '')
        g.step(a)
        w = g.view(s, 1 << 40)
        if w['turn']['number'] == v['turn']['number']:
            now = {r['id'] for r in w['runes']}
            was_energy = v['players']['me']['energy']
            e0 = sum(not x for x in runes.values()); e1 = sum((not r['exhausted']) for r in w['runes'] if r['ctrl'] == 1)
            for i, ex in runes.items():
                if i not in now:
                    if ex:
                        ex_rec += 1
                    else:
                        ready_rec += 1
                        k = re.sub(r' .*', '', txt)
                        kinds[k] = kinds.get(k, 0) + 1
    g.close()
    return n, rune_opt, ready_rec, ex_rec, kinds


if __name__ == '__main__':
    with Pool(12) as P:
        R = P.map(game, range(770000, 770060))
    k = {}
    for r in R:
        for x, y in r[4].items():
            k[x] = k.get(x, 0) + y
    print('games', len(R), 'decisions', sum(r[0] for r in R), 'decisions offering a rune as an object', sum(r[1] for r in R))
    print('runes recycled that were READY before the action', sum(r[2] for r in R), ' already exhausted', sum(r[3] for r in R), 'by verb', k)
