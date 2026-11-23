"""Does az_009944 receive and use different inputs for the Azir gear? Vocabulary sizes, card embeddings, sensitivity of the pick."""
import json, os
import tricks as T
import numpy as np, torch, playweb

A = T._A = playweb.NetAgent(os.path.join(playweb.SELFPLAY, 'runs', T.CK))
net = A.net
v = json.load(open(playweb.VOCAB_PATH))
v = v if isinstance(v, list) else v.get('vocab', v)
print('ckpt', T.CK.split('/')[-1], 'lib V', playweb.dims()['V'], 'vocab file', len(v), 'net special_idx', tuple(net.special_idx.shape), 'card_v2', getattr(net, 'card_v2', None),
      'R', tuple(net.R.shape))
names = ['b_f_sword', 'soul_sword', 'doran_s_shield', 'eye_of_the_herald', 'hand_hammer', 'brutalizer', 'discipline', 'back_off', 'other']
idx = [v.index(n) if n in v else v.index('other') for n in names]
with torch.no_grad():
    k = torch.tensor(idx)
    E = net.card_emb(k)
    R = net.R[k]
    print('raw vectors zero?', {n: bool((R[i].abs().sum() == 0)) for i, n in enumerate(names)})
    if getattr(net, 'card_v2', False):
        print('part norms: graph', net.W_card(net.R)[k].norm(dim=1).numpy().round(2), 'text', net.W_txt(net.R_txt)[k].norm(dim=1).numpy().round(2),
              'residual', net.E_res(k).norm(dim=1).numpy().round(2))
    D = torch.cdist(E, E).numpy()
    print('final embedding norms', E.norm(dim=1).numpy().round(2))
    print('distance between final card embeddings:')
    print('            ' + ' '.join(f'{n[:8]:>8s}' for n in names))
    for i, n in enumerate(names):
        print(f'{n[:12]:12s}' + ' '.join(f'{x:8.2f}' for x in D[i]))

# sensitivity: at real equip choices (Weaponmaster), swap the identity of one option's card and see how its probability moves
dk = os.path.join(T.DECKS, 'azir.txt')
gear_keys = {n: i for n, i in zip(names[:6], idx[:6])}
mv, n = [], 0
for seed in range(995000, 995012):
    g = playweb.Game(dk, dk, 1, seed)
    while True:
        st = g.status()
        if st['done'] or not st['need']:
            break
        o = g.observe()
        p, _ = T.ev(o)
        d = g.view(st['seat'], 1 << 40)['decision']
        if d['kind'] == 'target' and 'Equipment' in (d.get('question') or '') and st['nlegal'] >= 2:
            n += 1
            best = []
            rows = [t for t in range(o['tok'].shape[1]) if o['tok_mask'][0, t] and o['tok'][0, t, 0] == 1 and int(o['tok'][0, t, 1]) in gear_keys.values()]
            for t in rows[:3]:
                q = []
                for kk in gear_keys.values():
                    e = dict(o); e['tok'] = o['tok'].copy(); e['tok'][0, t, 1] = kk
                    q.append(T.ev(e)[0])
                q = np.stack(q)
                best.append(np.abs(q - q.mean(0)).sum(1).max() / 2)
            mv.append(max(best) if best else 0)
        g.step(int(p.argmax()))
    g.close()
print(f'equip prompts {n}: changing one gear\'s identity moves the choice distribution by at most (total variation) mean {np.mean(mv):.3f}, max {np.max(mv):.3f}')
