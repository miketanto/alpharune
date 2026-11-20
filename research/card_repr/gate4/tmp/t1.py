import sys, time, numpy as np, json
sys.path.insert(0, '/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate4')
import po as P
from rbenv import RBEnv
env = RBEnv(4, 'rengar', 'rengar', 12345)
rng = np.random.default_rng(0)
got = None
for it in range(4000):
    o = env.observe()
    for i in range(4):
        if o['need'][i] and o['glob'][i][1] >= 0 and P.env_resumable(env, i) and it > 300:
            got = P.env_hist(env, i); break
    if got: break
    ch = [rng.integers(o['nlegal'][i]) if o['need'][i] else 0 for i in range(4)]
    env.step(np.array(ch))
seed, hist = got
print('seed', seed, 'hist', len(hist))
po = P.Po(seed, hist, n_envs=64)
d = po.describe()
print(json.dumps({k: d[k] for k in ('turn_player','decider','turn','players','bfs')}))
units = [x for x in d['objs'] if x['unit'] and x['zone'] in ('base','bf')]
for u in units: print(u)
print(len(d['legal']), d['legal'][:5])
v0, e = po.add_variant('none'); print('v0', v0, e)
u = units[0]
v1, e = po.add_variant('kw_give %d A 2' % u['id']); print('v1', v1, e)
i1 = po.info(v1); print([x for x in i1['objs'] if x['id']==u['id']])
v2, e = po.add_variant('might %d 2' % u['id']); print('v2', v2, e, [x for x in po.info(v2)['objs'] if x['id']==u['id']])
v3, e = po.add_variant('to_base %d' % u['id']); print('v3', v3, e)
v4, e = po.add_variant('played 1'); print('v4', v4, e, po.info(v4)['players'])
ro = po.root_obs(v1); print(ro['nlegal'], ro['seat'], ro['tok_obj'][:10])
net = P.load_net('/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/selfplay/runs/gate4_B2/latest.pt', 'B2')
t=time.time()
jobs = [(v, s) for v in (v0, v1) for s in range(100, 164)]
res = P.play(po, net, jobs, 'cpu')
print('played', len(res), time.time()-t, 's')
for v in (v0, v1): w=[res[(v,s)] for s in range(100,164)]; print(v, np.mean([x==d['decider'] for x in w]), w.count(-1))
# determinism check: same seeds again
res2 = P.play(po, net, jobs[:16], 'cpu'); print('det', all(res2[k]==res[k] for k in res2))
