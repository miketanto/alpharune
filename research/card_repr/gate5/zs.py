"""Gate 5 test 2: zero-shot choices with unseen cards. Yi mirror games driven by the cf nets (SAMPLED; even
envs B2, odd envs id). At every external decision with >= 2 options, the decision is classified:
  unseen : the asking card (psrc token) or the source (act[1] src / act[2] ability source) of some option is a
           Yi-only card never seen in training;
  seen   : every named card is a Rengar-deck (trained) card.
A sample of each class (p_unseen, p_seen) is asked to ISMCTS-<sims> twice (two search seeds: the search's own
self-agreement is the ceiling) and the observation is stored, so any net can be scored later (score mode).

usage: zs.py collect --out zs_obs.npz [--decisions 300] ; zs.py score --inp zs_obs.npz --nets B2=..,id=..
"""
import argparse, ctypes, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'gate4'))
import po as P  # noqa: E402
from swap import CK, UNSEEN_UNITS, UNSEEN_OTHER, SEEN_UNITS, SEEN_OTHER, DECKS  # noqa: E402

UNSEEN = set(UNSEEN_UNITS + UNSEEN_OTHER + ['master_yi_wuju_bladesman', 'calm_rune'])
SEEN = set(SEEN_UNITS + SEEN_OTHER + ['rengar_pridestalker', 'body_rune', 'fury_rune', 'emperor_s_dais', 'seat_of_power',
                                      'star_spring', 'token:mech', 'token:sand_soldier'])
KEYS = ('glob', 'tok', 'tok_mask', 'act', 'act_mask', 'pk', 'psrc')


def classify(o, i, vocab):
    tok = o['tok'][i]; tm = o['tok_mask'][i]
    names = set()
    ps = int(o['psrc'][i])
    if 0 <= ps < len(tm) and tm[ps]:
        names.add(vocab[tok[ps, 1]])
    ask = set(names)
    for a in np.nonzero(o['act_mask'][i])[0]:
        for c in (1, 2):
            t = int(o['act'][i, a, c])
            if 0 <= t < len(tm) and tm[t]:
                names.add(vocab[tok[t, 1]])
    if names & UNSEEN:
        return 'unseen', sorted(names & UNSEEN), sorted(ask & UNSEEN)
    if names and names <= SEEN | {'hidden', 'keyword_ability', 'none', '<pad>'}:
        return 'seen', [], []
    return None, [], []


def collect(a):
    import torch
    from rbenv import RBEnv
    from model import load_vocab
    from selfplay import act_rows
    torch.set_num_threads(2)
    dev = 'cpu'
    L = P.lib()
    L.rbenv_advise.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int64]
    L.rbenv_advise.restype = ctypes.c_int
    vocab = load_vocab()[0]
    nets = {'B2': P.load_net(CK['B2'], 'B2', dev), 'id': P.load_net(CK['id'], 'id', dev)}
    rng = np.random.default_rng(a.seed)
    dk = DECKS['yi']
    env = RBEnv(a.n_envs, dk, dk, a.seed)
    drive = np.array(['B2' if i % 2 == 0 else 'id' for i in range(a.n_envs)])
    store = {k: [] for k in KEYS}
    meta = []
    cnt = {'unseen': 0, 'seen': 0}
    t0 = time.time()
    while min(cnt.values()) < a.decisions and time.time() - t0 < 60 * a.max_minutes:
        o = env.observe()
        rows = np.nonzero(o['need'])[0]
        for i in rows:
            if o['nlegal'][i] < 2:
                continue
            cls, un, ask = classify(o, i, vocab)
            if cls is None or cnt[cls] >= a.decisions or rng.random() > (a.p_unseen if cls == 'unseen' else a.p_seen):
                continue
            picks = [int(L.rbenv_advise(env._h, int(i), a.sims, int(rng.integers(1 << 40)))) for _ in range(2)]
            for k in KEYS:
                store[k].append(np.asarray(o[k][i]).copy())
            meta.append(dict(cls=cls, unseen=un, ask=ask, pk=int(o['pk'][i]), nlegal=int(o['nlegal'][i]), ismcts=picks,
                             drive=str(drive[i])))
            cnt[cls] += 1
            if sum(cnt.values()) % 25 == 0:
                print('[%5.0fs] %s' % (time.time() - t0, cnt), flush=True)
        ch = np.zeros(a.n_envs, np.int64)
        for nm in ('B2', 'id'):
            r = rows[drive[rows] == nm]
            if len(r):
                ac, _, _ = act_rows(nets[nm], o, r, dev, False)
                ch[r] = ac
        env.step(ch)
        env.results()
    np.savez_compressed(a.out, **{k: np.stack(v) for k, v in store.items()}, meta=json.dumps(meta))
    print('saved', cnt, '%.0fs' % (time.time() - t0))


def score(a):
    import torch
    torch.set_num_threads(2)
    from swap import load_nets
    z = np.load(a.inp)
    meta = json.loads(str(z['meta']))
    nets = load_nets(a.nets, 'cpu')
    obs = {k: z[k] for k in KEYS}
    obs['tok_mask'] = obs['tok_mask'].astype(bool); obs['act_mask'] = obs['act_mask'].astype(bool)
    out = {}
    for name, net in nets.items():
        picks = []
        for s in range(0, len(meta), 256):
            rows = np.arange(s, min(s + 256, len(meta)))
            pr, _ = P.net_eval(net, obs, rows, 'cpu')
            picks += pr.argmax(1).tolist()
        out[name] = picks
    json.dump(dict(meta=meta, picks=out), open(a.out, 'w'))
    print('scored', list(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['collect', 'score'])
    ap.add_argument('--out'); ap.add_argument('--inp'); ap.add_argument('--nets', default='')
    ap.add_argument('--decisions', type=int, default=300); ap.add_argument('--sims', type=int, default=400)
    ap.add_argument('--n-envs', type=int, default=16); ap.add_argument('--seed', type=int, default=9300001)
    ap.add_argument('--p-unseen', type=float, default=0.25); ap.add_argument('--p-seen', type=float, default=0.06)
    ap.add_argument('--max-minutes', type=float, default=40)
    a = ap.parse_args()
    {'collect': collect, 'score': score}[a.mode](a)


if __name__ == '__main__':
    main()
