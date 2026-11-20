p='/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate4/run.py'; s=open(p).read()
def rep(a,b):
    global s
    assert s.count(a)==1, a; s=s.replace(a,b)
rep("""    sel, cnt = select(pos, a.cap, np.random.default_rng(a.seed))
    k, m = (int(x) for x in a.shard.split('/'))
    sel = sel[k::m]
    print('selected', len(sel), 'positions;', sum(len(p['edits']) for p in sel), 'edits', flush=True)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line); done.add((r['seed'], len(r['hist'])))
""", """    k, m = (int(x) for x in a.shard.split('/'))
    order = np.random.default_rng(a.seed).permutation(len(pos))
    sel = [pos[i] for i in order][k::m]
    cap = max(1, -(-a.cap // m))   # per shard
    print('shard', k, 'of', m, ':', len(sel), 'candidate positions; cap per cell per shard', cap, flush=True)
    done = set()
    cnt = Counter()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line); done.add((r['seed'], len(r['hist'])))
            if not r.get('skipped'):
                for v in r['variants'][1:]:
                    if v['ok']:
                        cnt[v['cell']] += 1
""")
rep("""        t1 = time.time()
        po = P.Po(p['seed'], p['hist'], n_envs=a.n_envs)
        d0 = po.describe()
        dec = d0['decider']
        variants = [('orig', 'none', 1, '')] + [(e['cell'], e['edit'], e['orient'], e['note']) for e in p['edits']]
""", """        keep = [e for e in p['edits'] if cnt[e['cell']] < cap]
        if not keep:
            continue
        t1 = time.time()
        po = P.Po(p['seed'], p['hist'], n_envs=a.n_envs)
        d0 = po.describe()
        dec = d0['decider']
        # screen: positions already decided under the playout policy carry no signal
        v0, _ = po.add_variant('none')
        sres = P.play(po, pnet, [(v0, s) for s in seeds[:a.screen]], dev, greedy=not a.sample)
        sw = np.mean([1.0 if sres[(v0, s)] == dec else 0.0 for s in seeds[:a.screen]])
        if not (a.screen_lo <= sw <= 1 - a.screen_lo):
            with open(a.out, 'a') as f:
                f.write(json.dumps(dict(seed=p['seed'], hist=p['hist'], turn=p['turn'], skipped=True, screen_win=sw)) + '\n')
            po.close()
            print('[%5.0fs] skip seed %d turn %d: screen win %.2f' % (time.time() - t0, p['seed'], p['turn'], sw), flush=True)
            continue
        po.close()
        po = P.Po(p['seed'], p['hist'], n_envs=a.n_envs)
        variants = [('orig', 'none', 1, '')] + [(e['cell'], e['edit'], e['orient'], e['note']) for e in keep]
""")
rep("""        po.close()
        out = dict(""", """        po.close()
        for r in recs[1:]:
            if r['ok']:
                cnt[r['cell']] += 1
        out = dict(screen_win=sw, """)
rep("""    ap.add_argument('--max-minutes', type=float, default=0)""", """    ap.add_argument('--max-minutes', type=float, default=0)
    ap.add_argument('--screen', type=int, default=64)
    ap.add_argument('--screen-lo', type=float, default=0.08)""")
open(p,'w').write(s); print('ok')
