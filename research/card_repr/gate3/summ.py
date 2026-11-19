import json, sys
for tag in sys.argv[1:]:
    r = json.load(open(f'path_{tag}.json'))
    print('=====', tag)
    for pk, R in r.items():
        if pk.startswith('_'): continue
        dead = []; lo = []
        for grp in ('ref_token_fields', 'ask_token_fields', 'act_fields'):
            for f, m in R.get(grp, {}).items():
                if m['tv_mean'] < 1e-6: dead.append(f'{grp[:3]}.{f}')
                elif m['tv_mean'] < 1e-3: lo.append(f'{grp[:3]}.{f}={m["tv_mean"]:.0e}')
        top = sorted(((m['tv_mean'], f'{g[:3]}.{f}') for g in ('ref_token_fields','act_fields') for f, m in R.get(g, {}).items()), reverse=True)[:5]
        print(pk, 'DEAD', dead, '| glob_dead', R['glob_dead'])
        print('   weak', lo)
        print('   top', [(n, round(v, 4)) for v, n in top])
    print('--- schema v2')
    for pk, out in r['_schema_v2'].items():
        print(pk, {k: (round(v['dvalue'], 5), round(v['tv'], 5)) for k, v in out.items()})
