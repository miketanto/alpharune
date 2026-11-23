import json, glob, math, collections
def wilson(w, n, z=1.96):
    if n == 0: return (0, 0)
    p = w / n; d = 1 + z*z/n; c = p + z*z/(2*n); m = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (round((c-m)/d, 3), round((c+m)/d, 3))
for arm in ('none', 'play', 'play_use'):
    G = [g for f in sorted(glob.glob(f'{arm}_[0-9].json')) for g in json.load(open(f))['games']]
    if not G: continue
    w = sum(g['winner'] == g['treated'] for g in G); n = len(G)
    print(f'== arm {arm}: treated seat wins {w}/{n} = {w/n:.3f} {wilson(w, n)}')
    seats = [g['per'][str(g['treated'])] for g in G] if arm != 'none' else [p for g in G for p in g['per'].values()]
    off = sum(s['seal_play_offered'] for s in seats); tk = sum(s['seal_play_taken'] for s in seats)
    ao = sum(s['seal_act_offered'] for s in seats); at = sum(s['seal_act_taken'] for s in seats)
    print(f'   seal play offered {off}, taken {tk} ({tk/max(1,off):.2%}); seal activation offered {ao}, taken {at}; forced {sum(s["forced"] for s in seats)}')
    byturn = collections.defaultdict(list)
    for s in seats:
        for t, p, c in s['seal_play_p']: byturn[min(t, 12)].append((p, c))
    print('   by turn: ' + ', '.join(f't{t}: n={len(v)} p={sum(x for x,_ in v)/len(v):.3f} took={sum(c for _,c in v)/len(v):.2f}' for t, v in sorted(byturn.items())))
    # games where the treated seat played a seal at least once vs not
    if arm != 'none':
        played = [g for g in G if g['per'][str(g['treated'])]['seal_play_offered'] > 0]
        wp = sum(g['winner'] == g['treated'] for g in played)
        print(f'   games where treated seat was offered a Seal: {len(played)}, won {wp} ({wp/max(1,len(played)):.3f} {wilson(wp, len(played))})')
