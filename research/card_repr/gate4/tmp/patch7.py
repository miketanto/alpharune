p='/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate4/analyze.py'; s=open(p).read()
def rep(a,b,c=1):
    global s
    assert s.count(a)==c, (a, s.count(a)); s=s.replace(a,b)
rep("""            se_orig.append(""", """            LEVEL.append((float(wo[ok_o].mean()), {nm: (o['model'][nm]['v'] + 1) / 2 for nm in nets if nm in o['model']}))
            se_orig.append(""")
rep("""def rows_from(files, nets):""", """LEVEL = []   # (original win rate, {net: value head as a win probability}) per position


def rows_from(files, nets):""")
rep("""    hdr = '%-42s""", """    for nm in nets:
        w = np.array([x[0] for x in LEVEL if nm in x[1]]); v = np.array([x[1][nm] for x in LEVEL if nm in x[1]])
        if len(w) > 3:
            L.append('level check, %s: r(value head at the original root, playout win rate) = %.3f over %d positions; '
                     'mean |value - win| = %.3f' % (nm, pearson(v, w), len(w), float(np.abs(v - w).mean())))
    hdr = '%-42s""")
open(p,'w').write(s); print('ok')
