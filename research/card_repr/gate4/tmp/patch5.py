p='/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate4/analyze.py'; s=open(p).read()
def rep(a,b):
    global s
    assert s.count(a)==1, a; s=s.replace(a,b)
rep("""def verdict(c, nets, is_null):
    res = {}
    for nm in nets:
        s = c[nm]
        if is_null:
            res[nm] = 'NULL-OK' if s['model_abs'] <= 0.02 else 'NULL-FAIL'
            continue""", """def verdict(c, nets, is_null, ref=None):
    \"\"\"Mechanic cell: PASS = bootstrap r CI above 0 and sign agreement >= 0.7 on clear positions (n >= 8, >= 4
    clear); WEAK = r > 0.2 and sign agreement > 0.5. Null cell: NULL-OK = mean |dv| at most half the pooled
    mechanic cells' mean |dv| for that net and |mean dv| <= 0.01.\"\"\"
    res = {}
    for nm in nets:
        s = c[nm]
        if is_null:
            ok = s['model_abs'] <= 0.5 * ref[nm] and abs(s['model_mean']) <= 0.01
            res[nm] = 'NULL-OK' if ok else 'NULL-FAIL'
            continue""")
rep("""    order = sorted(by, key=lambda c: (c.startswith('null'), c))
    pooled_mech = []
""", """    order = sorted(by, key=lambda c: (c.startswith('null'), c))
    pooled_mech = []
    allm = [r for r in rows if not r['cell'].startswith('null')]
    ref = {nm: float(np.mean([abs(r['model_' + nm]) for r in allm])) if allm else 1.0 for nm in nets}
""")
rep("st['verdict'] = verdict(st, nets, c.startswith('null'))", "st['verdict'] = verdict(st, nets, c.startswith('null'), ref)")
open(p,'w').write(s); print('ok')
