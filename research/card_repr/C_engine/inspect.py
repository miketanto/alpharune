import json,sys
r=json.load(open('out/raw_probe.json'))
lay={b['block']:b['fields'] for b in r['layout']}
want=sys.argv[1:] 
for name,s in r['subjects'].items():
    if want and not any(w in name for w in want): continue
    print('==',name,s['card_type'],'err',s['errors'])
    for b,v in s['blocks'].items():
        nz=[(lay[b][i],x) for i,x in enumerate(v) if x]
        if nz: print('  ',b, ' '.join(f'{k}={x:g}' for k,x in nz[:40]))
