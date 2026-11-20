p='/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate4/run.py'; s=open(p).read()
a="""            r['legal'] = info['legal'][:ro['nlegal']]
"""
b="""            r['legal'] = info['legal'][:ro['nlegal']]
            # the edited object as the engine left it after resume + Cleanup (legality / edit-stuck check)
            parts = r['edit'].split()
            if len(parts) > 1 and parts[0] not in ('played', 'ss'):
                r['post'] = next((o for o in info['objs'] if o['id'] == int(parts[1])), None)
            elif parts[0] == 'played':
                r['post'] = info['players'][dec - 1]
            elif parts[0] == 'ss':
                r['post'] = info['bfs'][int(parts[1])]
"""
assert s.count(a)==1; s=s.replace(a,b); open(p,'w').write(s); print('ok')
