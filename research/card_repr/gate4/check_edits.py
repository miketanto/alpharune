"""Checks that every accepted edit is visible in the post-Cleanup root state (the 'post' record of run.py) and
counts failed (refused) edits. usage: python3 check_edits.py results_s*.jsonl"""
import json, sys
from collections import Counter
ok, bad, fail = Counter(), Counter(), Counter()
errs = Counter()
for fn in sys.argv[1:]:
    for line in open(fn):
        r = json.loads(line)
        if r.get('skipped'):
            continue
        for v in r['variants'][1:]:
            c = v['cell']
            if not v['ok']:
                fail[c] += 1; errs[(v['err'] or '')[:70]] += 1; continue
            p, e = v.get('post'), v['edit'].split()
            if e[0] == 'kw_give':
                good = p is not None and p['assault' if e[2] == 'A' else 'shield'] >= int(e[3])
            elif e[0] == 'kw_remove':
                good = p is not None and p['assault' if e[2] == 'A' else 'shield'] == 0
            elif e[0] == 'might':
                good = p is not None   # base Might checked against the original below is not stored; presence = still on board
            elif e[0] == 'to_base':
                good = p is not None and p['zone'] == 'base'
            elif e[0] == 'unhide':
                good = p is not None and p['zone'] == 'hand'
            elif e[0] == 'played':
                good = p is not None and p['cards_played'] == int(e[1])
            elif e[0] == 'ss':
                good = p is not None
            else:
                good = False
            (ok if good else bad)[c] += 1
print('edit visible after resume+Cleanup / not visible / refused, per cell:')
for c in sorted(set(ok) | set(bad) | set(fail)):
    print('  %-42s %4d %4d %4d' % (c, ok[c], bad[c], fail[c]))
print('refusal reasons:', dict(errs))
