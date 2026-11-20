import json, numpy as np
for l in open('/mnt/c/Users/sutanto4/Desktop/RiftboundAI/alpharune/research/card_repr/gate4/tmp/res_samp.jsonl'):
    r = json.loads(l)
    o = r['variants'][0]; wo = np.array(o['wins'])
    print('seed', r['seed'], 'turn', r['turn'], 'secs', r['secs'], 'orig', wo.mean().round(3), 'vB2', round(o['model']['B2']['v'],3), 'vid', round(o['model']['id']['v'],3))
    for v in r['variants'][1:]:
        if not v['ok']: print('   FAIL', v['cell'], v['err']); continue
        w = np.array(v['wins']); d = w - wo
        print('   %-38s dwin %+.3f (disc %.2f)  dvB2 %+.3f dvid %+.3f' % (v['cell'], d.mean(), (d!=0).mean(), v['model']['B2']['v']-o['model']['B2']['v'], v['model']['id']['v']-o['model']['id']['v']))
