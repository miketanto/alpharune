"""Derive the structured block's one-hot vocabularies from the B2 graphs -> B_graph/struct_vocab.json.

Sources: every B2 card graph (B2_code/cards/*.json, incl. each engine class in "_classes"), the token
units (graphs_code.TOKENS), the synthetic variants (extract.syn_edit / score_edit), plus the
schema-documented values in struct_block.SCHEMA_EXTRAS. Run once; the vocabulary is then frozen
(re-running it changes the layout and therefore every vector).
"""
import glob, importlib.util, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
CR = os.path.normpath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(CR, 'B2_code'))
import struct_block as S  # noqa: E402
_spec = importlib.util.spec_from_file_location('b2x', os.path.join(CR, 'B2_code', 'extract.py'))
X = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(X)

graphs, names = [], []
for f in sorted(glob.glob(os.path.join(CR, 'B2_code', 'cards', '*.json'))):
    g = json.load(open(f, encoding='utf-8'))
    for cid, cg in (g.get('_classes') or {}).items():
        graphs.append(X.strip(cg)); names.append(f'{os.path.basename(f)}@{cid}')
    graphs.append(X.strip({k: v for k, v in g.items() if k != '_classes'})); names.append(os.path.basename(f))
for k, tg in X.GC.TOKENS.items():
    graphs.append(X.strip(tg)); names.append(k)
cj = json.load(open(os.path.join(CR, 'cards.json'), encoding='utf-8'))
for s in cj['t2_minimal_pairs_synthetic']:
    bg = json.load(open(os.path.join(CR, 'B2_code', 'cards', s['base'] + '.json'), encoding='utf-8'))
    graphs.append(X.strip(X.syn_edit(bg, s['id']))); names.append('synthetic:' + s['id'])
for sid, base in X.SCORE_VARIANTS.items():
    bg = json.load(open(os.path.join(CR, 'B2_code', 'cards', base + '.json'), encoding='utf-8'))
    graphs.append(X.strip(X.score_edit(bg, sid))); names.append('synthetic:' + sid)
voc = S.derive_vocab(graphs)
nab = max(len(g['abilities']) for g in graphs)
json.dump({'source': 'B2 graphs: %d (cards, engine classes, tokens, synthetic variants) + SCHEMA_EXTRAS' % len(graphs),
           'max_abilities_seen': nab, 'fields': voc}, open(os.path.join(HERE, 'struct_vocab.json'), 'w'), indent=1)
idx, spans, n = S.layout(voc)
print('graphs', len(graphs), 'max abilities', nab, 'block dims', n)
for k, v in voc.items():
    print(' ', k, len(v), v[:30])
