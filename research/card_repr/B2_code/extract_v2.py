"""B2 v2: mechanics graphs for cards outside the original 51 (2026-11-21).

  python extract_v2.py                 build every card in v2_cards.json that has a builder; write
                                        out/vectors_v2.json (ALL v1 vectors, unchanged, + the new keys),
                                        v2/cards/<key>.json (graphs with citations), v2/report.json
  python extract_v2.py --check KEY ... build only those keys and print each graph's verb check

Builders: v2/graphs_*.py, each defining BUILDERS = {(key, def_id): fn} with fn() -> {"abilities": [...],
optional "kw_values", "flags", "ambush_enemy", "flow"} exactly like graphs_code.py (same helpers: import
them from graphs_code). Stats, keywords and keyword expansions come from the CardDef (extract.build_class).
The encoder is B's, unchanged (B_graph/build.py vectorize), so v1 vectors are bit-identical.
"""
import glob, importlib.util, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import extract as X   # noqa: E402  (imports scan, vocab_src, graphs_code, B's encoder)
GC = X.GC


def load_builders():
    out = {}
    for f in sorted(glob.glob(os.path.join(HERE, 'v2', 'graphs_*.py'))):
        spec = importlib.util.spec_from_file_location(os.path.basename(f)[:-3], f)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        for k, fn in m.BUILDERS.items():
            assert k not in out, f'duplicate builder {k} ({f})'
            out[k] = (fn, os.path.basename(f))
    return out


def main():
    check = sys.argv[2:] if len(sys.argv) > 1 and sys.argv[1] == '--check' else None
    cards = json.load(open(os.path.join(HERE, 'v2_cards.json')))
    builders = load_builders()
    for (k, did), (fn, _) in builders.items():
        GC.BUILDERS[(k, did)] = fn
    v1 = json.load(open(os.path.join(HERE, 'out', 'vectors.json')))
    vecs = dict(v1['vectors'])
    os.makedirs(os.path.join(HERE, 'v2', 'cards'), exist_ok=True)
    report, missing, problems = {}, [], []
    for c in cards:
        k, did, rel = c['key'], c['def_id'], c['file']
        if check is not None and k not in check:
            continue
        if (k, did) not in builders:
            missing.append(k)
            continue
        try:
            g, d = X.build_class(k, did, rel)
        except Exception as e:
            problems.append(f'{k}: build failed: {e!r}')
            continue
        cons = X.verb_consistency(g, rel)
        s = json.dumps(X.strip(g)).lower()
        kw_words = ({k.lower() for k in X.V.KEYWORDS} | {'hidden', 'ambush', 'accelerate', 'assault', 'shield'} |
                    {'fury', 'calm', 'mind', 'body', 'order', 'chaos'})   # keywords / domains: mechanics, not names
        for w in re.findall(r"[a-z']{5,}", c['name'].lower()):
            if w in s and w not in kw_words:   # a keyword word (Hidden Blade's [Hidden]) is mechanics, not a name
                problems.append(f"{k}: name word '{w}' appears in the graph")
        vec = X.vec(g)
        if not any(vec):
            problems.append(f'{k}: all-zero vector')
        report[k] = dict(builder=builders[(k, did)][1], verbs=sorted(X.verbs_of(g)), consistency=cons,
                         n_abilities=len(g['abilities']), flags=g.get('_meta', {}).get('flags', []))
        if check is not None:
            print(k, json.dumps(report[k], indent=1))
            print(json.dumps(X.strip(g), indent=1)[:3000])
            continue
        vecs[k] = vec
        json.dump(g, open(os.path.join(HERE, 'v2', 'cards', k + '.json'), 'w', encoding='utf-8'), indent=1)
    if check is not None:
        print('problems:', problems)
        return
    json.dump({'dim': v1['dim'], 'vectors': vecs}, open(os.path.join(HERE, 'out', 'vectors_v2.json'), 'w'))
    json.dump(dict(built=sorted(report), missing=missing, problems=problems, cards=report),
              open(os.path.join(HERE, 'v2', 'report.json'), 'w'), indent=1)
    print(f'built {len(report)}, missing {len(missing)} {missing}, problems {len(problems)}')
    for p in problems:
        print('  ', p)


if __name__ == '__main__':
    main()
