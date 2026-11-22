"""Cards of the given decks that have no B2 graph yet -> v2_cards.json (key, class file, def_id).

  python v2_list.py azir rengar diana
"""
import glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CR = os.path.dirname(HERE)
ROOT = os.path.normpath(os.path.join(CR, '..', '..'))


def slug(s):
    o = ''
    for c in s:
        if c.isalnum():
            o += c.lower()
        elif o and o[-1] != '_':
            o += '_'
    return o.strip('_')


def main():
    decks = sys.argv[1:] or ['azir', 'rengar', 'diana']
    have = set(json.load(open(os.path.join(HERE, 'out', 'vectors.json')))['vectors'])
    v2p = os.path.join(HERE, 'out', 'vectors_v2.json')
    if os.path.exists(v2p):
        have |= set(json.load(open(v2p))['vectors'])
    info = json.load(open(os.path.join(CR, 'cards_v2', 'keys_v2.json'), encoding='utf-8'))
    by_id = {}
    for f in glob.glob(os.path.join(ROOT, 'src', 'cards', '*', '*.cpp')):
        t = open(f, encoding='utf-8').read()
        for m in re.finditer(r'd\.id = (\d+);\s*d\.def_id = R"RB\(([^)]*)\)RB"', t):
            by_id[int(m.group(1))] = (m.group(2), os.path.relpath(f, ROOT).replace(os.sep, '/'))
    out, seen = [], set()
    for d in decks:
        for line in open(os.path.join(ROOT, 'decks', d + '.txt'), encoding='utf-8'):
            m = re.match(r'(\d+)\s+(.*)', line.strip())
            if not m:
                continue
            nm = m.group(2)
            mid = re.search(r'#(\d+)$', nm)
            k = slug(re.sub(r'\s*#\d+$', '', nm))
            if k not in info and ',' in nm:   # deck lists name legends "Champion, Title"; the engine name is "Title"
                k = slug(re.sub(r'\s*#\d+$', '', nm.split(',', 1)[1]))
            if k in have or k in seen:
                continue
            if k not in info:
                print('no key for', d, nm)
                continue
            seen.add(k)
            ids = info[k]['ids']
            i = int(mid.group(1)) if mid and int(mid.group(1)) in ids else ids[0]
            did, rel = by_id.get(i, (None, None))
            out.append(dict(key=k, deck=d, card_id=i, def_id=did, file=rel, type=info[k]['type'], name=info[k]['name'],
                            all_ids=ids))
    json.dump(out, open(os.path.join(HERE, 'v2_cards.json'), 'w'), indent=1)
    print(len(out), 'cards')
    for o in out:
        print(o['deck'], o['key'], o['card_id'], o['def_id'], o['file'], o['type'])


if __name__ == '__main__':
    main()
