"""Gate 5 test 2 analysis: agreement of each net's greedy choice with ISMCTS-400 (two search seeds).
agree = mean over decisions of the fraction of the two searches the net matches; 'conf' = only decisions where
the two searches agree with each other (agreement with that common answer); ceiling = search self-agreement;
chance = mean 1/nlegal. CIs: bootstrap over decisions. B2 - id: paired bootstrap.
usage: python3 zs_analyze.py zs_scored.json [more scored files with other nets]"""
import json, sys
import numpy as np


def ci(x, B=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = [x[rng.integers(len(x), size=len(x))].mean() for _ in range(B)]
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    d = json.load(open(sys.argv[1]))
    for f in sys.argv[2:]:
        e = json.load(open(f)); assert e['meta'] == d['meta']; d['picks'].update(e['picks'])
    meta = d['meta']; nets = list(d['picks'])
    groups = [('unseen (any option / asker)', lambda m: m['cls'] == 'unseen'),
              ('  unseen card is the ASKER', lambda m: m['cls'] == 'unseen' and m['ask']),
              ('  unseen, main phase (pk 3)', lambda m: m['cls'] == 'unseen' and m['pk'] == 3),
              ('seen only (control)', lambda m: m['cls'] == 'seen')]
    out = {}
    print('%-30s %4s %5s %5s | ' % ('decisions', 'n', 'chance', 'self') + ' | '.join('%-24s' % (n + ' agree [CI] (conf)') for n in nets))
    for name, f in groups:
        idx = [i for i, m in enumerate(meta) if f(m)]
        if not idx:
            continue
        ch = np.mean([1 / meta[i]['nlegal'] for i in idx])
        selfa = np.mean([meta[i]['ismcts'][0] == meta[i]['ismcts'][1] for i in idx])
        line = '%-30s %4d %5.2f %5.2f | ' % (name, len(idx), ch, selfa)
        out[name] = dict(n=len(idx), chance=ch, self=selfa)
        conf = [i for i in idx if meta[i]['ismcts'][0] == meta[i]['ismcts'][1]]
        for n in nets:
            ag = [np.mean([d['picks'][n][i] == s for s in meta[i]['ismcts']]) for i in idx]
            cf = [d['picks'][n][i] == meta[i]['ismcts'][0] for i in conf]
            m, lo, hi = ci(ag)
            line += '%.3f [%.3f,%.3f] (%.2f) | ' % (m, lo, hi, np.mean(cf) if cf else np.nan)
            out[name][n] = dict(agree=m, ci=[lo, hi], conf=float(np.mean(cf)) if cf else None, n_conf=len(conf))
        print(line)
        if 'B2' in nets and 'id' in nets:
            dd = [np.mean([d['picks']['B2'][i] == s for s in meta[i]['ismcts']]) - np.mean([d['picks']['id'][i] == s for s in meta[i]['ismcts']]) for i in idx]
            m, lo, hi = ci(dd, seed=1)
            same = np.mean([d['picks']['B2'][i] == d['picks']['id'][i] for i in idx])
            print('%-30s B2 - id = %+.3f [%+.3f, %+.3f]; B2 and id pick the same option %.2f' % ('', m, lo, hi, same))
            out[name]['B2_minus_id'] = [m, lo, hi]
    if len(sys.argv) > 1:
        json.dump(out, open(sys.argv[1].replace('.json', '_summary.json'), 'w'), default=float, indent=1)


if __name__ == '__main__':
    main()
