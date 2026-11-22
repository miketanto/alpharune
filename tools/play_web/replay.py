"""Replays of recorded search games (research/card_repr/search/replay_search.py `play` output).

A recorded game is a decision log (declog_seed<S>.json): for every external decision of a game, the
choice index applied and the search / network annotations (rbs_declog rows, layout below). The game
is played again in libplayweb (same engine, same seed, same rbenv start and autopay), so any position
is rebuilt by applying the first i choices -- a whole game takes a few tens of milliseconds, so there
is no frame cache: going back recreates the game.

rbs_declog row (rbsearch.cpp):
  0 searched  1 chosen index  2 network argmax  3 prior of chosen  4 prior of argmax
  5 visits of chosen  6 visits of argmax  7 total visits  8 Q of chosen  9 Q of argmax
  10 value (root, decider's view, -1..1)  11 decider seat  12..17 top three by visits (index, visits)
"""
import glob
import json
import os

import playweb

DECK_GUESSES = ('azir', 'rengar', 'diana', 'yi_rebuild', 'yi')


def find_logs(root):
    out = []
    for p in sorted(glob.glob(os.path.join(root, '**', 'declog_seed*.json'), recursive=True)):
        if '__pycache__' in p:
            continue
        out.append(p)
    return out


def _replays_cleanly(deck_p1, deck_p2, seed, game):
    """Does the log replay to its recorded end (every decider matches, same winner, every choice used)?"""
    try:
        g = playweb.Game(deck_p1, deck_p2, 1, seed)
    except RuntimeError:
        return False
    log = game['declog']
    k = 0
    try:
        while True:
            st = g.status()
            if st['done'] or not st['need']:
                break
            if k >= len(log) and game.get('done') is False:
                break
            if k >= len(log) or st['seat'] != int(log[k][11]):
                return False
            g.step(int(log[k][1]))
            k += 1
        st = g.status()
        if game.get('done') is False:   # a saved game that was not finished: the log ends at a pending decision
            return k == len(log)
        return k == len(log) and st['done'] and st['winner'] == int(game.get('winner', st['winner']))
    except ValueError:
        return False
    finally:
        g.close()


class Catalog:
    """Every decision log under the research tree, with its decks worked out once."""

    def __init__(self, root, deck_dir):
        self.root = root
        self.deck_dir = deck_dir
        self._decks = {}   # path -> (p1 deck name, p2 deck name) or None

    def deck_path(self, name):
        return os.path.join(self.deck_dir, name + '.txt')

    def decks_of(self, path, D):
        if path in self._decks:
            return self._decks[path]
        found = None
        g0 = D['games'][0]
        named = D.get('deck')
        cands = []
        if named:
            cands.append((named, D.get('deck_b') or named))
        cands += [(a, b) for a in DECK_GUESSES for b in DECK_GUESSES if (a, b) not in cands]
        for a, b in cands:
            if not (os.path.isfile(self.deck_path(a)) and os.path.isfile(self.deck_path(b))):
                continue
            # slot layout of replay_search.py: the deck pair is (deck, deck_b) for every slot
            if _replays_cleanly(self.deck_path(a), self.deck_path(b), D['seed'], g0):
                found = (a, b)
                break
        self._decks[path] = found
        return found

    def entries(self):
        out = []
        for p in find_logs(self.root):
            try:
                D = json.load(open(p))
            except (OSError, ValueError):
                continue
            decks = self.decks_of(p, D)
            rel = os.path.relpath(p, self.root)
            for i, g in enumerate(D.get('games', [])):
                out.append(dict(id=f'{rel}#{i}', file=rel, slot=i, seed=D['seed'], sims=D.get('sims'),
                                name=D.get('name') or os.path.basename(os.path.dirname(p)),
                                search_seat=g.get('search_seat'), human_seat=g.get('human_seat'), done=g.get('done', True), winner=g.get('winner'), turns=g.get('turns'),
                                decisions=len(g.get('declog', [])), decks=decks,
                                mtime=os.path.getmtime(p)))
        out.sort(key=lambda e: (-e['mtime'], e['file'], e['slot']))
        return out


class Replay:
    def __init__(self, catalog, entry_id, seat, both_hands):
        rel, slot = entry_id.rsplit('#', 1)
        path = os.path.normpath(os.path.join(catalog.root, rel))
        if not path.startswith(os.path.normpath(catalog.root)) or not os.path.isfile(path):
            raise ValueError('no replay ' + entry_id)
        D = json.load(open(path))
        self.game_rec = D['games'][int(slot)]
        self.log = self.game_rec['declog']
        decks = catalog.decks_of(path, D)
        if not decks:
            raise ValueError('could not work out the decks of this replay (none of %s replays it)' % ', '.join(DECK_GUESSES))
        self.decks = decks
        self.p1, self.p2 = catalog.deck_path(decks[0]), catalog.deck_path(decks[1])
        self.seed = int(D['seed'])
        self.sims = D.get('sims')
        self.seat = 2 if int(seat) == 2 else 1
        self.both = bool(both_hands)
        ss = int(self.game_rec.get('search_seat') or 0)
        nm = D.get('name') or os.path.basename(os.path.dirname(path))
        self.labels = {ss: f'search N={self.sims}', 3 - ss: 'network greedy'} if ss in (1, 2) else {1: 'P1', 2: 'P2'}
        self.human = int(self.game_rec.get('human_seat') or 0)
        if D.get('labels'):
            self.labels = {int(k): v for k, v in D['labels'].items()}
        self.meta = dict(replay=True, file=rel, slot=int(slot), seed=self.seed, decks=list(decks), name=nm,
                         labels={str(k): v for k, v in self.labels.items()}, search_seat=ss,
                         n=len(self.log), recorded_winner=self.game_rec.get('winner'))
        self.g = None
        self.i = 0
        self.log_next = 0
        self._restart()
        self.turns = self._scan_turns()
        self.meta['turn_starts'] = self.turns

    def _restart(self):
        if self.g:
            self.g.close()
        self.g = playweb.Game(self.p1, self.p2, self.seat, self.seed)
        self.i = 0
        self.log_next = 0

    def _scan_turns(self):
        """[(decision index, turn number, turn player seat)] where a new turn begins."""
        out = []
        last = None
        self._restart()
        while True:
            st = self.g.status()
            if st['done'] or not st['need'] or self.i >= len(self.log):
                break
            t = self.g.view(1, 1 << 40)['turn']
            key = (t['number'], t['turn_player'])
            if key != last:
                out.append([self.i, t['number'], 1 if t['turn_player'] == 1 else 2])
                last = key
            self.g.step(int(self.log[self.i][1]))
            self.i += 1
        self._restart()
        return out

    def set_view(self, seat=None, both=None):
        if seat is not None:
            self.seat = 2 if int(seat) == 2 else 1
        if both is not None:
            self.both = bool(both)
        self.log_next = 0

    def goto(self, i):
        i = max(0, min(int(i), len(self.log)))
        reset = False
        if i < self.i:
            self._restart()
            reset = True
        while self.i < i:
            st = self.g.status()
            if st['done'] or not st['need']:
                break
            self.g.step(int(self.log[self.i][1]))
            self.i += 1
        if reset:
            self.log_next = 0
        return self.frame(log_reset=reset or self.log_next == 0)

    def frame(self, log_reset=False):
        v = self.g.view(self.seat, self.log_next)
        self.log_next = v.get('log_next', self.log_next)
        other = 3 - self.seat
        if self.both:
            o = self.g.view(other, 1 << 40)
            v['opp_hand_cards'] = o.get('hand', [])
            v.setdefault('cards', {}).update(o.get('cards', {}))
        v['meta'] = self.meta
        v['replay'] = self.annotation()
        v['replay']['log_reset'] = log_reset
        return v

    def annotation(self):
        a = dict(i=self.i, n=len(self.log))
        st = self.g.status()
        if st['done'] or self.i >= len(self.log):
            a['done'] = True
            a['winner'] = st['winner']
            return a
        d = self.log[self.i]
        seat = int(d[11])
        dv = self.g.view(seat, 1 << 40)
        opts = {o['index']: o for o in (dv.get('decision') or {}).get('options', [])}
        dec = dv.get('decision') or {}
        text = lambda k: (opts.get(int(k)) or {}).get('text') or (opts.get(int(k)) or {}).get('label') or f'option {int(k)}'
        objs = lambda k: (opts.get(int(k)) or {}).get('objs', [])
        ch, am = int(d[1]), int(d[2])
        a['human'] = seat == self.human
        a.update(seat=seat, who=self.labels.get(seat, f'P{seat}'), searched=bool(d[0]), value=round(float(d[10]), 3),
                 kind=dec.get('kind'), question=dec.get('question'), n_options=len(opts),
                 chosen=dict(index=ch, text=text(ch), objs=objs(ch), prior=round(float(d[3]), 3)))
        if d[0]:
            tot = max(1.0, float(d[7]))
            top = []
            for t in range(3):
                k, nv = int(d[12 + 2 * t]), float(d[13 + 2 * t])
                if k >= 0:
                    top.append(dict(index=k, text=text(k), share=round(nv / tot, 3), chosen=k == ch))
            a.update(visits=int(tot), top=top, q=round(float(d[8]), 3))
            if am != ch:
                a['override'] = dict(index=am, text=text(am), prior=round(float(d[4]), 3), q=round(float(d[9]), 3))
        cards = dv.get('cards') or {}
        a['cards'] = cards
        return a
