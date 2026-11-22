#!/usr/bin/env python3
"""Play Riftbound against a self-play network in the browser (tools/play_web/README.md).

    CUDA_VISIBLE_DEVICES= taskset -c 12-15 python3 tools/play_web/server.py [--port 8765]

then open http://localhost:8765. One game at a time; stdlib only (plus numpy / torch for the
network). The human is shown only what its seat may know: every frame is pw_view(human seat),
built in C++ from the engine's per-observer queries, and the log carries only the lines whose
audience includes that seat.
"""
import argparse
import glob
import json
import os
import random
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import playweb  # noqa: E402
import replay  # noqa: E402

STATIC = os.path.join(HERE, 'static')
DECK_DIR = os.path.join(REPO, 'decks')
DEFAULT_CKPT = os.path.join(playweb.SELFPLAY, 'runs', 'ov_main', 'league_02080.pt')
MAX_AGENT_FRAMES = 400
CATALOG = replay.Catalog(playweb.CARD_REPR, DECK_DIR)


def list_decks():
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(DECK_DIR, '*.txt')))


def list_checkpoints():
    runs = os.path.join(playweb.SELFPLAY, 'runs')
    found = set(glob.glob(os.path.join(runs, 'ov_main', '*.pt')))
    found |= set(glob.glob(os.path.join(runs, '*', 'latest.pt')))
    found |= set(glob.glob(os.path.join(runs, 'g1_main', '*.pt')))
    out = []
    for p in found:
        try:
            out.append((os.path.getmtime(p), p))
        except OSError:
            pass
    out.sort(reverse=True)
    rel = [os.path.relpath(p, runs) for _, p in out[:60]]
    d = os.path.relpath(DEFAULT_CKPT, runs)
    if os.path.isfile(DEFAULT_CKPT):
        rel = [d] + [r for r in rel if r != d]
    return rel


class Session:
    def __init__(self):
        self.lock = threading.Lock()
        self.game = None
        self.agent = None
        self.agent_ckpt = None
        self.human = 1
        self.meta = {}
        self.log_next = 0

    def agent_seat(self):
        return 3 - self.human

    def new(self, deck, agent_deck, ckpt, seat, seed):
        runs = os.path.join(playweb.SELFPLAY, 'runs')
        ck = ckpt if os.path.isabs(ckpt) else os.path.join(runs, ckpt)
        if not os.path.isfile(ck):
            raise ValueError('no checkpoint ' + ckpt)
        for d in (deck, agent_deck):
            if d not in list_decks():
                raise ValueError('no deck ' + d)
        if self.agent is None or self.agent_ckpt != ck:
            t0 = time.time()
            self.agent = playweb.NetAgent(ck)
            self.agent_ckpt = ck
            print(f'loaded {ck} in {time.time() - t0:.1f}s', flush=True)
        if seat not in (1, 2):
            seat = random.choice((1, 2))
        seed = int(seed) if seed else random.randrange(1, 2 ** 31 - 1)
        dh, da = os.path.join(DECK_DIR, deck + '.txt'), os.path.join(DECK_DIR, agent_deck + '.txt')
        p1, p2 = (dh, da) if seat == 1 else (da, dh)
        if self.game:
            self.game.close()
        self.game = playweb.Game(p1, p2, seat, seed)
        self.human = seat
        self.log_next = 0
        self.meta = dict(deck=deck, agent_deck=agent_deck, checkpoint=os.path.relpath(ck, runs), seat=seat, seed=seed)
        return self.run_agent()

    def frame(self):
        v = self.game.view(self.human, self.log_next)
        self.log_next = v.get('log_next', self.log_next)
        v['meta'] = self.meta
        return v

    def run_agent(self):
        """Play the network's decisions until the human is asked or the game ends. Returns the
        frames the human sees along the way (one after each network decision), then the
        current one."""
        frames = []
        g = self.game
        while True:
            st = g.status()
            if st['done'] or not st['need'] or st['seat'] == self.human:
                break
            obs = g.observe()
            a, value = self.agent.choose(obs, st['nlegal'])
            g.step(a)
            f = self.frame()
            f['agent_moved'] = True
            if len(frames) < MAX_AGENT_FRAMES:
                frames.append(f)
        frames.append(self.frame())
        return frames

    def act(self, index):
        st = self.game.status()
        if st['done'] or not st['need'] or st['seat'] != self.human:
            raise ValueError('it is not your decision')
        if not (0 <= index < st['nlegal']):
            raise ValueError('bad option')
        self.game.step(index)
        first = self.frame()
        return [first] + self.run_agent()


S = Session()
R = {'replay': None}   # the replay being watched (one at a time, like the game)


class Handler(BaseHTTPRequestHandler):
    server_version = 'playweb/1'

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype='application/json'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _host_ok(self):
        host = (self.headers.get('Host') or '').split(':')[0]
        return host in ('localhost', '127.0.0.1', '[::1]', '') or os.environ.get('PLAYWEB_ANY_HOST')

    def do_GET(self):
        if not self._host_ok():
            return self._send(403, {'error': 'forbidden host'})
        u = urlparse(self.path)
        if u.path in ('/', '/index.html'):
            return self._static('index.html')
        if u.path.startswith('/static/'):
            return self._static(u.path[len('/static/'):])
        if u.path == '/api/setup':
            return self._send(200, dict(decks=list_decks(), checkpoints=list_checkpoints(),
                                        default_checkpoint=os.path.relpath(DEFAULT_CKPT, os.path.join(playweb.SELFPLAY, 'runs')),
                                        default_agent_deck='rengar', has_game=S.game is not None))
        if u.path == '/api/replays':
            with S.lock:
                return self._send(200, {'replays': CATALOG.entries()})
        if u.path == '/api/state':
            with S.lock:
                if not S.game:
                    return self._send(200, {'frames': []})
                S.log_next = 0 if parse_qs(u.query).get('full') else S.log_next
                return self._send(200, {'frames': [S.frame()]})
        self._send(404, {'error': 'not found'})

    def do_POST(self):
        if not self._host_ok():
            return self._send(403, {'error': 'forbidden host'})
        n = int(self.headers.get('Content-Length') or 0)
        try:
            body = json.loads(self.rfile.read(n) or b'{}')
        except ValueError:
            return self._send(400, {'error': 'bad json'})
        u = urlparse(self.path)
        try:
            with S.lock:
                if u.path == '/api/new':
                    frames = S.new(body.get('deck', 'rengar'), body.get('agent_deck', 'rengar'),
                                   body.get('checkpoint') or DEFAULT_CKPT, int(body.get('seat') or 0), body.get('seed'))
                    return self._send(200, {'frames': frames})
                if u.path == '/api/replay/open':
                    if R['replay']:
                        R['replay'].g.close()
                    R['replay'] = replay.Replay(CATALOG, body['id'], int(body.get('seat') or 1), body.get('both', True))
                    return self._send(200, {'frames': [R['replay'].frame(log_reset=True)]})
                if u.path in ('/api/replay/goto', '/api/replay/view'):
                    rp = R['replay']
                    if not rp:
                        return self._send(400, {'error': 'no replay open'})
                    if u.path == '/api/replay/view':
                        rp.set_view(body.get('seat'), body.get('both'))
                        return self._send(200, {'frames': [rp.frame(log_reset=True)]})
                    return self._send(200, {'frames': [rp.goto(int(body['i']))]})
                if u.path == '/api/act':
                    if not S.game:
                        return self._send(400, {'error': 'no game'})
                    return self._send(200, {'frames': S.act(int(body['index']))})
        except ValueError as e:
            return self._send(400, {'error': str(e)})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self._send(500, {'error': f'{type(e).__name__}: {e}'})
        self._send(404, {'error': 'not found'})

    def _static(self, name):
        p = os.path.normpath(os.path.join(STATIC, name))
        if not p.startswith(STATIC) or not os.path.isfile(p):
            return self._send(404, {'error': 'not found'})
        ctype = {'.html': 'text/html; charset=utf-8', '.js': 'application/javascript', '.css': 'text/css',
                 '.svg': 'image/svg+xml'}.get(os.path.splitext(p)[1], 'application/octet-stream')
        with open(p, 'rb') as f:
            self._send(200, f.read(), ctype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--bind', default='127.0.0.1')
    a = ap.parse_args()
    playweb.lib()   # fail early if the library is missing
    srv = ThreadingHTTPServer((a.bind, a.port), Handler)
    print(f'playweb: http://localhost:{a.port}', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
