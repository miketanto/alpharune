'use strict';
// Riftbound vs Network — browser client for tools/play_web/server.py.
// Every frame is what the human's seat may know (pw_view): players are relative
// (1 = you, 2 = the network), and options carry the engine's legal index.

const $ = (s, r = document) => r.querySelector(s);
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const sleep = ms => new Promise(r => setTimeout(r, ms));

let F = null;              // the frame on screen
let cards = {};            // card id -> printed facts (accumulated)
let queue = [];
let pumping = false;
let busy = false;          // a request is in flight
let logLines = [];         // {cls, text}
let lastActor = 0;         // 1 me / 2 opp, from the last CHOSE line
let mullSel = new Set();
let pendingDest = null;    // {objId, opts} while a destination is being picked
let mySeat = 1;
let RP = null;             // replay annotation of the frame on screen (null when playing)
let rpTimer = 0;           // autoplay
let rpBusy = false;

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({ error: 'bad response' }));
  if (!r.ok || j.error) throw new Error(j.error || r.statusText);
  return j;
}

// ---------------- setup ----------------
async function initSetup() {
  try {
    const s = await api('/api/setup');
    const fill = (sel, items, def) => { sel.innerHTML = items.map(i => `<option${i === def ? ' selected' : ''}>${esc(i)}</option>`).join(''); };
    fill($('#s-deck'), s.decks, s.default_deck || 'rengar');
    fill($('#s-agent-deck'), s.decks, s.default_agent_deck);
    fill($('#s-ckpt'), s.checkpoints, s.default_checkpoint);
    $('#s-resume').classList.toggle('hidden', !s.has_game);
  } catch (e) { $('#s-err').textContent = 'Setup failed: ' + e.message; }
  initReplays();
}
$('#s-start').onclick = async () => {
  $('#s-err').textContent = '';
  $('#s-start').disabled = true; $('#s-start').textContent = 'Starting…';
  try {
    const j = await api('/api/new', { deck: $('#s-deck').value, agent_deck: $('#s-agent-deck').value, checkpoint: $('#s-ckpt').value,
      sims: +$('#s-sims').value, seat: +$('#s-seat').value, seed: $('#s-seed').value.trim() || null });
    resetGame();
    showGame();
    enqueue(j.frames);
  } catch (e) { $('#s-err').textContent = e.message; }
  $('#s-start').disabled = false; $('#s-start').textContent = 'Start game';
};
$('#s-resume').onclick = async () => {
  const j = await api('/api/state?full=1');
  resetGame(); showGame(); enqueue(j.frames);
};
function resetGame() { F = null; cards = {}; queue = []; logLines = []; lastActor = 0; mullSel.clear(); $('#log').innerHTML = ''; $('#gameover').classList.add('hidden'); }
function showGame() { $('#setup').classList.add('hidden'); $('#game').classList.remove('hidden'); }
function showSetup() { $('#game').classList.add('hidden'); $('#setup').classList.remove('hidden'); initSetup(); }

// ---------------- frames ----------------
function enqueue(frames) { queue.push(...frames); if (!pumping) pump(); }
async function pump() {
  pumping = true;
  while (queue.length) {
    const f = queue.shift();
    const before = F ? boardSig(F) : '';
    apply(f);
    if (f.agent_moved && queue.length) {
      const changed = boardSig(f) !== before;
      await sleep(changed ? (queue.length > 40 ? 200 : 520) : 40);
    }
  }
  pumping = false;
  render();
}
function boardSig(f) {
  return JSON.stringify([f.board.map(o => [o.id, o.loc, o.exhausted, o.might, o.damage]), f.chain.length, f.players.me.score,
    f.players.opp.score, f.players.opp.hand_count, f.battlefields.map(b => [b.controller, b.facedown.length])]);
}
function apply(f) {
  if (f.view_error) { toast('View error: ' + f.view_error, 'opp'); return; }
  const prev = F;
  Object.assign(cards, f.cards || {});
  mySeat = f.me;
  RP = f.replay || null;
  if (RP) {
    Object.assign(cards, RP.cards || {});
    f.decision = null;   // a replay is watched, never answered: nothing on the board is clickable
    if (RP.log_reset) { logLines = []; $('#log').innerHTML = ''; lastActor = 0; }
  }
  F = f;
  addLog(f.log || []);
  if (!prev || (prev.decision?.kind !== f.decision?.kind)) mullSel.clear();
  pendingDest = null;
  render(RP && RP.log_reset ? null : prev, !!f.agent_moved || !!RP);
  if (f.agent_moved) agentToast(f.log || []);
  if (RP) markChosen();
  if (f.game_over && !RP) showGameOver();
}

// ---------------- labels ----------------
// Playing: "You" / "Opponent". Watching a replay: the seat and what played it ("P1 · search N=64").
function seatLabel(seat) { const l = F?.meta?.labels?.[String(seat)]; return `P${seat}` + (l ? ` · ${l}` : ''); }
function relSeat(rel) { return rel === 1 ? mySeat : 3 - mySeat; }
function relName(rel) { return RP ? seatLabel(relSeat(rel)) : rel === 1 ? 'you' : 'opponent'; }

// ---------------- log ----------------
const OPP = () => `P${3 - mySeat}`, ME = () => `P${mySeat}`;
function prettyLog(msg) {
  let m = msg.replace(/^\[D:[0-9.]+\]\s*/, '');
  if (!RP) m = m.replace(new RegExp(`\\b${ME()}\\b`, 'g'), 'You').replace(new RegExp(`\\b${OPP()}\\b`, 'g'), 'Opponent');
  m = m.replace(/\s*\(id=\d+,?\s*/g, ' (').replace(/\(\)/g, '').replace(/ \(\s*\)/g, '');
  m = m.replace(/BF#(\d+)/g, (_, i) => bfName(+i));
  m = m.replace(/\s*\(chain_item=\d+(, ability=-?\d+)?(, spell=\d+)?\)/g, '');
  return m;
}
function addLog(lines) {
  const verbose = $('#log-verbose').checked;
  const box = $('#log');
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  for (const l of lines) {
    const raw = l.msg.replace(/^\[D:[0-9.]+\]\s*/, '');
    const cm = raw.match(/^CHOSE: \[(P[12])\]/);
    if (cm) lastActor = cm[1] === ME() ? 1 : 2;
    let cls = lastActor === 1 ? 'me' : lastActor === 2 ? 'opp' : 'sys';
    if (/^SCORE/.test(raw)) cls = 'score';
    if (/^[^A-Za-z0-9[]/.test(raw)) { cls = 'sys'; lastActor = 0; }   // phase / turn markers
    if (/\bTURN \d+\b/.test(raw)) cls = 'turn';
    const noise = /^(DECISION|CHOICE-REQUEST|FINALIZE-CHOICES|TGT_PROMPT|PROMPT|TARGET \(finalize\))|pick rune|PassPriority|PassFocus|^CHOSE: \[P[12]\] (MakeChoice|PassPriority|PassFocus)$/.test(raw);
    const entry = { cls, text: prettyLog(l.msg), noise };
    logLines.push(entry);
    if (verbose || !noise) { const d = el('div', 'ln new ' + cls); d.textContent = entry.text; box.appendChild(d); }
  }
  while (box.childElementCount > 1500) box.firstChild.remove();
  if (atBottom || lines.length) box.scrollTop = box.scrollHeight;
}
$('#log-verbose').onchange = () => {
  const box = $('#log'); box.innerHTML = '';
  const v = $('#log-verbose').checked;
  for (const e of logLines.slice(-1500)) if (v || !e.noise) { const d = el('div', 'ln ' + e.cls); d.textContent = e.text; box.appendChild(d); }
  box.scrollTop = box.scrollHeight;
};
function agentToast(lines) {
  // The most telling line the network's move produced, as the human may read it.
  let best = null;
  for (const l of lines) {
    const raw = l.msg.replace(/^\[D:[0-9.]+\]\s*/, '');
    if (/^(PLAY|MOVE|SCORE|HIDE|ACTIVATE|ATTACK|COMBAT|CONQUER)/.test(raw)) best = raw;
    else if (!best && new RegExp(`^CHOSE: \\[${OPP()}\\] (?!PassPriority|PassFocus|MakeChoice)`).test(raw)) best = raw;
  }
  if (best) toast('Opponent ' + toastPhrase(prettyLog(best)), 'opp');
}
function toastPhrase(t) {
  let m;
  if ((m = t.match(/^PLAY: (.+?)(?: \(.*)?$/))) return 'plays ' + m[1];
  if ((m = t.match(/^MOVE: (.+?) -> (.+)$/))) return `moves ${m[1]} → ${m[2].replace(/Base\(.*\)/, 'base')}`;
  if ((m = t.match(/^SCORE: Opponent \+(\d+) via (\w+) -> (\d+)/))) return `scores ${m[1]} (${m[2]}) — now ${m[3]}`;
  if ((m = t.match(/^SCORE: You \+(\d+) via (\w+) -> (\d+)/))) return `lets you score — you have ${m[3]}`;
  t = t.replace(/^CHOSE: \[Opponent\] /, '');
  if (/^EndTurn/.test(t)) return 'ends the turn';
  if ((m = t.match(/^Mulligan (\d+)/))) return m[1] === '0' ? 'keeps its hand' : `mulligans ${m[1]}`;
  if ((m = t.match(/^StandardMove units=\[(.+)\]/))) return 'moves ' + m[1];
  if ((m = t.match(/^PlayCard (.+)/))) return 'plays ' + m[1];
  return '· ' + t;
}
let toastTimer = 0;
function toast(text, who) {
  const t = $('#toast'); t.textContent = text; t.className = who === 'me' ? 'me' : ''; t.classList.remove('hidden');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.add('hidden'), 1500);
}

// ---------------- helpers ----------------
function bfName(slot) {
  const b = F?.battlefields?.find(x => x.slot === slot);
  const c = b && cards[b.card];
  if (!c) return `Battlefield ${slot + 1}`;
  if (F.battlefields.filter(x => x.card === b.card).length < 2) return c.name;
  return `${c.name} (${b.contributed_by === 1 ? 'yours' : b.contributed_by === 2 ? "opponent's" : slot + 1})`;
}
function destName(d) { return d === 'base' ? 'your base' : d && d.startsWith('bf') ? bfName(+d.slot || +d.slice(2)) : ''; }
const VERB = { PlayCard: 'Play', PlayReaction: 'Play', PlayActionCard: 'Play', StandardMove: 'Move', ActivateAbility: 'Activate',
  ActivateReactionAbility: 'Activate', ActivateActionAbility: 'Activate', HideCard: 'Hide', PassPriority: 'Pass', PassFocus: 'Pass focus',
  EndTurn: 'End turn', MulliganDecision: 'Mulligan' };
function optText(o) {
  let t = o.text || o.label || '';
  t = t.replace(/^([A-Z][A-Za-z]+) /, (m, w) => (VERB[w] ? VERB[w] + ' ' : m));
  t = t.replace(/\bbattlefield (\d+)\b/g, (_, i) => bfName(+i));
  t = t.replace(/ in hand\b/g, '').replace(/\byour (?=[A-Z])/g, '');
  return t.charAt(0).toUpperCase() + t.slice(1);
}
function options() { return F?.decision?.options || []; }
function optsFor(id) { return options().filter(o => o.objs.includes(id)); }
function choose(o) {
  if (busy || pumping) return;
  closeMenu();
  busy = true;
  renderPrompt();
  api('/api/act', { index: o.index }).then(j => { busy = false; enqueue(j.frames); })
    .catch(e => { busy = false; toast(e.message, 'opp'); render(); });
}
function isPassy(o) { return o.objs.length === 0 && /^(EndTurn|PassPriority|PassFocus)$/.test(o.type); }

// ---------------- rendering ----------------
function render(prev, byAgent) {
  if (!F) return;
  $('#zoom').classList.add('hidden');
  renderTracks(); renderSides(); renderRows(); renderBases(); renderRunes(); renderBattlefields(); renderHand();
  renderTurn(); renderPrompt(); renderChain();
  if (prev) flashChanges(prev, byAgent);
}

function cardEl(o, opts = {}) {
  const c = o.card != null ? cards[o.card] : null;
  const d = el('div', 'card');
  d.dataset.id = o.id;
  if (c) { d.style.backgroundImage = `url("${c.img}")`; d.classList.add('has-art'); d.dataset.card = o.card; }
  if (o.facedown || (o.card == null && !o.name)) d.classList.add('facedown');
  else if (!c) d.classList.add('noart');   // a token with no card of its own
  if (o.exhausted) d.classList.add('exhausted');
  if (o.ctrl) d.classList.add(o.ctrl === 1 ? 'ctrl-me' : 'ctrl-opp');
  if (o.lethal) d.classList.add('lethal');
  if (o.name) d.appendChild(el('div', 'nm', esc(o.name)));
  if (o.type === 'Unit' && o.might != null && !opts.noStats) {
    const printed = c ? c.might : o.might;
    d.appendChild(el('div', 'might' + (o.might > printed ? ' up' : ''), o.might));
    if (o.damage) d.appendChild(el('div', 'dmg', o.damage));
  }
  const st = el('div', 'status');
  if (o.stunned) st.appendChild(el('span', 'chip stun', 'Stunned'));
  if (o.combat === 'attacker') st.appendChild(el('span', 'chip atk', 'Attacker'));
  if (o.combat === 'defender') st.appendChild(el('span', 'chip def', 'Defender'));
  if (o.buffs) st.appendChild(el('span', 'chip', `Buff ${o.buffs}`));
  if (o.token) st.appendChild(el('span', 'chip tok', 'Token'));
  if (o.empowered) st.appendChild(el('span', 'chip', 'Empowered'));
  if (st.childElementCount) d.appendChild(st);
  hookZoom(d, o);
  const acts = optsFor(o.id);
  if (acts.length && !pumping && !busy) {
    d.classList.add('actionable');
    d.onclick = ev => { ev.stopPropagation(); onCardClick(o, acts, ev); };
  }
  if (F.decision?.kind === 'mulligan' && mullSel.has(o.id)) d.classList.add('selected');
  return d;
}
function withGear(u) {
  const gear = F.board.filter(g => g.attached_to === u.id);
  if (!gear.length) return cardEl(u);
  const w = el('div', 'attach-row');
  w.appendChild(cardEl(u));
  for (const g of gear) {
    const c = cards[g.card];
    const gd = el('div', 'gear', esc(g.name));
    if (c) gd.style.backgroundImage = `url("${c.img}")`;
    gd.dataset.id = g.id;
    hookZoom(gd, g);
    const acts = optsFor(g.id);
    if (acts.length && !pumping && !busy) { gd.classList.add('actionable'); gd.style.cursor = 'pointer'; gd.onclick = ev => { ev.stopPropagation(); onCardClick(g, acts, ev); }; }
    w.appendChild(gd);
  }
  return w;
}

function renderTracks() {
  const vs = F.turn.victory_score || 8;
  for (const [id, p] of [['#opp-track', F.players.opp], ['#me-track', F.players.me]]) {
    const t = $(id); t.innerHTML = '';
    t.appendChild(el('div', 'score-big', `${p.score}<span class="tiny"> / ${vs}</span>`));
    for (let i = 1; i <= vs; i++) t.appendChild(el('div', 'pip' + (i <= p.score ? ' on' : ''), i));
  }
}
function renderSides() {
  for (const [id, rel, p] of [['#opp-side', 2, F.players.opp], ['#me-side', 1, F.players.me]]) {
    const s = $(id); s.innerHTML = '';
    const leg = F.legends.find(l => l.ctrl === rel);
    if (leg) { s.appendChild(el('div', 'slot-label', 'Legend')); s.appendChild(cardEl(leg, { noStats: true })); }
    if (p.champion_zone) { s.appendChild(el('div', 'slot-label', 'Champion')); s.appendChild(cardEl(p.champion_zone)); }
  }
}
function pileHtml(p, who) {
  return `<span class="pile">Deck <b>${p.deck_count}</b></span><span class="pile">Rune deck <b>${p.rune_deck_count}</b></span>` +
    `<span class="pile click" data-pile="${who}-trash">Trash <b>${p.trash.length}</b></span>` +
    (p.banishment.length ? `<span class="pile click" data-pile="${who}-ban">Banished <b>${p.banishment.length}</b></span>` : '') +
    (p.xp ? `<span class="pile">XP <b>${p.xp}</b></span>` : '') +
    `<span class="pile pool"><span>Runes <b>${p.ready_runes}</b>/${p.ready_runes + p.exhausted_runes} ready</span>` +
    `<span class="energy">⚡<b>${p.energy}</b></span><span class="power">◆<b>${p.power}</b></span></span>`;
}
function renderRows() {
  const oh = $('#opp-hand'); oh.innerHTML = '';
  const n = F.players.opp.hand_count;
  oh.classList.toggle('faceup', !!F.opp_hand_cards);
  if (F.opp_hand_cards) for (const o of F.opp_hand_cards) oh.appendChild(cardEl(o, { noStats: true }));
  else for (let i = 0; i < n; i++) oh.appendChild(el('div', 'back'));
  if (F.opp_known_hand.length && !F.opp_hand_cards) {
    const k = el('span', 'known-chip', `you know ${F.opp_known_hand.length} of these card${F.opp_known_hand.length > 1 ? 's' : ''} ▸`);
    k.onclick = () => openModal('Cards you know are in the opponent\'s hand', F.opp_known_hand.map(c => ({ id: -1, card: c, name: cards[c]?.name, type: cards[c]?.type })));
    oh.appendChild(k);
  }
  $('#opp-piles').innerHTML = pileHtml(F.players.opp, 'opp');
  $('#me-piles').innerHTML = pileHtml(F.players.me, 'me');
  document.querySelectorAll('[data-pile]').forEach(e => e.onclick = () => {
    const [who, kind] = e.dataset.pile.split('-');
    const p = who === 'me' ? F.players.me : F.players.opp;
    openModal(`${who === 'me' ? 'Your' : 'Opponent\'s'} ${kind === 'trash' ? 'trash' : 'banishment'}`, kind === 'trash' ? p.trash : p.banishment);
  });
}
function renderBases() {
  for (const [id, rel] of [['#opp-base', 2], ['#me-base', 1]]) {
    const z = $(id), c = $('.cards', z); c.innerHTML = '';
    $('.zone-label', z).textContent = RP ? `${seatLabel(relSeat(rel))} base` : rel === 1 ? 'Your base' : 'Opponent base';
    for (const o of F.board.filter(o => o.loc === 'base' && o.ctrl === rel && o.attached_to == null)) c.appendChild(withGear(o));
    z.classList.remove('drop'); z.onclick = null;
  }
}
function renderRunes() {
  for (const [id, rel] of [['#opp-runes', 2], ['#me-runes', 1]]) {
    const z = $(id); z.innerHTML = '';
    // Runes are shown as cards (their art, turned when exhausted) in a row of their own, with a
    // domain-coloured edge so the colours read at a glance.
    for (const r of F.runes.filter(r => r.ctrl === rel)) {
      const c = cards[r.card];
      const dom = c?.domains?.[0] || 'none';
      const d = cardEl(r, { noStats: true });
      d.classList.add('rune-card', `d-${dom}`);
      if (!c) d.appendChild(el('div', 'rune-dom', esc(dom)));
      z.appendChild(d);
    }
  }
}
function renderBattlefields() {
  const z = $('#bfs'); z.innerHTML = '';
  for (const b of F.battlefields) {
    const c = cards[b.card];
    const d = el('div', 'bf' + (b.controller === 1 ? ' ctrl-me' : b.controller === 2 ? ' ctrl-opp' : '') + (b.combat ? ' in-combat' : ''));
    d.dataset.slot = b.slot;
    const art = el('div', 'art'); if (c) art.style.backgroundImage = `url("${c.img}")`; d.appendChild(art);
    const top = el('div', 'half opp'), mid = el('div', 'mid'), bot = el('div', 'half me');
    const loc = 'bf' + b.slot;
    for (const o of F.board.filter(o => o.loc === loc && o.attached_to == null)) (o.ctrl === 1 ? bot : top).appendChild(withGear(o));
    for (const o of b.facedown) (o.owner === 1 ? bot : top).appendChild(cardEl(o));
    const name = el('span', 'bf-name', esc(bfName(b.slot)));
    if (c) hookZoom(name, { id: -1, card: b.card, name: c.name, type: 'Battlefield' });
    const badges = el('span', 'badges');
    const ctl = b.controller === 1 ? `<span class="badge me">${RP ? `P${relSeat(1)} controls` : 'You control'}</span>` :
      b.controller === 2 ? `<span class="badge opp">${RP ? `P${relSeat(2)} controls` : 'Opponent controls'}</span>` : '<span class="badge">Uncontrolled</span>';
    badges.innerHTML = ctl + (b.combat ? '<span class="badge combat">Combat</span>' : b.showdown ? '<span class="badge showdown">Showdown</span>' : '') +
      (b.contested ? '<span class="badge">Contested</span>' : '') + (b.scored_by_me ? '<span class="badge me">Scored</span>' : '');
    mid.append(name, badges);
    d.append(top, mid, bot);
    z.appendChild(d);
  }
}
function renderHand() {
  const h = $('#hand'); h.innerHTML = '';
  const n = F.hand.length;
  F.hand.forEach((o, i) => {
    const d = cardEl(o, { noStats: true });
    const c = cards[o.card];
    if (c) d.appendChild(el('div', 'cost', c.energy));
    const rot = n > 1 ? (i - (n - 1) / 2) * Math.min(5, 26 / n) : 0;
    const lift = Math.abs(i - (n - 1) / 2) * 3;
    if (!d.classList.contains('selected')) d.style.transform = `rotate(${rot}deg) translateY(${lift}px)`;
    if (!d.classList.contains('actionable') && F.decision) d.classList.add('unplayable');
    h.appendChild(d);
  });
}
const PHASE = { Setup: 'Setup', Mulligan: 'Mulligan', AwakenPhase: 'Awaken', BeginningStep: 'Beginning', ScoringStep: 'Scoring',
  ChannelPhase: 'Channel', DrawPhase: 'Draw', MainPhase: 'Main', EndingStep: 'Ending', ExpirationStep: 'Expiration', GameOver: 'Game over' };
function renderTurn() {
  const t = F.turn;
  const mine = t.turn_player === 1;
  const whose = RP ? `${seatLabel(relSeat(t.turn_player))}'s turn` : mine ? 'Your turn' : 'Opponent\'s turn';
  const tail = RP ? `replay · seed ${F.meta.seed} · ${F.meta.decks.join(' vs ')}` : `you are Player ${F.me} · ${F.meta?.checkpoint || ''}${F.meta?.sims ? ` + search ${F.meta.sims}` : ''}`;
  $('#turn-box').innerHTML = `<div class="t1">Turn ${t.number} · <span class="whose ${mine ? 'me' : 'opp'}">${esc(whose)}</span></div>` +
    `<div class="dim">${PHASE[t.phase] || t.phase} phase${t.showdown ? ' · showdown' : ''}${t.closed ? ' · chain active' : ''}` +
    `${t.focus ? ` · focus: ${esc(relName(t.focus))}` : ''} · ${esc(tail)}</div>`;
}
function renderPrompt() {
  const box = $('#prompt-box'); box.innerHTML = ''; box.className = '';
  if (!F) return;
  if (RP) return renderReplay(box);
  if (F.game_over) { box.className = 'waiting'; box.innerHTML = `<div class="q">Game over</div><button class="primary" id="again">New game</button>`; $('#again').onclick = showSetup; return; }
  const d = F.decision;
  if (busy || pumping || !d) {
    box.className = 'waiting';
    box.innerHTML = `<div class="q"><span class="spinner"></span>${busy ? 'Applying your choice…' : 'The network is playing…'}</div>`;
    return;
  }
  box.className = 'mine';
  const src = d.source_card != null && cards[d.source_card] ? cards[d.source_card].name : '';
  const q = d.kind === 'main phase action' ? 'Your move' : d.kind === 'chain priority' ? 'You have priority' :
    d.kind === 'showdown focus' ? 'You have focus' : (d.question || d.kind);
  box.appendChild(el('div', 'q', esc(q.charAt(0).toUpperCase() + q.slice(1))));
  const sub = src ? `${esc(src)} — ${esc(d.kind)}` : esc(d.kind);
  box.appendChild(el('div', 'sub', sub + (d.at_finalization ? ' (as it is put on the chain)' : '')));
  if (d.kind === 'mulligan') return renderMulligan(box, d);
  const free = d.options.filter(o => o.objs.length === 0);
  const withObj = d.options.filter(o => o.objs.length > 0);
  if (withObj.length) box.appendChild(el('div', 'sub', 'Glowing cards can be clicked.'));
  const row = el('div', 'opts');
  for (const o of free) {
    const b = el('button', isPassy(o) || o.role === 'yes' ? 'main' : '', esc(optText(o)));
    b.onclick = () => choose(o);
    hookOptZoom(b, o);
    row.appendChild(b);
  }
  box.appendChild(row);
  if (withObj.length) {
    const det = el('details'); if (!free.length && withObj.length <= 6) det.open = true;
    det.appendChild(el('summary', '', `All options (${d.options.length})`));
    const all = el('div', 'all');
    for (const o of d.options) { const b = el('button', '', esc(optText(o))); b.onclick = () => choose(o); hookOptZoom(b, o); all.appendChild(b); }
    det.appendChild(all);
    box.appendChild(det);
  }
}
function renderMulligan(box, d) {
  box.appendChild(el('div', 'mull-hint', 'Click up to two cards in your hand to set aside, then confirm.'));
  const sel = [...mullSel].sort((a, b) => a - b).join(',');
  const match = d.options.find(o => [...o.objs].sort((a, b) => a - b).join(',') === sel);
  const b = el('button', 'primary', mullSel.size ? `Set aside ${mullSel.size} and draw` : 'Keep this hand');
  b.disabled = !match;
  b.onclick = () => match && choose(match);
  box.appendChild(b);
}
function renderChain() {
  const z = $('#chain-box'); z.innerHTML = '';
  z.appendChild(el('div', 'box-head', `Chain${F.chain.length ? ` (${F.chain.length})` : ' — empty'}`));
  const name = id => { const o = findObj(id); return o ? (o.name || 'a facedown card') : 'something'; };
  for (const c of [...F.chain].reverse()) {
    const cd = cards[c.card];
    const it = el('div', 'chain-item ' + (c.ctrl === 1 ? 'me' : 'opp') + (c.resolving ? ' resolving' : ''));
    const th = el('div', 'thumb'); if (cd) th.style.backgroundImage = `url("${cd.img}")`;
    it.appendChild(th);
    const tg = c.targets.length ? ` → ${c.targets.map(name).join(', ')}` : '';
    it.appendChild(el('div', '', `<div class="ci-t">${esc(cd ? cd.name : 'Hidden')}</div><div class="ci-s">${RP ? `P${relSeat(c.ctrl)}'s` : c.ctrl === 1 ? 'Yours' : 'Opponent\'s'} ${esc(c.kind)}${c.resolving ? ' · resolving' : c.finalized ? '' : ' · pending'}${esc(tg)}</div>`));
    if (cd) hookZoom(it, { id: -1, card: c.card, name: cd.name, type: cd.type });
    it.onmouseenter = () => c.targets.forEach(t => document.querySelectorAll(`[data-id="${t}"]`).forEach(e => e.classList.add('targeted')));
    it.onmouseleave = () => document.querySelectorAll('.targeted').forEach(e => e.classList.remove('targeted'));
    z.appendChild(it);
  }
}
function findObj(id) {
  if (!F) return null;
  for (const list of [F.board, F.hand, F.runes, F.legends, F.players.me.trash, F.players.opp.trash, F.players.me.banishment, F.players.opp.banishment])
    for (const o of list) if (o.id === id) return o;
  for (const b of F.battlefields) for (const o of b.facedown) if (o.id === id) return o;
  for (const p of [F.players.me, F.players.opp]) if (p.champion_zone && p.champion_zone.id === id) return p.champion_zone;
  return null;
}
function flashChanges(prev, byAgent) {
  const old = new Map();
  for (const o of prev.board) old.set(o.id, o);
  for (const b of prev.battlefields) for (const o of b.facedown) old.set(o.id, o);
  const cls = byAgent ? 'flash' : 'flash-me';
  const check = o => {
    const p = old.get(o.id);
    if (!p || p.loc !== o.loc || p.exhausted !== o.exhausted || p.might !== o.might || p.damage !== o.damage)
      document.querySelectorAll(`.card[data-id="${o.id}"]`).forEach(e => { e.classList.remove('flash', 'flash-me'); void e.offsetWidth; e.classList.add(cls); });
  };
  F.board.forEach(check);
  F.battlefields.forEach(b => b.facedown.forEach(check));
}

// ---------------- interaction ----------------
function onCardClick(o, acts, ev) {
  const d = F.decision;
  if (d.kind === 'mulligan') {
    if (mullSel.has(o.id)) mullSel.delete(o.id); else if (mullSel.size < 2) mullSel.add(o.id);
    render(); return;
  }
  // An answer that names exactly this object: take it at once.
  const exact = acts.filter(a => a.objs.length === 1 && a.role === 'object');
  if (exact.length === 1 && acts.length === 1) return choose(exact[0]);
  // Several destinations for one card: let the board zones be clicked too.
  const dests = acts.filter(a => a.dest);
  if (dests.length > 1) markDrops(dests);
  openMenu(ev, o, acts);
}
function markDrops(opts) {
  for (const o of opts) {
    const z = o.dest === 'base' ? $('#me-base') : document.querySelector(`.bf[data-slot="${o.dest.slice(2)}"]`);
    if (!z) continue;
    z.classList.add('drop');
    z.onclick = ev => { ev.stopPropagation(); choose(o); };
  }
}
function clearDrops() { document.querySelectorAll('.drop').forEach(z => { z.classList.remove('drop'); z.onclick = null; }); }
function openMenu(ev, o, acts) {
  $('#zoom').classList.add('hidden');
  const m = $('#menu'); m.innerHTML = '';
  m.appendChild(el('div', 'mh', esc(o.name || 'Card')));
  for (const a of acts) {
    let t = optText(a);
    const b = el('button', '', esc(t)); b.onclick = e => { e.stopPropagation(); choose(a); }; m.appendChild(b);
  }
  const c = el('button', 'ghost', 'Cancel'); c.onclick = e => { e.stopPropagation(); closeMenu(); }; m.appendChild(c);
  m.classList.remove('hidden');
  const r = m.getBoundingClientRect();
  m.style.left = Math.min(ev.clientX + 8, innerWidth - r.width - 8) + 'px';
  m.style.top = Math.min(ev.clientY + 8, innerHeight - r.height - 8) + 'px';
}
function closeMenu() { $('#menu').classList.add('hidden'); clearDrops(); }
document.addEventListener('click', e => { if (!$('#menu').contains(e.target)) closeMenu(); });
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') { closeMenu(); $('#modal').classList.add('hidden'); }
  if (RP && !$('#game').classList.contains('hidden') && !/^(INPUT|SELECT|BUTTON)$/.test(document.activeElement?.tagName || '')) {
    const k = { ArrowRight: e.shiftKey ? 'turn+' : 1, ArrowLeft: e.shiftKey ? 'turn-' : -1, PageDown: 10, PageUp: -10, Home: 'start', End: 'end' }[e.key];
    if (k != null) { e.preventDefault(); rpStop(); rpStep(k); return; }
    if (e.key === ' ') { e.preventDefault(); rpToggle(); return; }
  }
  if ((e.key === ' ' || e.key === 'Enter') && F?.decision && !busy && !pumping && document.activeElement === document.body) {
    const pass = options().find(isPassy); if (pass) { e.preventDefault(); choose(pass); }
  }
});

// ---------------- zoom ----------------
function hookZoom(node, o) {
  node.addEventListener('mouseenter', ev => showZoom(o, ev));
  node.addEventListener('mousemove', ev => placeZoom(ev));
  node.addEventListener('mouseleave', () => $('#zoom').classList.add('hidden'));
}
function hookOptZoom(node, opt) {
  const id = opt.objs[0];
  const c = opt.named_card != null ? opt.named_card : null;
  if (id != null) {
    node.addEventListener('mouseenter', ev => { const o = findObj(id); if (o) showZoom(o, ev); document.querySelectorAll(`[data-id="${id}"]`).forEach(e => e.classList.add('targeted')); });
    node.addEventListener('mouseleave', () => { $('#zoom').classList.add('hidden'); document.querySelectorAll('.targeted').forEach(e => e.classList.remove('targeted')); });
  } else if (c != null) hookZoom(node, { id: -1, card: c, name: cards[c]?.name });
}
function showZoom(o, ev) {
  const z = $('#zoom');
  const c = o.card != null ? cards[o.card] : null;
  if (!c && !o.name) { z.classList.add('hidden'); return; }
  let h = c ? `<img src="${esc(c.img)}" alt="">` : '';
  h += `<div class="zt">${esc(c ? c.name : o.name)}</div>`;
  if (c) h += `<div class="zs">${esc(c.type)}${c.domains.length ? ' · ' + esc(c.domains.join(' / ')) : ''}${c.type !== 'Rune' && c.type !== 'Battlefield' && c.type !== 'Legend' ? ` · cost ${c.energy}${c.power ? ' + ' + c.power + '◆' : ''}` : ''}${c.type === 'Unit' ? ` · Might ${c.might}` : ''}${c.tags.length ? ' · ' + esc(c.tags.join(', ')) : ''}</div>`;
  if (c && c.text) h += `<div class="zx">${esc(c.text)}</div>`;
  const live = [];
  if (o.type === 'Unit' && o.might != null && o.id >= 0) live.push(`Might now ${o.might}${o.damage ? `, ${o.damage} damage` : ''}`);
  if (o.keywords && o.keywords.length) live.push(o.keywords.join(' · '));
  if (o.exhausted) live.push('Exhausted');
  if (live.length) h += `<div class="zl">${esc(live.join(' — '))}</div>`;
  z.innerHTML = h; z.classList.remove('hidden'); placeZoom(ev);
}
function placeZoom(ev) {
  const z = $('#zoom'); const w = 300, h = z.offsetHeight || 480;
  let x = ev.clientX + 24; if (x + w > innerWidth - 8) x = ev.clientX - w - 24;
  let y = Math.min(Math.max(8, ev.clientY - h / 2), innerHeight - h - 8);
  z.style.left = x + 'px'; z.style.top = y + 'px';
}

// ---------------- modal / game over ----------------
function openModal(title, list) {
  $('#modal-title').textContent = title;
  const b = $('#modal-body'); b.innerHTML = '';
  if (!list.length) b.appendChild(el('div', 'dim', 'Empty.'));
  for (const o of list) b.appendChild(cardEl(o, { noStats: true }));
  $('#modal').classList.remove('hidden');
}
$('#modal-close').onclick = () => $('#modal').classList.add('hidden');
$('#modal').onclick = e => { if (e.target.id === 'modal') $('#modal').classList.add('hidden'); };
function showGameOver() {
  const g = $('#gameover');
  const w = F.winner;
  const title = F.error ? 'Game ended' : w === 1 ? 'Victory' : w === 2 ? 'Defeat' : 'Draw';
  g.innerHTML = `<div class="go"><h2 class="${w === 1 ? 'win' : 'loss'}">${title}</h2>` +
    `<div>${F.players.me.score} – ${F.players.opp.score} after ${F.turn.number} turns</div>` +
    (F.error ? `<div class="err">${esc(F.error)}</div>` : '') +
    `<div class="row" style="justify-content:center"><button class="ghost" id="go-board">View board</button><button class="primary" id="go-new">New game</button></div></div>`;
  g.classList.remove('hidden');
  $('#go-board').onclick = () => g.classList.add('hidden');
  $('#go-new').onclick = () => { g.classList.add('hidden'); showSetup(); };
}

initSetup();

// ---------------- replays ----------------
// A recorded search game (server replay.py): every frame is the position before decision i, from the
// watched seat, with the decision's annotation (who decides, what was chosen, the search's visit
// shares and value). The board is never clickable; arrows / the slider move through the game.
async function initReplays() {
  try {
    const r = await api('/api/replays');
    const sel = $('#r-list');
    sel.innerHTML = r.replays.map(e => {
      const d = e.decks ? e.decks.join(' vs ') : 'decks unknown';
      const ss = e.human_seat ? `you are P${e.human_seat}` : e.search_seat ? `search on P${e.search_seat}` : '';
      return `<option value="${esc(e.id)}"${e.decks ? '' : ' disabled'}>${esc(e.name)} · seed ${e.seed} · ${esc(d)} · ${ss} · ${e.done ? `P${e.winner} won` : 'unfinished'}, ${e.turns} turns</option>`;
    }).join('') || '<option disabled>No recorded games found</option>';
  } catch (e) { $('#r-err').textContent = 'Replays: ' + e.message; }
}
$('#r-watch').onclick = async () => {
  $('#r-err').textContent = '';
  const id = $('#r-list').value;
  if (!id) return;
  $('#r-watch').disabled = true; $('#r-watch').textContent = 'Loading…';
  try {
    const j = await api('/api/replay/open', { id, seat: +$('#r-seat').value, both: $('#r-both').checked });
    rpStop(); resetGame(); showGame(); enqueue(j.frames);
  } catch (e) { $('#r-err').textContent = e.message; }
  $('#r-watch').disabled = false; $('#r-watch').textContent = 'Watch';
};

function rpTarget(k, i) {
  const n = RP.n, ts = F.meta.turn_starts || [];
  if (k === 'start') return 0;
  if (k === 'end') return n;
  if (k === 'turn+') { const t = ts.find(x => x[0] > i); return t ? t[0] : n; }
  if (k === 'turn-') { const prev = ts.filter(x => x[0] < i); return prev.length ? prev[prev.length - 1][0] : 0; }
  return i + k;
}
// Moves are queued as a target position: key presses made while a frame is loading add up
// instead of being dropped, and only the final position is fetched.
let rpWant = null;
async function rpGoto(i) {
  if (!RP) return;
  rpWant = Math.max(0, Math.min(RP.n, i));
  if (rpBusy) return;
  rpBusy = true;
  try {
    while (RP && rpWant != null && rpWant !== RP.i) {
      const want = rpWant;
      const j = await api('/api/replay/goto', { i: want });
      apply(j.frames[0]);
    }
  } catch (e) { toast(e.message, 'opp'); rpStop(); }
  rpWant = null;
  rpBusy = false;
}
function rpStep(k) { if (RP) rpGoto(rpTarget(k, rpWant ?? RP.i)); }
function rpStop() { clearTimeout(rpTimer); rpTimer = 0; if (RP) renderPrompt(); }
function rpToggle() {
  if (rpTimer) return rpStop();
  const tick = async () => {
    if (!RP || RP.done) return rpStop();
    await rpGoto(RP.i + 1);
    if (rpTimer) rpTimer = setTimeout(tick, rpSpeed);
  };
  rpTimer = setTimeout(tick, 10);
  renderPrompt();
}
async function rpView(seat, both) {
  try { const j = await api('/api/replay/view', { seat, both }); resetGame(); apply(j.frames[0]); }
  catch (e) { toast(e.message, 'opp'); }
}
let rpSpeed = 900;
function renderReplay(box) {
  box.className = 'replay';
  const m = F.meta, a = RP;
  const turn = (m.turn_starts || []).filter(x => x[0] <= a.i).pop();
  const speeds = [[1800, 'slow'], [900, 'normal'], [400, 'fast'], [120, 'very fast']];
  box.innerHTML = `<div class="rp-head"><span class="t">Replay</span><span class="dim">decision ${a.i} / ${a.n}${turn ? ` · turn ${turn[1]}` : ''}</span></div>` +
    `<input type="range" id="rp-slider" min="0" max="${a.n}" value="${a.i}">` +
    `<div class="rp-ctl">` +
    `<button data-k="start" title="Start (Home)">⏮</button><button data-k="turn-" title="Previous turn (Shift+Left)">«</button>` +
    `<button data-k="-1" title="Previous decision (Left)">‹</button>` +
    `<button id="rp-play" class="${rpTimer ? 'on' : ''}" title="Play / pause (Space)">${rpTimer ? '⏸' : '▶'}</button>` +
    `<button data-k="1" title="Next decision (Right)">›</button><button data-k="turn+" title="Next turn (Shift+Right)">»</button>` +
    `<button data-k="end" title="End (End)">⏭</button>` +
    `<select id="rp-speed" title="Autoplay speed">${speeds.map(([v, t]) => `<option value="${v}"${v === rpSpeed ? ' selected' : ''}>${t}</option>`).join('')}</select>` +
    `</div>` +
    `<div class="rp-view"><label>Watch as <select id="rp-seat"><option value="1"${mySeat === 1 ? ' selected' : ''}>${esc(seatLabel(1))}</option>` +
    `<option value="2"${mySeat === 2 ? ' selected' : ''}>${esc(seatLabel(2))}</option></select></label>` +
    `<label><input type="checkbox" id="rp-both"${F.opp_hand_cards ? ' checked' : ''}> both hands</label>` +
    `<button class="ghost" id="rp-exit">Exit</button></div>` +
    `<div class="rp-dec" id="rp-dec"></div>`;
  box.querySelectorAll('.rp-ctl [data-k]').forEach(b => b.onclick = () => { rpStop(); const k = b.dataset.k; rpStep(/^-?\d+$/.test(k) ? +k : k); b.blur(); });
  $('#rp-play').onclick = e => { e.target.blur(); rpToggle(); };
  $('#rp-speed').onchange = e => { rpSpeed = +e.target.value; e.target.blur(); };
  $('#rp-slider').onchange = e => { rpStop(); rpGoto(+e.target.value); e.target.blur(); };
  $('#rp-seat').onchange = e => { rpStop(); rpView(+e.target.value, $('#rp-both').checked); };
  $('#rp-both').onchange = e => { rpStop(); rpView(mySeat, e.target.checked); };
  $('#rp-exit').onclick = () => { rpStop(); RP = null; showSetup(); };
  const dz = $('#rp-dec');
  if (a.done) {
    dz.innerHTML = `<div class="who">Game over — ${a.winner ? esc(seatLabel(a.winner)) + ' wins' : 'no winner'}</div>` +
      `<div class="dim">P${mySeat} ${F.players.me.score} – P${3 - mySeat} ${F.players.opp.score}${F.error ? ' · ' + esc(F.error) : ''}</div>`;
    return;
  }
  const sg = v => (v >= 0 ? '+' : '') + v.toFixed(2);
  const win = v => `${Math.round((v + 1) * 50)}%`;
  let h = `<div><span class="who s${a.seat}">${esc(seatLabel(a.seat))}</span> decides <span class="dim">· ${esc(a.kind || '')} · ${a.n_options} options</span></div>` +
    `<div class="dim">${a.question && a.kind !== 'main phase action' ? esc(a.question) + ' · ' : ''}${a.human ? '' : `value ${sg(a.value)} (its win chance ≈ ${win(a.value)})`}</div>` +
    `<div class="chose">→ ${esc(optText(a.chosen))}</div>`;
  if (a.searched) {
    h += `<div class="dim">search: ${a.visits} visits; top options by visits</div>`;
    for (const t of a.top) {
      h += `<div class="rp-bar${t.chosen ? ' chosen' : ''}"><div class="fill" style="width:${(100 * t.share).toFixed(1)}%"></div>` +
        `<span>${(100 * t.share).toFixed(0)}% · ${esc(optText(t))}</span></div>`;
    }
    if (a.override) {
      h += `<div class="rp-ovr">Search overrode the network: alone it would have played “${esc(optText(a.override))}” ` +
        `(prior ${(100 * a.override.prior).toFixed(0)}%). Search Q ${sg(a.override.q)} for that vs ${sg(a.q)} for the chosen move.</div>`;
    }
  } else if (!a.human) h += `<div class="dim">network, greedy: ${(100 * a.chosen.prior).toFixed(0)}% on its pick</div>`;
  dz.innerHTML = h;
}
function markChosen() {
  document.querySelectorAll('.targeted').forEach(e => e.classList.remove('targeted'));
  if (!RP || RP.done || !RP.chosen) return;
  for (const id of RP.chosen.objs) document.querySelectorAll(`[data-id="${id}"]`).forEach(e => e.classList.add('targeted'));
}
