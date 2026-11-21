// playweb.cpp -- one human-vs-network game for the browser UI (tools/play_web/README.md).
//
// C ABI for ctypes (tools/play_web/playweb.py). Built in the WSL build copy only
// (playweb.cmake, build.sh), like research/card_repr/selfplay/rbenv: it #includes that
// directory's rbenv.cpp so the network's observation is exactly the one it was trained on
// (rbe::encode / Encoder of decision_export.cpp). Nothing here is part of src/.
//
// Two faces of one game:
//   * the network's seat: pw_observe fills the rbenv observation buffers (n_envs = 1);
//   * the human's seat: pw_view returns JSON with only what that player may know,
//     built from the engine's per-observer queries: ObserverView (board, chain,
//     piles, known hand cards), knowledge::identityVisible (which card an object is),
//     describePrompt (the decision's words and options) and LogEvent::visibleTo
//     (which log lines the player may read).
// Rune payments are answered automatically for both seats (autopay), as in training.


#include "research/card_repr/selfplay/rbenv/rbenv.cpp"


#include "engine/effect_executor.h"
#include "engine/knowledge_tracker.h"
#include "engine/observer_view.h"
#include "engine/prompt_description.h"

#include <nlohmann/json.hpp>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace pw {
using namespace riftbound;
using json = nlohmann::json;

struct LogLine { long seq; int level; std::string msg; bool p1, p2; };

struct Game {
    rbe::Handle h;                 // n = 1; envs[0] is the game
    PlayerId human = PlayerId::Player1;
    bool done = false;
    int winner = 0;                // 1 / 2 seat, 0 none
    std::string error;
    std::vector<LogLine> log;
    long seq = 0;
    boost::signals2::scoped_connection log_conn;
    bool autopay = true;
};

int seatOf(PlayerId p) { return p == PlayerId::Player1 ? 1 : p == PlayerId::Player2 ? 2 : 0; }

// rbe::startGame without the restart machinery, and with the log listener attached
// before the game begins (so the setup lines are kept).
void start(Game& g, int64_t seed) {
    using namespace rbe;
    Env& e = g.h.envs[0];
    e.hist.clear();
    e.game_seed = seed;
    const uint64_t es = engineSeedOf(static_cast<uint64_t>(seed));
    e.bus = std::make_unique<EventBus>();
    e.engine = std::make_unique<GameEngine>(*g_db, *e.bus, *g_reg);
    e.length = 0; e.n_ext = 0; e.need = false;
    for (auto& t : e.mtrack) t = MoveTrack{};
    g.log_conn = e.bus->on_log.connect([&g](const LogEvent& ev) {
        g.log.push_back(LogLine{g.seq++, static_cast<int>(ev.level), ev.message,
                                ev.visibleTo(PlayerId::Player1), ev.visibleTo(PlayerId::Player2)});
    });
    e.engine->beginGame(g.h.da, g.h.db, es);
}

// rbe::advance for one game that is not restarted when it ends.
void advance(Game& g) {
    using namespace rbe;
    Env& e = g.h.envs[0];
    try {
        while (true) {
            const StepResult& sr = e.engine->currentStep();
            if (sr.kind == StepKind::Done) {
                g.done = true;
                g.winner = winnerCode(e.engine->stepResult().winner);
                return;
            }
            if (e.length >= kDecisionCap) { g.done = true; g.error = "decision cap"; return; }
            const int n = static_cast<int>(sr.legal.size());
            const bool rune = sr.prompt.kind == PromptKind::RuneForEnergy || sr.prompt.kind == PromptKind::RuneForPower;
            const PlayerId dec = deciderOf(e.engine->state(), sr.legal, sr.prompt);
            if (n <= 1 || (g.autopay && rune)) {
                ++e.length; ++g.h.c.auto_;
                applyTracked(e, sr, dec, 0);
                continue;
            }
            encode(g.h, e, sr, dec);   // the network's observation (also sets e.seat / e.dec)
            e.need = true;
            return;
        }
    } catch (const std::exception& ex) {
        g.done = true;
        g.error = std::string("engine error: ") + ex.what();
    }
}

// ---- the human's view -------------------------------------------------------

const char* phaseName(TurnPhase p) { return toString(p); }

std::string locString(const std::optional<LocationId>& loc) {
    if (!loc) return "";
    if (const auto* b = std::get_if<BattlefieldLocation>(&*loc)) return "bf" + std::to_string(b->id.raw());
    return "base";
}

struct ViewBuilder {
    const GameState& s;
    PlayerId me;
    const CardDB& db;
    ObserverView ov;
    json cards = json::object();   // card id -> printed facts, for every card named in the view

    ViewBuilder(const GameState& st, PlayerId m, const CardDB& d)
        : s(st), me(m), db(d), ov(st, m, d, rbe::g_reg) {}

    int rel(PlayerId p) const { return p == PlayerId::None ? 0 : p == me ? 1 : 2; }

    void noteCard(CardDefId c) {
        if (!c.valid()) return;
        const std::string k = std::to_string(c.raw());
        if (cards.contains(k)) return;
        const CardDef& d = db.get(c);
        json j;
        j["name"] = d.name;
        j["img"] = d.image_url;
        j["type"] = toString(d.card_type);
        j["energy"] = d.energy_cost;
        j["power"] = d.power_cost;
        j["might"] = d.might;
        j["text"] = d.ability_text;
        json doms = json::array();
        for (Domain dm : d.domains) doms.push_back(toString(dm));
        j["domains"] = doms;
        json tags = json::array();
        for (const auto& t : d.tags) tags.push_back(t);
        j["tags"] = tags;
        cards[k] = j;
    }

    // The card of an object as this player may see it (kInvalidId: face down / hidden).
    CardDefId visibleCard(GameObjectId id) const {
        if (!s.objectExists(id)) return kInvalidId;
        if (!knowledge::identityVisible(s, id, me)) return kInvalidId;
        return s.getObject(id).card_def_id;
    }

    json object(GameObjectId id) {
        json j;
        j["id"] = id.raw();
        if (!s.objectExists(id)) return j;
        const GameObject& o = s.getObject(id);
        const bool known = knowledge::identityVisible(s, id, me);
        const CardDefId c = known ? o.card_def_id : kInvalidId;
        noteCard(c);
        j["card"] = c.valid() ? json(c.raw()) : json(nullptr);
        j["name"] = known ? std::string(o.name.get()) : std::string();
        j["type"] = known ? toString(o.card_type) : "";
        j["owner"] = rel(o.owner);
        j["ctrl"] = rel(o.controller);
        j["token"] = known && o.isToken();
        j["facedown"] = o.is_hidden || o.zone == ZoneType::FacedownZone;
        return j;
    }

    json permanent(const PermanentFacts& f) {
        json j = object(f.id);
        j["might"] = f.might;
        j["damage"] = f.damage;
        j["lethal"] = f.lethal;
        j["exhausted"] = f.exhausted;
        j["stunned"] = f.stunned;
        j["buffs"] = f.buffs;
        j["empowered"] = f.empowered;
        j["loc"] = locString(f.location);
        j["ctrl"] = rel(f.controller);
        j["attached_to"] = f.attached_to.valid() ? json(f.attached_to.raw()) : json(nullptr);
        j["combat"] = f.designation == CombatDesignation::Attacker ? "attacker"
                    : f.designation == CombatDesignation::Defender ? "defender" : "";
        json kws = json::array();
        for (uint32_t k = 0; k < 32; ++k)
            if (f.keywords & (1u << k)) {
                std::string name = toString(static_cast<Keyword>(k));
                if (static_cast<Keyword>(k) == Keyword::Assault && f.assault) name += " " + std::to_string(f.assault);
                if (static_cast<Keyword>(k) == Keyword::Shield && f.shield) name += " " + std::to_string(f.shield);
                if (static_cast<Keyword>(k) == Keyword::Deflect && f.deflect) name += " " + std::to_string(f.deflect);
                kws.push_back(name);
            }
        j["keywords"] = kws;
        return j;
    }

    json player(PlayerId p) {
        const PlayerState& P = s.players[playerIndex(p)];
        const ResourceFacts r = ov.resources(p);
        json j;
        j["score"] = P.score;
        j["xp"] = P.xp;
        j["hand_count"] = ov.handSize(p);
        j["deck_count"] = ov.deckSize(p);
        j["rune_deck_count"] = ov.runeDeckSize(p);
        j["energy"] = r.pool.energy + r.pool.unit_energy + r.pool.spell_energy + r.pool.showdown_energy;
        int power = r.pool.universal_power + r.pool.spell_power + r.pool.gear_power;
        for (int d = 0; d < static_cast<int>(Domain::Count); ++d) power += r.pool.power[d];
        j["power"] = power;
        j["ready_runes"] = r.ready_runes;
        j["exhausted_runes"] = r.exhausted_runes;
        json trash = json::array(), ban = json::array();
        for (GameObjectId id : ov.publicPile(p, ZoneType::Trash)) trash.push_back(object(id));
        for (GameObjectId id : ov.publicPile(p, ZoneType::Banishment)) ban.push_back(object(id));
        j["trash"] = trash;
        j["banishment"] = ban;
        j["champion_zone"] = P.champion_zone.valid() && s.objectExists(P.champion_zone) ? object(P.champion_zone) : json(nullptr);
        return j;
    }

    json build(const rbe::Env& e, const Game& g, long log_since) {
        json v;
        v["me"] = seatOf(me);
        const TimingFacts t = ov.timing();
        v["turn"] = {{"number", t.turn_number}, {"phase", phaseName(t.phase)}, {"turn_player", rel(t.turn_player)},
                     {"showdown", t.ns == NeutralShowdownState::Showdown}, {"closed", t.oc == OpenClosedState::Closed},
                     {"priority", t.priority_holder ? rel(*t.priority_holder) : 0},
                     {"focus", t.focus_holder ? rel(*t.focus_holder) : 0},
                     {"victory_score", s.effectiveVictoryScore()}};
        const PlayerId opp = me == PlayerId::Player1 ? PlayerId::Player2 : PlayerId::Player1;
        v["players"] = {{"me", player(me)}, {"opp", player(opp)}};

        json hand = json::array();
        for (GameObjectId id : ov.ownHand()) hand.push_back(object(id));
        v["hand"] = hand;
        json oh = json::array();
        for (const CardKnowledge& k : ov.knownInHand(opp)) {
            if (!k.identity_known) continue;
            noteCard(k.card_def_id);
            oh.push_back(k.card_def_id.raw());
        }
        v["opp_known_hand"] = oh;

        json board = json::array(), runes = json::array(), legends = json::array();
        for (const PermanentFacts& f : ov.board()) {
            if (!s.objectExists(f.id)) continue;
            const GameObject& o = s.getObject(f.id);
            if (o.card_type == CardType::Battlefield) continue;
            json j = permanent(f);
            if (o.isRune()) runes.push_back(j);
            else if (o.card_type == CardType::Legend) legends.push_back(j);
            else board.push_back(j);
        }
        v["board"] = board;
        v["runes"] = runes;
        v["legends"] = legends;

        json bfs = json::array();
        for (const BattlefieldFacts& b : ov.battlefields()) {
            json j;
            j["slot"] = b.id.raw();
            noteCard(b.card);
            j["card"] = b.card.valid() ? json(b.card.raw()) : json(nullptr);
            j["controller"] = b.controller ? rel(*b.controller) : 0;
            j["contested"] = b.contested;
            j["showdown"] = b.showdown_in_progress;
            j["combat"] = b.combat_in_progress;
            j["staged"] = b.showdown_staged || b.combat_staged;
            j["attacker"] = b.attacker ? rel(*b.attacker) : 0;
            json fd = json::array();
            if (b.id.raw() >= 0 && static_cast<size_t>(b.id.raw()) < s.battlefields.size())
                for (GameObjectId id : s.battlefields[b.id].facedown) fd.push_back(object(id));
            j["facedown"] = fd;
            const BattlefieldState& bs = s.battlefields[b.id];
            j["contributed_by"] = rel(bs.contributed_by);
            j["scored_by_me"] = s.players[playerIndex(me)].battlefields_scored_this_turn.count(b.id) > 0;
            bfs.push_back(j);
        }
        v["battlefields"] = bfs;

        json chain = json::array();
        for (const ChainItemFacts& c : ov.chain()) {
            json j;
            j["pos"] = c.position;
            j["resolving"] = c.resolving;
            j["ctrl"] = rel(c.controller);
            noteCard(c.card);
            j["card"] = c.card.valid() ? json(c.card.raw()) : json(nullptr);
            static const char* kinds[] = {"spell", "activated", "triggered", "keyword", "permanent", "equip"};
            j["kind"] = kinds[static_cast<int>(c.kind)];
            j["finalized"] = c.finalized;
            j["source"] = c.source.valid() ? json(c.source.raw()) : json(nullptr);
            json tg = json::array();
            for (auto id : c.targets) tg.push_back(id.raw());
            for (auto id : c.chosen) tg.push_back(id.raw());
            j["targets"] = tg;
            chain.push_back(j);
        }
        v["chain"] = chain;

        // The decision, if it is this player's.
        v["decision"] = nullptr;
        v["waiting_for"] = 0;
        if (!g.done && e.need) {
            const StepResult& sr = e.engine->currentStep();
            v["waiting_for"] = rel(e.dec);
            if (e.dec == me) {
                const PromptDescription d = describePrompt(s, sr.prompt, sr.legal, me, db);
                json dj;
                dj["kind"] = toString(d.kind);
                dj["question"] = d.question;
                dj["text"] = d.text;
                dj["at_finalization"] = d.at_finalization;
                if (d.has_source) { noteCard(d.source_card); dj["source_card"] = d.source_card.valid() ? json(d.source_card.raw()) : json(nullptr); }
                if (sr.prompt.source.valid() && s.objectExists(sr.prompt.source) && knowledge::identityVisible(s, sr.prompt.source, me))
                    dj["source_obj"] = sr.prompt.source.raw();
                json opts = json::array();
                for (const OptionDescription& od : d.options) {
                    const Intent& in = sr.legal.at(od.index);
                    json oj;
                    oj["index"] = od.index;
                    oj["role"] = toString(od.role);
                    oj["type"] = toString(od.type);
                    oj["text"] = od.text;
                    oj["label"] = od.label;
                    oj["value"] = od.value;
                    if (od.named_card.valid()) { noteCard(od.named_card); oj["named_card"] = od.named_card.raw(); }
                    for (CardDefId c : od.cards) noteCard(c);
                    // Objects the option names, for the board to highlight: only ones this
                    // player can see as objects (its own cards, public zones, the board).
                    std::vector<GameObjectId> ids;
                    auto add = [&](GameObjectId id) {
                        if (!id.valid() || !s.objectExists(id)) return;
                        const GameObject& o = s.getObject(id);
                        const bool on_board = o.location.has_value() || o.zone == ZoneType::FacedownZone;
                        if (!on_board && !knowledge::identityVisible(s, id, me)) return;
                        for (auto x : ids) if (x == id) return;
                        ids.push_back(id);
                    };
                    add(in.card);
                    add(in.ability_source);
                    for (auto id : in.units_to_move) add(id);
                    for (auto id : in.chosen_objects) add(id);
                    for (auto id : in.targets) add(id);
                    for (auto id : in.cards_to_mulligan) add(id);
                    for (const auto& da : in.damage_assignments) add(da.target_unit);
                    json oi = json::array();
                    for (auto id : ids) oi.push_back(id.raw());
                    oj["objs"] = oi;
                    std::string dest = locString(in.play_location);
                    if (dest.empty()) dest = locString(in.move_destination);
                    if (dest.empty() && in.chosen_battlefield.valid()) dest = "bf" + std::to_string(in.chosen_battlefield.raw());
                    oj["dest"] = dest;
                    opts.push_back(oj);
                }
                dj["options"] = opts;
                v["decision"] = dj;
            }
        }
        v["game_over"] = g.done;
        v["winner"] = g.done ? (g.winner == 0 ? 0 : (g.winner == seatOf(me) ? 1 : 2)) : 0;
        v["error"] = g.error;

        json log = json::array();
        for (const LogLine& l : g.log) {
            if (l.seq < log_since) continue;
            if (!(me == PlayerId::Player1 ? l.p1 : l.p2)) continue;
            log.push_back({{"seq", l.seq}, {"lvl", l.level}, {"msg", l.msg}});
        }
        v["log"] = log;
        v["log_next"] = g.seq;
        v["cards"] = cards;
        return v;
    }
};

int copyOut(const std::string& s, char* out, int cap) {
    const int n = static_cast<int>(s.size());
    if (n + 1 > cap) return -(n + 1);
    std::memcpy(out, s.data(), n);
    out[n] = 0;
    return n;
}

}  // namespace pw

extern "C" {

// subjects / vocab: as rbenv_init (the network's card keys and vocabulary).
int pw_init(const char* subjects, const char* vocab, char* err, int errlen) {
    return rbenv_init(subjects, vocab, "", err, errlen);
}

void pw_dims(int32_t* out) { rbenv_dims(out); }

// deck_p1 / deck_p2: deck list paths. human_seat 1 or 2. Returns a handle or null (err).
void* pw_create(const char* deck_p1, const char* deck_p2, int human_seat, int64_t seed, char* err, int errlen) {
    using namespace rbe;
    try {
        if (!g_inited) throw std::runtime_error("pw_init first");
        auto g = std::make_unique<pw::Game>();
        g->human = human_seat == 2 ? PlayerId::Player2 : PlayerId::Player1;
        Handle& h = g->h;
        h.n = 1; h.seed = seed; h.next_seed = seed + 1;
        h.deck_a = deck_p1; h.deck_b = deck_p2;
        h.autopay = true;
        h.da = DeckValidator::loadFromDeckList(h.deck_a, *g_db);
        h.db = DeckValidator::loadFromDeckList(h.deck_b, *g_db);
        h.envs.resize(1);
        Env& e = h.envs[0];
        e.idx = 0;
        e.glob.assign(G, 0); e.tok.assign(T * F, 0); e.act.assign(A * K, 0);
        e.tmask.assign(T, 0); e.amask.assign(A, 0);
        pw::start(*g, seed);
        pw::advance(*g);
        return g.release();
    } catch (const std::exception& ex) {
        std::snprintf(err, errlen, "%s", ex.what());
        return nullptr;
    }
}

void pw_destroy(void* p) { delete static_cast<pw::Game*>(p); }

// out: [done, winner seat, need, decider seat, n_legal, prompt kind]
void pw_status(void* p, int32_t* out) {
    auto& g = *static_cast<pw::Game*>(p);
    const auto& e = g.h.envs[0];
    out[0] = g.done; out[1] = g.winner; out[2] = !g.done && e.need;
    out[3] = (!g.done && e.need) ? e.seat : 0;
    out[4] = (!g.done && e.need) ? e.n_legal : 0;
    out[5] = (!g.done && e.need) ? e.pk : 0;
}

// The network's observation of the pending decision (rbenv layout, n_envs = 1).
void pw_observe(void* p, int32_t* glob, int32_t* tok, uint8_t* tmask, int32_t* act, uint8_t* amask,
                int32_t* pk, int32_t* psrc) {
    using namespace rbe;
    auto& g = *static_cast<pw::Game*>(p);
    const Env& e = g.h.envs[0];
    std::memcpy(glob, e.glob.data(), G * 4);
    std::memcpy(tok, e.tok.data(), T * F * 4);
    std::memcpy(tmask, e.tmask.data(), T);
    std::memcpy(act, e.act.data(), A * K * 4);
    std::memcpy(amask, e.amask.data(), A);
    *pk = e.pk;
    *psrc = e.psrc;
}

// Apply choice k (index into the engine's legal list) for whoever decides, then run to the
// next decision that is not answered automatically. 0 ok, -1 bad choice (err).
int pw_step(void* p, int k, char* err, int errlen) {
    using namespace rbe;
    auto& g = *static_cast<pw::Game*>(p);
    Env& e = g.h.envs[0];
    if (g.done || !e.need) { std::snprintf(err, errlen, "no decision pending"); return -1; }
    if (k < 0 || k >= e.n_legal) { std::snprintf(err, errlen, "choice %d outside 0..%d", k, e.n_legal - 1); return -1; }
    e.need = false;
    ++e.length; ++e.n_ext;
    try { applyTracked(e, e.engine->currentStep(), e.dec, k); }
    catch (const std::exception& ex) {
        g.done = true; g.error = std::string("engine error: ") + ex.what();
        return 0;
    }
    pw::advance(g);
    return 0;
}

// JSON of what `seat` (1 / 2) may know now, with the log lines from `log_since` that it may
// read. Returns the length, or -(needed bytes) when cap is too small.
int pw_view(void* p, int seat, long log_since, char* out, int cap) {
    auto& g = *static_cast<pw::Game*>(p);
    const auto& e = g.h.envs[0];
    std::string s;
    try {
        pw::ViewBuilder vb(e.engine->state(), seat == 2 ? riftbound::PlayerId::Player2 : riftbound::PlayerId::Player1,
                           *rbe::g_db);
        s = vb.build(e, g, log_since).dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
    } catch (const std::exception& ex) {
        s = nlohmann::json({{"view_error", ex.what()}}).dump();
    }
    return pw::copyOut(s, out, cap);
}

}  // extern "C"
