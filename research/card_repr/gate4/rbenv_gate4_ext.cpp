
// ===========================================================================
// Gate 4 (research/card_repr/gate4): counterfactual playouts.
// (Source of truth: research/card_repr/gate4/rbenv_gate4_ext.cpp, appended to
// selfplay/rbenv/rbenv.cpp.)
// A position = (game seed, every applied choice index since the game began:
// Env::hist). rbenv_po_create replays it in a fresh engine; it must stop at a
// top-level main-phase decision (GameEngine::isResumableDecision). Each
// variant = one edit of a copy of that GameState through the engine's own
// editing path (StateEditor / the GameObject fields the engine's own effects
// write), then a probe engine resumes it (resumeFromSnapshot runs the Main
// Phase's settle + Cleanup, which recomputes derived state) and must arrive at
// a main-phase decision of the same player: the post-Cleanup state and that
// decision are the variant's root. Playout jobs (variant, seed): the root is
// resumed with the engine RNG seeded from the seed and both Main Decks
// reshuffled with it (hidden to both players); every decision is external
// (the network plays both seats); autopay as usual.
// ===========================================================================
#include "io/state_editor.h"
#include <deque>
#include <random>
#include <sstream>

namespace rbe {

struct PoVariant {
    std::string edit;
    GameState st;          // post-Cleanup state at the root decision
    StepResult step;       // the root decision
    std::vector<int32_t> glob, tok, act;
    std::vector<uint8_t> tmask, amask;
    std::vector<int64_t> tok_obj;
    int seat = 0, pk = 0, psrc = -1, nlegal = 0;
    std::string info;      // JSON: describeState + legal option strings
};

struct Po {
    int64_t game_seed = 0;
    GameState s0;
    StepResult step0;
    std::mt19937_64 rng0;
    std::vector<PoVariant> vars;
    std::deque<std::pair<int, int64_t>> jobs;
};

std::string jesc(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '\n') { o += "\\n"; continue; }
        if (c == '"' || c == '\\') o += '\\';
        o += c;
    }
    return o;
}

const char* zoneName(ZoneType z) {
    switch (z) {
        case ZoneType::MainDeck: return "deck";
        case ZoneType::RuneDeck: return "runedeck";
        case ZoneType::Hand: return "hand";
        case ZoneType::Base: return "base";
        case ZoneType::Trash: return "trash";
        case ZoneType::Banishment: return "banish";
        case ZoneType::ChampionZone: return "champ";
        case ZoneType::LegendZone: return "legend";
        case ZoneType::BattlefieldZone: return "bf";
        case ZoneType::Chain: return "chain";
        case ZoneType::FacedownZone: return "facedown";
    }
    return "?";
}

int pcode(PlayerId p) { return p == PlayerId::Player1 ? 1 : (p == PlayerId::Player2 ? 2 : 0); }

// JSON description of a state: what the Python side needs to choose edits.
std::string describeState(const GameState& s, const StepResult& sr) {
    std::ostringstream o;
    o << "{\"turn_player\":" << pcode(s.turn.turn_player) << ",\"decider\":" << pcode(sr.perspective)
      << ",\"turn\":" << s.turn.turn_number << ",\"phase\":" << static_cast<int>(s.turn.phase)
      << ",\"game_over\":" << (s.game_over ? 1 : 0);
    o << ",\"players\":[";
    for (int i = 0; i < 2; ++i) {
        const auto& P = s.players[i];
        o << (i ? "," : "") << "{\"score\":" << P.score << ",\"cards_played\":" << P.cards_played_this_turn
          << ",\"hand\":" << P.hand.size() << ",\"deck\":" << P.main_deck.size() << "}";
    }
    o << "],\"bfs\":[";
    for (size_t b = 0; b < s.battlefields.size(); ++b) {
        const auto& bf = s.battlefields[b];
        const bool has = s.objectExists(bf.card_object_id);
        std::string name = has ? std::string(s.getObject(bf.card_object_id).name.get()) : std::string();
        o << (b ? "," : "") << "{\"slot\":" << b << ",\"name\":\"" << jesc(name) << "\",\"obj\":" << bf.card_object_id.raw()
          << ",\"controller\":" << (bf.controller ? pcode(*bf.controller) : 0) << ",\"facedown\":[";
        for (size_t j = 0; j < bf.facedown.size(); ++j) o << (j ? "," : "") << bf.facedown[j].raw();
        o << "]";
        if (has) {
            const auto& cc = s.getObject(bf.card_object_id).card_counters;
            for (int p = 0; p < 2; ++p) {
                auto it = cc.find("__starspring_turn_p" + std::to_string(p));
                o << ",\"ss_used_p" << (p + 1) << "\":" << ((it != cc.end() && it->second == s.turn.turn_number + 1) ? 1 : 0);
            }
        }
        o << "}";
    }
    o << "],\"objs\":[";
    bool first = true;
    for (const auto& [id, ob] : s.objects) {
        const ZoneType z = ob.zone;
        if (!(z == ZoneType::Hand || z == ZoneType::Base || z == ZoneType::BattlefieldZone || z == ZoneType::FacedownZone ||
              z == ZoneType::ChampionZone)) continue;
        if (!(ob.isUnit() || z == ZoneType::Hand || z == ZoneType::FacedownZone || ob.isGear())) continue;
        int loc = -1;
        if (auto bid = ob.battlefieldId()) loc = static_cast<int>(bid->raw());
        if (loc < 0 && ob.hidden_at.valid()) loc = static_cast<int>(ob.hidden_at.raw());
        o << (first ? "" : ",") << "{\"id\":" << id.raw() << ",\"name\":\"" << jesc(ob.name) << "\",\"key\":\""
          << jesc(cardKey(ob.card_def_id, &ob)) << "\",\"owner\":" << pcode(ob.owner) << ",\"ctrl\":" << pcode(ob.controller)
          << ",\"zone\":\"" << zoneName(z) << "\",\"loc\":" << loc << ",\"unit\":" << (ob.isUnit() ? 1 : 0)
          << ",\"might\":" << ob.current_might << ",\"base_might\":" << ob.base_might << ",\"damage\":" << ob.damage_marked
          << ",\"assault\":" << ob.assault_value << ",\"shield\":" << ob.shield_value
          << ",\"exhausted\":" << (ob.is_exhausted ? 1 : 0) << ",\"stunned\":" << (ob.is_stunned ? 1 : 0)
          << ",\"hidden\":" << (ob.is_hidden ? 1 : 0) << ",\"token\":" << (ob.isToken() ? 1 : 0)
          << ",\"combat\":" << static_cast<int>(ob.combat_designation) << "}";
        first = false;
    }
    o << "],\"legal\":[";
    for (size_t a = 0; a < sr.legal.size(); ++a) o << (a ? "," : "") << "\"" << jesc(sr.legal[a].describe()) << "\"";
    o << "]}";
    return o.str();
}

// One edit on a copy of the state. Ops separated by ';', fields by ' ':
//   none | kw_give <id> A|S <v> | kw_remove <id> A|S | might <id> <delta> | to_base <id> | to_bf <id> <slot>
//   unhide <id> | played <n> | ss <slot> <0|1>   (Star Spring stamp of the turn player)
void applyEdit(GameState& s, const std::string& spec) {
    std::stringstream all(spec);
    std::string one;
    StateEditor ed;
    while (std::getline(all, one, ';')) {
        std::stringstream ss(one);
        std::string op;
        ss >> op;
        if (op.empty() || op == "none") continue;
        auto obj = [&]() -> GameObject& {
            uint32_t id = 0;
            ss >> id;
            const auto gid = GameObjectId::fromRaw(id);
            if (!s.objectExists(gid)) throw std::runtime_error("edit: no object " + std::to_string(id));
            return s.getObject(gid);
        };
        if (op == "kw_give" || op == "kw_remove") {
            GameObject& o = obj();
            std::string k;
            ss >> k;
            const bool A = k == "A";
            int& val = A ? o.assault_value : o.shield_value;
            int& tmp = A ? o.temp_assault_value : o.temp_shield_value;
            if (op == "kw_give") {   // as EffectExecutor::giveTemporaryKeyword ("this turn")
                int v = 0;
                ss >> v;
                o.keywords.set(A ? Keyword::Assault : Keyword::Shield);
                o.temp_keywords.set(A ? Keyword::Assault : Keyword::Shield);
                val += v;
                tmp += v;
            } else {                 // loses it this turn: the expiration step restores val - tmp
                if (val <= 0) throw std::runtime_error("edit: kw_remove on a unit without it");
                tmp -= val;
                val = 0;
            }
            o.recomputeMight();
        } else if (op == "might") {
            GameObject& o = obj();
            int d = 0;
            ss >> d;
            o.base_might = std::max(0, o.base_might + d);
            o.recomputeMight();
        } else if (op == "to_base") {
            GameObject& o = obj();
            const auto r = ed.moveObject(s, o.id, o.controller, ZoneType::Base);
            if (!r.ok) throw std::runtime_error("edit: " + r.error);
        } else if (op == "to_bf") {
            GameObject& o = obj();
            int slot = 0;
            ss >> slot;
            const auto r = ed.moveObjectToBattlefield(s, o.id, BattlefieldId::fromRaw(slot));
            if (!r.ok) throw std::runtime_error("edit: " + r.error);
        } else if (op == "unhide") {
            GameObject& o = obj();
            const GameObjectId id = o.id;
            const auto r = ed.moveObject(s, id, o.owner, ZoneType::Hand);
            if (!r.ok) throw std::runtime_error("edit: " + r.error);
            GameObject& o2 = s.getObject(id);
            o2.is_hidden = false;
            o2.hidden_at = BattlefieldId{};
            o2.hidden_on_turn = -1;
        } else if (op == "played") {
            int n = 0;
            ss >> n;
            s.player(s.turn.turn_player).cards_played_this_turn = n;
        } else if (op == "ss") {
            int slot = 0, used = 0;
            ss >> slot >> used;
            auto& bfo = s.getObject(s.battlefields.at(slot).card_object_id);
            const std::string key = "__starspring_turn_p" + std::to_string(playerIndex(s.turn.turn_player));
            bfo.card_counters[key] = used ? s.turn.turn_number + 1 : 0;
        } else {
            throw std::runtime_error("edit: unknown op " + op);
        }
    }
}

void poStart(Handle& h, Env& e) {
    Po& po = *h.po;
    e.engine.reset();
    e.opp.reset();
    e.length = 0; e.n_ext = 0; e.need = false;
    for (auto& t : e.mtrack) t = MoveTrack{};
    e.hist.clear();
    if (po.jobs.empty()) { e.idle = true; e.var = -1; return; }
    e.idle = false;
    const auto job = po.jobs.front();
    po.jobs.pop_front();
    e.var = job.first; e.po_seed = job.second; e.game_seed = job.second;
    const PoVariant& pv = po.vars.at(job.first);
    GameState st = pv.st;
    std::mt19937_64 sh(static_cast<uint64_t>(job.second) * 0x9E3779B97F4A7C15ull + 17);
    for (auto& P : st.players) std::shuffle(P.main_deck.begin(), P.main_deck.end(), sh);
    e.bus = std::make_unique<EventBus>();
    e.engine = std::make_unique<GameEngine>(*g_db, *e.bus, *g_reg);
    e.engine->resumeFromSnapshot(std::move(st), std::mt19937_64(static_cast<uint64_t>(job.second) ^ 0x5DEECE66Dull), pv.step);
}

}  // namespace rbe

extern "C" {

// History of env i (for a later rbenv_po_create). Returns its length; copies up to cap.
int rbenv_hist(void* p, int i, int64_t* seed_out, int32_t* out, int cap) {
    auto& e = static_cast<rbe::Handle*>(p)->envs[i];
    *seed_out = e.game_seed;
    const int n = static_cast<int>(e.hist.size());
    for (int k = 0; k < n && k < cap; ++k) out[k] = e.hist[k];
    return n;
}

// 1 if env i waits at a resumable (top-level main-phase) decision.
int rbenv_resumable(void* p, int i) {
    auto& e = static_cast<rbe::Handle*>(p)->envs[i];
    if (!e.need || !e.engine) return 0;
    return riftbound::GameEngine::isResumableDecision(e.engine->state(), e.engine->currentStep().legal) ? 1 : 0;
}

// Token -> object id of env i's current observation (T int64, -1 none).
void rbenv_tok_obj(void* p, int i, int64_t* out) {
    auto& e = static_cast<rbe::Handle*>(p)->envs[i];
    for (int t = 0; t < rbe::T; ++t) out[t] = t < static_cast<int>(e.tok_obj.size()) ? e.tok_obj[t] : -1;
}

void* rbenv_po_create(int n_envs, const char* deck_a, const char* deck_b, int64_t game_seed,
                      const int32_t* hist, int nhist, char* err, int errlen) {
    using namespace rbe;
    try {
        if (!g_inited) throw std::runtime_error("rbenv_init first");
        auto h = std::make_unique<Handle>();
        h->n = n_envs; h->seed = game_seed; h->next_seed = game_seed;
        h->deck_a = deck_a; h->deck_b = deck_b; h->autopay = true;
        h->da = DeckValidator::loadFromDeckList(h->deck_a, *g_db);
        h->db = DeckValidator::loadFromDeckList(h->deck_b, *g_db);
        Env b;   // replay
        b.game_seed = game_seed;
        b.bus = std::make_unique<EventBus>();
        b.engine = std::make_unique<GameEngine>(*g_db, *b.bus, *g_reg);
        b.engine->beginGame(h->da, h->db, engineSeedOf(static_cast<uint64_t>(game_seed)));
        for (int k = 0; k < nhist; ++k) {
            const StepResult& sr = b.engine->currentStep();
            if (sr.kind == StepKind::Done) throw std::runtime_error("replay: game ended early");
            applyTracked(b, sr, deciderOf(b.engine->state(), sr.legal, sr.prompt), hist[k]);
        }
        const StepResult& sr = b.engine->currentStep();
        if (sr.kind != StepKind::NeedDecision || !GameEngine::isResumableDecision(b.engine->state(), sr.legal))
            throw std::runtime_error("replay: not at a resumable main-phase decision");
        auto po = std::make_shared<Po>();
        po->game_seed = game_seed;
        po->s0 = b.engine->state();
        po->step0 = sr;
        po->rng0 = b.engine->rngState();
        h->po = po;
        h->envs.resize(n_envs);
        for (int i = 0; i < n_envs; ++i) {
            Env& e = h->envs[i];
            e.idx = i; e.idle = true;
            e.glob.assign(G, 0); e.tok.assign(T * F, 0); e.act.assign(A * K, 0);
            e.tmask.assign(T, 0); e.amask.assign(A, 0);
        }
        return h.release();
    } catch (const std::exception& ex) {
        std::snprintf(err, errlen, "%s", ex.what());
        return nullptr;
    }
}

// Description of the base position (JSON). Returns the length needed.
int rbenv_po_describe(void* p, char* out, int cap) {
    auto& po = *static_cast<rbe::Handle*>(p)->po;
    const std::string s = rbe::describeState(po.s0, po.step0);
    std::snprintf(out, cap, "%s", s.c_str());
    return static_cast<int>(s.size());
}

// Adds a variant. Returns its index, or -1 (err) if the edit fails or the edited
// position does not arrive at a main-phase decision of the same player.
int rbenv_po_add_variant(void* p, const char* edit, char* err, int errlen) {
    using namespace rbe;
    auto& h = *static_cast<Handle*>(p);
    auto& po = *h.po;
    try {
        GameState s1 = po.s0;
        applyEdit(s1, edit);
        Env pr;
        pr.bus = std::make_unique<EventBus>();
        pr.engine = std::make_unique<GameEngine>(*g_db, *pr.bus, *g_reg);
        try {
            pr.engine->resumeFromSnapshot(std::move(s1), po.rng0, po.step0);
        } catch (const std::logic_error& ex) {
            if (std::string(ex.what()).find("did not reach") == std::string::npos) throw;
            // the edit changed the options: the engine's own decision after Cleanup is the root
        }
        const StepResult& sr = pr.engine->currentStep();
        if (sr.kind != StepKind::NeedDecision) throw std::runtime_error("variant: game over after the edit");
        if (sr.perspective != po.step0.perspective) throw std::runtime_error("variant: another player decides");
        if (!GameEngine::isResumableDecision(pr.engine->state(), sr.legal))
            throw std::runtime_error("variant: not a main-phase decision after Cleanup");
        PoVariant v;
        v.edit = edit;
        v.st = pr.engine->state();
        v.step = sr;
        pr.glob.assign(G, 0); pr.tok.assign(T * F, 0); pr.act.assign(A * K, 0);
        pr.tmask.assign(T, 0); pr.amask.assign(A, 0);
        const PlayerId dec = deciderOf(pr.engine->state(), sr.legal, sr.prompt);
        encode(h, pr, sr, dec);
        v.glob = pr.glob; v.tok = pr.tok; v.act = pr.act; v.tmask = pr.tmask; v.amask = pr.amask;
        v.tok_obj = pr.tok_obj; v.seat = pr.seat; v.pk = pr.pk; v.psrc = pr.psrc; v.nlegal = pr.n_legal;
        v.info = describeState(v.st, v.step);
        po.vars.push_back(std::move(v));
        return static_cast<int>(po.vars.size()) - 1;
    } catch (const std::exception& ex) {
        std::snprintf(err, errlen, "%s", ex.what());
        return -1;
    }
}

int rbenv_po_variant_info(void* p, int v, char* out, int cap) {
    const auto& s = static_cast<rbe::Handle*>(p)->po->vars.at(v).info;
    std::snprintf(out, cap, "%s", s.c_str());
    return static_cast<int>(s.size());
}

// Root observation of variant v (one row of observe()); meta = seat, pk, psrc, nlegal.
void rbenv_po_root_obs(void* p, int v, int32_t* glob, int32_t* tok, uint8_t* tmask, int32_t* act, uint8_t* amask,
                       int32_t* meta, int64_t* tok_obj) {
    using namespace rbe;
    const auto& x = static_cast<Handle*>(p)->po->vars.at(v);
    std::memcpy(glob, x.glob.data(), G * 4);
    std::memcpy(tok, x.tok.data(), T * F * 4);
    std::memcpy(tmask, x.tmask.data(), T);
    std::memcpy(act, x.act.data(), A * K * 4);
    std::memcpy(amask, x.amask.data(), A);
    meta[0] = x.seat; meta[1] = x.pk; meta[2] = x.psrc; meta[3] = x.nlegal;
    for (int t = 0; t < T; ++t) tok_obj[t] = t < static_cast<int>(x.tok_obj.size()) ? x.tok_obj[t] : -1;
}

// Queue playout jobs (variant, seed) and start the idle envs.
void rbenv_po_start(void* p, const int32_t* vars, const int64_t* seeds, int n) {
    using namespace rbe;
    auto& h = *static_cast<Handle*>(p);
    for (int k = 0; k < n; ++k) h.po->jobs.emplace_back(vars[k], seeds[k]);
    for (auto& e : h.envs)
        if (!e.need && !e.engine) { e.idle = false; advance(h, e); }
}

// Envs waiting on a decision.
int rbenv_po_busy(void* p) {
    auto& h = *static_cast<rbe::Handle*>(p);
    int n = 0;
    for (auto& e : h.envs) n += e.need ? 1 : 0;
    return n;
}

// Playout results: rows of (var, seed, winner, turns, error 0/1). Returns the count and removes them.
int rbenv_po_results(void* p, int64_t* out, int cap) {
    auto& h = *static_cast<rbe::Handle*>(p);
    int n = 0;
    for (const auto& r : h.results) {
        if (n >= cap) break;
        int64_t* row = out + 5 * n;
        row[0] = r.var; row[1] = r.po_seed; row[2] = r.winner; row[3] = r.turns; row[4] = r.error.empty() ? 0 : 1;
        ++n;
    }
    h.results.erase(h.results.begin(), h.results.begin() + n);
    return n;
}

}  // extern "C"
