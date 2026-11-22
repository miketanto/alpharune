// rbenv.cpp -- vectorised Riftbound environment for no-search self-play
// (research/card_repr/SELFPLAY.md, "Python interface contract"). C ABI for
// ctypes (rbenv.py). Research code: built only in the WSL build copy
// (rbenv.cmake, build.sh), never part of src/.
//
// Reuses research/card_repr/smoke/data/src/decision_export.cpp verbatim (its
// Encoder, card keys, registerVariant, makeAgent) by #including it with its
// main() renamed. Observations are therefore the exporter's, written as int32
// fields instead of JSON (layout: README.txt).

#define main decision_export_main
#include "research/card_repr/smoke/data/src/decision_export.cpp"
#undef main

#include <cstring>
#include <memory>
#include <mutex>
#include <set>

namespace rbe {
using namespace riftbound;
using namespace riftbound::probe;
using namespace riftbound::probe::dx;

// Schema v2 (gate 1, 2026-11-19): T 64 -> 96 (pile tokens, known opponent hand),
// F 17 -> 24 (memory flags, effective cost, affordable), K 46 -> 49 (move consequences).
// Schema v3 (gate 3): K 49 -> 56 (optional-cost consequences, act 49..55).
// Schema v4 (gate 3 completion): K 56 -> 61. TriggerOrder options carry the
// chain item they order (33 ci) and its source (1 src); MoveJoin options carry
// the move's destination (30, 46..48), the units already moving (56..59,
// leading unit first; 60 their count) and "move no more units" as dec (39).
constexpr int kSchemaVersion = 4;
constexpr int G = 72, T = 96, F = 24, A = 96, K = 61;
constexpr int kMaxList = 4, kMaxMu = 2, kMaxDm = 6, kMaxChosen = 4;
constexpr int kDecisionCap = 20000;

std::unordered_map<std::string, int> g_vocab;
int g_vocab_other = 0;
std::string g_variant;
bool g_inited = false;

struct Counters { long trunc_tok = 0, trunc_act = 0, trunc_list = 0, unk_key = 0, ext = 0, auto_ = 0;
                  long trig_opts = 0, trig_mapped = 0, trig_amb = 0; };   // v4: TriggerOrder mapping

// RBENV_LOG_UNK=1: print each unknown card key once (stderr), to find keys the vocabulary lacks.
std::mutex g_unk_mu;
std::set<std::string> g_unk_seen;
int vocabOf(const std::string& k, Counters& c) {
    auto it = g_vocab.find(k);
    if (it != g_vocab.end()) return it->second;
    ++c.unk_key;
    static const bool log = std::getenv("RBENV_LOG_UNK") != nullptr;
    if (log) {
        std::lock_guard<std::mutex> lk(g_unk_mu);
        if (g_unk_seen.insert(k).second) std::fprintf(stderr, "[rbenv] unknown card key: %s\n", k.c_str());
    }
    return g_vocab_other;
}
}  // namespace rbe
#include "research/card_repr/selfplay/rbenv/history.h"
namespace rbe {

struct Result { int env; int64_t game_seed; int winner; int length; int n_ext; int turns; std::string error;
                int var = -1; int64_t po_seed = 0; };   // gate 4 playouts

struct Env {
    int idx = 0;
    int64_t game_seed = 0;
    std::unique_ptr<EventBus> bus;
    std::unique_ptr<GameEngine> engine;
    std::unique_ptr<AgentInterface> opp;
    int length = 0, n_ext = 0;
    bool need = false;
    int seat = 0, pk = 0, psrc = -1, n_legal = 0;
    std::vector<int32_t> glob, tok, act;
    std::vector<uint8_t> tmask, amask;
    MoveTrack mtrack[3];      // v4: per decider (PlayerId), the Standard Move being built
    PlayerId dec = PlayerId::None;
    std::vector<int32_t> hist;   // gate 4: every applied choice index since game start (replay)
    std::vector<int64_t> tok_obj; // gate 4: object id per token (-1 chain / pile)
    std::vector<HistEv> hev;      // history.h: public action history (external decisions, both seats)
    std::vector<int32_t> histobs; // history.h: the decider's encoded history at the pending decision
    int var = -1; int64_t po_seed = 0; bool idle = false;   // gate 4 playout job
};

// v4: every applied choice goes through here so MoveJoin knows its move.
void applyTracked(Env& e, const StepResult& sr, PlayerId dec, int k) {
    if (k >= 0 && k < static_cast<int>(sr.legal.size()))
        noteChoice(e.mtrack[static_cast<int>(dec) % 3], sr.prompt, sr.legal[k]);
    e.hist.push_back(k);
    HistPending hp = histBegin(e.engine->state(), sr, dec, k);   // before: sr is the engine's current step
    e.engine->applyChoice(k);
    histEnd(e.engine->state(), hp, e.hev);
}

struct Po;   // gate 4 playout state (end of file)
struct Handle;
void poStart(Handle& h, Env& e);
struct Handle {
    int n = 0;
    int64_t seed = 0, next_seed = 0;
    std::string deck_a, deck_b, opponent;
    bool autopay = true;
    DeckSubmission da, db;
    std::vector<Env> envs;
    std::vector<Result> results;
    Counters c;
    std::string err;
    std::shared_ptr<Po> po;   // gate 4: playout handle
};

// Thread-safe core of encode (search M0): writes one observation of decision `sr` of
// state `s` (its engine `eng`, may be null) for `dec` into caller-owned buffers, counting
// into caller-owned counters. Touches no Env / Handle.
struct ObsOut {
    int32_t* glob; int32_t* tok; int32_t* act; uint8_t* tmask; uint8_t* amask; int64_t* tok_obj;
    int psrc = -1;
};
void encodeRaw(const GameState& s, const GameEngine* eng, const MoveTrack* mt, const StepResult& sr,
               PlayerId dec, Counters& c, ObsOut& o) {
    Encoder enc(s, dec, eng);
    enc.setMove(mt);
    enc.build();
    std::fill(o.glob, o.glob + G, 0);
    std::fill(o.tok, o.tok + T * F, 0);
    std::fill(o.act, o.act + A * K, 0);
    std::fill(o.tmask, o.tmask + T, 0);
    std::fill(o.amask, o.amask + A, 0);
    const auto& legal = sr.legal;
    const Prompt& p = sr.prompt;
    enc.prepare(legal, p);   // v4: TriggerOrder option -> chain item
    c.trig_opts += p.kind == PromptKind::TriggerOrder ? static_cast<long>(legal.size()) : 0;
    c.trig_mapped += enc.trigMapped();
    c.trig_amb += enc.trigAmbiguous();
    // Actions first (same ref() order as Encoder::encode/action, so appended
    // tokens get the same indices as in the exporter's JSON).
    const int na = static_cast<int>(legal.size());
    if (na > A) c.trunc_act += na - A;
    for (int a = 0; a < na; ++a) {
        const Intent& i = legal[a];
        int32_t buf[K];
        for (int k = 0; k < K; ++k) buf[k] = -1;
        buf[0] = static_cast<int>(i.type);
        buf[1] = i.card.valid() ? enc.ref(i.card) : -1;
        const uint64_t trig = enc.trigItem(i, p);   // v4
        if (buf[1] < 0 && trig) { const auto sc = enc.trigSource(trig); if (sc.valid()) buf[1] = enc.ref(sc); }
        if (i.ability_source.valid()) { buf[2] = enc.ref(i.ability_source); buf[3] = i.ability_index; }
        auto list = [&](int off, int cap, int cnt_slot, const std::vector<GameObjectId>& v) {
            int j = 0;
            for (auto id : v) { const int r = enc.ref(id); if (j < cap) buf[off + j] = r; ++j; }
            if (j > cap) c.trunc_list++;
            buf[cnt_slot] = j;
        };
        list(4, kMaxList, 41, i.targets);
        list(8, kMaxList, 42, i.chosen_objects);
        list(12, kMaxList, 43, i.units_to_move);
        list(16, kMaxMu, 44, i.cards_to_mulligan);
        {
            int j = 0;
            for (const auto& d : i.damage_assignments) {
                const int r = enc.ref(d.target_unit);
                if (j < kMaxDm) { buf[18 + 2 * j] = r; buf[19 + 2 * j] = d.damage; }
                ++j;
            }
            if (j > kMaxDm) c.trunc_list++;
            buf[45] = j;
        }
        buf[30] = enc.actionDst(i, p);
        {
            const auto q = enc.consequences(buf[30]);
            buf[46] = q[0]; buf[47] = q[1]; buf[48] = q[2];
        }
        buf[31] = i.chosen_value.has_value() ? 1 : 0;
        buf[32] = i.chosen_value.has_value() ? static_cast<int32_t>(*i.chosen_value) : 0;
        if (i.chosen_value.has_value() && p.kind == PromptKind::ChainItem)
            buf[33] = enc.itemRef(static_cast<uint64_t>(*i.chosen_value));
        else if (trig)
            buf[33] = enc.itemRef(trig);
        buf[34] = i.chosen_card_def.valid() ? vocabOf(cardKey(i.chosen_card_def, nullptr), c) : 0;
        buf[35] = i.chosen_battlefield.valid() ? static_cast<int32_t>(i.chosen_battlefield.raw()) : -1;
        buf[36] = static_cast<int>(i.play_source);
        buf[37] = i.use_alt_play_cost ? 1 : 0;
        buf[38] = i.pay_location_cost ? 1 : 0;
        buf[39] = (i.declines || Encoder::moveJoinStop(i, p)) ? 1 : 0;
        buf[40] = i.type == IntentType::PlayFirstDecision ? (i.choose_first ? 1 : 0) : -1;
        {   // schema v3 (gate 3): optional-cost consequences, Encoder::optCost
            const auto q = enc.optCost(i, p);
            for (int j = 0; j < 7; ++j) buf[49 + j] = q[j];
        }
        {   // schema v4: MoveJoin, the units already moving (leading unit first) and their count
            const auto mg = enc.moveGroup(p);
            int j = 0;
            for (auto id : mg) { const int r = enc.ref(id); if (j < kMaxList) buf[56 + j] = r; ++j; }
            if (j > kMaxList) c.trunc_list++;
            buf[60] = p.kind == PromptKind::MoveJoin ? j : -1;
        }
        if (a < A) { std::memcpy(&o.act[a * K], buf, sizeof buf); o.amask[a] = 1; }
    }
    o.psrc = p.source.valid() ? enc.ref(p.source) : -1;
    const auto g = enc.global(p);
    for (int k = 0; k < G && k < static_cast<int>(g.size()); ++k) o.glob[k] = static_cast<int32_t>(g[k]);
    // tokens
    const auto& all = enc.all();
    std::unordered_map<uint64_t, int> obj_idx;
    for (size_t t = 0; t < all.size(); ++t) if (all[t].kind != 'k') obj_idx[all[t].obj] = static_cast<int>(t);
    auto idx = [&](uint64_t r) { auto it = obj_idx.find(r); return it == obj_idx.end() ? -1 : it->second; };
    const int nt = static_cast<int>(all.size());
    if (nt > T) c.trunc_tok += nt - T;
    std::fill(o.tok_obj, o.tok_obj + T, int64_t{-1});
    for (int t = 0; t < nt && t < T; ++t) {
        const Tok& tk = all[t];
        o.tok_obj[t] = (tk.kind == 'c' || tk.kind == 'b') ? static_cast<int64_t>(tk.obj) : -1;
        int32_t* row = &o.tok[t * F];
        for (int k = 0; k < F; ++k) row[k] = 0;
        row[0] = tk.kind == 'c' ? 1 : (tk.kind == 'b' ? 2 : (tk.kind == 'k' ? 3 : 4));
        row[1] = vocabOf(tk.key, c);
        int k = 2;
        for (long x : tk.f) if (k < F) row[k++] = static_cast<int32_t>(x);
        if (tk.kind == 'c') {
            row[16] = tk.refs.empty() ? -1 : idx(tk.refs[0]);
            // v2 [17..23]: play ordinal, hidden this turn, eff energy, eff power, affordable, assault, shield
            for (size_t j = 0; j < tk.x.size() && 17 + j < static_cast<size_t>(F); ++j)
                row[17 + j] = static_cast<int32_t>(tk.x[j]);
        } else if (tk.kind == 'k') {
            // 2..9 the 8 fields, 10 source token, 11..14 chosen tokens, 15 n chosen, 16 unused
            row[10] = idx(tk.refs[0]);
            for (int j = 11; j <= 14; ++j) row[j] = -1;
            int n = 0;
            for (size_t j = 1; j < tk.refs.size(); ++j, ++n) if (n < kMaxChosen) row[11 + n] = idx(tk.refs[j]);
            if (n > kMaxChosen) c.trunc_list++;
            row[15] = n;
            // v2 [16] additional cost paid, [17] source play ordinal, [18] triggering subject token
            row[16] = tk.x.size() > 0 ? static_cast<int32_t>(tk.x[0]) : 0;
            row[17] = tk.x.size() > 1 ? static_cast<int32_t>(tk.x[1]) : 0;
            row[18] = tk.subj ? idx(tk.subj) : -1;
        }
        o.tmask[t] = 1;
    }
    // references to truncated tokens become -1
    if (nt > T) {
        auto fix = [](int32_t& v) { if (v >= T) v = -1; };
        for (int t = 0; t < T; ++t) {
            int32_t* row = &o.tok[t * F];
            if (row[0] == 1) fix(row[16]);
            else if (row[0] == 3) { fix(row[10]); for (int j = 11; j <= 14; ++j) fix(row[j]); fix(row[18]); }
        }
        for (int a = 0; a < na && a < A; ++a) {
            int32_t* r = &o.act[a * K];
            for (int j : {1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24, 26, 28, 33, 56, 57, 58, 59}) fix(r[j]);
        }
        if (o.psrc >= T) o.psrc = -1;
    }
}

void encode(Handle& h, Env& e, const StepResult& sr, PlayerId dec) {
    e.tok_obj.resize(T);
    ObsOut o{e.glob.data(), e.tok.data(), e.act.data(), e.tmask.data(), e.amask.data(), e.tok_obj.data()};
    encodeRaw(e.engine->state(), e.engine.get(), &e.mtrack[static_cast<int>(dec) % 3], sr, dec, h.c, o);
    e.psrc = o.psrc;
    const Prompt& p = sr.prompt;
    const int na = static_cast<int>(sr.legal.size());
    e.seat = dec == PlayerId::Player1 ? 1 : 2;
    e.dec = dec;
    e.pk = static_cast<int>(p.kind);
    e.n_legal = na;
    e.histobs.resize(kHistLen * kHistF);
    encodeHist(e.engine->state(), e.hev, dec, h.c, e.histobs.data());
}

void startGame(Handle& h, Env& e) {
    if (h.po) { poStart(h, e); return; }
    e.hist.clear();
    e.hev.clear();
    e.game_seed = h.next_seed++;
    const uint64_t es = engineSeedOf(static_cast<uint64_t>(e.game_seed));
    e.engine.reset();
    e.opp.reset();
    e.bus = std::make_unique<EventBus>();
    e.engine = std::make_unique<GameEngine>(*g_db, *e.bus, *g_reg);
    e.length = 0; e.n_ext = 0; e.need = false;
    for (auto& t : e.mtrack) t = MoveTrack{};
    if (!h.opponent.empty()) {
        GameSpec g; g.deck_a = h.deck_a; g.deck_b = h.deck_b; g.seed = e.game_seed;
        IsMctsAgent* m = nullptr;
        e.opp = makeAgent(h.opponent, g, es, 1, &m);
    }
    e.engine->beginGame(h.da, h.db, es);
}

void finish(Handle& h, Env& e, const std::string& error) {
    Result r;
    r.env = e.idx; r.game_seed = e.game_seed; r.length = e.length; r.n_ext = e.n_ext;
    r.winner = 0; r.turns = 0; r.error = error; r.var = e.var; r.po_seed = e.po_seed;
    if (error.empty()) {
        const GameResult& gr = e.engine->stepResult();
        r.winner = winnerCode(gr.winner);
        r.turns = gr.total_turns;
    }
    h.results.push_back(std::move(r));
}

// Runs env e until an external decision is pending; finished games are
// recorded and restarted.
void advance(Handle& h, Env& e) {
    for (int guard = 0; guard < 1000; ++guard) {
        try {
            if (!e.engine) { startGame(h, e); if (e.idle) return; }
            while (true) {
                const StepResult& sr = e.engine->currentStep();
                if (sr.kind == StepKind::Done) { finish(h, e, ""); break; }
                if (e.length >= kDecisionCap) { finish(h, e, "decision cap"); break; }
                const int n = static_cast<int>(sr.legal.size());
                const bool rune = sr.prompt.kind == PromptKind::RuneForEnergy || sr.prompt.kind == PromptKind::RuneForPower;
                const PlayerId dec = deciderOf(e.engine->state(), sr.legal, sr.prompt);
                if (n <= 1 || (h.autopay && rune)) {
                    ++e.length; ++h.c.auto_;
                    applyTracked(e, sr, dec, 0);
                    continue;
                }
                if (e.opp && dec == PlayerId::Player2) {
                    const Intent pick = e.opp->decide(e.engine->state(), sr.legal, sr.prompt);
                    const int k = indexOf(sr.legal, pick);
                    if (k < 0) throw std::runtime_error("opponent answered an option not in the legal list");
                    ++e.length;
                    applyTracked(e, sr, dec, k);
                    continue;
                }
                encode(h, e, sr, dec);
                e.need = true;
                return;
            }
        } catch (const std::exception& ex) {
            finish(h, e, std::string("error: ") + ex.what());
        }
        e.engine.reset();
        e.opp.reset();
        e.bus.reset();
    }
    throw std::runtime_error("rbenv: 1000 consecutive games ended without a decision");
}

std::vector<std::string> parseJsonStrings(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open vocab " + path);
    std::string s((std::istreambuf_iterator<char>(in)), {});
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '"') continue;
        std::string v;
        for (++i; i < s.size() && s[i] != '"'; ++i) { if (s[i] == '\\') ++i; v += s[i]; }
        out.push_back(v);
    }
    return out;
}

}  // namespace rbe

extern "C" {

// Process-wide setup: card table (+ variants, comma-separated), card keys, vocabulary.
// Returns 0, or -1 with the message in err.
int rbenv_init(const char* subjects, const char* vocab, const char* variant, char* err, int errlen) {
    using namespace rbe;
    try {
        const std::string v = variant ? variant : "";
        if (g_inited) {
            if (v != g_variant) throw std::runtime_error("rbenv: one variant set per process (have '" + g_variant + "')");
            return 0;
        }
        auto& shared = const_cast<openspiel::SharedCards&>(openspiel::sharedCards());
        // variant: one name or a comma-separated list. Each variant has its own class id and
        // card key, so several can be registered at once (a deck may then mix them).
        if (!v.empty()) {
            std::vector<std::string> names;
            std::string cur;
            for (char ch : v + ",") {
                if (ch == ',') { if (!cur.empty()) names.push_back(cur); cur.clear(); }
                else if (ch != ' ') cur += ch;
            }
            for (const auto& n : names) {
                if (!findVariant(n)) throw std::runtime_error("unknown variant " + n);
                registerVariant(shared.registry, n);
            }
            shared.db.buildFromClasses(shared.registry);
        }
        g_reg = &shared.registry;
        g_db = &shared.db;
        loadKeys(subjects);
        const auto words = parseJsonStrings(vocab);
        for (size_t i = 0; i < words.size(); ++i) g_vocab[words[i]] = static_cast<int>(i);
        if (!g_vocab.count("other")) throw std::runtime_error("vocab has no 'other' entry");
        g_vocab_other = g_vocab["other"];
        g_variant = v;
        g_inited = true;
        return 0;
    } catch (const std::exception& ex) {
        std::snprintf(err, errlen, "%s", ex.what());
        return -1;
    }
}

int rbenv_schema_version() { return rbe::kSchemaVersion; }

void rbenv_dims(int32_t* out) {  // G T F A K V
    out[0] = rbe::G; out[1] = rbe::T; out[2] = rbe::F; out[3] = rbe::A; out[4] = rbe::K;
    out[5] = static_cast<int32_t>(rbe::g_vocab.size());
}

void* rbenv_create(int n_envs, const char* deck_a, const char* deck_b, int64_t seed, int autopay,
                   const char* opponent, char* err, int errlen) {
    using namespace rbe;
    try {
        if (!g_inited) throw std::runtime_error("rbenv_init first");
        auto h = std::make_unique<Handle>();
        h->n = n_envs; h->seed = seed; h->next_seed = seed;
        h->deck_a = deck_a; h->deck_b = deck_b;
        h->opponent = opponent ? opponent : "";
        h->autopay = autopay != 0;
        h->da = DeckValidator::loadFromDeckList(h->deck_a, *g_db);
        h->db = DeckValidator::loadFromDeckList(h->deck_b, *g_db);
        h->envs.resize(n_envs);
        for (int i = 0; i < n_envs; ++i) {
            Env& e = h->envs[i];
            e.idx = i;
            e.glob.assign(G, 0); e.tok.assign(T * F, 0); e.act.assign(A * K, 0);
            e.tmask.assign(T, 0); e.amask.assign(A, 0);
        }
        // game seeds: env i starts with seed+i; each new game takes the next unused seed
        for (int i = 0; i < n_envs; ++i) advance(*h, h->envs[i]);
        return h.release();
    } catch (const std::exception& ex) {
        std::snprintf(err, errlen, "%s", ex.what());
        return nullptr;
    }
}

void rbenv_destroy(void* p) { delete static_cast<rbe::Handle*>(p); }

void rbenv_observe(void* p, int32_t* glob, int32_t* tok, uint8_t* tmask, int32_t* act, uint8_t* amask,
                   uint8_t* need, int32_t* seat, int32_t* pk, int32_t* psrc, int32_t* nlegal) {
    using namespace rbe;
    auto& h = *static_cast<Handle*>(p);
    for (int i = 0; i < h.n; ++i) {
        const Env& e = h.envs[i];
        need[i] = e.need;
        seat[i] = e.need ? e.seat : 0;
        pk[i] = e.need ? e.pk : 0;
        psrc[i] = e.need ? e.psrc : -1;
        nlegal[i] = e.need ? e.n_legal : 0;
        std::memcpy(glob + i * G, e.glob.data(), G * 4);
        std::memcpy(tok + static_cast<size_t>(i) * T * F, e.tok.data(), T * F * 4);
        std::memcpy(tmask + i * T, e.tmask.data(), T);
        std::memcpy(act + static_cast<size_t>(i) * A * K, e.act.data(), A * K * 4);
        std::memcpy(amask + i * A, e.amask.data(), A);
    }
}

// history.h: dims {kHistLen, kHistF}; the pending decision's history per env (zeros without one).
void rbenv_hist_dims(int32_t* out) { out[0] = rbe::kHistLen; out[1] = rbe::kHistF; }
void rbenv_observe_hist(void* p, int32_t* hist) {
    using namespace rbe;
    auto& h = *static_cast<Handle*>(p);
    const size_t n1 = static_cast<size_t>(kHistLen) * kHistF;
    for (int i = 0; i < h.n; ++i) {
        const Env& e = h.envs[i];
        if (e.need && e.histobs.size() == n1) std::memcpy(hist + i * n1, e.histobs.data(), n1 * 4);
        else std::memset(hist + i * n1, 0, n1 * 4);
    }
}

// choices[i] for envs with need. Returns 0, or -1 (err) on a bad choice: the
// env's game is then ended with an error result and restarted.
int rbenv_step(void* p, const int32_t* choices, char* err, int errlen) {
    using namespace rbe;
    auto& h = *static_cast<Handle*>(p);
    int rc = 0;
    try {
        for (int i = 0; i < h.n; ++i) {
            Env& e = h.envs[i];
            if (!e.need) continue;
            e.need = false;
            const int c = choices[i];
            if (c < 0 || c >= e.n_legal || c >= A) {
                std::snprintf(err, errlen, "env %d: choice %d outside 0..%d", i, c, std::min(e.n_legal, A) - 1);
                rc = -1;
                finish(h, e, "error: bad choice from caller");
                e.engine.reset(); e.opp.reset(); e.bus.reset();
            } else {
                ++e.length; ++e.n_ext; ++h.c.ext;
                try { applyTracked(e, e.engine->currentStep(), e.dec, c); }
                catch (const std::exception& ex) {
                    finish(h, e, std::string("error: ") + ex.what());
                    e.engine.reset(); e.opp.reset(); e.bus.reset();
                }
            }
            advance(h, e);
        }
    } catch (const std::exception& ex) {
        std::snprintf(err, errlen, "%s", ex.what());
        return -2;
    }
    return rc;
}

}  // extern "C"
namespace rbe {
// Schema v2 field check (gate 1): decodes each pending env's int32 observation
// and compares the new fields with an independent read of the engine state
// (not through the Encoder / ObserverView). Adds into out[0..kVerifyN):
//  0 decisions checked
//  1 pile mismatches (sum of pile counts per (owner, zone) != pile size, or a
//    per-card count != a direct count of the pile)   2 pile tokens seen
//  3 known-hand mismatches (opp-hand tokens vs GameKnowledgeView known+confirmed
//    hand cards; key must equal the true card)        4 known opp-hand tokens seen
//  5 cost mismatches: eff energy/power differ from the printed cost where no
//    modifier can apply (all in-scope cards but Noxus Hopeful), or Noxus Hopeful
//    not printed-2 exactly when Legion is on (cards played this turn >= 1)
//  6 hand cards with a cost checked                    7 Noxus Hopeful checked
//  8 a hand card offered as a PlayCard from hand while affordable == 0
//  9 affordable == 1 hand cards in a main-phase decision with an empty chain
// 10 ... of which a PlayCard from hand is offered
// 11 hidden-this-turn mismatches (vs hidden_on_turn == turn, and a facedown play
//    of such a card offered)                           12 hidden-this-turn = 1 seen
// 13 Star Spring stamps set seen                       14 chain items with paid = 1
// 15 chain items with play ordinal > 0                 16 Location prompts mapped
//    to a real slot (dst >= 1)                         17 Location options left unmapped
// 18 move consequence mismatches (dst controller vs bf.controller)   19 moves checked
// 20 Sabotage prompts: opp hand size                   21 ... opp-hand tokens visible then
// 22 max tokens in one observation
constexpr int kVerifyN = 23;
void verifyEnv(Handle& h, Env& e, int64_t* out) {
    const GameState& s = e.engine->state();
    const StepResult& sr = e.engine->currentStep();
    const PlayerId me = e.seat == 1 ? PlayerId::Player1 : PlayerId::Player2;
    const PlayerId op = me == PlayerId::Player1 ? PlayerId::Player2 : PlayerId::Player1;
    out[0]++;
    int ntok = 0;
    for (int t = 0; t < T; ++t) ntok += e.tmask[t];
    out[22] = std::max<int64_t>(out[22], ntok);
    Counters dummy;
    auto vocab = [&](const std::string& k) { return vocabOf(k, dummy); };
    auto row = [&](int t) { return &e.tok[t * F]; };
    // 1. piles
    std::map<std::tuple<int, int, int>, long> got;   // (owner, zone, vocab) -> count
    std::map<std::pair<int, int>, long> got_sum;
    for (int t = 0; t < ntok; ++t) {
        const int32_t* r = row(t);
        if (r[0] != 4) continue;
        out[2]++;
        got[{r[2], r[3], r[1]}] += r[4];
        got_sum[{r[2], r[3]}] += r[4];
    }
    auto checkPile = [&](PlayerId pl, ZoneType z, const std::vector<GameObjectId>& ids) {
        const int ow = pl == me ? 1 : 2, zc = static_cast<int>(z);
        std::map<int, long> want;
        for (auto id : ids) { const auto& o = s.getObject(id); want[vocab(cardKey(o.card_def_id, &o))]++; }
        if (got_sum[{ow, zc}] != static_cast<long>(ids.size())) out[1]++;
        for (auto& [v, n] : want) if (got[{ow, zc, v}] != n) out[1]++;
    };
    checkPile(me, ZoneType::Trash, s.player(me).trash);
    checkPile(op, ZoneType::Trash, s.player(op).trash);
    checkPile(me, ZoneType::Banishment, s.player(me).banishment);
    checkPile(op, ZoneType::Banishment, s.player(op).banishment);
    checkPile(me, ZoneType::MainDeck, s.player(me).main_deck);
    // 2. known opponent hand
    {
        knowledge::GameKnowledgeView kv(s, me);
        std::multiset<int> want;
        for (const auto& k : kv.knownHiddenCards(op))
            if (k.last_known_zone == ZoneType::Hand && k.zone_confirmed) {
                const auto& o = s.getObject(k.object);
                want.insert(vocab(cardKey(o.card_def_id, &o)));   // the TRUE card
            }
        std::multiset<int> have;
        for (int t = 0; t < ntok; ++t) {
            const int32_t* r = row(t);
            if (r[0] == 1 && r[2 + 1] == 2 && r[2 + 2] == static_cast<int>(ZoneType::Hand)) have.insert(r[1]);
        }
        out[4] += static_cast<int64_t>(have.size());
        // action-referenced opp hand cards (Sabotage's choosable ones) are tokens too: have >= want
        for (int v : want) { auto it = have.find(v); if (it == have.end()) out[3]++; else have.erase(it); }
        if (sr.prompt.source.valid() && s.objectExists(sr.prompt.source) &&
            cardKey(s.getObject(sr.prompt.source).card_def_id, nullptr) == "sabotage" &&
            sr.prompt.kind != PromptKind::MainPhase && sr.prompt.kind != PromptKind::ChainPriority) {
            out[20] += static_cast<int64_t>(s.player(op).hand.size());
            for (int t = 0; t < ntok; ++t) { const int32_t* r = row(t); if (r[0] == 1 && r[3] == 2 && r[4] == 2) out[21]++; }
        }
    }
    // 3. cost / affordable / hidden: map token -> object through the hand order is not
    // available, so match own hand tokens by key against the hand's objects.
    const bool main_decision = sr.prompt.kind == PromptKind::MainPhase && s.chain.items.empty();
    std::set<uint64_t> offered;
    std::set<uint64_t> offered_facedown;
    for (const auto& i : sr.legal) {
        if (i.type == IntentType::PlayCard && i.card.valid()) {
            if (i.play_source == Intent::PlaySource::Hand) offered.insert(i.card.raw());
            if (i.play_source == Intent::PlaySource::Hidden) offered_facedown.insert(i.card.raw());
        }
    }
    for (int t = 0; t < ntok; ++t) {
        const int32_t* r = row(t);
        if (r[0] != 1 || r[3] != 1 || r[4] != static_cast<int>(ZoneType::Hand)) continue;
        // candidate objects in own hand with this key
        for (auto hid : s.player(me).hand) {
            const auto& o = s.getObject(hid);
            if (vocab(cardKey(o.card_def_id, &o)) != r[1]) continue;
            const auto& def = g_db->get(o.card_def_id);
            const std::string key = cardKey(o.card_def_id, &o);
            out[6]++;
            if (key == "noxus_hopeful") {
                out[7]++;
                const bool legion = s.player(me).cards_played_this_turn >= 1;
                const int want_e = std::max(0, def.energy_cost - (legion ? 2 : 0));
                if (r[19] != want_e || r[20] != def.power_cost) out[5]++;
            } else if (r[19] != def.energy_cost || r[20] != def.power_cost) {
                out[5]++;
            }
            if (offered.count(hid.raw()) && r[21] == 0) out[8]++;
            if (main_decision && s.turn.turn_player == me && r[21] == 1) {
                out[9]++;
                if (offered.count(hid.raw())) out[10]++;
            }
            break;   // copies share a key and a cost
        }
    }
    for (int t = 0; t < ntok; ++t) {
        const int32_t* r = row(t);
        if (r[0] != 1 || r[13] != 1 || r[3] != 1) continue;   // own facedown cards
        if (r[18] == 1) out[12]++;
    }
    {
        int want_hid = 0;
        for (const auto& b : s.battlefields)
            for (auto fd : b.facedown)
                if (s.objectExists(fd)) {
                    const auto& o = s.getObject(fd);
                    if (o.controller == me && o.hidden_on_turn >= s.turn.turn_number) {
                        ++want_hid;
                        if (offered_facedown.count(fd.raw())) out[11]++;
                    }
                }
        int have_hid = 0;
        for (int t = 0; t < ntok; ++t) { const int32_t* r = row(t); if (r[0] == 1 && r[3] == 1 && r[13] == 1 && r[18] == 1) ++have_hid; }
        if (have_hid != want_hid) out[11]++;
    }
    for (int t = 0; t < ntok; ++t) {
        const int32_t* r = row(t);
        if (r[0] == 2 && (r[15] || r[16])) out[13]++;
        if (r[0] == 3 && r[16] == 1) out[14]++;
        if (r[0] == 3 && r[17] > 0) out[15]++;
    }
    // actions: Location prompts and move consequences
    for (int a = 0; a < e.n_legal && a < A; ++a) {
        const int32_t* r = &e.act[a * K];
        const Intent& i = sr.legal[a];
        if (sr.prompt.kind == PromptKind::Location && i.type == IntentType::MakeChoice && !i.declines) {
            if (r[30] >= 1) out[16]++; else out[17]++;
        }
        if (i.type == IntentType::StandardMove && i.move_destination.has_value()) {
            out[19]++;
            const auto* bl = std::get_if<BattlefieldLocation>(&*i.move_destination);
            if (bl) {
                const auto& bf = s.battlefields[bl->id];
                const int want = bf.controller.has_value() ? (*bf.controller == me ? 1 : 2) : 0;
                if (r[46] != want || r[30] != 1 + static_cast<int>(bl->id.raw())) out[18]++;
            } else if (r[46] != -1 || r[30] != 0) {
                out[18]++;
            }
        }
    }
    (void)h;
}
}  // namespace rbe

extern "C" {

// Adds the schema-v2 field checks of every env with a pending decision into out[0..23).
void rbenv_verify(void* p, int64_t* out) {
    auto& h = *static_cast<rbe::Handle*>(p);
    for (auto& e : h.envs) if (e.need && e.engine) rbe::verifyEnv(h, e, out);
}

int rbenv_results_count(void* p) { return static_cast<int>(static_cast<rbe::Handle*>(p)->results.size()); }

// row: env, game_seed, winner, length, n_ext, turns
void rbenv_result_get(void* p, int i, int64_t* row, char* err, int errlen) {
    const auto& r = static_cast<rbe::Handle*>(p)->results.at(i);
    row[0] = r.env; row[1] = r.game_seed; row[2] = r.winner; row[3] = r.length; row[4] = r.n_ext; row[5] = r.turns;
    std::snprintf(err, errlen, "%s", r.error.c_str());
}

void rbenv_results_clear(void* p) { static_cast<rbe::Handle*>(p)->results.clear(); }

// trunc_tok, trunc_act, trunc_list, unk_key, ext decisions, auto decisions
void rbenv_counters(void* p, int64_t* out) {
    const auto& c = static_cast<rbe::Handle*>(p)->c;
    out[0] = c.trunc_tok; out[1] = c.trunc_act; out[2] = c.trunc_list; out[3] = c.unk_key; out[4] = c.ext; out[5] = c.auto_;
    out[6] = c.trig_opts; out[7] = c.trig_mapped; out[8] = c.trig_amb;   // v4
}

}  // extern "C"

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
// ---- PART A ----
// (inside namespace rbe; agents/determinizer.h is included at file scope by splice.py)
CardDefId defOfKey(const std::string& key) {
    for (const auto& kv : g_keys)
        if (kv.second == key) return CardDefId::fromRaw(kv.first);
    throw std::runtime_error("swap: unknown card key " + key);
}
void swapCard(GameObject& o, const std::string& key) {
    if (o.is_hidden) throw std::runtime_error("swap: facedown object");
    if (o.isToken()) throw std::runtime_error("swap: token");
    if (o.attached_to.has_value() || !o.attachments.empty()) throw std::runtime_error("swap: attachments");
    const CardDef& def = g_db->get(defOfKey(key));
    const bool ex = o.is_exhausted, st = o.is_stunned;
    const int dmg = o.damage_marked;
    const ZoneType z = o.zone;
    const bool onboard = z == ZoneType::Base || z == ZoneType::BattlefieldZone;
    if (onboard && !(o.isUnit() && def.card_type == CardType::Unit))
        throw std::runtime_error("swap: board swap must be unit->unit");
    riftbound::becomeCard(o, def);
    if (onboard) { o.is_exhausted = ex; o.is_stunned = st; o.damage_marked = dmg; }
    o.recomputeMight();
}
// ---- END PART A ----
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
        } else if (op == "swap") {   // gate 5
            GameObject& o = obj();
            std::string key;
            ss >> key;
            swapCard(o, key);
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
    e.hev.clear();
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

// ---- PART B ----
extern "C" {
int rbenv_advise(void* p, int i, int sims, int64_t seed) {
    using namespace rbe;
    auto& h = *static_cast<Handle*>(p);
    auto& e = h.envs[i];
    if (!e.need || !e.engine) return -1;
    const StepResult& sr = e.engine->currentStep();
    const uint64_t es = engineSeedOf(static_cast<uint64_t>(e.game_seed));
    IsMctsAgent a(h.deck_a, h.deck_b, "", es, static_cast<uint64_t>(seed), sims);
    const Intent pick = a.decide(e.engine->state(), sr.legal, sr.prompt);
    return indexOf(sr.legal, pick);
}
}  // extern "C"
// ---- END PART B ----

// ---- search (research/card_repr/search, M0/M1) ----
#include "research/card_repr/search/rbsearch.cpp"
