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

namespace rbe {
using namespace riftbound;
using namespace riftbound::probe;
using namespace riftbound::probe::dx;

constexpr int G = 72, T = 64, F = 17, A = 96, K = 46;
constexpr int kMaxList = 4, kMaxMu = 2, kMaxDm = 6, kMaxChosen = 4;
constexpr int kDecisionCap = 20000;

std::unordered_map<std::string, int> g_vocab;
int g_vocab_other = 0;
std::string g_variant;
bool g_inited = false;

struct Counters { long trunc_tok = 0, trunc_act = 0, trunc_list = 0, unk_key = 0, ext = 0, auto_ = 0; };

int vocabOf(const std::string& k, Counters& c) {
    auto it = g_vocab.find(k);
    if (it != g_vocab.end()) return it->second;
    ++c.unk_key;
    return g_vocab_other;
}

struct Result { int env; int64_t game_seed; int winner; int length; int n_ext; int turns; std::string error; };

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
};

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
};

void encode(Handle& h, Env& e, const StepResult& sr, PlayerId dec) {
    const GameState& s = e.engine->state();
    Encoder enc(s, dec);
    enc.build();
    std::fill(e.glob.begin(), e.glob.end(), 0);
    std::fill(e.tok.begin(), e.tok.end(), 0);
    std::fill(e.act.begin(), e.act.end(), 0);
    std::fill(e.tmask.begin(), e.tmask.end(), 0);
    std::fill(e.amask.begin(), e.amask.end(), 0);
    const auto& legal = sr.legal;
    const Prompt& p = sr.prompt;
    // Actions first (same ref() order as Encoder::encode/action, so appended
    // tokens get the same indices as in the exporter's JSON).
    const int na = static_cast<int>(legal.size());
    if (na > A) h.c.trunc_act += na - A;
    for (int a = 0; a < na; ++a) {
        const Intent& i = legal[a];
        int32_t buf[K];
        for (int k = 0; k < K; ++k) buf[k] = -1;
        buf[0] = static_cast<int>(i.type);
        buf[1] = i.card.valid() ? enc.ref(i.card) : -1;
        if (i.ability_source.valid()) { buf[2] = enc.ref(i.ability_source); buf[3] = i.ability_index; }
        auto list = [&](int off, int cap, int cnt_slot, const std::vector<GameObjectId>& v) {
            int j = 0;
            for (auto id : v) { const int r = enc.ref(id); if (j < cap) buf[off + j] = r; ++j; }
            if (j > cap) h.c.trunc_list++;
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
            if (j > kMaxDm) h.c.trunc_list++;
            buf[45] = j;
        }
        buf[30] = i.play_location.has_value() ? locCode(i.play_location) : locCode(i.move_destination);
        buf[31] = i.chosen_value.has_value() ? 1 : 0;
        buf[32] = i.chosen_value.has_value() ? static_cast<int32_t>(*i.chosen_value) : 0;
        if (i.chosen_value.has_value() && p.kind == PromptKind::ChainItem)
            buf[33] = enc.itemRef(static_cast<uint64_t>(*i.chosen_value));
        buf[34] = i.chosen_card_def.valid() ? vocabOf(cardKey(i.chosen_card_def, nullptr), h.c) : 0;
        buf[35] = i.chosen_battlefield.valid() ? static_cast<int32_t>(i.chosen_battlefield.raw()) : -1;
        buf[36] = static_cast<int>(i.play_source);
        buf[37] = i.use_alt_play_cost ? 1 : 0;
        buf[38] = i.pay_location_cost ? 1 : 0;
        buf[39] = i.declines ? 1 : 0;
        buf[40] = i.type == IntentType::PlayFirstDecision ? (i.choose_first ? 1 : 0) : -1;
        if (a < A) { std::memcpy(&e.act[a * K], buf, sizeof buf); e.amask[a] = 1; }
    }
    e.psrc = p.source.valid() ? enc.ref(p.source) : -1;
    const auto g = enc.global(p);
    for (int k = 0; k < G && k < static_cast<int>(g.size()); ++k) e.glob[k] = static_cast<int32_t>(g[k]);
    // tokens
    const auto& all = enc.all();
    std::unordered_map<uint64_t, int> obj_idx;
    for (size_t t = 0; t < all.size(); ++t) if (all[t].kind != 'k') obj_idx[all[t].obj] = static_cast<int>(t);
    auto idx = [&](uint64_t r) { auto it = obj_idx.find(r); return it == obj_idx.end() ? -1 : it->second; };
    const int nt = static_cast<int>(all.size());
    if (nt > T) h.c.trunc_tok += nt - T;
    for (int t = 0; t < nt && t < T; ++t) {
        const Tok& tk = all[t];
        int32_t* row = &e.tok[t * F];
        for (int k = 0; k < F; ++k) row[k] = 0;
        row[0] = tk.kind == 'c' ? 1 : (tk.kind == 'b' ? 2 : 3);
        row[1] = vocabOf(tk.key, h.c);
        int k = 2;
        for (long x : tk.f) if (k < F) row[k++] = static_cast<int32_t>(x);
        if (tk.kind == 'c') {
            row[16] = tk.refs.empty() ? -1 : idx(tk.refs[0]);
        } else if (tk.kind == 'k') {
            // 2..9 the 8 fields, 10 source token, 11..14 chosen tokens, 15 n chosen, 16 unused
            row[10] = idx(tk.refs[0]);
            for (int j = 11; j <= 14; ++j) row[j] = -1;
            int n = 0;
            for (size_t j = 1; j < tk.refs.size(); ++j, ++n) if (n < kMaxChosen) row[11 + n] = idx(tk.refs[j]);
            if (n > kMaxChosen) h.c.trunc_list++;
            row[15] = n;
        }
        e.tmask[t] = 1;
    }
    // references to truncated tokens become -1
    if (nt > T) {
        auto fix = [](int32_t& v) { if (v >= T) v = -1; };
        for (int t = 0; t < T; ++t) {
            int32_t* row = &e.tok[t * F];
            if (row[0] == 1) fix(row[16]);
            else if (row[0] == 3) { fix(row[10]); for (int j = 11; j <= 14; ++j) fix(row[j]); }
        }
        for (int a = 0; a < na && a < A; ++a) {
            int32_t* r = &e.act[a * K];
            for (int j : {1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24, 26, 28, 33}) fix(r[j]);
        }
        if (e.psrc >= T) e.psrc = -1;
    }
    e.seat = dec == PlayerId::Player1 ? 1 : 2;
    e.pk = static_cast<int>(p.kind);
    e.n_legal = na;
}

void startGame(Handle& h, Env& e) {
    e.game_seed = h.next_seed++;
    const uint64_t es = engineSeedOf(static_cast<uint64_t>(e.game_seed));
    e.engine.reset();
    e.opp.reset();
    e.bus = std::make_unique<EventBus>();
    e.engine = std::make_unique<GameEngine>(*g_db, *e.bus, *g_reg);
    e.length = 0; e.n_ext = 0; e.need = false;
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
    r.winner = 0; r.turns = 0; r.error = error;
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
            if (!e.engine) startGame(h, e);
            while (true) {
                const StepResult& sr = e.engine->currentStep();
                if (sr.kind == StepKind::Done) { finish(h, e, ""); break; }
                if (e.length >= kDecisionCap) { finish(h, e, "decision cap"); break; }
                const int n = static_cast<int>(sr.legal.size());
                const bool rune = sr.prompt.kind == PromptKind::RuneForEnergy || sr.prompt.kind == PromptKind::RuneForPower;
                const PlayerId dec = deciderOf(e.engine->state(), sr.legal, sr.prompt);
                if (n <= 1 || (h.autopay && rune)) {
                    ++e.length; ++h.c.auto_;
                    e.engine->applyChoice(0);
                    continue;
                }
                if (e.opp && dec == PlayerId::Player2) {
                    const Intent pick = e.opp->decide(e.engine->state(), sr.legal, sr.prompt);
                    const int k = indexOf(sr.legal, pick);
                    if (k < 0) throw std::runtime_error("opponent answered an option not in the legal list");
                    ++e.length;
                    e.engine->applyChoice(k);
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

// Process-wide setup: card table (+ one variant), card keys, vocabulary.
// Returns 0, or -1 with the message in err.
int rbenv_init(const char* subjects, const char* vocab, const char* variant, char* err, int errlen) {
    using namespace rbe;
    try {
        const std::string v = variant ? variant : "";
        if (g_inited) {
            if (v != g_variant) throw std::runtime_error("rbenv: one variant per process (have '" + g_variant + "')");
            return 0;
        }
        auto& shared = const_cast<openspiel::SharedCards&>(openspiel::sharedCards());
        if (!v.empty()) {
            if (!findVariant(v)) throw std::runtime_error("unknown variant " + v);
            registerVariant(shared.registry, v);
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
                try { e.engine->applyChoice(c); }
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
}

}  // extern "C"
