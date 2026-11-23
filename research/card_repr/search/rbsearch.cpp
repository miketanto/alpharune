// rbsearch.cpp -- network-guided determinized PUCT (research/card_repr/SEARCH_PLAN.md, M0 + M1).
// #included at the end of selfplay/rbenv/rbenv.cpp (same TU: Env, encodeRaw, applyTracked,
// deciderOf, engineSeedOf, the card keys and vocabulary). Never part of src/.
//
// Evaluation harness: n_games game slots, each plays exactly one game. Both seats are
// played by the network; a game's search seat (1 or 2, or 0 = nobody) searches every
// external decision with >= 2 options, the other seat plays the network's argmax.
// Python drives it with a leaf queue:
//     while not finished: B = collect(); rows(...); net(rows) -> logp, v; feed(logp, v)
// collect() advances every game (parallel over games on a thread pool) until it needs the
// network: one row for a greedy decision, or up to `leaves` search leaves (virtual loss).
//
// One simulation: build a world at the root decision (fresh engine: resume the last
// top-level main-phase snapshot of the live game and replay the choices since, or replay
// from the start before the first one), redraw what the root player cannot see
// (WorldSampler + GameEngine::editHiddenInformation, as IsMctsAgent::makeWorld), descend
// by PUCT (single-option decisions and rune payments auto-taken, as rbenv does), and
// stop at the first node that is new or offers an option it has no prior for: that leaf
// is encoded from its decider's view and evaluated by the network. Nodes are keyed by the
// root player's information set (openspiel::infoSetIdFor with player = root player:
// single-observer, as IsMctsSearch); children by action id; the root's children by index
// into the live legal list (world options matched by Intent equality).

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <mutex>
#include <thread>

namespace rbs {
using namespace riftbound;
using namespace rbe;
using Clock = std::chrono::steady_clock;

inline int64_t nsSince(Clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t).count();
}

inline uint64_t mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Persistent thread pool: run(n, f) calls f(0..n-1) on the workers and returns when done.
class Pool {
public:
    explicit Pool(int n) {
        for (int i = 0; i < n; ++i) th_.emplace_back([this] { loop(); });
    }
    ~Pool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
            ++gen_;
        }
        cv_.notify_all();
        for (auto& t : th_) t.join();
    }
    void run(int n, const std::function<void(int)>& f) {
        if (th_.empty()) { for (int i = 0; i < n; ++i) f(i); return; }
        {
            std::lock_guard<std::mutex> lk(m_);
            fn_ = &f; n_ = n; next_ = 0; active_ = static_cast<int>(th_.size()); ++gen_;
        }
        cv_.notify_all();
        std::unique_lock<std::mutex> lk(m_);
        done_.wait(lk, [&] { return active_ == 0; });
        fn_ = nullptr;
    }
private:
    void loop() {
        int seen = 0;
        while (true) {
            const std::function<void(int)>* f;
            int n;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return gen_ != seen; });
                seen = gen_;
                if (stop_) return;
                f = fn_; n = n_;
            }
            while (true) {
                const int i = next_.fetch_add(1);
                if (i >= n) break;
                (*f)(i);
            }
            std::lock_guard<std::mutex> lk(m_);
            if (--active_ == 0) done_.notify_one();
        }
    }
    std::vector<std::thread> th_;
    std::mutex m_;
    std::condition_variable cv_, done_;
    const std::function<void(int)>* fn_ = nullptr;
    int n_ = 0, active_ = 0, gen_ = 0;
    std::atomic<int> next_{0};
    bool stop_ = false;
};

struct ObsBuf {
    std::vector<int32_t> glob, tok, act;
    std::vector<uint8_t> tmask, amask;
    std::vector<int64_t> tok_obj;
    std::vector<int32_t> hist;              // history.h, kHistLen x kHistF
    int psrc = -1, pk = 0, nlegal = 0, seat = 0;
    ObsBuf() : glob(G), tok(T * F), act(A * K), tmask(T), amask(A), tok_obj(T), hist(kHistLen * kHistF) {}
};

void encodeTo(ObsBuf& b, const GameEngine& eng, const MoveTrack* mt, const StepResult& sr, PlayerId dec,
              Counters& c, const std::vector<HistEv>& hv) {
    ObsOut o{b.glob.data(), b.tok.data(), b.act.data(), b.tmask.data(), b.amask.data(), b.tok_obj.data()};
    encodeRaw(eng.state(), &eng, mt, sr, dec, c, o);
    encodeHist(eng.state(), hv, dec, c, b.hist.data());
    b.psrc = o.psrc;
    b.pk = static_cast<int>(sr.prompt.kind);
    b.nlegal = static_cast<int>(sr.legal.size());
    b.seat = dec == PlayerId::Player1 ? 1 : 2;
}

bool sameObs(const ObsBuf& a, const ObsBuf& b, std::string* why) {
    auto cmp = [&](const char* name, const auto& x, const auto& y) {
        if (x == y) return true;
        if (why && why->empty()) {
            size_t k = 0;
            while (k < x.size() && x[k] == y[k]) ++k;
            *why = std::string(name) + "[" + std::to_string(k) + "]";
        }
        return false;
    };
    bool ok = cmp("glob", a.glob, b.glob);
    ok = cmp("tok", a.tok, b.tok) && ok;
    ok = cmp("tmask", a.tmask, b.tmask) && ok;
    ok = cmp("act", a.act, b.act) && ok;
    ok = cmp("amask", a.amask, b.amask) && ok;
    ok = cmp("hist", a.hist, b.hist) && ok;
    if (a.psrc != b.psrc || a.pk != b.pk || a.nlegal != b.nlegal) {
        ok = false;
        if (why && why->empty()) *why = "meta";
    }
    return ok;
}

struct Child { float P = -1.f; int N = 0; float W = 0.f; };
struct Node {
    bool expanded = false;
    float v0 = 0.f;                         // net value at first expansion, node decider's view
    std::unordered_map<int, Child> ch;      // action id (root: live legal index)
};
struct Step { Node* node; int key; int sign; };
struct Leaf {
    std::vector<Step> path;
    Node* node = nullptr;
    int sign = 1;                           // +1 if the leaf's decider is the root player
    std::vector<int> keys;                  // child key of each option j < nopt (-1: none)
    int nopt = 0;
};

// Timing / counters per game (summed by rbs_stats).
struct Stats {
    int64_t sims = 0, leaves = 0, terminals = 0, fails = 0, reexpand = 0, fallbacks = 0;
    int64_t searches = 0, greedy = 0, depth_sum = 0, collisions = 0;
    int64_t ns_build = 0, ns_edit = 0, ns_descend = 0, ns_encode = 0, ns_teardown = 0, ns_snap = 0;
    int64_t replay_steps = 0;
    int64_t changed = 0, dl_fallback = 0;   // searched move != net argmax; decklist redraw fell back
};

struct Snap {
    bool have = false;
    GameState st;
    std::mt19937_64 rng;
    StepResult step;
    size_t hist_len = 0;
};

struct Cfg {
    int sims = 64, leaves = 4;
    float cpuct = 2.0f, fpu_red = 0.0f, vl = 1.0f;
    bool perfect = false;
    int deck_model = 0;   // 0 legend pool (WorldSampler, honest default); 1 public decklist (redrawFromDecklist)
    int max_depth = 4000;
    // E2 generator mode (seat_mode 2, search/gen_targets.py): both seats are the network and SAMPLE
    // their moves from its policy (temperature 1); each external decision is searched with
    // probability p_search, the search's root statistics are recorded as a training target
    // (GenRec), and the move played is still sampled from the network's root prior (positions
    // stay on the network's own sampled-play distribution).
    bool gen = false;
    float p_search = 1.f;
    // AlphaZero mode (az/, 2026-11-21; needs gen): the move played is the SEARCH's (visit-proportional for
    // the first az_temp_moves external decisions of the game, then the most visited), Dirichlet noise
    // (dir_alpha, dir_eps) on the root prior of full searches; playout cap randomisation: a decision gets
    // a full search (sims, recorded as a training target) with probability p_full, else sims_fast
    // simulations, not recorded.
    bool az = false;
    int az_temp_moves = 30, sims_fast = 16;
    float dir_alpha = 0.3f, dir_eps = 0.25f, p_full = 1.f;
    // Runes as actions (az/TAPS.md): which seats are networks that may tap: 0 none, 1 both, 2 the search seat,
    // 3 the other seat.
    int runes = 0;
    float dir_tap = 1.f;   // a tap's share of the root noise, relative to a move's (the noise becomes visits, visits the target)
};

// E2: one searched decision. obs = the live decision encoded for the decider with the
// training encoder (loose visibility, as rbenv); P / N / Q per live option index (< A):
// net prior, root visits, mean value (decider's view; -2 = unvisited / not offered).
struct GenRec {
    ObsBuf obs;
    std::vector<float> P, N, Q;
    float v0 = 0.f, vs = 0.f;   // net value at the root; visit-weighted root value
    int seat = 0;
};

struct SHandle;

struct Game {
    int idx = 0;
    int64_t seed = 0;
    int search_seat = 0;
    Env e;
    Snap snap;
    enum Mode { Idle, Greedy, Search, Done } mode = Idle;
    bool started = false;
    PlayerId dec = PlayerId::None;
    std::vector<Intent> live;
    // search
    std::unordered_map<uint64_t, Node> nodes;
    Node root;
    PlayerId root_player = PlayerId::None;
    int issued = 0;
    int decision_no = 0;
    std::mt19937_64 rng;
    std::vector<Leaf> leaves;               // in flight (row r -> leaves[r] while mode == Search)
    std::vector<ObsBuf> bufs;
    int nrows = 0;
    // Per external decision of the live game (replay / annotation; rbs_declog): searched, choice,
    // argmax, p_choice, p_argmax, n_choice, n_argmax, n_total, q_choice, q_argmax, net value (decider),
    // decider seat, then the top 3 by visits: index, N (x3). -1 = not applicable.
    std::vector<std::array<float, 18>> dlog;
    // Per dlog entry: every option of the decision, (live index, prior, visits, mean value for the decider; -2 =
    // unvisited). A searched decision: its root; a greedy one: the network's policy, visits 0 (rbs_root).
    std::vector<std::vector<std::array<float, 4>>> droot;
    Counters c;
    Stats st;
    int winner = 0, turns = 0, n_search = 0, n_greedy = 0;
    // Seats (mask 1 | 2) offered their runes' own abilities (GameEngine::setManualRunes; needs RB_MANUAL_RUNES).
    // tap_seat: those that are networks (Cfg::runes): offered inside the search's worlds too, and asked nothing
    // when taps are the only alternative to one move. The others (a human opponent, tools/play_web) are asked
    // everything in the live game and have no taps in the worlds.
    int manual_seat = 0, tap_seat = 0;
    int64_t taps[4] = {0, 0, 0, 0};   // live: exhaust, recycle, external decisions offering a tap, external decisions
    std::string err;
    std::unique_ptr<AgentInterface> opp;   // C++ agent on the non-search seat (opponent spec), else greedy net
    // E2 generator mode
    std::mt19937_64 grng;
    std::vector<GenRec> recs;
    int cur_sims = 0;            // this decision's simulation budget
    bool cur_record = false;     // this decision's search is a training target
    int mu = 0;                  // matchup (deck pair) of this slot
};

struct SHandle {
    Cfg cfg;
    DeckSubmission da, db;
    std::string deck_a_path, deck_b_path, opp_spec;
    // matchups: slot i plays pair i % size (deck_a / deck_b given as '|'-separated lists of equal length)
    std::vector<std::string> pa, pb;
    std::vector<DeckSubmission> das, dbs;
    std::vector<Game> games;
    std::unique_ptr<Pool> pool;
    std::unique_ptr<WorldSampler> sampler;
    std::vector<int> offs;                  // row offset of each game (after collect)
    int total_rows = 0;
};

// ---------------------------------------------------------------- the live game
void searchInit(SHandle& h, Game& g) {
    g.nodes.clear();
    g.root = Node{};
    g.issued = 0;
    g.leaves.clear();
    g.root_player = g.dec;
    g.rng.seed(mix64(static_cast<uint64_t>(g.seed) * 1000003ull + static_cast<uint64_t>(g.decision_no) * 7919ull +
                     static_cast<uint64_t>(g.search_seat)));
    (void)h;
}

// Runs the live game to its next external decision (>= 2 options, not a rune payment).
void advanceReal(SHandle& h, Game& g) {
    Env& e = g.e;
    if (!g.started) {
        e.bus = std::make_unique<EventBus>();
        e.engine = std::make_unique<GameEngine>(*g_db, *e.bus, *g_reg);
        const uint64_t es = engineSeedOf(static_cast<uint64_t>(g.seed));
        if (!h.opp_spec.empty() && g.search_seat != 0) {
            GameSpec gs; gs.deck_a = h.pa[g.mu]; gs.deck_b = h.pb[g.mu]; gs.seed = static_cast<uint64_t>(g.seed);
            IsMctsAgent* m = nullptr;
            g.opp = makeAgent(h.opp_spec, gs, es, g.search_seat == 1 ? 1 : 0, &m);   // seat index of the opponent
        }
        if (h.cfg.runes == 1) g.tap_seat = 3;
        else if (h.cfg.runes == 2) g.tap_seat = g.search_seat;
        else if (h.cfg.runes == 3) g.tap_seat = g.search_seat ? 3 - g.search_seat : 0;
        g.manual_seat |= g.tap_seat;
        manualRunes(*e.engine, g.manual_seat, true);
        e.engine->beginGame(h.das[g.mu], h.dbs[g.mu], es);
        g.started = true;
    }
    while (true) {
        const StepResult& sr = e.engine->currentStep();
        if (sr.kind == StepKind::Done) {
            const GameResult& gr = e.engine->stepResult();
            g.winner = winnerCode(gr.winner);
            g.turns = gr.total_turns;
            g.mode = Game::Done;
            return;
        }
        if (e.length >= kDecisionCap) throw std::runtime_error("decision cap");
        if ((!g.snap.have || g.snap.hist_len != e.hist.size()) &&
            GameEngine::isResumableDecision(e.engine->state(), sr.legal)) {
            const auto t0 = Clock::now();
            g.snap.st = e.engine->state();
            g.snap.rng = e.engine->rngState();
            g.snap.step = sr;
            g.snap.hist_len = e.hist.size();
            g.snap.have = true;
            g.st.ns_snap += nsSince(t0);
        }
        const PlayerId dec = deciderOf(e.engine->state(), sr.legal, sr.prompt);
        const int only = autoOption(e.engine->state(), sr, true, seatIn(g.tap_seat, dec));
        if (only >= 0) {
            ++e.length;
            applyTracked(e, sr, dec, only);
            continue;
        }
        g.dec = dec;
        g.live = sr.legal;
        ++g.decision_no;
        const int seat = dec == PlayerId::Player1 ? 1 : 2;
        if (g.opp && seat != g.search_seat) {
            const Intent pick = g.opp->decide(e.engine->state(), sr.legal, sr.prompt);
            const int k = indexOf(sr.legal, pick);
            if (k < 0) throw std::runtime_error("opponent answered an option not in the legal list");
            ++e.length; ++e.n_ext;
            applyTracked(e, sr, dec, k);
            ++g.n_greedy;
            continue;
        }
        bool do_search = h.cfg.sims > 0 && seat == g.search_seat;
        g.cur_sims = h.cfg.sims;
        g.cur_record = h.cfg.gen;
        if (h.cfg.gen)
            do_search = h.cfg.sims > 0 &&
                        std::uniform_real_distribution<float>(0.f, 1.f)(g.grng) < h.cfg.p_search;
        if (h.cfg.az) {
            do_search = h.cfg.sims > 0;
            g.cur_record = std::uniform_real_distribution<float>(0.f, 1.f)(g.grng) < h.cfg.p_full;
            g.cur_sims = g.cur_record ? h.cfg.sims : std::max(1, h.cfg.sims_fast);
        }
        if (do_search) {
            g.mode = Game::Search;
            ++g.n_search;
            ++g.st.searches;
            searchInit(h, g);
            if (h.cfg.gen && g.cur_record) {   // the stored observation: training encoder (loose), live state
                GenRec r;
                riftbound::probe::dx::g_vis_force = 0;
                encodeTo(r.obs, *e.engine, &e.mtrack[static_cast<int>(dec) % 3], sr, dec, g.c, e.hev);
                riftbound::probe::dx::g_vis_force = -1;
                r.seat = seat;
                g.recs.push_back(std::move(r));
            }
        } else {
            g.mode = Game::Greedy;
            ++g.n_greedy;
            ++g.st.greedy;
        }
        return;
    }
}

void applyReal(Game& g, int k) {
    Env& e = g.e;
    ++e.length; ++e.n_ext;
    ++g.taps[3];
    if (g.tap_seat) {
        const StepResult& sr = e.engine->currentStep();
        const GameState& s = e.engine->state();
        for (const auto& i : sr.legal) if (isRuneAbility(s, i)) { ++g.taps[2]; break; }
        if (k >= 0 && k < static_cast<int>(sr.legal.size()) && isRuneAbility(s, sr.legal[k]))
            ++g.taps[sr.legal[k].ability_index == 0 ? 0 : 1];
    }
    applyTracked(e, e.engine->currentStep(), g.dec, k);
    g.mode = Game::Idle;
}

// ---------------------------------------------------------------- worlds
// "Public decklist" deck model (SEARCH_PLAN.md leak guards: the optional open-decklist setting).
// Applied after WorldSampler::sample: every opponent card in a hand / Main Deck / facedown pile
// whose identity AND place the observer does not know (R, computed on the truth before sampling)
// is redrawn from the opponent's decklist minus every other card of theirs (all of which the
// observer knows: public, or known at their place); facedown slots get [Hidden] cards. The same
// for the opponent's rune deck. A card known by identity but not by place goes back into the pool
// (its zone restriction is not kept: the observer knows a little less, never more). If the counts
// do not add up (an effect made extra copies), the WorldSampler world is kept (dl_fallback).
struct DeckRedraw {
    std::vector<GameObjectId> R, Rr;   // opponent card slots / rune slots to redraw
};
DeckRedraw redrawSlots(const GameState& s, PlayerId obs) {
    DeckRedraw d;
    const PlayerId opp = obs == PlayerId::Player1 ? PlayerId::Player2 : PlayerId::Player1;
    const knowledge::GameKnowledgeView kv(s, obs);
    auto atPlace = [&](GameObjectId id) {
        const CardKnowledge k = kv.about(id);
        if (!k.identity_known) return false;
        const ZoneType z = s.getObject(id).zone;
        if (z == ZoneType::MainDeck || z == ZoneType::RuneDeck)
            return k.deck_position != DeckPosition::Unknown && k.deck_rank_span == 1;
        return k.zone_confirmed;
    };
    std::vector<GameObjectId> ids;
    for (const auto& [id, o] : s.objects) if (o.owner == opp && !o.is_token) ids.push_back(id);
    std::sort(ids.begin(), ids.end());
    for (auto id : ids) {
        const GameObject& o = s.getObject(id);
        const ZoneType z = o.zone;
        if (z == ZoneType::RuneDeck) { if (!atPlace(id)) d.Rr.push_back(id); continue; }
        if (z != ZoneType::Hand && z != ZoneType::MainDeck && z != ZoneType::FacedownZone) continue;
        if (!atPlace(id)) d.R.push_back(id);
    }
    return d;
}
bool redrawFromDecklist(GameState& s, PlayerId obs, const DeckSubmission& dl, const DeckRedraw& d,
                        std::mt19937_64& rng) {
    const PlayerId opp = obs == PlayerId::Player1 ? PlayerId::Player2 : PlayerId::Player1;
    std::set<GameObjectId> inR(d.R.begin(), d.R.end()), inRr(d.Rr.begin(), d.Rr.end());
    std::map<CardDefId, int> pool, rpool;
    for (auto c : dl.main_deck) pool[c]++;
    if (dl.chosen_champion.valid()) pool[dl.chosen_champion]++;
    for (auto c : dl.rune_deck) rpool[c]++;
    for (const auto& [id, o] : s.objects) {
        if (o.owner != opp || o.is_token) continue;
        if (o.card_type == CardType::Rune) { if (!inRr.count(id)) rpool[o.card_def_id]--; continue; }
        if (o.card_type == CardType::Legend || o.card_type == CardType::Battlefield) continue;
        if (!inR.count(id)) pool[o.card_def_id]--;
    }
    bool ok = true;
    auto expand = [&](const std::map<CardDefId, int>& m) {
        std::vector<CardDefId> v;
        for (const auto& [c, n] : m) { if (n < 0) ok = false; for (int k = 0; k < n; ++k) v.push_back(c); }
        return v;
    };
    std::vector<CardDefId> cards = expand(pool), runes = expand(rpool);
    if (!ok || cards.size() != d.R.size() || runes.size() != d.Rr.size()) return false;
    std::shuffle(cards.begin(), cards.end(), rng);
    std::shuffle(runes.begin(), runes.end(), rng);
    // facedown slots first, from the [Hidden] cards of the pool
    std::vector<GameObjectId> order;
    for (auto id : d.R) if (s.getObject(id).zone == ZoneType::FacedownZone) order.push_back(id);
    const size_t nfd = order.size();
    for (auto id : d.R) if (s.getObject(id).zone != ZoneType::FacedownZone) order.push_back(id);
    std::vector<CardDefId> assign(order.size());
    std::vector<bool> used(cards.size(), false);
    for (size_t j = 0; j < nfd; ++j) {
        size_t k = 0;
        while (k < cards.size() && (used[k] || !g_db->get(cards[k]).keywords.has(Keyword::Hidden))) ++k;
        if (k == cards.size()) return false;
        used[k] = true;
        assign[j] = cards[k];
    }
    size_t k = 0;
    for (size_t j = nfd; j < order.size(); ++j) {
        while (used[k]) ++k;
        used[k] = true;
        assign[j] = cards[k];
    }
    for (size_t j = 0; j < order.size(); ++j) becomeCard(s.getObject(order[j]), g_db->get(assign[j]));
    for (size_t j = 0; j < d.Rr.size(); ++j) becomeCard(s.getObject(d.Rr[j]), g_db->get(runes[j]));
    return true;
}

struct SWorld {
    std::unique_ptr<EventBus> bus;
    std::unique_ptr<GameEngine> eng;
};

// A fresh engine at the live game's current decision; hidden information redrawn for
// `observer` unless perfect. `draw_seed`, `eng_seed` from the caller's RNG.
SWorld buildWorld(SHandle& h, Game& g, PlayerId observer, bool perfect, uint64_t draw_seed, uint64_t eng_seed) {
    const auto t0 = Clock::now();
    SWorld w;
    w.bus = std::make_unique<EventBus>();
    w.eng = std::make_unique<GameEngine>(*g_db, *w.bus, *g_reg);
    size_t from = 0;
    manualRunes(*w.eng, g.manual_seat, true);
    if (g.snap.have) {
        w.eng->resumeFromSnapshot(g.snap.st, g.snap.rng, g.snap.step);
        from = g.snap.hist_len;
    } else {
        w.eng->beginGame(h.das[g.mu], h.dbs[g.mu], engineSeedOf(static_cast<uint64_t>(g.seed)));
    }
    for (size_t k = from; k < g.e.hist.size(); ++k) {
        if (w.eng->currentStep().kind != StepKind::NeedDecision) throw std::runtime_error("world replay: game over");
        w.eng->applyChoice(g.e.hist[k]);
    }
    g.st.replay_steps += static_cast<int64_t>(g.e.hist.size() - from);
    if (w.eng->currentStep().kind != StepKind::NeedDecision ||
        w.eng->currentStep().legal.size() != g.live.size())
        throw std::runtime_error("world replay: not at the live decision");
    manualRunes(*w.eng, g.manual_seat & ~g.tap_seat, false);
    g.st.ns_build += nsSince(t0);
    if (!perfect) {
        const auto t1 = Clock::now();
        w.eng->editHiddenInformation(
            [&](GameState& s) {
                std::mt19937_64 d(draw_seed);
                DeckRedraw rd;
                if (h.cfg.deck_model == 1) rd = redrawSlots(s, observer);
                h.sampler->sample(s, observer, d);
                if (h.cfg.deck_model == 1) {
                    const DeckSubmission& dl = observer == PlayerId::Player1 ? h.dbs[g.mu] : h.das[g.mu];
                    if (!redrawFromDecklist(s, observer, dl, rd, d)) ++g.st.dl_fallback;
                }
            },
            eng_seed);
        g.st.ns_edit += nsSince(t1);
    }
    return w;
}

void teardown(Game& g, SWorld& w) {
    const auto t0 = Clock::now();
    w.eng.reset();
    w.bus.reset();
    g.st.ns_teardown += nsSince(t0);
}

// ---------------------------------------------------------------- PUCT
int selectPuct(const Cfg& cfg, const Node& nd, const std::vector<int>& keys, int nopt) {
    float sumN = 0.f;
    for (int j = 0; j < nopt; ++j) {
        if (keys[j] < 0) continue;
        auto it = nd.ch.find(keys[j]);
        if (it != nd.ch.end()) sumN += static_cast<float>(it->second.N);
    }
    const float sq = std::sqrt(std::max(1.f, sumN));
    int best = -1;
    float bs = -1e30f;
    for (int j = 0; j < nopt; ++j) {
        if (keys[j] < 0) continue;
        auto it = nd.ch.find(keys[j]);
        if (it == nd.ch.end() || it->second.P < 0.f) continue;
        const Child& c = it->second;
        const float q = c.N > 0 ? c.W / static_cast<float>(c.N) : nd.v0 - cfg.fpu_red;
        const float u = q + cfg.cpuct * c.P * sq / (1.f + static_cast<float>(c.N));
        if (u > bs) { bs = u; best = j; }
    }
    return best;
}

void addVirtual(const Cfg& cfg, Node& nd, int key) {
    Child& c = nd.ch[key];
    c.N += 1;
    c.W -= cfg.vl;
}

void revertPath(const Cfg& cfg, const std::vector<Step>& path) {
    for (const Step& s : path) {
        Child& c = s.node->ch[s.key];
        c.N -= 1;
        c.W += cfg.vl;
    }
}

void backup(const Cfg& cfg, const std::vector<Step>& path, float v_root) {
    for (const Step& s : path) {
        Child& c = s.node->ch[s.key];
        c.W += cfg.vl + static_cast<float>(s.sign) * v_root;
    }
}

enum class SimOut { Leaf, Terminal, Fail };

// One simulation; on Leaf the observation is in g.bufs[g.nrows] and the leaf in g.leaves.
SimOut simulate(SHandle& h, Game& g) {
    const Cfg& cfg = h.cfg;
    const uint64_t draw_seed = g.rng(), eng_seed = g.rng();
    std::vector<Step> path;
    SWorld w;
    try {
        w = buildWorld(h, g, g.root_player, cfg.perfect, draw_seed, eng_seed);
        const auto t0 = Clock::now();
        GameEngine& eng = *w.eng;
        MoveTrack mt[3] = {g.e.mtrack[0], g.e.mtrack[1], g.e.mtrack[2]};
        std::vector<HistEv> hv = g.e.hev;   // history.h: the live history + this simulation's path
        // root: world options -> live indices
        {
            const StepResult& sr = eng.currentStep();
            const int n = static_cast<int>(sr.legal.size());
            const int nopt = std::min(n, A);
            std::vector<int> keys(nopt, -1);
            for (int j = 0; j < nopt; ++j) {
                if (j < static_cast<int>(g.live.size()) && sr.legal[j] == g.live[j]) { keys[j] = j; continue; }
                for (size_t i = 0; i < g.live.size(); ++i)
                    if (sr.legal[j] == g.live[i]) { keys[j] = static_cast<int>(i); break; }
            }
            if (!g.root.expanded) {
                g.st.ns_descend += nsSince(t0);
                const auto t1 = Clock::now();
                if (static_cast<int>(g.bufs.size()) <= g.nrows) g.bufs.emplace_back();
                encodeTo(g.bufs[g.nrows], eng, &mt[static_cast<int>(g.root_player) % 3], sr, g.root_player, g.c, hv);
                g.st.ns_encode += nsSince(t1);
                Leaf lf;
                lf.node = &g.root; lf.sign = 1; lf.keys = std::move(keys); lf.nopt = nopt;
                g.leaves.push_back(std::move(lf));
                teardown(g, w);
                return SimOut::Leaf;
            }
            const int j = selectPuct(cfg, g.root, keys, nopt);
            if (j < 0) throw std::runtime_error("root: no selectable option in this world");
            addVirtual(cfg, g.root, keys[j]);
            path.push_back({&g.root, keys[j], 1});
            noteChoice(mt[static_cast<int>(g.root_player) % 3], sr.prompt, sr.legal[j]);
            HistPending hp = histBegin(eng.state(), sr, g.root_player, j);
            eng.applyChoice(j);
            histEnd(eng.state(), hp, hv);
        }
        for (int depth = 0;; ++depth) {
            // auto-advance
            while (true) {
                const StepResult& sr = eng.currentStep();
                if (sr.kind == StepKind::Done) break;
                const PlayerId d = deciderOf(eng.state(), sr.legal, sr.prompt);
                const int only = autoOption(eng.state(), sr, true, seatIn(g.tap_seat, d));
                if (only < 0) break;
                noteChoice(mt[static_cast<int>(d) % 3], sr.prompt, sr.legal[only]);
                eng.applyChoice(only);
            }
            const StepResult& sr = eng.currentStep();
            if (sr.kind == StepKind::Done) {
                const PlayerId win = eng.stepResult().winner;
                const float v = win == g.root_player ? 1.f : (win == PlayerId::None ? 0.f : -1.f);
                backup(cfg, path, v);
                g.st.depth_sum += static_cast<int64_t>(path.size());
                g.st.ns_descend += nsSince(t0);
                teardown(g, w);
                return SimOut::Terminal;
            }
            const PlayerId dec = deciderOf(eng.state(), sr.legal, sr.prompt);
            const int sign = dec == g.root_player ? 1 : -1;
            const uint64_t key = openspiel::infoSetIdFor(eng.state(), sr.legal, dec, g.root_player, *g_db);
            Node& nd = g.nodes[key];
            const int n = static_cast<int>(sr.legal.size());
            const int nopt = std::min(n, A);
            std::vector<int> ids = openspiel::actionIds(sr.legal, eng.state());
            ids.resize(nopt);
            bool need = !nd.expanded || depth >= cfg.max_depth;
            if (!need)
                for (int j = 0; j < nopt; ++j) {
                    auto it = nd.ch.find(ids[j]);
                    if (it == nd.ch.end() || it->second.P < 0.f) { need = true; ++g.st.reexpand; break; }
                }
            if (need) {
                g.st.ns_descend += nsSince(t0);
                const auto t1 = Clock::now();
                if (static_cast<int>(g.bufs.size()) <= g.nrows) g.bufs.emplace_back();
                encodeTo(g.bufs[g.nrows], eng, &mt[static_cast<int>(dec) % 3], sr, dec, g.c, hv);
                g.st.ns_encode += nsSince(t1);
                g.st.depth_sum += static_cast<int64_t>(path.size());
                Leaf lf;
                lf.path = std::move(path); lf.node = &nd; lf.sign = sign; lf.keys = std::move(ids); lf.nopt = nopt;
                g.leaves.push_back(std::move(lf));
                teardown(g, w);
                return SimOut::Leaf;
            }
            const int j = selectPuct(cfg, nd, ids, nopt);
            addVirtual(cfg, nd, ids[j]);
            path.push_back({&nd, ids[j], sign});
            noteChoice(mt[static_cast<int>(dec) % 3], sr.prompt, sr.legal[j]);
            HistPending hp = histBegin(eng.state(), sr, dec, j);
            eng.applyChoice(j);
            histEnd(eng.state(), hp, hv);
        }
    } catch (const std::exception& ex) {
        revertPath(cfg, path);
        ++g.st.fails;
        if (g.err.empty() && std::getenv("RBS_VERBOSE")) std::fprintf(stderr, "[rbs] sim failed: %s\n", ex.what());
        if (w.eng) teardown(g, w);
        return SimOut::Fail;
    }
}

int bestRoot(const Game& g) {
    int best = -1, bn = -1;
    float bp = -1.f;
    const int nopt = std::min(static_cast<int>(g.live.size()), A);
    for (int i = 0; i < nopt; ++i) {
        auto it = g.root.ch.find(i);
        if (it == g.root.ch.end() || it->second.P < 0.f) continue;
        const Child& c = it->second;
        if (c.N > bn || (c.N == bn && c.P > bp)) { bn = c.N; bp = c.P; best = i; }
    }
    return best;
}

// One game's part of collect(): advance to the next point where the network is needed.
void collectGame(SHandle& h, Game& g) {
    g.nrows = 0;
    g.leaves.clear();
    try {
        for (int guard = 0; guard < 100000; ++guard) {
            if (g.mode == Game::Done) return;
            if (g.mode == Game::Idle) { advanceReal(h, g); continue; }
            if (g.mode == Game::Greedy) {
                if (g.bufs.empty()) g.bufs.emplace_back();
                encodeTo(g.bufs[0], *g.e.engine, &g.e.mtrack[static_cast<int>(g.dec) % 3], g.e.engine->currentStep(),
                         g.dec, g.c, g.e.hev);
                g.nrows = 1;
                return;
            }
            // Search
            if (g.issued >= g.cur_sims) {
                int k = bestRoot(g);
                if (k < 0) {   // root never expanded
                    ++g.st.fallbacks; g.mode = Game::Greedy;
                    if (h.cfg.gen && g.cur_record && !g.recs.empty()) g.recs.pop_back();
                    continue;
                }
                if (h.cfg.az) {   // AlphaZero: record (full searches) and play the search's move
                    if (g.cur_record && !g.recs.empty()) {
                        GenRec& r = g.recs.back();
                        r.P.assign(A, 0.f); r.N.assign(A, 0.f); r.Q.assign(A, -2.f);
                        float tw = 0.f, tn = 0.f;
                        for (const auto& [i, c] : g.root.ch) {
                            if (i < 0 || i >= A) continue;
                            r.P[i] = std::max(0.f, c.P); r.N[i] = static_cast<float>(c.N);
                            if (c.N > 0) { r.Q[i] = c.W / static_cast<float>(c.N); tw += c.W; tn += static_cast<float>(c.N); }
                        }
                        r.v0 = g.root.v0; r.vs = tn > 0.f ? tw / tn : g.root.v0;
                    }
                    if (g.decision_no <= h.cfg.az_temp_moves) {   // visit-proportional
                        std::vector<std::pair<int, int>> vis;
                        float tot = 0.f;
                        for (const auto& [i, c] : g.root.ch)
                            if (i >= 0 && i < A && c.P >= 0.f && c.N > 0) { vis.push_back({i, c.N}); tot += static_cast<float>(c.N); }
                        std::sort(vis.begin(), vis.end());
                        if (tot > 0.f) {
                            float u = std::uniform_real_distribution<float>(0.f, tot)(g.grng);
                            for (const auto& [i, n] : vis) { k = i; u -= static_cast<float>(n); if (u <= 0.f) break; }
                        }
                    }
                    applyReal(g, k);
                    continue;
                }
                if (h.cfg.gen) {
                    GenRec& r = g.recs.back();
                    r.P.assign(A, 0.f); r.N.assign(A, 0.f); r.Q.assign(A, -2.f);
                    float tw = 0.f, tn = 0.f;
                    for (const auto& [i, c] : g.root.ch) {
                        if (i < 0 || i >= A) continue;
                        r.P[i] = std::max(0.f, c.P); r.N[i] = static_cast<float>(c.N);
                        if (c.N > 0) { r.Q[i] = c.W / static_cast<float>(c.N); tw += c.W; tn += static_cast<float>(c.N); }
                    }
                    r.v0 = g.root.v0; r.vs = tn > 0.f ? tw / tn : g.root.v0;
                    // move: sampled from the network's prior (not the search's choice)
                    float tot = 0.f;
                    for (int i = 0; i < A; ++i) tot += r.P[i];
                    if (tot > 0.f) {
                        float u = std::uniform_real_distribution<float>(0.f, tot)(g.grng);
                        int pick = -1;
                        for (int i = 0; i < A; ++i) { if (r.P[i] <= 0.f) continue; pick = i; u -= r.P[i]; if (u <= 0.f) break; }
                        if (pick >= 0) k = pick;
                    }
                    if (k != bestRoot(g)) ++g.st.changed;
                    applyReal(g, k);
                    continue;
                }
                {
                    int am = -1; float ap = -1.f;   // the network's argmax at the root
                    for (const auto& [i, c] : g.root.ch) if (c.P > ap || (c.P == ap && i < am)) { ap = c.P; am = i; }
                    if (am != k) ++g.st.changed;
                    std::array<float, 18> d{};
                    d.fill(-1.f);
                    int tot = 0;
                    std::vector<std::pair<int, int>> byN;
                    for (const auto& [i, c] : g.root.ch) { tot += c.N; byN.push_back({c.N, i}); }
                    std::sort(byN.begin(), byN.end(), [](const auto& a, const auto& b) {
                        return a.first > b.first || (a.first == b.first && a.second < b.second); });
                    const Child& ck = g.root.ch.at(k);
                    const Child& ca = g.root.ch.at(am);
                    d[0] = 1; d[1] = k; d[2] = am; d[3] = ck.P; d[4] = ca.P; d[5] = ck.N; d[6] = ca.N; d[7] = tot;
                    d[8] = ck.N ? ck.W / ck.N : 0.f; d[9] = ca.N ? ca.W / ca.N : 0.f; d[10] = g.root.v0;
                    d[11] = static_cast<float>(static_cast<int>(g.dec));
                    for (int t = 0; t < 3 && t < static_cast<int>(byN.size()); ++t) {
                        d[12 + 2 * t] = byN[t].second; d[13 + 2 * t] = byN[t].first; }
                    g.dlog.push_back(d);
                    {
                        std::vector<std::array<float, 4>> rt;
                        for (const auto& [i, c] : g.root.ch)
                            if (i >= 0 && c.P >= 0.f)
                                rt.push_back({static_cast<float>(i), c.P, static_cast<float>(c.N), c.N ? c.W / c.N : -2.f});
                        std::sort(rt.begin(), rt.end());
                        g.droot.push_back(std::move(rt));
                    }
                }
                applyReal(g, k);
                continue;
            }
            const int target = g.root.expanded ? h.cfg.leaves : 1;
            int tries = 0;
            while (g.nrows < target && g.issued < g.cur_sims && tries < 4 * target) {
                ++tries;
                const SimOut r = simulate(h, g);
                ++g.issued;
                if (r == SimOut::Leaf) { ++g.nrows; ++g.st.leaves; if (!g.root.expanded) break; }
                else if (r == SimOut::Terminal) { ++g.st.terminals; ++g.st.sims; }
            }
            if (g.nrows > 0) return;
        }
        throw std::runtime_error("collect: no progress");
    } catch (const std::exception& ex) {
        g.err = std::string("error: ") + ex.what();
        g.mode = Game::Done;
        g.nrows = 0;
        g.leaves.clear();
    }
}

void feedGame(SHandle& h, Game& g, const float* logp, const float* v) {
    if (g.nrows == 0) return;
    try {
        if (g.mode == Game::Greedy) {
            const int nopt = std::min(static_cast<int>(g.live.size()), A);
            int best = 0;
            for (int j = 1; j < nopt; ++j) if (logp[j] > logp[best]) best = j;
            if (h.cfg.gen) {   // E2 generator: sample (temperature 1), as the training actors
                float tot = 0.f;
                for (int j = 0; j < nopt; ++j) tot += std::exp(logp[j]);
                float u = std::uniform_real_distribution<float>(0.f, tot)(g.grng);
                for (int j = 0; j < nopt; ++j) { best = j; u -= std::exp(logp[j]); if (u <= 0.f) break; }
                applyReal(g, best);
                g.leaves.clear(); g.nrows = 0;
                return;
            }
            {
                std::array<float, 18> d{};
                d.fill(-1.f);
                d[0] = 0; d[1] = best; d[2] = best; d[3] = std::exp(logp[best]); d[4] = d[3];
                d[10] = v[0]; d[11] = static_cast<float>(static_cast<int>(g.dec));
                g.dlog.push_back(d);
                std::vector<std::array<float, 4>> rt;
                for (int j = 0; j < nopt; ++j) rt.push_back({static_cast<float>(j), std::exp(logp[j]), 0.f, -2.f});
                g.droot.push_back(std::move(rt));
            }
            applyReal(g, best);
        } else if (g.mode == Game::Search) {
            for (int r = 0; r < g.nrows; ++r) {
                Leaf& lf = g.leaves[r];
                const float* lp = logp + static_cast<size_t>(r) * A;
                for (int j = 0; j < lf.nopt; ++j) {
                    if (lf.keys[j] < 0) continue;
                    Child& c = lf.node->ch[lf.keys[j]];
                    if (c.P < 0.f) c.P = std::exp(lp[j]);
                }
                const float vl = v[r];
                if (h.cfg.az && g.cur_record && lf.node == &g.root && !lf.node->expanded && h.cfg.dir_eps > 0.f) {
                    std::gamma_distribution<float> gam(h.cfg.dir_alpha, 1.f);
                    std::vector<float> d(lf.nopt, 0.f);
                    float ds = 0.f;
                    for (int j = 0; j < lf.nopt; ++j) if (lf.keys[j] >= 0) {
                        d[j] = gam(g.grng);
                        if (g.tap_seat && j < static_cast<int>(g.live.size()) && isRuneAbility(g.e.engine->state(), g.live[j])) d[j] *= h.cfg.dir_tap;
                        ds += d[j];
                    }
                    if (ds > 0.f)
                        for (int j = 0; j < lf.nopt; ++j) {
                            if (lf.keys[j] < 0) continue;
                            Child& c = lf.node->ch[lf.keys[j]];
                            c.P = (1.f - h.cfg.dir_eps) * c.P + h.cfg.dir_eps * d[j] / ds;
                        }
                }
                if (!lf.node->expanded) { lf.node->expanded = true; lf.node->v0 = vl; }
                backup(h.cfg, lf.path, static_cast<float>(lf.sign) * vl);
                ++g.st.sims;
            }
        }
    } catch (const std::exception& ex) {
        g.err = std::string("error: ") + ex.what();
        g.mode = Game::Done;
    }
    g.leaves.clear();
    g.nrows = 0;
}

}  // namespace rbs

extern "C" {

// cfg_f: cpuct, fpu_red, vl.  cfg_i: sims, leaves, threads, perfect, seat_mode, deck_model (0 pool, 1 decklist)
// seat_mode 0: game i searches as seat 1 if i even, 2 if odd, game seed = seed + i/2 (seat-balanced,
//              common seeds); 1: nobody searches (greedy vs greedy), same seeds.
void* rbs_create(int n_games, const char* deck_a, const char* deck_b, int64_t seed, const int32_t* cfg_i,
                 const float* cfg_f, const char* opponent, char* err, int errlen) {
    using namespace rbs;
    try {
        if (!rbe::g_inited) throw std::runtime_error("rbenv_init first");
        auto h = std::make_unique<SHandle>();
        h->cfg.sims = cfg_i[0];
        h->cfg.leaves = std::max(1, cfg_i[1]);
        const int threads = std::max(0, cfg_i[2]);
        h->cfg.perfect = cfg_i[3] != 0;
        h->cfg.deck_model = cfg_i[5];
        const int seat_mode = cfg_i[4];
        h->cfg.cpuct = cfg_f[0];
        h->cfg.fpu_red = cfg_f[1];
        h->cfg.vl = cfg_f[2];
        h->cfg.gen = seat_mode == 2;
        h->cfg.p_search = cfg_f[3];
        // az (cfg_i[6..8], cfg_f[4..6]); callers pass arrays of 12 (rbsearch.py)
        h->cfg.az = cfg_i[6] != 0 && h->cfg.gen;
        h->cfg.az_temp_moves = cfg_i[7];
        h->cfg.sims_fast = cfg_i[8];
        h->cfg.runes = cfg_i[9];
#ifndef RB_MANUAL_RUNES
        if (h->cfg.runes) throw std::runtime_error("this engine has no manual runes");
#endif
        h->cfg.dir_alpha = cfg_f[4]; h->cfg.dir_eps = cfg_f[5]; h->cfg.p_full = cfg_f[6];
        if (cfg_f[7] > 0.f) h->cfg.dir_tap = cfg_f[7];
        h->deck_a_path = deck_a; h->deck_b_path = deck_b; h->opp_spec = opponent ? opponent : "";
        auto split = [](const std::string& x) {
            std::vector<std::string> out;
            std::string cur;
            for (char ch : x + "|") {
                if (ch == '|') { if (!cur.empty()) out.push_back(cur); cur.clear(); }
                else cur += ch;
            }
            return out;
        };
        h->pa = split(deck_a); h->pb = split(deck_b);
        if (h->pa.empty() || h->pa.size() != h->pb.size()) throw std::runtime_error("deck lists: equal non-empty '|' lists");
        for (size_t m = 0; m < h->pa.size(); ++m) {
            h->das.push_back(DeckValidator::loadFromDeckList(h->pa[m], *g_db));
            h->dbs.push_back(DeckValidator::loadFromDeckList(h->pb[m], *g_db));
        }
        h->da = h->das[0]; h->db = h->dbs[0];
        h->sampler = std::make_unique<WorldSampler>(*g_db);
        h->games.resize(n_games);
        for (int i = 0; i < n_games; ++i) {
            Game& g = h->games[i];
            g.idx = i;
            g.mu = i % static_cast<int>(h->pa.size());
            g.seed = seed + i / 2;
            g.search_seat = seat_mode == 0 ? (i % 2 == 0 ? 1 : 2) : 0;
            g.e.idx = i;
            g.e.game_seed = g.seed;
            if (h->cfg.gen) { g.seed = seed + i; g.e.game_seed = g.seed; }
            g.grng.seed(mix64(static_cast<uint64_t>(g.seed) * 2654435761ull + 12345ull));
        }
        h->pool = std::make_unique<Pool>(threads);
        return h.release();
    } catch (const std::exception& ex) {
        std::snprintf(err, errlen, "%s", ex.what());
        return nullptr;
    }
}

void rbs_destroy(void* p) { delete static_cast<rbs::SHandle*>(p); }

// Advances every game to its next network request. Returns the number of rows.
int rbs_collect(void* p) {
    using namespace rbs;
    auto& h = *static_cast<SHandle*>(p);
    const int n = static_cast<int>(h.games.size());
    std::function<void(int)> f = [&](int i) { collectGame(h, h.games[i]); };
    h.pool->run(n, f);
    h.offs.assign(n, 0);
    int tot = 0;
    for (int i = 0; i < n; ++i) { h.offs[i] = tot; tot += h.games[i].nrows; }
    h.total_rows = tot;
    return tot;
}

// Copies the rows of the last collect. meta per row: pk, psrc, nlegal, seat, game, kind (0 greedy, 1 leaf)
void rbs_rows(void* p, int32_t* glob, int32_t* tok, uint8_t* tmask, int32_t* act, uint8_t* amask, int32_t* meta) {
    using namespace rbs;
    auto& h = *static_cast<SHandle*>(p);
    std::function<void(int)> f = [&](int i) {
        const Game& g = h.games[i];
        for (int r = 0; r < g.nrows; ++r) {
            const ObsBuf& b = g.bufs[r];
            const size_t o = static_cast<size_t>(h.offs[i] + r);
            std::memcpy(glob + o * G, b.glob.data(), G * 4);
            std::memcpy(tok + o * T * F, b.tok.data(), T * F * 4);
            std::memcpy(tmask + o * T, b.tmask.data(), T);
            std::memcpy(act + o * A * K, b.act.data(), A * K * 4);
            std::memcpy(amask + o * A, b.amask.data(), A);
            int32_t* m = meta + o * 6;
            m[0] = b.pk; m[1] = b.psrc; m[2] = b.nlegal; m[3] = b.seat; m[4] = i;
            m[5] = g.mode == Game::Search ? 1 : 0;
        }
    };
    h.pool->run(static_cast<int>(h.games.size()), f);
}

// history.h: the rows' histories (rows, kHistLen, kHistF), same row order as rbs_rows.
void rbs_rows_hist(void* p, int32_t* hist) {
    using namespace rbs;
    auto& h = *static_cast<SHandle*>(p);
    const size_t n1 = static_cast<size_t>(kHistLen) * kHistF;
    std::function<void(int)> f = [&](int i) {
        const Game& g = h.games[i];
        for (int r = 0; r < g.nrows; ++r)
            std::memcpy(hist + static_cast<size_t>(h.offs[i] + r) * n1, g.bufs[r].hist.data(), n1 * 4);
    };
    h.pool->run(static_cast<int>(h.games.size()), f);
}

// logp (rows, A) float32, v (rows) float32, in row order of the last collect.
void rbs_feed(void* p, const float* logp, const float* v) {
    using namespace rbs;
    auto& h = *static_cast<SHandle*>(p);
    std::function<void(int)> f = [&](int i) {
        feedGame(h, h.games[i], logp + static_cast<size_t>(h.offs[i]) * A, v + h.offs[i]);
    };
    h.pool->run(static_cast<int>(h.games.size()), f);
}

int rbs_n_done(void* p) {
    int n = 0;
    for (const auto& g : static_cast<rbs::SHandle*>(p)->games) n += g.mode == rbs::Game::Done ? 1 : 0;
    return n;
}

// Per game: seed, search_seat, done, winner, turns, n_search, n_greedy, error(0/1), sims
void rbs_results(void* p, int64_t* out) {
    const auto& h = *static_cast<rbs::SHandle*>(p);
    for (size_t i = 0; i < h.games.size(); ++i) {
        const auto& g = h.games[i];
        int64_t* r = out + i * 9;
        r[0] = g.seed; r[1] = g.search_seat; r[2] = g.mode == rbs::Game::Done ? 1 : 0; r[3] = g.winner;
        r[4] = g.turns; r[5] = g.n_search; r[6] = g.n_greedy; r[7] = g.err.empty() ? 0 : 1; r[8] = g.st.sims;
    }
}

// Decision log of game i (18 floats per external decision, see Game::dlog); returns the count.
// Every option of game i's decision d (an index into its decision log): live index, prior, visits, mean value for
// the decider (-2 unvisited), 4 floats each; returns how many (-1: no such decision).
int rbs_root(void* p, int i, int d, float* out, int cap) {
    const auto& g = static_cast<rbs::SHandle*>(p)->games.at(i);
    if (d < 0 || d >= static_cast<int>(g.droot.size())) return -1;
    const auto& rt = g.droot[d];
    const int n = static_cast<int>(rt.size());
    for (int r = 0; r < n && r < cap; ++r) for (int k = 0; k < 4; ++k) out[r * 4 + k] = rt[r][k];
    return n;
}

int rbs_declog(void* p, int i, float* out, int cap) {
    const auto& g = static_cast<rbs::SHandle*>(p)->games.at(i);
    const int n = static_cast<int>(g.dlog.size());
    for (int r = 0; r < n && r < cap; ++r) for (int k = 0; k < 18; ++k) out[r * 18 + k] = g.dlog[r][k];
    return n;
}

int rbs_error(void* p, int i, char* out, int cap) {
    const auto& g = static_cast<rbs::SHandle*>(p)->games.at(i);
    std::snprintf(out, cap, "%s", g.err.c_str());
    return static_cast<int>(g.err.size());
}

// Summed stats: sims leaves terminals fails reexpand fallbacks searches greedy depth_sum collisions
//   ns_build ns_edit ns_descend ns_encode ns_teardown ns_snap replay_steps unk_key trunc_tok trunc_act
void rbs_stats(void* p, int64_t* out) {
    const auto& h = *static_cast<rbs::SHandle*>(p);
    for (int k = 0; k < 22; ++k) out[k] = 0;
    for (const auto& g : h.games) {
        const auto& s = g.st;
        const int64_t v[22] = {s.sims, s.leaves, s.terminals, s.fails, s.reexpand, s.fallbacks, s.searches, s.greedy,
                               s.depth_sum, s.collisions, s.ns_build, s.ns_edit, s.ns_descend, s.ns_encode,
                               s.ns_teardown, s.ns_snap, s.replay_steps, g.c.unk_key, g.c.trunc_tok, g.c.trunc_act,
                               s.changed, s.dl_fallback};
        for (int k = 0; k < 22; ++k) out[k] += v[k];
    }
}

// Runes as actions, summed over the games in their slots: taps to exhaust, to recycle, external decisions that
// offered a tap, external decisions.
void rbs_taps(void* p, int64_t* out) {
    const auto& h = *static_cast<rbs::SHandle*>(p);
    for (int k = 0; k < 4; ++k) { out[k] = 0; for (const auto& g : h.games) out[k] += g.taps[k]; }
}
void rbs_taps_game(void* p, int i, int64_t* out) {
    const auto& g = static_cast<rbs::SHandle*>(p)->games.at(i);
    for (int k = 0; k < 4; ++k) out[k] = g.taps[k];
}

// E2 generator: copies and clears the records of game i (only once it is Done, so the outcome z
// is known; returns 0 otherwise). meta per record: pk, psrc, nlegal, seat. fl per record (3A + 3):
// P[A], N[A], Q[A], v0, vs, z (decider's view: +1 win, -1 loss, 0 draw / error).
int rbs_gen_take(void* p, int i, int cap, int32_t* glob, int32_t* tok, uint8_t* tmask, int32_t* act, uint8_t* amask,
                 int32_t* meta, float* fl, int32_t* hist) {
    using namespace rbs;
    auto& g = static_cast<SHandle*>(p)->games.at(i);
    if (g.mode != Game::Done) return 0;
    const int n = std::min(cap, static_cast<int>(g.recs.size()));
    for (int r = 0; r < n; ++r) {
        const GenRec& x = g.recs[r];
        std::memcpy(glob + static_cast<size_t>(r) * G, x.obs.glob.data(), G * 4);
        std::memcpy(tok + static_cast<size_t>(r) * T * F, x.obs.tok.data(), T * F * 4);
        std::memcpy(tmask + static_cast<size_t>(r) * T, x.obs.tmask.data(), T);
        std::memcpy(act + static_cast<size_t>(r) * A * K, x.obs.act.data(), A * K * 4);
        std::memcpy(amask + static_cast<size_t>(r) * A, x.obs.amask.data(), A);
        if (hist) std::memcpy(hist + static_cast<size_t>(r) * kHistLen * kHistF, x.obs.hist.data(), kHistLen * kHistF * 4);
        int32_t* m = meta + r * 4;
        m[0] = x.obs.pk; m[1] = x.obs.psrc; m[2] = x.obs.nlegal; m[3] = x.seat;
        float* f = fl + static_cast<size_t>(r) * (3 * A + 3);
        std::memcpy(f, x.P.data(), A * 4);
        std::memcpy(f + A, x.N.data(), A * 4);
        std::memcpy(f + 2 * A, x.Q.data(), A * 4);
        f[3 * A] = x.v0; f[3 * A + 1] = x.vs;
        f[3 * A + 2] = (!g.err.empty() || g.winner == 0) ? 0.f : (g.winner == x.seat ? 1.f : -1.f);
    }
    g.recs.clear();
    return n;
}

// E2 generator: replaces game slot i (must be Done) by a fresh game with seed `seed`.
int rbs_restart(void* p, int i, int64_t seed) {
    using namespace rbs;
    auto& h = *static_cast<SHandle*>(p);
    Game& old = h.games.at(i);
    if (old.mode != Game::Done) return -1;
    old.e.engine.reset();      // engine before its event bus
    old.e.bus.reset();
    Game g;
    g.idx = i; g.seed = seed; g.search_seat = 0;
    g.mu = old.mu;
    g.e.idx = i; g.e.game_seed = seed;
    g.grng.seed(mix64(static_cast<uint64_t>(seed) * 2654435761ull + 12345ull));
    h.games[i] = std::move(g);
    return 0;
}

// M0 leak test, at every game now waiting at a real decision (after a collect, before feed):
// `worlds` worlds per decision; each must encode byte-identically to the truth for the decider.
// out (adds): 0 decisions main-phase  1 decisions other  2 worlds main  3 worlds other
//   4 mismatches main  5 mismatches other  6 worlds whose hidden cards differ from the truth
//   7 WorldSampler::check problems  8 world build failures  9 opp-hidden objects compared
//   10 of which redrawn to another card
//   11 negative control: worlds whose encoding for the OPPONENT (whose hidden cards were redrawn)
//      differs from the truth encoded for the opponent (must be > 0: the comparison can see a change)
// msg: the first mismatch / failure.
void rbs_leaktest(void* p, int worlds, int64_t* out, char* msg, int cap) {
    using namespace rbs;
    auto& h = *static_cast<SHandle*>(p);
    std::mutex mu;
    std::string first;
    std::function<void(int)> f = [&](int i) {
        Game& g = h.games[i];
        if (g.mode != Game::Greedy && g.mode != Game::Search) return;
        const GameEngine& truth = *g.e.engine;
        const StepResult& sr = truth.currentStep();
        const PlayerId dec = g.dec;
        const PlayerId opp = dec == PlayerId::Player1 ? PlayerId::Player2 : PlayerId::Player1;
        const bool main = sr.prompt.kind == PromptKind::MainPhase;
        ObsBuf ref, got, ref_opp, got_opp;
        Counters cc;
        encodeTo(ref, truth, &g.e.mtrack[static_cast<int>(dec) % 3], sr, dec, cc, g.e.hev);
        int64_t loc[12] = {0};
        encodeTo(ref_opp, truth, &g.e.mtrack[static_cast<int>(opp) % 3], sr, opp, cc, g.e.hev);
        loc[main ? 0 : 1]++;
        std::mt19937_64 rng(mix64(static_cast<uint64_t>(g.seed) * 31 + static_cast<uint64_t>(g.decision_no)));
        std::string why;
        const PlayerId saved_root = g.root_player;
        for (int w = 0; w < worlds; ++w) {
            loc[main ? 2 : 3]++;
            try {
                const uint64_t ds = rng(), es = rng();
                SWorld wd = buildWorld(h, g, dec, false, ds, es);
                encodeTo(got, *wd.eng, &g.e.mtrack[static_cast<int>(dec) % 3], wd.eng->currentStep(), dec, cc, g.e.hev);
                std::string wy;
                if (!sameObs(ref, got, &wy)) {
                    loc[main ? 4 : 5]++;
                    if (std::getenv("RBS_LEAK_DUMP") || why.empty()) {
                        std::string d = "game " + std::to_string(i) + " pk " + std::to_string(static_cast<int>(sr.prompt.kind)) + ": " + wy;
                        // the first differing token: both rows, its object in truth and world
                        for (int t = 0; t < T; ++t) {
                            bool same = ref.tmask[t] == got.tmask[t];
                            for (int k = 0; k < F && same; ++k) same = ref.tok[t * F + k] == got.tok[t * F + k];
                            if (same) continue;
                            auto rowStr = [&](const ObsBuf& b, const GameState& s) {
                                std::string r = "[";
                                for (int k = 0; k < F; ++k) r += std::to_string(b.tok[t * F + k]) + (k + 1 < F ? "," : "]");
                                const int64_t oid = b.tok_obj[t];
                                if (oid >= 0) {
                                    const auto id = GameObjectId::fromRaw(static_cast<uint32_t>(oid));
                                    if (s.objectExists(id)) {
                                        const auto& o = s.getObject(id);
                                        r += " obj " + std::to_string(oid) + " " + cardKey(o.card_def_id, &o) + " zone " +
                                             std::to_string(static_cast<int>(o.zone.get())) + " owner " +
                                             std::to_string(static_cast<int>(o.owner)) + " vis " +
                                             std::to_string(knowledge::identityVisible(s, id, dec) ? 1 : 0);
                                    }
                                }
                                return r;
                            };
                            d += " tok " + std::to_string(t) + " truth " + rowStr(ref, truth.state()) + " world " +
                                 rowStr(got, wd.eng->state());
                            // where the world's object sits in the truth
                            for (int t2 = 0; t2 < T; ++t2)
                                if (ref.tmask[t2] && ref.tok_obj[t2] == got.tok_obj[t]) {
                                    const int tt = t;
                                    t = t2;
                                    d += " | truth tok " + std::to_string(t2) + " " + rowStr(ref, truth.state());
                                    t = tt;
                                }
                            break;
                        }
                        if (std::getenv("RBS_LEAK_DUMP")) std::fprintf(stderr, "[leak] %s\n", d.c_str());
                        if (why.empty()) why = d;
                    }
                }
                const GameState& a = truth.state();
                const GameState& b = wd.eng->state();
                bool differ = false;
                for (const auto& [id, o] : a.objects) {
                    if (o.owner != opp) continue;
                    const ZoneType z = o.zone;
                    if (z != ZoneType::Hand && z != ZoneType::MainDeck && z != ZoneType::FacedownZone) continue;
                    if (knowledge::identityVisible(a, id, dec)) continue;
                    loc[9]++;
                    if (b.objectExists(id) && b.getObject(id).card_def_id != o.card_def_id) { loc[10]++; differ = true; }
                }
                if (differ) loc[6]++;
                loc[7] += static_cast<int64_t>(h.sampler->check(a, b, dec).size());
                encodeTo(got_opp, *wd.eng, &g.e.mtrack[static_cast<int>(opp) % 3], wd.eng->currentStep(), opp, cc, g.e.hev);
                if (!sameObs(ref_opp, got_opp, nullptr)) loc[11]++;
            } catch (const std::exception& ex) {
                loc[8]++;
                if (why.empty()) why = std::string("build: ") + ex.what();
            }
        }
        g.root_player = saved_root;
        std::lock_guard<std::mutex> lk(mu);
        for (int k = 0; k < 12; ++k) out[k] += loc[k];
        if (first.empty() && !why.empty()) first = why;
    };
    h.pool->run(static_cast<int>(h.games.size()), f);
    std::snprintf(msg, cap, "%s", first.c_str());
}

}  // extern "C"

extern "C" {
// Strict visibility of the encoder (decision_export.cpp g_strict_vis), process-wide. The search
// harness turns it on for both seats; rbenv training observations leave it off.
void rbs_set_strict_vis(int on) { riftbound::probe::dx::g_strict_vis = on != 0; }
int rbs_get_strict_vis() { return riftbound::probe::dx::g_strict_vis ? 1 : 0; }
}
