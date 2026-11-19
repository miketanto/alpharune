// autopay_match.cpp -- does auto-paying rune costs weaken the ISMCTS teacher?
//
// Research tool (WSL build copy only, EXCLUDE_FROM_ALL target autopay_match,
// autopay_match.cmake). Never part of src/. It reuses the decision exporter
// (research/card_repr/smoke/data/src/decision_export.cpp: Seat, GameCtx,
// replay/verify, fingerprints) by #including it with its main() renamed.
//
// Agents (per seat, agent seed = engine_seed*2 + seat as in the exporter):
//   ismcts:sims=N    IsMctsAgent (the teacher), unchanged.
//   autopay:sims=N   the same IsMctsAgent, except that at every
//                    PromptKind::RuneForEnergy / RuneForPower decision it
//                    returns legal.front() (the engine's default order)
//                    WITHOUT calling the agent. All other decisions go to the
//                    agent. The agent's internal search still branches on
//                    rune choices inside its tree.
//   Mirror: IsMctsAgent's LiveMirror re-syncs from GameState::action_history
//   at the next decision it is asked, replaying every token made since (its
//   own skipped rune answers included, exactly like the opponent's
//   decisions). --check-mirror additionally runs an independent LiveMirror
//   at every decision of every seat and counts decisions where it is not an
//   exact replay of the live game.
//
// Modes
//   play   <games.txt> <out_prefix> [--shard i n] [--check-mirror]
//          games.txt lines: <gid> <deck A> <deck B> <agent A> <agent B> <seed> <swapcand>
//          writes <out_prefix>.games.jsonl (one line per game, below).
//   verify <jobs.txt>   = decision_export verify (seed + choices replay,
//          fingerprint + observation hash at every multi-option decision).
//
// Game line: gid, deck_a, deck_b, agent_a, agent_b, seed, engine_seed, winner
// (1 deck A / 2 deck B / 0 none), term, turns, scores, n_seq, secs,
// side_a / side_b: {dec (decisions asked), multi (>=2 options), searches
// (IsMctsSearch runs), secs (wall time inside this seat's decide), rune
// (rune prompts), rune_multi, autopaid (rune prompts answered with
// legal.front() without the agent), autopaid_multi, rune_searched, rune_nonfront
// (searched rune prompts where the agent did NOT pick legal.front()),
// unsearched_multi (multi-option decisions after setup the agent answered
// without a search: a broken mirror shows here), mirror_checked, mirror_bad},
// choices (chosen index for every decision), checks ([seq, fp, hash]).

#define main decision_export_unused_main
#include "research/card_repr/smoke/data/src/decision_export.cpp"
#undef main

namespace riftbound {
namespace probe {
namespace ap {

using namespace riftbound::probe::dx;

struct SideStats {
    long dec = 0, multi = 0, searches = 0, rune = 0, rune_multi = 0, autopaid = 0, autopaid_multi = 0,
         rune_searched = 0, rune_nonfront = 0, unsearched_multi = 0, mirror_checked = 0, mirror_bad = 0;
    double secs = 0;
    std::string json() const {
        std::ostringstream o;
        o << "{\"dec\":" << dec << ",\"multi\":" << multi << ",\"searches\":" << searches << ",\"secs\":" << fnum(secs)
          << ",\"rune\":" << rune << ",\"rune_multi\":" << rune_multi << ",\"autopaid\":" << autopaid
          << ",\"autopaid_multi\":" << autopaid_multi << ",\"rune_searched\":" << rune_searched
          << ",\"rune_nonfront\":" << rune_nonfront << ",\"unsearched_multi\":" << unsearched_multi
          << ",\"mirror_checked\":" << mirror_checked << ",\"mirror_bad\":" << mirror_bad << "}";
        return o.str();
    }
};

bool isRune(PromptKind k) { return k == PromptKind::RuneForEnergy || k == PromptKind::RuneForPower; }

class CountingAgent : public AgentInterface {
public:
    CountingAgent(std::unique_ptr<IsMctsAgent> a, bool autopay, SideStats& st,
                  std::unique_ptr<LiveMirror> check)
        : inner_(std::move(a)), autopay_(autopay), st_(st), check_(std::move(check)) {}
    Intent selectAction(const GameState& s, const std::vector<Intent>& legal) override {
        return decide(s, legal, Prompt{});
    }
    Intent decide(const GameState& s, const std::vector<Intent>& legal, const Prompt& p) override {
        const auto t0 = std::chrono::steady_clock::now();
        ++st_.dec;
        const bool multi = legal.size() >= 2, rune = isRune(p.kind);
        st_.multi += multi;
        st_.rune += rune;
        st_.rune_multi += rune && multi;
        if (check_) {
            ++st_.mirror_checked;
            const auto* root = check_->rootFor(s, legal);
            if (!check_->exact() || !root) ++st_.mirror_bad;
        }
        Intent out;
        if (autopay_ && rune && !legal.empty()) {
            out = legal.front();
            ++st_.autopaid;
            st_.autopaid_multi += multi;
        } else {
            out = inner_->decide(s, legal, p);
            const SearchSummary& ss = inner_->lastSearch();
            if (ss.searched) {
                ++st_.searches;
                if (rune) {
                    ++st_.rune_searched;
                    if (!(out == legal.front())) ++st_.rune_nonfront;
                }
            } else if (multi && !s.battlefields.empty()) {
                ++st_.unsearched_multi;
            }
        }
        st_.secs += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return out;
    }
private:
    std::unique_ptr<IsMctsAgent> inner_;
    bool autopay_;
    SideStats& st_;
    std::unique_ptr<LiveMirror> check_;
};

std::unique_ptr<AgentInterface> makeSide(const std::string& spec, const GameSpec& g, uint64_t es, int seat,
                                         SideStats& st, bool check_mirror) {
    const uint64_t derived = es * 2 + seat;
    bool autopay;
    std::string rest;
    if (spec.rfind("ismcts:sims=", 0) == 0) { autopay = false; rest = spec.substr(12); }
    else if (spec.rfind("autopay:sims=", 0) == 0) { autopay = true; rest = spec.substr(13); }
    else throw std::invalid_argument("unknown agent " + spec);
    auto a = std::make_unique<IsMctsAgent>(g.deck_a, g.deck_b, "", es, derived, std::stoi(rest));
    std::unique_ptr<LiveMirror> chk;
    if (check_mirror) {
        auto game = ::open_spiel::LoadGame(buildGameStr(g.deck_a, g.deck_b, es, MatchFormat::BestOf1));
        chk = std::make_unique<LiveMirror>(game, es, derived ^ 0x77u);
    }
    return std::make_unique<CountingAgent>(std::move(a), autopay, st, std::move(chk));
}

void play(const std::vector<GameSpec>& games, const std::string& prefix, int shard, int nshards, bool check_mirror) {
    std::ofstream gout(prefix + ".games.jsonl");
    const auto T0 = std::chrono::steady_clock::now();
    for (size_t gi = 0; gi < games.size(); ++gi) {
        if (static_cast<int>(gi % nshards) != shard) continue;
        GameCtx ctx;
        ctx.g = games[gi];
        ctx.record = true;   // only for the fingerprint/hash checks; decision lines are dropped
        const auto t0 = std::chrono::steady_clock::now();
        const uint64_t es = engineSeedOf(ctx.g.seed);
        SideStats sa, sb;
        RunOut r;
        try {
            const DeckSubmission a = DeckValidator::loadFromDeckList(ctx.g.deck_a, *g_db);
            const DeckSubmission b = DeckValidator::loadFromDeckList(ctx.g.deck_b, *g_db);
            auto i1 = makeSide(ctx.g.agent_a, ctx.g, es, 0, sa, check_mirror);
            auto i2 = makeSide(ctx.g.agent_b, ctx.g, es, 1, sb, check_mirror);
            Seat s1(std::move(i1), nullptr, ctx.g.agent_a, ctx), s2(std::move(i2), nullptr, ctx.g.agent_b, ctx);
            EventBus events;
            GameEngine engine(*g_db, events, *g_reg);
            r.res = engine.runGame(a, b, s1, s2, es);
            r.term = r.res.termination_reason;
        } catch (const std::exception& ex) {
            r.term = std::string("error: ") + ex.what();
        }
        r.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const int w = winnerCode(r.res.winner);
        std::ostringstream o;
        o << "{\"gid\":" << jstr(ctx.g.gid) << ",\"deck_a\":" << jstr(ctx.g.deck_a) << ",\"deck_b\":" << jstr(ctx.g.deck_b)
          << ",\"agent_a\":" << jstr(ctx.g.agent_a) << ",\"agent_b\":" << jstr(ctx.g.agent_b)
          << ",\"seed\":" << ctx.g.seed << ",\"engine_seed\":" << es << ",\"winner\":" << w
          << ",\"term\":" << jstr(r.term) << ",\"turns\":" << r.res.total_turns << ",\"scores\":["
          << r.res.final_scores[0] << "," << r.res.final_scores[1] << "],\"n_seq\":" << ctx.seq
          << ",\"secs\":" << fnum(r.secs) << ",\"side_a\":" << sa.json() << ",\"side_b\":" << sb.json()
          << ",\"choices\":[";
        for (size_t j = 0; j < ctx.choices.size(); ++j) o << (j ? "," : "") << ctx.choices[j];
        o << "],\"checks\":[";
        for (size_t j = 0; j < ctx.checks.size(); ++j) o << (j ? "," : "") << ctx.checks[j];
        o << "]}\n";
        gout << o.str();
        gout.flush();
        std::cerr << ctx.g.gid << " " << r.term << " seq " << ctx.seq << " " << r.secs << " s\n";
    }
    std::cerr << "shard " << shard << " done in "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - T0).count() << " s\n";
}

}  // namespace ap
}  // namespace probe
}  // namespace riftbound

int main(int argc, char** argv) {
    using namespace riftbound;
    using namespace riftbound::probe;
    if (argc < 3) { std::cerr << "usage: see autopay_match.cpp\n"; return 2; }
    std::string subjects = "research/card_repr/C_engine/out/subjects.txt";
    int shard = 0, nshards = 1;
    bool check_mirror = false;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shard") { shard = std::stoi(argv[++i]); nshards = std::stoi(argv[++i]); }
        else if (a == "--check-mirror") check_mirror = true;
        else if (a == "--subjects") subjects = argv[++i];
        else pos.push_back(a);
    }
    auto& shared = const_cast<openspiel::SharedCards&>(openspiel::sharedCards());
    g_reg = &shared.registry;
    g_db = &shared.db;
    dx::loadKeys(subjects);
    const std::string mode = pos[0];
    if (mode == "play") {
        const auto games = dx::readGames(pos[1]);
        std::cerr << games.size() << " games in " << pos[1] << "\n";
        ap::play(games, pos[2], shard, nshards, check_mirror);
        return 0;
    }
    if (mode == "verify") { dx::verifyMode(pos[1]); return 0; }
    return 2;
}
