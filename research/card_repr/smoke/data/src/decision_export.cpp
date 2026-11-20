// decision_export.cpp -- decision dataset for the minimal network
// (research/card_repr/NETWORK.md; smoke test of card representations).
//
// Research tool, built only in the WSL build copy as an EXCLUDE_FROM_ALL
// target (decision_export.cmake). Never part of src/; test-only card classes
// come from ../../../C_engine/card_probe.cpp.
//
// Modes
//   play   <games.txt> <out_prefix> [--variant V] [--shard i n]
//          plays every game line, records every decision with >= 2 legal
//          options of both seats: an observation from the decider's view
//          (global features + entity tokens keyed by CARD KEY), the legal
//          actions as kind + token references + scalars, the agent's choice,
//          and for an ismcts agent its root visit distribution, per-option
//          mean return and root value; the game result is appended at the end.
//          Writes <out_prefix>.decisions.jsonl and <out_prefix>.games.jsonl.
//          games.txt lines: <gid> <deck A> <deck B> <agent A> <agent B> <seed> <swapcand 0|1>
//   verify <jobs.txt> [--variant V]
//          replays games from their seed + recorded choice indices (no agent)
//          and checks the state fingerprint and the hash of the exported
//          observation+actions at every recorded decision, and the winner.
//          jobs lines: <gid> <deck A> <deck B> <seed> <winner> <n> c1..cn <m> (seq fp hash)*m
//   swap   <jobs.txt> <out.jsonl> --variant V [--sims N] [--seeds K]
//          replays a game to a main-phase decision where a base card is in
//          the decider's hand and playable, then searches the position with
//          the teacher (IsMctsSearch from a snapshot) K times, once as it is
//          and once with every copy of the base card in the decider's hand
//          replaced by the variant V; records observation, actions and the
//          search results of both versions.
//          jobs lines: <jid> <gid> <deck A> <deck B> <seed> <seq> <fp> <base key> <n> c1..cn
//
// --variant V registers ONE test card into the engine's shared card table
// (the table the OpenSpiel game, the search agents' mirror and the world
// sampler read) before anything else runs, so a variant deck list can name it.
// Base games register nothing.

#include "agents/mcts_agent.cpp"   // IsMctsSearch, RiftboundHeuristicEvaluator, buildGameStr (TU-local)

#define CARD_PROBE_NO_MAIN
#include "research/card_repr/C_engine/card_probe.cpp"

#include "agents/random_agent.h"
#include "agents/random_player_agent.h"
#include "engine/knowledge_tracker.h"
#include "engine/observer_view.h"
#include "rules/deck_validator.h"

#include <array>
#include <random>
#include <set>
#include <unordered_map>

namespace riftbound {
namespace probe {
namespace dx {

// ─── Variants ─────────────────────────────────────────────────────────────
// Same classes, ids and def tags as card_probe.cpp's registerSynthetics, one
// at a time. primal_strength is a real engine card (154): nothing to add.
struct VariantInfo {
    std::string name;       // as in cards.json / vectors.json
    std::string key;        // card key written in the observation
    uint32_t id;            // engine class id
    std::vector<uint32_t> base_ids;
    std::string base_key;
};

const std::vector<VariantInfo>& variants() {
    static const std::vector<VariantInfo> v = {
        {"primal_strength", "primal_strength", 154, {419}, "punch_first"},
        {"synthetic:punch_first__amount4", "synthetic:punch_first__amount4", 1611, {419}, "punch_first"},
        {"synthetic:punch_first__energy3", "synthetic:punch_first__energy3", 1612, {419}, "punch_first"},
        {"synthetic:punch_first__power0", "synthetic:punch_first__power0", 1613, {419}, "punch_first"},
        {"synthetic:punch_first__reaction", "synthetic:punch_first__reaction", 1614, {419}, "punch_first"},
        {"synthetic:discipline__draw2", "synthetic:discipline__draw2", 1615, {58}, "discipline"},
        {"synthetic:discipline__enemy", "synthetic:discipline__enemy", 1616, {58}, "discipline"},
        {"synthetic:lonely_poro__draw2", "synthetic:lonely_poro__draw2", 1617, {359, 777}, "lonely_poro"},
        {"synthetic:defy__cost2", "synthetic:defy__cost2", 1618, {45}, "defy"},
        {"synthetic:sabotage__unit", "synthetic:sabotage__unit", 1619, {156}, "sabotage"},
        {"synthetic:pit_rookie__might3", "synthetic:pit_rookie__might3", 1620, {136}, "pit_rookie"},
        {"synthetic:punch_first__score1", "synthetic:punch_first__score1", 1631, {419}, "punch_first"},
        {"synthetic:punch_first__score_nounits", "synthetic:punch_first__score_nounits", 1632, {419}, "punch_first"},
        {"synthetic:punch_first__kill", "synthetic:punch_first__kill", 1633, {419}, "punch_first"},
        {"synthetic:pit_rookie__score1", "synthetic:pit_rookie__score1", 1634, {136}, "pit_rookie"},
        {"synthetic:pit_rookie__score_alone", "synthetic:pit_rookie__score_alone", 1635, {136}, "pit_rookie"},
        {"synthetic:sabotage__score1", "synthetic:sabotage__score1", 1636, {156}, "sabotage"},
        {"synthetic:sabotage__score_opphand4", "synthetic:sabotage__score_opphand4", 1637, {156}, "sabotage"},
        {"synthetic:punch_first__score_bfunit", "synthetic:punch_first__score_bfunit", 1638, {419}, "punch_first"},
        {"synthetic:punch_first__score_behind", "synthetic:punch_first__score_behind", 1639, {419}, "punch_first"},
        {"synthetic:punch_first__minus5", "synthetic:punch_first__minus5", 1640, {419}, "punch_first"},
        {"synthetic:punch_first__ctrl_scores", "synthetic:punch_first__ctrl_scores", 1641, {419}, "punch_first"},
        {"synthetic:punch_first__opp_of_ctrl_scores", "synthetic:punch_first__opp_of_ctrl_scores", 1642, {419}, "punch_first"},
    };
    return v;
}

const VariantInfo* findVariant(const std::string& n) {
    for (const auto& v : variants()) if (v.name == n) return &v;
    return nullptr;
}

void registerVariant(CardRegistry& reg, const std::string& name) {
    constexpr CardDefId kPunch = 419, kDisc = 58, kPoro = 777, kDefy = 45, kSab = 156, kRookie = 136;
    auto add = [&](uint32_t id, std::unique_ptr<Card> c) { reg.registerCard(CardDefId::fromRaw(id), std::move(c)); };
    if (name == "primal_strength") return;
    if (name == "synthetic:punch_first__amount4") add(1611, std::make_unique<PunchFirstVar>(derive(reg, kPunch, 1611, "g"), 4));
    else if (name == "synthetic:punch_first__energy3") { auto d = derive(reg, kPunch, 1612, "h"); d.energy_cost = 3; add(1612, std::make_unique<PunchFirstVar>(d, 5)); }
    else if (name == "synthetic:punch_first__power0") { auto d = derive(reg, kPunch, 1613, "i"); d.power_cost = 0; add(1613, std::make_unique<PunchFirstVar>(d, 5)); }
    else if (name == "synthetic:punch_first__reaction") {
        auto d = derive(reg, kPunch, 1614, "j");
        d.keywords.clear(Keyword::Action); d.keywords.set(Keyword::Reaction);
        add(1614, std::make_unique<PunchFirstVar>(d, 5));
    }
    else if (name == "synthetic:discipline__draw2") add(1615, std::make_unique<DisciplineVar>(derive(reg, kDisc, 1615, "k"), 2, false));
    else if (name == "synthetic:discipline__enemy") add(1616, std::make_unique<DisciplineVar>(derive(reg, kDisc, 1616, "l"), 1, true));
    else if (name == "synthetic:lonely_poro__draw2") add(1617, std::make_unique<LonelyPoroVar>(derive(reg, kPoro, 1617, "m"), 2));
    else if (name == "synthetic:defy__cost2") add(1618, std::make_unique<CounterVar>(derive(reg, kDefy, 1618, "n"), CounterVar::Rule::EnergyAndPowerCap, 2, 1));
    else if (name == "synthetic:sabotage__unit") add(1619, std::make_unique<SabotageVar>(derive(reg, kSab, 1619, "o"), true));
    else if (name == "synthetic:pit_rookie__might3") { auto d = derive(reg, kRookie, 1620, "p"); d.might = 3; add(1620, std::make_unique<PitRookieVar>(d)); }
    else if (name == "synthetic:punch_first__score1") add(1631, std::make_unique<ScoreSpellVar>(derive(reg, kPunch, 1631, "q1"), ScoreCond::Always));
    else if (name == "synthetic:punch_first__score_nounits") add(1632, std::make_unique<ScoreSpellVar>(derive(reg, kPunch, 1632, "q2"), ScoreCond::NoFriendlyUnits));
    else if (name == "synthetic:punch_first__kill") add(1633, std::make_unique<PunchKillVar>(derive(reg, kPunch, 1633, "q3")));
    else if (name == "synthetic:pit_rookie__score1") add(1634, std::make_unique<ScoreUnitVar>(derive(reg, kRookie, 1634, "q4"), ScoreCond::Always));
    else if (name == "synthetic:pit_rookie__score_alone") add(1635, std::make_unique<ScoreUnitVar>(derive(reg, kRookie, 1635, "q5"), ScoreCond::NoOtherFriendlyUnits));
    else if (name == "synthetic:sabotage__score1") add(1636, std::make_unique<ScoreSpellVar>(derive(reg, kSab, 1636, "q6"), ScoreCond::Always));
    else if (name == "synthetic:sabotage__score_opphand4") add(1637, std::make_unique<ScoreSpellVar>(derive(reg, kSab, 1637, "q7"), ScoreCond::OppHand4));
    else if (name == "synthetic:punch_first__score_bfunit") add(1638, std::make_unique<ScoreSpellVar>(derive(reg, kPunch, 1638, "q8"), ScoreCond::FriendlyUnitAtBattlefield));
    else if (name == "synthetic:punch_first__score_behind") add(1639, std::make_unique<ScoreSpellVar>(derive(reg, kPunch, 1639, "q9"), ScoreCond::Behind));
    else if (name == "synthetic:punch_first__minus5") add(1640, std::make_unique<PunchFirstVar>(derive(reg, kPunch, 1640, "q10"), -5));
    else if (name == "synthetic:punch_first__ctrl_scores") add(1641, std::make_unique<PunchScoreTargetVar>(derive(reg, kPunch, 1641, "q11"), false));
    else if (name == "synthetic:punch_first__opp_of_ctrl_scores") add(1642, std::make_unique<PunchScoreTargetVar>(derive(reg, kPunch, 1642, "q12"), true));
    else throw std::invalid_argument("unknown variant " + name);
}

// ─── Card keys ────────────────────────────────────────────────────────────
std::map<uint32_t, std::string> g_keys;   // class id -> card key

std::string slug(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (std::isalnum(static_cast<unsigned char>(c))) o += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        else if (!o.empty() && o.back() != '_') o += '_';
    }
    while (!o.empty() && o.back() == '_') o.pop_back();
    return o;
}

std::string cardKey(CardDefId id, const GameObject* o) {
    if (!id.valid()) {
        if (o && o->is_token) return "token:" + slug(o->name);
        return o ? "other:" + slug(o->name) : "none";
    }
    auto it = g_keys.find(id.raw());
    if (it != g_keys.end()) return it->second;
    if (o && o->is_token) return "token:" + slug(o->name);
    if (o) return "other:" + slug(o->name);
    return "other:" + std::to_string(id.raw());
}

void loadKeys(const std::string& subjects) {
    std::ifstream in(subjects);
    std::string name; uint32_t id; int rn;
    while (in >> name >> id >> rn) {
        if (rn || name.rfind("class:", 0) != 0) continue;
        g_keys[id] = name.substr(6, name.rfind(':') - 6);
    }
    for (const auto& v : variants()) g_keys[v.id] = v.key;
}

// ─── Fingerprint (as situations/sampler.cpp) ──────────────────────────────
uint64_t mix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

uint64_t fingerprint(const GameState& s) {
    uint64_t h = 1469598103934665603ull;
    auto add = [&](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
    add(s.decision_index);
    add(s.turn.turn_number);
    add(static_cast<uint64_t>(s.turn.phase));
    add(static_cast<uint64_t>(s.turn.turn_player));
    for (const auto& p : s.players) {
        add(p.score); add(p.xp); add(p.main_deck.size()); add(p.trash.size());
        for (auto c : p.hand) { add(c.raw()); if (s.objectExists(c)) add(s.getObject(c).card_def_id.raw()); }
        add(p.rune_pool.energy);
    }
    std::vector<std::pair<uint32_t, uint64_t>> objs;
    for (const auto& [id, o] : s.objects.onBoard()) {
        uint64_t v = o.card_def_id.raw();
        v = v * 31 + static_cast<uint64_t>(o.current_might + 100);
        v = v * 31 + static_cast<uint64_t>(o.damage_marked);
        v = v * 31 + (o.battlefieldId().has_value() ? o.battlefieldId()->raw() + 2 : 1);
        v = v * 31 + o.is_exhausted;
        objs.push_back({id.raw(), v});
    }
    std::sort(objs.begin(), objs.end());
    for (auto& [i, v] : objs) { add(i); add(v); }
    add(s.chain.items.size());
    return h;
}

uint64_t fnv(const std::string& s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) h = (h ^ c) * 1099511628211ull;
    return h;
}

std::string hex(uint64_t v) { char b[20]; std::snprintf(b, 20, "%016llx", (unsigned long long)v); return b; }

std::string fnum(double v) {
    if (v == static_cast<long long>(v)) return std::to_string(static_cast<long long>(v));
    char buf[32]; std::snprintf(buf, 32, "%.4g", v); return buf;
}

// ─── Observation ──────────────────────────────────────────────────────────
// Zone codes (ZoneType order): 0 MainDeck 1 RuneDeck 2 Hand 3 Base 4 Trash
// 5 Banishment 6 ChampionZone 7 LegendZone 8 BattlefieldZone 9 Chain 10 Facedown
int locOf(const GameObject& o) {
    if (!o.location.has_value()) return -1;
    auto bf = o.battlefieldId();
    return bf.has_value() ? 1 + static_cast<int>(bf->raw()) : 0;
}

int locCode(const std::optional<LocationId>& l) {
    if (!l.has_value()) return -1;
    if (const auto* b = std::get_if<BattlefieldLocation>(&*l)) return 1 + static_cast<int>(b->id.raw());
    return 0;
}

struct Tok {
    char kind = 'c';        // c card object, b battlefield, k chain item, p pile (schema v2)
    std::string key;
    std::vector<long> f;    // situational fields (positional, SCHEMA.txt)
    std::vector<long> x;    // schema v2 extra fields (c: 7, k: 2), written after the v1 ones
    uint64_t obj = 0;       // object id (c, b) or chain item id (k)
    std::vector<uint64_t> refs;   // objects referenced (attached_to for c; source, targets for k)
    uint64_t subj = 0;      // k: the object whose event triggered it (0 none)
    std::string content() const {
        std::string s(1, kind); s += key;
        for (long v : f) { s += ','; s += std::to_string(v); }
        for (long v : x) { s += ';'; s += std::to_string(v); }
        return s;
    }
};

// Schema v2 (gate 1): GameEngine::netCardCost is private; read it through the
// standard explicit-instantiation access idiom (no edit of src/).
struct NetCostTag {
    using type = std::pair<int, int> (GameEngine::*)(PlayerId, GameObjectId, const Card::PlayCostAdjust&,
                                                     int, int) const;
    friend type netCostPtr(NetCostTag);
};
template <typename Tag, typename Tag::type M>
struct NetCostRob { friend typename Tag::type netCostPtr(Tag) { return M; } };
template struct NetCostRob<NetCostTag, &GameEngine::netCardCost>;

// (energy, power) `card` costs `player` to play now, by the engine's own cost
// computation (printed cost after every modifier, e.g. Legion's -2).
inline std::pair<int, int> engineNetCost(const GameEngine& e, PlayerId player, GameObjectId card) {
    return (e.*netCostPtr(NetCostTag{}))(player, card, Card::PlayCostAdjust{}, -1, -1);
}

// Schema v4 (gate 3): the Standard Move a MoveJoin prompt belongs to. The
// engine keeps the destination and the units already moving in locals of
// GameEngine::executeStandardMove, so the caller follows the decider's own
// choices: the StandardMove chosen, then each unit added at MoveJoin.
struct MoveTrack {
    int dst = -1;                         // destination code (0 base, 1+i battlefield i)
    std::vector<GameObjectId> moving;     // leading unit first, then the units joined
};
inline void noteChoice(MoveTrack& t, const Prompt& p, const Intent& chosen) {
    if (chosen.type == IntentType::StandardMove) {
        t.dst = chosen.move_destination.has_value() ? locCode(chosen.move_destination) : -1;
        t.moving = chosen.units_to_move;
    } else if (p.kind == PromptKind::MoveJoin && !chosen.chosen_objects.empty()) {
        t.moving.push_back(chosen.chosen_objects.front());
    }
}

// Search M0 (research/card_repr/search, 2026-11-20): strict visibility, OFF by default (training,
// the exporter and rbenv observations are unchanged). knowledge::identityVisible (src/) reports an
// opponent's hand / facedown / deck card as visible when the observer knows the card by identity
// but not its slot (CardKnowledge::zone_confirmed false, or a deck card known only within a group);
// the token then pins that identity to the object's real slot, which the observer does not know
// (WorldSampler places such a card at a random slot). With g_strict_vis such a card is "hidden",
// so every world sampled for the observer encodes identically to the truth (search/m0.py leak).
inline bool g_strict_vis = false;
inline bool visibleTo(const GameState& s, GameObjectId id, PlayerId me) {
    const bool v = knowledge::identityVisible(s, id, me);
    if (!v || !g_strict_vis || !s.objectExists(id)) return v;
    const GameObject& o = s.getObject(id);
    if (o.owner == me) return true;
    const ZoneType z = o.zone;
    if (z != ZoneType::Hand && z != ZoneType::FacedownZone && z != ZoneType::MainDeck) return true;
    const CardKnowledge k = knowledge::GameKnowledgeView(s, me).about(id);
    if (z == ZoneType::MainDeck) return k.deck_position != DeckPosition::Unknown && k.deck_rank_span == 1;
    return k.zone_confirmed;
}

class Encoder {
public:
    // `eng` (optional, schema v2): the engine that owns `s`; needed for the
    // effective-cost / affordable fields (written -1 without it).
    Encoder(const GameState& s, PlayerId me, const GameEngine* eng = nullptr)
        : s_(s), me_(me), eng_(eng),
          ov_(g_db ? std::make_unique<ObserverView>(s, me, *g_db, g_reg) : nullptr) {}

    int rel(PlayerId p) const { return p == PlayerId::None ? 0 : (p == me_ ? 1 : 2); }
    PlayerId opp() const { return me_ == PlayerId::Player1 ? PlayerId::Player2 : PlayerId::Player1; }

    Tok cardTok(GameObjectId id) const {
        const auto& o = s_.getObject(id);
        Tok t;
        t.kind = 'c';
        t.obj = id.raw();
        const bool vis = visibleTo(s_, id, me_);
        t.key = vis ? cardKey(o.card_def_id, &o) : "hidden";
        const bool board = o.location.has_value();
        t.f = {rel(o.controller), rel(o.owner), static_cast<long>(o.zone.get()), locOf(o),
               board ? o.is_exhausted : 0, board ? o.is_stunned : 0, board ? o.damage_marked : 0,
               board ? o.current_might : 0, board ? o.buff_count : 0, board ? o.temp_might_bonus : 0,
               o.is_token ? 1 : 0, o.is_hidden ? 1 : 0, static_cast<long>(o.combat_designation),
               static_cast<long>(o.attachments.size())};
        if (o.attached_to.has_value()) t.refs.push_back(o.attached_to->raw());
        // v2 extras: play ordinal this turn, hidden this turn, effective cost
        // (energy, power), affordable now, effective Assault, Shield.
        const long ord = playOrdinalThisTurn(s_, id);
        const long hid_now = (o.is_hidden && o.hidden_on_turn >= 0 &&
                              o.hidden_on_turn >= s_.turn.turn_number) ? 1 : 0;
        long ce = -1, cp = -1, aff = -1;
        const bool own_playable_zone = o.owner == me_ &&
            (o.zone == ZoneType::Hand || o.zone == ZoneType::ChampionZone);
        if (eng_ && own_playable_zone && o.card_def_id.valid()) {
            const auto c = engineNetCost(*eng_, me_, id);
            ce = c.first; cp = c.second;
            aff = eng_->testHook_canAfford(me_, id) ? 1 : 0;
        }
        long as = 0, sh = 0;
        if (board && ov_) { const auto pf = ov_->permanent(id); as = pf.assault; sh = pf.shield; }
        t.x = {ord, hid_now, ce, cp, aff, as, sh};
        return t;
    }

    // Destination code of an action (0 base, 1+i battlefield i, -1 none).
    // Location prompts answered by a value (Thrill of the Hunt: an index into
    // the battlefields its owner may play a unit to, battlefieldTakesUnitsFrom)
    // are mapped back to the real slot.
    int actionDst(const Intent& i, const Prompt& p) const {
        if (p.kind == PromptKind::MoveJoin) return mt_ ? mt_->dst : -1;   // v4: the move's destination
        const int d = i.play_location.has_value() ? locCode(i.play_location) : locCode(i.move_destination);
        if (d >= 0) return d;
        if (p.kind == PromptKind::Location && i.type == IntentType::MakeChoice && i.chosen_value.has_value() &&
            !i.declines) {
            const PlayerId who = i.player != PlayerId::None ? i.player : me_;
            int k = 0;
            for (const auto& bf : s_.battlefields) {
                if (!battlefieldTakesUnitsFrom(s_, who, bf)) continue;
                if (k++ == *i.chosen_value) return 1 + static_cast<int>(bf.id.raw());
            }
        }
        return -1;
    }

    // Consequences of putting a unit of the decider at `dst`: destination
    // controller (-1 n/a, 0 none, 1 self, 2 opp), would start a combat as
    // attacker (enemy units there), would contest (not controlled by self).
    // Schema v3 (gate 3): what an optional-cost answer costs. For Accelerate and
    // AdditionalCost prompts (p.source = the card being played), per option:
    //   [0] extra Energy this answer adds to the card's cost (0 for the decline)
    //   [1] extra Power it adds
    //   [2] runes the whole play would RECYCLE for Power with this answer
    //   [3] Domain index of those runes (Fury 0 .. Order 5; -1 none)
    //   [4] own runes on the board READY after the play's payment
    //   [5] own runes on the board in total after the payment
    //   [6] the unit enters ready (1) / exhausted (0); -1 not a unit decision
    // -1 everywhere for other prompts. The payment is an estimate from the
    // engine's cost (netCardCost, as card [19..20]) and the pool / board, paid
    // in the engine's rune order (schema v4: exact; see below).
    // Accelerate's extra is [1] + one Power of the unit's Domains (CR 805.1.b.1;
    // the optional-cost discount, not in the Rengar deck, is not applied).
    std::array<int, 7> optCost(const Intent& i, const Prompt& p) const {
        std::array<int, 7> r = {-1, -1, -1, -1, -1, -1, -1};
        if (p.kind != PromptKind::Accelerate && p.kind != PromptKind::AdditionalCost) return r;
        if (!eng_ || !p.source.valid() || !s_.objects.count(p.source)) return r;
        const auto& card = s_.getObject(p.source);
        if (!card.card_def_id.valid()) return r;
        const bool yes = !i.declines;
        std::pair<int, int> base{0, 0};
        try { base = engineNetCost(*eng_, me_, p.source); } catch (...) { return r; }
        // v4 (gate 3 completion): the payment is replayed rune by rune in the
        // engine's own order, which is what autopay (option 0 of every rune
        // prompt) takes:
        //  1. AdditionalCost "yes": resource::payIn, called by
        //     GameEngine::maybePayOptionalAdditionalCost before the card's cost:
        //     pool first; Energy from ready runes that cannot pay the Power
        //     first, lowest id first; Power from runes that pay it, exhausted
        //     first, lowest id first.
        //  2. The card's cost (with Accelerate's [1][C] for "yes"):
        //     GameEngine::beginPaymentOf / advanceCostPayment: pool first; Energy
        //     from the ready runes in base in the object table's order
        //     (readyRunesInBase); Power from the first rune in that order of the
        //     card's Domains, ready or exhausted (matchingRunesInBase).
        int xe = 0, xp = 0, reduce = 0;
        const std::vector<Domain> cdoms = card.domains.get();
        unsigned xmask = 0;                       // the optional cost's Power Domains
        if (p.kind == PromptKind::Accelerate) {
            if (yes) { xe = 1; xp = 1; }
            r[6] = yes ? 1 : 0;
        } else {
            const Card* c = g_reg ? g_reg->get(card.card_def_id) : nullptr;
            if (c) {
                const auto oc = c->optionalAdditionalCost(s_, me_);
                if (oc.valid && yes) {
                    xe = oc.energy; xp = oc.power; reduce = oc.reduce_energy;
                    xmask = oc.any_domain ? 0x3Fu : (1u << static_cast<int>(oc.power_domain));
                }
            }
        }
        r[0] = xe; r[1] = xp;
        struct R { uint64_t id; unsigned dm; bool ex; bool gone; };
        std::vector<R> runes;   // in the object table's order at the base (readyRunesInBase)
        for (const auto& [id, o] : s_.objects.atLocation(LocationId{BaseLocation{me_}})) {
            if (!o.isRune() || o.controller != me_) continue;
            unsigned dm = 0;
            for (auto d : o.domains.get()) if (d != Domain::Count) dm |= 1u << static_cast<int>(d);
            runes.push_back({id.raw(), dm, o.is_exhausted, false});
        }
        unsigned cmask = 0;
        for (auto d : cdoms) if (d != Domain::Count) cmask |= 1u << static_cast<int>(d);
        auto rp = s_.player(me_).rune_pool;      // a copy, spent below
        const bool unit = card.isUnit();
        int rec = 0, rec_dom = -1;
        auto recycle = [&](R& x) {
            x.gone = true; ++rec;
            if (rec_dom < 0) for (int d = 0; d < 6; ++d) if (x.dm & (1u << d)) { rec_dom = d; break; }
        };
        auto poolEnergy = [&](int need) {
            int& re = unit ? rp.unit_energy : rp.spell_energy;
            int k = std::min(re, need); re -= k; need -= k;
            k = std::min(rp.energy, need); rp.energy -= k; need -= k;
            return need;
        };
        auto poolPower = [&](int need, unsigned mask) {
            if (!unit) { int k = std::min(rp.spell_power, need); rp.spell_power -= k; need -= k; }
            for (int d = 0; d < 6; ++d)
                if (mask & (1u << d)) { int k = std::min(rp.power[d], need); rp.power[d] -= k; need -= k; }
            int k = std::min(rp.universal_power, need); rp.universal_power -= k; need -= k;
            return need;
        };
        if (xe > 0 || xp > 0) if (p.kind == PromptKind::AdditionalCost) {   // 1. payIn
            std::vector<R*> byId;
            for (auto& x : runes) byId.push_back(&x);
            std::sort(byId.begin(), byId.end(), [](const R* a, const R* b) { return a->id < b->id; });
            int e = poolEnergy(xe), pw = poolPower(xp, xmask);
            for (int pass = 0; pass < 2 && e > 0; ++pass)            // non-paying first
                for (auto* x : byId)
                    if (e > 0 && !x->gone && !x->ex && ((x->dm & xmask) != 0) == (pass == 1)) { x->ex = true; --e; }
            for (int pass = 0; pass < 2 && pw > 0; ++pass)           // exhausted first
                for (auto* x : byId)
                    if (pw > 0 && !x->gone && (x->dm & xmask) && x->ex == (pass == 0)) { recycle(*x); --pw; }
        }
        {   // 2. the card's cost
            const bool acc = p.kind == PromptKind::Accelerate;
            int e = poolEnergy(std::max(0, base.first - reduce) + (acc ? xe : 0));
            int pw = poolPower(base.second + (acc ? xp : 0), cmask);
            for (auto& x : runes) if (e > 0 && !x.gone && !x.ex) { x.ex = true; --e; }
            for (auto& x : runes) if (pw > 0 && !x.gone && (x.dm & cmask)) { recycle(x); --pw; }
        }
        int ready = 0, total = 0;
        for (const auto& x : runes) if (!x.gone) { ++total; ready += x.ex ? 0 : 1; }
        r[2] = rec; r[3] = rec_dom; r[4] = ready; r[5] = total;
        return r;
    }

    // ---- schema v4 (gate 3) ----
    void setMove(const MoveTrack* m) { mt_ = m; }
    // MoveJoin: the units already in the move (leading unit first); empty otherwise.
    std::vector<GameObjectId> moveGroup(const Prompt& p) const {
        if (p.kind != PromptKind::MoveJoin || !mt_) return {};
        return mt_->moving;
    }
    // MoveJoin's "Move no more units" option (no unit chosen): marked as the decline.
    static bool moveJoinStop(const Intent& i, const Prompt& p) {
        return p.kind == PromptKind::MoveJoin && i.type == IntentType::MakeChoice && i.chosen_objects.empty();
    }
    // GameEngine::triggerOrderLabel, mirrored (it is private).
    std::string trigLabel(const ChainItem& ci) const {
        std::string l = s_.objectExists(ci.source) ? std::string(s_.getObject(ci.source).name.get()) : std::string("?");
        if (ci.keyword_ability.has_value()) l += std::string(" [") + toString(*ci.keyword_ability) + "]";
        if (ci.equipped_unit != kInvalidId && s_.objectExists(ci.equipped_unit))
            l += " on " + std::string(s_.getObject(ci.equipped_unit).name.get());
        else if (ci.triggering_subject != kInvalidId && ci.triggering_subject != ci.source &&
                 s_.objectExists(ci.triggering_subject))
            l += " (" + std::string(s_.getObject(ci.triggering_subject).name.get()) + ")";
        return l;
    }
    // TriggerOrder: option chosen_value k -> the chain item it orders. The
    // engine (GameEngine::orderPendingTriggers) offers distinct[k] of the
    // decider's still-unordered items of the oldest pending batch; that batch
    // already stands in its canonical order on the chain (canonicalBatchOrder),
    // and the options keep that order. Rebuilt here: the batch's pending items
    // of the decider in chain order, matched to the options' labels in order,
    // skipping items that are the same ability as one already matched. Exact
    // unless two different items share a label and the earlier one was
    // ordered already (counted in trig_amb_).
    void prepare(const std::vector<Intent>& legal, const Prompt& p) {
        trig_.clear(); trig_amb_ = 0;
        if (p.kind != PromptKind::TriggerOrder || legal.empty()) return;
        const PlayerId who = legal[0].player != PlayerId::None ? legal[0].player : me_;
        const ChainItem* oldest = nullptr;
        for (const auto& ci : s_.chain.items) if (ci.status == ChainItemStatus::Pending) { oldest = &ci; break; }
        if (!oldest || oldest->trigger_batch == 0) return;
        std::vector<const ChainItem*> cand;
        for (const auto& ci : s_.chain.items)
            if (ci.status == ChainItemStatus::Pending && ci.trigger_batch == oldest->trigger_batch && ci.controller == who)
                cand.push_back(&ci);
        auto same = [](const ChainItem& a, const ChainItem& b) {
            return a.source == b.source && a.controller == b.controller && a.card_def_id == b.card_def_id &&
                   a.fired_trigger == b.fired_trigger && a.triggering_subject == b.triggering_subject &&
                   a.equipped_unit == b.equipped_unit && a.keyword_ability == b.keyword_ability;
        };
        std::vector<std::pair<int, const Intent*>> opts;
        for (const auto& i : legal) if (i.chosen_value.has_value()) opts.emplace_back(*i.chosen_value, &i);
        std::sort(opts.begin(), opts.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        std::vector<const ChainItem*> matched;
        size_t pos = 0;
        const std::string pre = "Order \xE2\x80\x94 ";
        for (const auto& [k, ip] : opts) {
            for (size_t j = pos; j < cand.size(); ++j) {
                bool dup = false;
                for (auto* m : matched) if (same(*m, *cand[j])) dup = true;
                if (dup) continue;
                const std::string lj = trigLabel(*cand[j]);
                if (ip->choice_label != pre + lj) continue;
                for (size_t q = j + 1; q < cand.size(); ++q)
                    if (!same(*cand[q], *cand[j]) && trigLabel(*cand[q]) == lj) { ++trig_amb_; break; }
                trig_[k] = cand[j]->id.raw();
                matched.push_back(cand[j]);
                pos = j + 1;
                break;
            }
        }
    }
    // The chain item (raw id) a TriggerOrder option orders; 0 when unknown.
    uint64_t trigItem(const Intent& i, const Prompt& p) const {
        if (p.kind != PromptKind::TriggerOrder || !i.chosen_value.has_value()) return 0;
        auto it = trig_.find(*i.chosen_value);
        return it == trig_.end() ? 0 : it->second;
    }
    GameObjectId trigSource(uint64_t item) const {
        for (const auto& ci : s_.chain.items) if (ci.id.raw() == item) return ci.source;
        return kInvalidId;
    }
    int trigAmbiguous() const { return trig_amb_; }
    int trigMapped() const { return static_cast<int>(trig_.size()); }

    std::array<int, 3> consequences(int dst) const {
        std::array<int, 3> r = {-1, 0, 0};
        if (dst < 1) return r;
        for (const auto& bf : s_.battlefields) {
            if (1 + static_cast<int>(bf.id.raw()) != dst) continue;
            r[0] = bf.controller.has_value() ? rel(*bf.controller) : 0;
            bool enemy = false;
            for (const auto& [oid, o] : s_.objects.onBoard())
                if (o.isUnit() && o.controller != me_ && o.battlefieldId() == bf.id) { enemy = true; break; }
            r[1] = enemy ? 1 : 0;
            r[2] = (!bf.controller.has_value() || *bf.controller != me_) ? 1 : 0;
        }
        return r;
    }

    void addObj(GameObjectId id) {
        if (!s_.objectExists(id) || seen_.count(id.raw())) return;
        seen_.insert(id.raw());
        toks_.push_back(cardTok(id));
    }

    void build() {
        // own hand
        for (auto h : s_.player(me_).hand) addObj(h);
        // v2: opponent hand cards the observer knows (knowledge tracker via ObserverView)
        if (ov_) {
            for (const auto& k : ov_->knownInHand(opp())) {
                if (!s_.objectExists(k.object) || seen_.count(k.object.raw())) continue;
                seen_.insert(k.object.raw());
                Tok t = cardTok(k.object);
                if (t.key == "hidden" && k.identity_known) t.key = cardKey(k.card_def_id, &s_.getObject(k.object));
                toks_.push_back(t);
            }
        }
        // v2: aggregated public piles (trash, banishment of both) and own remaining deck:
        // one token per (pile, owner, card) with a count.
        {
            auto pile = [&](PlayerId owner, ZoneType z, const std::vector<GameObjectId>& ids) {
                std::map<std::string, long> cnt;
                for (auto id : ids) {
                    if (!s_.objectExists(id)) continue;
                    const auto& o = s_.getObject(id);
                    cnt[cardKey(o.card_def_id, &o)]++;
                }
                for (const auto& [k, n] : cnt) {
                    Tok t; t.kind = 'p'; t.key = k;
                    t.f = {rel(owner), static_cast<long>(z), n};
                    ptoks_.push_back(t);
                }
            };
            for (PlayerId pl : {me_, opp()}) {
                pile(pl, ZoneType::Trash, s_.player(pl).trash);
                pile(pl, ZoneType::Banishment, s_.player(pl).banishment);
            }
            pile(me_, ZoneType::MainDeck, s_.player(me_).main_deck);
        }
        // board and legend zones (runes are aggregated in the global features,
        // battlefield cards are battlefield tokens)
        for (const auto& [id, o] : s_.objects.inPlay()) {
            if (o.card_type == CardType::Rune || o.card_type == CardType::Battlefield) continue;
            addObj(id);
        }
        for (const auto& p : s_.players) {
            if (p.champion_zone.valid() && s_.objectExists(p.champion_zone) &&
                s_.getObject(p.champion_zone).zone == ZoneType::ChampionZone)
                addObj(p.champion_zone);
            if (p.legend_zone.valid()) addObj(p.legend_zone);
        }
        for (const auto& b : s_.battlefields) for (auto fd : b.facedown) addObj(fd);
        // chain: the spell objects, then one token per chain item
        auto chainTok = [&](const ChainItem& ci, int pos, bool resolving) {
            if (ci.source.valid() && s_.objectExists(ci.source)) {
                const auto& so = s_.getObject(ci.source);
                if (so.zone == ZoneType::Chain) addObj(ci.source);
            }
            Tok t;
            t.kind = 'k';
            t.obj = ci.id.raw();
            const GameObject* so = (ci.source.valid() && s_.objectExists(ci.source)) ? &s_.getObject(ci.source) : nullptr;
            const bool vis = !so || so->owner == me_ || visibleTo(s_, ci.source, me_) || so->location.has_value();
            t.key = ci.keyword_ability.has_value() ? "keyword_ability" : (vis ? cardKey(ci.card_def_id, so) : "hidden");
            t.f = {rel(ci.controller), pos, resolving ? 1 : 0, ci.is_spell ? 1 : 0, ci.is_ability ? 1 : 0,
                   ci.is_activated_ability ? 1 : 0, static_cast<long>(ci.status), ci.is_equip ? 1 : 0};
            t.refs.push_back(ci.source.valid() ? ci.source.raw() : 0);
            for (auto x : ci.chosen_objects) t.refs.push_back(x.raw());
            // v2: optional additional cost paid (Rampage's __rampage_paid, or the
            // generic Card::kAdditionalCostPaid flag on the source), play ordinal of
            // the source this turn (Darius), triggering subject (as a token index).
            long paid = 0;
            if (so) {
                for (const std::string key : {std::string("__rampage_paid"), std::string(Card::kAdditionalCostPaid)}) {
                    const auto it = so->card_counters.find(key);
                    if (it != so->card_counters.end() && it->second == 1) paid = 1;
                }
            }
            t.x = {paid, so ? static_cast<long>(playOrdinalThisTurn(s_, ci.source)) : 0};
            t.subj = ci.triggering_subject.valid() ? ci.triggering_subject.raw() : 0;
            if (t.subj && s_.objectExists(ci.triggering_subject)) addObj(ci.triggering_subject);
            ktoks_.push_back(t);
        };
        const int n = static_cast<int>(s_.chain.items.size());
        for (int i = 0; i < n; ++i) chainTok(s_.chain.items[i], n - 1 - i, false);
        if (s_.chain.resuming.has_value()) chainTok(*s_.chain.resuming, -1, true);
        // battlefields
        for (const auto& b : s_.battlefields) {
            Tok t;
            t.kind = 'b';
            t.obj = b.card_object_id.valid() ? b.card_object_id.raw() : 0;
            const GameObject* bo = s_.objectExists(b.card_object_id) ? &s_.getObject(b.card_object_id) : nullptr;
            t.key = bo ? cardKey(bo->card_def_id, bo) : "none";
            long fd_self = 0, fd_opp = 0;
            for (auto fd : b.facedown)
                if (s_.objectExists(fd)) (s_.getObject(fd).controller == me_ ? fd_self : fd_opp)++;
            const PlayerId opp = me_ == PlayerId::Player1 ? PlayerId::Player2 : PlayerId::Player1;
            t.f = {static_cast<long>(b.id.raw()), b.controller.has_value() ? rel(*b.controller) : 0,
                   b.is_contested ? 1 : 0, rel(b.contested_by), b.showdown_in_progress ? 1 : 0,
                   b.combat_in_progress ? 1 : 0, b.attacker.has_value() ? rel(*b.attacker) : 0,
                   b.showdown_staged || b.combat_staged ? 1 : 0, fd_self, fd_opp,
                   static_cast<long>(s_.player(me_).battlefields_scored_this_turn.count(b.id)),
                   static_cast<long>(s_.player(opp).battlefields_scored_this_turn.count(b.id)),
                   b.is_token ? 1 : 0};
            // v2: "first time each turn" stamp on the battlefield card, per player
            // (Star Spring's __starspring_turn_p<i> == turn + 1).
            long st_self = 0, st_opp = 0;
            if (bo) {
                for (PlayerId pl : {me_, opp}) {
                    const std::string key = "__starspring_turn_p" + std::to_string(playerIndex(pl));
                    const auto it = bo->card_counters.find(key);
                    if (it != bo->card_counters.end() && it->second == s_.turn.turn_number + 1)
                        (pl == me_ ? st_self : st_opp) = 1;
                }
            }
            t.f.push_back(st_self); t.f.push_back(st_opp);
            btoks_.push_back(t);
            bf_by_obj_[t.obj] = 0;
        }
        finalize();
    }

    // Canonical order: by content, ties by id (identical content only).
    void finalize() {
        auto order = [](std::vector<Tok>& v) {
            std::stable_sort(v.begin(), v.end(), [](const Tok& a, const Tok& b) {
                const auto ca = a.content(), cb = b.content();
                if (ca != cb) return ca < cb;
                return a.obj < b.obj;
            });
        };
        order(toks_); order(btoks_); order(ktoks_); order(ptoks_);
        all_.clear();
        for (auto* v : {&btoks_, &toks_, &ktoks_, &ptoks_}) for (auto& t : *v) all_.push_back(t);
        reindex();
    }

    void reindex() {
        obj_idx_.clear(); item_idx_.clear();
        for (size_t i = 0; i < all_.size(); ++i) {
            if (all_[i].kind == 'k') item_idx_[all_[i].obj] = static_cast<int>(i);
            else if (all_[i].kind != 'p') obj_idx_[all_[i].obj] = static_cast<int>(i);
        }
    }

    // Token index of an object; an object no token covers yet (a card in the
    // opponent's hand chosen by Sabotage, a card looked at) is appended.
    int ref(GameObjectId id) {
        if (!id.valid()) return -1;
        auto it = obj_idx_.find(id.raw());
        if (it != obj_idx_.end()) return it->second;
        if (!s_.objectExists(id)) return -1;
        seen_.insert(id.raw());
        all_.push_back(cardTok(id));
        obj_idx_[id.raw()] = static_cast<int>(all_.size() - 1);
        return static_cast<int>(all_.size() - 1);
    }

    int itemRef(uint64_t item) const {
        auto it = item_idx_.find(item);
        return it == item_idx_.end() ? -1 : it->second;
    }

    std::vector<long> global(const Prompt& p) const {
        const auto& me = s_.player(me_);
        const PlayerId oppid = me_ == PlayerId::Player1 ? PlayerId::Player2 : PlayerId::Player1;
        const auto& op = s_.player(oppid);
        std::vector<long> g = {
            s_.turn.turn_number, static_cast<long>(s_.turn.phase), s_.turn.turn_player == me_,
            s_.turn.starting_player == me_, me.score, op.score, me.xp, op.xp, s_.effectiveVictoryScore(),
            static_cast<long>(me.hand.size()), static_cast<long>(op.hand.size()),
            static_cast<long>(me.main_deck.size()), static_cast<long>(op.main_deck.size()),
            static_cast<long>(me.trash.size()), static_cast<long>(op.trash.size()),
            static_cast<long>(me.rune_deck.size()), static_cast<long>(op.rune_deck.size()),
            static_cast<long>(me.banishment.size()), static_cast<long>(op.banishment.size())};
        for (const auto* pl : {&me, &op}) {
            const auto& rp = pl->rune_pool;
            g.push_back(rp.energy);
            for (int d = 0; d < static_cast<int>(Domain::Count); ++d) g.push_back(rp.power[d]);
            g.push_back(rp.universal_power);
            g.push_back(rp.unit_energy + rp.spell_energy + rp.showdown_energy);
            g.push_back(rp.spell_power + rp.gear_power);
        }
        // runes on the board: [self ready x6, self exhausted x6, opp ready x6, opp exhausted x6]
        std::vector<long> runes(24, 0);
        for (const auto& [id, o] : s_.objects.onBoard()) {
            if (o.card_type != CardType::Rune) continue;
            const std::vector<Domain>& doms = o.domains;
            const int d = doms.empty() ? 0 : static_cast<int>(doms[0]);
            const int side = o.controller == me_ ? 0 : 12;
            runes[side + (o.is_exhausted ? 6 : 0) + d]++;
        }
        g.insert(g.end(), runes.begin(), runes.end());
        g.push_back(static_cast<long>(s_.chain.items.size()));
        g.push_back(static_cast<long>(s_.turn.ns_state));
        g.push_back(static_cast<long>(s_.turn.oc_state));
        g.push_back(s_.turn.priority_holder.has_value() ? rel(*s_.turn.priority_holder) : 0);
        g.push_back(s_.turn.focus_holder.has_value() ? rel(*s_.turn.focus_holder) : 0);
        g.push_back(static_cast<long>(p.kind));
        g.push_back(me.cards_played_this_turn);
        g.push_back(op.cards_played_this_turn);
        g.push_back(p.at_finalization ? 1 : 0);
        return g;
    }

    std::string action(const Intent& i, const Prompt& p) {
        std::ostringstream o;
        o << "{\"t\":" << static_cast<int>(i.type);
        auto list = [&](const char* k, const std::vector<GameObjectId>& v) {
            if (v.empty()) return;
            o << ",\"" << k << "\":[";
            for (size_t j = 0; j < v.size(); ++j) o << (j ? "," : "") << ref(v[j]);
            o << "]";
        };
        const uint64_t trig = trigItem(i, p);   // v4
        if (i.card.valid()) o << ",\"src\":" << ref(i.card);
        else if (trig) { const auto sc = trigSource(trig); if (sc.valid()) o << ",\"src\":" << ref(sc); }
        if (i.ability_source.valid()) o << ",\"abs\":" << ref(i.ability_source) << ",\"ai\":" << i.ability_index;
        list("tg", i.targets);
        list("ch", i.chosen_objects);
        list("mv", i.units_to_move);
        list("mu", i.cards_to_mulligan);
        list("mg", moveGroup(p));   // v4: MoveJoin, the units already moving (leading unit first)
        if (!i.damage_assignments.empty()) {
            o << ",\"dm\":[";
            for (size_t j = 0; j < i.damage_assignments.size(); ++j)
                o << (j ? "," : "") << "[" << ref(i.damage_assignments[j].target_unit) << ","
                  << i.damage_assignments[j].damage << "]";
            o << "]";
        }
        const int dst = actionDst(i, p);
        if (dst >= 0) {
            o << ",\"dst\":" << dst;
            if (dst >= 1) { const auto q = consequences(dst); o << ",\"dc\":" << q[0] << ",\"atk\":" << q[1] << ",\"con\":" << q[2]; }
        }
        if (i.chosen_value.has_value()) {
            o << ",\"v\":" << *i.chosen_value;
            if (p.kind == PromptKind::ChainItem) {
                const int ci = itemRef(static_cast<uint64_t>(*i.chosen_value));
                if (ci >= 0) o << ",\"ci\":" << ci;
            } else if (trig) {
                const int ci = itemRef(trig);
                if (ci >= 0) o << ",\"ci\":" << ci;
            }
        }
        if (i.chosen_card_def.valid()) o << ",\"nc\":" << jstr(cardKey(i.chosen_card_def, nullptr));
        if (i.chosen_battlefield.valid()) o << ",\"bf\":" << i.chosen_battlefield.raw();
        if (i.play_source != Intent::PlaySource::Hand) o << ",\"ps\":" << static_cast<int>(i.play_source);
        if (i.use_alt_play_cost) o << ",\"alt\":1";
        if (i.pay_location_cost) o << ",\"plc\":1";
        if (i.declines || moveJoinStop(i, p)) o << ",\"dec\":1";
        if (i.type == IntentType::PlayFirstDecision) o << ",\"first\":" << (i.choose_first ? 1 : 0);
        {   // schema v3: optional-cost consequences (Accelerate / AdditionalCost)
            const auto q = optCost(i, p);
            if (q[0] >= 0) { o << ",\"oc\":["; for (int j = 0; j < 7; ++j) o << (j ? "," : "") << q[j]; o << "]"; }
        }
        o << "}";
        return o.str();
    }

    std::string tokens() const {
        std::ostringstream o;
        o << "[";
        for (size_t i = 0; i < all_.size(); ++i) {
            const Tok& t = all_[i];
            o << (i ? "," : "") << "[\"" << t.kind << "\"," << jstr(t.key);
            for (long x : t.f) o << "," << x;
            if (t.kind == 'c') {
                int a = -1;
                if (!t.refs.empty()) { auto it = obj_idx_.find(t.refs[0]); if (it != obj_idx_.end()) a = it->second; }
                o << "," << a;
                for (long x : t.x) o << "," << x;
            } else if (t.kind == 'k') {
                auto idx = [&](uint64_t r) { auto it = obj_idx_.find(r); return it == obj_idx_.end() ? -1 : it->second; };
                o << "," << idx(t.refs[0]) << ",[";
                for (size_t j = 1; j < t.refs.size(); ++j) o << (j > 1 ? "," : "") << idx(t.refs[j]);
                o << "]";
                for (long x : t.x) o << "," << x;
                o << "," << (t.subj ? idx(t.subj) : -1);
            }
            o << "]";
        }
        o << "]";
        return o.str();
    }

    const std::vector<Tok>& all() const { return all_; }

    // Everything the network sees at this decision: global, tokens, actions.
    std::string encode(const std::vector<Intent>& legal, const Prompt& p, int* psrc_out = nullptr) {
        prepare(legal, p);
        std::vector<std::string> acts;
        for (const auto& i : legal) acts.push_back(action(i, p));
        const int psrc = p.source.valid() ? ref(p.source) : -1;
        if (psrc_out) *psrc_out = psrc;
        std::ostringstream o;
        o << "\"pk\":" << jstr(toString(p.kind)) << ",\"psrc\":" << psrc << ",\"g\":[";
        const auto g = global(p);
        for (size_t i = 0; i < g.size(); ++i) o << (i ? "," : "") << g[i];
        o << "],\"tok\":" << tokens() << ",\"act\":[";
        for (size_t i = 0; i < acts.size(); ++i) o << (i ? "," : "") << acts[i];
        o << "]";
        return o.str();
    }

private:
    const GameState& s_;
    PlayerId me_;
    const GameEngine* eng_ = nullptr;
    std::unique_ptr<ObserverView> ov_;
    std::set<uint64_t> seen_;
    std::vector<Tok> toks_, btoks_, ktoks_, ptoks_, all_;
    std::map<uint64_t, int> obj_idx_, item_idx_, bf_by_obj_;
    const MoveTrack* mt_ = nullptr;          // v4
    std::map<int, uint64_t> trig_;           // v4: TriggerOrder value -> chain item
    int trig_amb_ = 0;
};

// ─── Games ────────────────────────────────────────────────────────────────
struct GameSpec {
    std::string gid, deck_a, deck_b, agent_a, agent_b;
    uint64_t seed = 0;
    int swapcand = 0;
};

uint64_t engineSeedOf(uint64_t s) {
    s &= 0x7FFFFFFFu;
    return s == 0 ? 1 : s;
}

struct StopGame : std::runtime_error { StopGame() : std::runtime_error("stop") {} };

const std::set<uint32_t>& swapBaseIds() {
    static const std::set<uint32_t> s = {419, 58, 359, 777, 156, 136, 45};
    return s;
}

PlayerId deciderOf(const GameState& s, const std::vector<Intent>& legal, const Prompt& p) {
    return p.decider != PlayerId::None ? p.decider : decidingPlayer(s, legal);
}

struct GameCtx {
    GameSpec g;
    std::string variant;
    bool record = true;
    int seq = 0;
    std::vector<int> choices;
    std::vector<std::string> recs;        // decision lines without the closing ,"z":..}
    std::vector<PlayerId> rec_dec;
    std::vector<std::string> checks;      // [seq, fp, hash]
    std::map<std::string, long> key_count;
    std::vector<std::string> swap_cands;
    std::map<int, MoveTrack> mtrack;      // v4: per decider, the Standard Move being built (MoveJoin)
    std::set<std::string> swap_seen;
    long n_multi = 0;
    // replay
    const std::vector<int>* replay = nullptr;
    std::function<bool(int, const GameState&, const std::vector<Intent>&, const Prompt&)> on_seq;  // true: stop
    const GameEngine* engine = nullptr;   // v2: the live engine (effective cost / affordable fields)
};

int indexOf(const std::vector<Intent>& legal, const Intent& pick) {
    for (size_t k = 0; k < legal.size(); ++k) if (legal[k] == pick) return static_cast<int>(k);
    return -1;
}

class Seat : public AgentInterface {
public:
    Seat(std::unique_ptr<AgentInterface> inner, IsMctsAgent* ismcts, std::string teacher, GameCtx& ctx)
        : inner_(std::move(inner)), ismcts_(ismcts), teacher_(std::move(teacher)), ctx_(ctx) {}
    Intent selectAction(const GameState& s, const std::vector<Intent>& legal) override {
        return decide(s, legal, Prompt{});
    }
    Intent decide(const GameState& s, const std::vector<Intent>& legal, const Prompt& p) override {
        const int k = ctx_.seq++;
        if (k > 20000) throw std::runtime_error("decision cap");
        if (ctx_.replay) {
            if (ctx_.on_seq && ctx_.on_seq(k, s, legal, p)) throw StopGame();
            if (k >= static_cast<int>(ctx_.replay->size())) throw std::runtime_error("replay ran past its choices");
            const Intent& rp = legal.at((*ctx_.replay)[k]);
            noteChoice(ctx_.mtrack[static_cast<int>(deciderOf(s, legal, p))], p, rp);
            return rp;
        }
        const PlayerId dec = deciderOf(s, legal, p);
        std::string head;
        if (ctx_.record && legal.size() >= 2) {
            Encoder enc(s, dec, ctx_.engine);
            enc.setMove(&ctx_.mtrack[static_cast<int>(dec)]);
            enc.build();
            const std::string body = enc.encode(legal, p);
            ++ctx_.n_multi;
            for (const auto& t : enc.all()) ctx_.key_count[t.key]++;
            const uint64_t fp = fingerprint(s);
            ctx_.checks.push_back("[" + std::to_string(k) + ",\"" + hex(fp) + "\",\"" + hex(fnv(body)) + "\"]");
            std::ostringstream o;
            o << "{\"gid\":" << jstr(ctx_.g.gid) << ",\"seq\":" << k << ",\"d\":" << s.decision_index
              << ",\"seat\":" << static_cast<int>(dec) << ",\"teacher\":" << jstr(teacher_) << ","
              << body;
            head = o.str();
        }
        // swap-test candidates: main phase, base card in the decider's hand, its play offered
        if (ctx_.g.swapcand && p.kind == PromptKind::MainPhase) {
            for (auto h : s.player(dec).hand) {
                if (!s.objectExists(h)) continue;
                const uint32_t cid = s.getObject(h).card_def_id.raw();
                if (!swapBaseIds().count(cid)) continue;
                bool offered = false;
                for (const auto& i : legal) if (i.type == IntentType::PlayCard && i.card == h) offered = true;
                if (!offered) continue;
                const std::string key = g_keys[cid];
                const std::string sk = key + "|" + std::to_string(s.turn.turn_number) + "|" + std::to_string(int(dec));
                if (!ctx_.swap_seen.insert(sk).second) continue;
                ctx_.swap_cands.push_back("{\"seq\":" + std::to_string(k) + ",\"d\":" + std::to_string(s.decision_index) +
                                          ",\"turn\":" + std::to_string(s.turn.turn_number) + ",\"seat\":" +
                                          std::to_string(int(dec)) + ",\"base\":" + jstr(key) +
                                          ",\"fp\":\"" + hex(fingerprint(s)) + "\"}");
            }
        }
        const Intent pick = inner_->decide(s, legal, p);
        const int idx = indexOf(legal, pick);
        if (idx < 0) throw std::runtime_error("agent answered an option not in the legal list");
        noteChoice(ctx_.mtrack[static_cast<int>(dec)], p, pick);
        ctx_.choices.push_back(idx);
        if (!head.empty()) {
            std::ostringstream o;
            o << head << ",\"pi\":" << idx;
            if (ismcts_) {
                const SearchSummary& ss = ismcts_->lastSearch();
                o << ",\"searched\":" << (ss.searched ? 1 : 0);
                if (ss.searched) {
                    o << ",\"vis\":[";
                    for (size_t j = 0; j < ss.visits.size(); ++j) o << (j ? "," : "") << ss.visits[j];
                    o << "],\"q\":[";
                    for (size_t j = 0; j < ss.value.size(); ++j) o << (j ? "," : "") << fnum(ss.value[j]);
                    o << "],\"rv\":" << fnum(ss.root_value);
                }
            }
            ctx_.recs.push_back(o.str());
            ctx_.rec_dec.push_back(dec);
        }
        return pick;
    }
private:
    std::unique_ptr<AgentInterface> inner_;
    IsMctsAgent* ismcts_;
    std::string teacher_;
    GameCtx& ctx_;
};

std::unique_ptr<AgentInterface> makeAgent(const std::string& spec, const GameSpec& g, uint64_t es, int seat,
                                          IsMctsAgent** ismcts) {
    const uint64_t derived = es * 2 + seat;
    double pass = 0;
    *ismcts = nullptr;
    if (spec == "random") return std::make_unique<RandomAgent>(derived);
    if (isRandomPlayerSpec(spec, &pass)) return std::make_unique<RandomPlayerAgent>(derived, pass);
    if (spec.rfind("ismcts:sims=", 0) == 0) {
        const int sims = std::stoi(spec.substr(12));
        auto a = std::make_unique<IsMctsAgent>(g.deck_a, g.deck_b, "", es, derived, sims);
        *ismcts = a.get();
        return a;
    }
    throw std::invalid_argument("unknown agent " + spec);
}

class NullAgent : public AgentInterface {
public:
    Intent selectAction(const GameState&, const std::vector<Intent>& legal) override { return legal.at(0); }
};

struct RunOut { GameResult res; std::string term; double secs = 0; };

RunOut runOne(GameCtx& ctx, bool with_agents) {
    const auto t0 = std::chrono::steady_clock::now();
    const GameSpec& g = ctx.g;
    const uint64_t es = engineSeedOf(g.seed);
    const DeckSubmission a = DeckValidator::loadFromDeckList(g.deck_a, *g_db);
    const DeckSubmission b = DeckValidator::loadFromDeckList(g.deck_b, *g_db);
    IsMctsAgent *m1 = nullptr, *m2 = nullptr;
    std::unique_ptr<AgentInterface> i1, i2;
    if (with_agents) {
        i1 = makeAgent(g.agent_a, g, es, 0, &m1);
        i2 = makeAgent(g.agent_b, g, es, 1, &m2);
    } else {
        i1 = std::make_unique<NullAgent>();
        i2 = std::make_unique<NullAgent>();
    }
    Seat s1(std::move(i1), m1, g.agent_a, ctx), s2(std::move(i2), m2, g.agent_b, ctx);
    EventBus events;
    GameEngine engine(*g_db, events, *g_reg);
    ctx.engine = &engine;
    RunOut r;
    try {
        r.res = engine.runGame(a, b, s1, s2, es);
        r.term = r.res.termination_reason;
    } catch (const StopGame&) {
        r.term = "stopped";
    } catch (const std::exception& ex) {
        r.term = std::string("error: ") + ex.what();
    }
    r.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return r;
}

std::string deckKeys(const std::string& path) {
    const DeckSubmission d = DeckValidator::loadFromDeckList(path, *g_db);
    std::map<std::string, int> c;
    auto add = [&](CardDefId id) { if (id.valid()) c[cardKey(id, nullptr)]++; };
    add(d.legend); add(d.chosen_champion);
    for (auto x : d.main_deck) add(x);
    for (auto x : d.rune_deck) add(x);
    for (auto x : d.battlefields) add(x);
    std::ostringstream o;
    o << "{";
    bool f = true;
    for (auto& [k, n] : c) { o << (f ? "" : ",") << jstr(k) << ":" << n; f = false; }
    o << "}";
    return o.str();
}

std::string validation(const std::string& path) {
    try {
        const DeckSubmission d = DeckValidator::loadFromDeckList(path, *g_db);
        DeckValidator v(*g_db);
        auto r = v.validate(d);
        if (r.is_legal) return "ok";
        std::string all;
        for (const auto& e : r.errors) all += e + "; ";
        return all;
    } catch (const std::exception& ex) {
        return std::string("exception: ") + ex.what();
    }
}

int winnerCode(PlayerId w) { return w == PlayerId::Player1 ? 1 : (w == PlayerId::Player2 ? 2 : 0); }

void playMode(const std::vector<GameSpec>& games, const std::string& prefix, int shard, int nshards,
              const std::string& variant) {
    std::ofstream dec(prefix + ".decisions.jsonl"), gout(prefix + ".games.jsonl");
    std::map<std::string, std::string> val_cache;
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t gi = 0; gi < games.size(); ++gi) {
        if (static_cast<int>(gi % nshards) != shard) continue;
        GameCtx ctx;
        ctx.g = games[gi];
        ctx.variant = variant;
        RunOut r = runOne(ctx, true);
        const int w = winnerCode(r.res.winner);
        for (size_t j = 0; j < ctx.recs.size(); ++j) {
            const int z = w == 0 ? 0 : (w == static_cast<int>(ctx.rec_dec[j]) ? 1 : -1);
            dec << ctx.recs[j] << ",\"z\":" << z << "}\n";
        }
        for (const auto* p : {&ctx.g.deck_a, &ctx.g.deck_b})
            if (!val_cache.count(*p)) val_cache[*p] = validation(*p);
        std::ostringstream o;
        o << "{\"gid\":" << jstr(ctx.g.gid) << ",\"deck_a\":" << jstr(ctx.g.deck_a) << ",\"deck_b\":" << jstr(ctx.g.deck_b)
          << ",\"agent_a\":" << jstr(ctx.g.agent_a) << ",\"agent_b\":" << jstr(ctx.g.agent_b)
          << ",\"seed\":" << ctx.g.seed << ",\"engine_seed\":" << engineSeedOf(ctx.g.seed)
          << ",\"variant\":" << jstr(variant) << ",\"winner\":" << w << ",\"term\":" << jstr(r.term)
          << ",\"turns\":" << r.res.total_turns << ",\"scores\":[" << r.res.final_scores[0] << ","
          << r.res.final_scores[1] << "],\"n_seq\":" << ctx.seq << ",\"n_multi\":" << ctx.n_multi
          << ",\"n_rec\":" << ctx.recs.size() << ",\"secs\":" << fnum(r.secs)
          << ",\"valid_a\":" << jstr(val_cache[ctx.g.deck_a]) << ",\"valid_b\":" << jstr(val_cache[ctx.g.deck_b])
          << ",\"deck_cards_a\":" << deckKeys(ctx.g.deck_a) << ",\"deck_cards_b\":" << deckKeys(ctx.g.deck_b)
          << ",\"obs_keys\":{";
        bool f = true;
        for (auto& [k, n] : ctx.key_count) { o << (f ? "" : ",") << jstr(k) << ":" << n; f = false; }
        o << "},\"choices\":[";
        for (size_t j = 0; j < ctx.choices.size(); ++j) o << (j ? "," : "") << ctx.choices[j];
        o << "],\"checks\":[";
        for (size_t j = 0; j < ctx.checks.size(); ++j) o << (j ? "," : "") << ctx.checks[j];
        o << "],\"swap_cands\":[";
        for (size_t j = 0; j < ctx.swap_cands.size(); ++j) o << (j ? "," : "") << ctx.swap_cands[j];
        o << "]}\n";
        gout << o.str();
        dec.flush(); gout.flush();
        std::cerr << ctx.g.gid << " " << r.term << " seq " << ctx.seq << " rec " << ctx.recs.size() << " "
                  << r.secs << " s\n";
    }
    std::cerr << "shard " << shard << " done in "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s\n";
}

// ─── verify ───────────────────────────────────────────────────────────────
void verifyMode(const std::string& jobs) {
    std::ifstream in(jobs);
    std::string line;
    long games = 0, games_ok = 0, checks = 0, fp_bad = 0, hash_bad = 0, win_bad = 0, len_bad = 0;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        GameSpec g;
        int winner, n, m;
        ss >> g.gid >> g.deck_a >> g.deck_b >> g.seed >> winner >> n;
        std::vector<int> ch(n);
        for (auto& c : ch) ss >> c;
        ss >> m;
        std::map<int, std::pair<std::string, std::string>> want;
        for (int j = 0; j < m; ++j) { int sq; std::string fp, h; ss >> sq >> fp >> h; want[sq] = {fp, h}; }
        GameCtx ctx;
        ctx.g = g;
        ctx.record = false;
        ctx.replay = &ch;
        long gf = 0, gh = 0;
        ctx.on_seq = [&](int k, const GameState& s, const std::vector<Intent>& legal, const Prompt& p) {
            auto it = want.find(k);
            if (it == want.end()) return false;
            ++checks;
            Encoder enc(s, deciderOf(s, legal, p), ctx.engine);
            enc.build();
            const std::string body = enc.encode(legal, p);
            if (hex(fingerprint(s)) != it->second.first) ++gf;
            if (hex(fnv(body)) != it->second.second) ++gh;
            return false;
        };
        RunOut r = runOne(ctx, false);
        ++games;
        const bool wok = winnerCode(r.res.winner) == winner, lok = ctx.seq == n;
        fp_bad += gf; hash_bad += gh; win_bad += !wok; len_bad += !lok;
        if (!gf && !gh && wok && lok) ++games_ok;
        else std::cout << "MISMATCH " << g.gid << " fp " << gf << " hash " << gh << " winner " << wok << " len "
                       << ctx.seq << "/" << n << " term " << r.term << "\n";
    }
    std::cout << "{\"games\":" << games << ",\"games_ok\":" << games_ok << ",\"decisions_checked\":" << checks
              << ",\"fp_mismatch\":" << fp_bad << ",\"hash_mismatch\":" << hash_bad << ",\"winner_mismatch\":"
              << win_bad << ",\"length_mismatch\":" << len_bad << "}\n";
}

// ─── swap ─────────────────────────────────────────────────────────────────
void becomeDef(GameObject& o, CardDefId id) {
    const auto& def = g_db->get(id);
    o.card_def_id = def.id;
    o.name = def.name;
    o.card_type = def.card_type;
    o.super_type = def.super_type;
    o.keywords = def.keywords;
    o.domains = def.domains;
    o.tags = def.tags;
    o.base_might = def.might;
    o.current_might = def.might;
    o.assault_value = def.assault_value;
    o.shield_value = def.shield_value;
    o.deflect_value = def.deflect_value;
    o.might_bonus = def.might_bonus;
}

std::vector<Intent> legalOf(const GameState& st) {
    EventBus events;
    ProbeAgent a1(false), a2(false);
    GameEngine e(*g_db, events, *g_reg);
    e.mutableState() = st;
    e.testHook_setAgents(&a1, &a2);
    e.testHook_cleanup();
    return e.generateLegalActions();
}

bool g_tgt = false;
void swapMode(const std::string& jobs, const std::string& out_path, const VariantInfo& v, int sims, int nseeds) {
    std::ifstream in(jobs);
    std::ofstream out(out_path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto t0 = std::chrono::steady_clock::now();
        std::istringstream ss(line);
        std::string jid, base_key, fp_want;
        GameSpec g;
        int seq, n;
        ss >> jid >> g.gid >> g.deck_a >> g.deck_b >> g.seed >> seq >> fp_want >> base_key >> n;
        std::vector<int> ch(n);
        for (auto& c : ch) ss >> c;
        GameCtx ctx;
        ctx.g = g;
        ctx.record = false;
        ctx.replay = &ch;
        std::optional<GameState> st;
        std::vector<Intent> legal0;
        Prompt prompt;
        ctx.on_seq = [&](int k, const GameState& s, const std::vector<Intent>& legal, const Prompt& p) {
            if (k != seq) return false;
            st = s; legal0 = legal; prompt = p;
            return true;
        };
        RunOut r = runOne(ctx, false);
        std::ostringstream o;
        o << "{\"jid\":" << jstr(jid) << ",\"gid\":" << jstr(g.gid) << ",\"seq\":" << seq << ",\"base\":" << jstr(base_key)
          << ",\"variant\":" << jstr(v.key) << ",\"sims\":" << sims;
        if (!st.has_value()) {
            o << ",\"error\":" << jstr("replay did not reach the decision: " + r.term) << "}\n";
            out << o.str(); out.flush();
            continue;
        }
        const PlayerId dec = deciderOf(*st, legal0, prompt);
        const bool fp_ok = hex(fingerprint(*st)) == fp_want;
        o << ",\"d\":" << st->decision_index << ",\"seat\":" << static_cast<int>(dec) << ",\"turn\":"
          << st->turn.turn_number << ",\"fp_ok\":" << (fp_ok ? 1 : 0);
        // the swapped version
        GameState sw = *st;
        std::vector<uint64_t> swapped;
        for (auto h : sw.player(dec).hand) {
            if (!sw.objectExists(h)) continue;
            auto& obj = sw.getObject(h);
            if (std::find(v.base_ids.begin(), v.base_ids.end(), obj.card_def_id.raw()) == v.base_ids.end()) continue;
            becomeDef(obj, CardDefId::fromRaw(v.id));
            swapped.push_back(h.raw());
        }
        o << ",\"n_swapped\":" << swapped.size();
        auto game = ::open_spiel::LoadGame(buildGameStr(g.deck_a, g.deck_b, engineSeedOf(g.seed), MatchFormat::BestOf1));
        const auto& gdb = static_cast<const openspiel::RiftboundGame&>(*game).cardDb();
        WorldSampler sampler(gdb);
        o << ",\"versions\":{";
        bool firstv = true;
        for (int ver = 0; ver < 2; ++ver) {
            const GameState& vs = ver == 0 ? *st : sw;
            std::string err;
            std::vector<Intent> legal;
            try { legal = legalOf(vs); } catch (const std::exception& ex) { err = std::string("legal: ") + ex.what(); }
            o << (firstv ? "" : ",") << (ver == 0 ? "\"orig\":{" : "\"swap\":{");
            firstv = false;
            if (ver == 0) {
                bool same = legal.size() == legal0.size();
                for (size_t k = 0; same && k < legal.size(); ++k) same = legal[k] == legal0[k];
                o << "\"legal_same_as_game\":" << (same ? 1 : 0) << ",";
                if (!same) legal = legal0;   // the game's list is the decision that was searched in play
            }
            Encoder enc(vs, dec);
            enc.build();
            o << enc.encode(legal, prompt);
            // actions that play / use one of the (swapped) hand copies
            o << ",\"card_acts\":[";
            bool f = true;
            for (size_t k = 0; k < legal.size(); ++k) {
                const auto& i = legal[k];
                if (std::find(swapped.begin(), swapped.end(), i.card.valid() ? i.card.raw() : ~0ull) != swapped.end()) {
                    o << (f ? "" : ",") << k; f = false;
                }
            }
            o << "],\"searches\":[";
            for (int sd = 0; sd < nseeds && err.empty() && legal.size() >= 1; ++sd) {
                const uint64_t seed = mix(0x5EED0000ull + static_cast<uint64_t>(sd));
                try {
                    StepResult step;
                    step.kind = StepKind::NeedDecision;
                    step.perspective = dec;
                    step.legal = legal;
                    step.prompt = prompt;
                    auto root = openspiel::RiftboundState::makeFromSnapshot(game, engineSeedOf(g.seed), vs, step,
                                                                            GameResult{}, mix(seed ^ 1));
                    IsMctsSearch search(seed, std::make_shared<RiftboundHeuristicEvaluator>(), 2.0, sims, true);
                    auto make_world = [&]() {
                        std::unique_ptr<openspiel::RiftboundState> world(
                            static_cast<openspiel::RiftboundState*>(root->Clone().release()));
                        std::mt19937_64& rng = search.rng();
                        const uint64_t draw_seed = rng();
                        const uint64_t engine_rng_seed = rng();
                        world->editHiddenInformation(
                            [&](GameState& gs) { std::mt19937_64 draw(draw_seed); sampler.sample(gs, dec, draw); },
                            engine_rng_seed);
                        return world;
                    };
                    const SearchSummary ss = legal.size() == 1 ? SearchSummary{} : search.step(make_world, legal);
                    o << (sd ? "," : "") << "{\"seed\":\"" << hex(seed) << "\",\"pi\":" << ss.chosen << ",\"rv\":"
                      << fnum(ss.root_value) << ",\"vis\":[";
                    for (size_t j = 0; j < ss.visits.size(); ++j) o << (j ? "," : "") << ss.visits[j];
                    o << "],\"q\":[";
                    for (size_t j = 0; j < ss.value.size(); ++j) o << (j ? "," : "") << fnum(ss.value[j]);
                    o << "]}";
                } catch (const std::exception& ex) {
                    o << (sd ? "," : "") << "{\"seed\":\"" << hex(seed) << "\",\"error\":" << jstr(ex.what()) << "}";
                }
            }
            o << "]";
            // --tgt: play the first card action, answer the decider's non-target
            // prompts with their first option, and search the target prompt.
            if (g_tgt && err.empty()) {
                std::vector<int> ca;
                for (size_t k = 0; k < legal.size(); ++k)
                    if (std::find(swapped.begin(), swapped.end(), legal[k].card.valid() ? legal[k].card.raw() : ~0ull) != swapped.end())
                        ca.push_back(static_cast<int>(k));
                if (!ca.empty()) {
                    try {
                        StepResult step0;
                        step0.kind = StepKind::NeedDecision; step0.perspective = dec; step0.legal = legal; step0.prompt = prompt;
                        auto t = openspiel::RiftboundState::makeFromSnapshot(game, engineSeedOf(g.seed), vs, step0, GameResult{}, mix(0x7A47ull));
                        {
                            auto ids = openspiel::actionIds(t->legalIntents(), t->engineState());
                            t->ApplyAction(ids[ca[0]]);
                        }
                        int guard = 0; bool found = false;
                        while (!t->IsTerminal() && guard++ < 30) {
                            const auto pk = t->decisionPrompt().kind;
                            if (pk == PromptKind::Target || pk == PromptKind::TargetFirst) { found = true; break; }
                            if (pk == PromptKind::MainPhase || pk == PromptKind::ChainPriority) break;
                            auto ids = openspiel::actionIds(t->legalIntents(), t->engineState());
                            t->ApplyAction(ids[0]);
                        }
                        o << ",\"tgt\":{\"found\":" << (found ? 1 : 0) << ",\"steps\":" << guard;
                        if (found) {
                            const std::vector<Intent> tl = t->legalIntents();
                            const Prompt tp = t->decisionPrompt();
                            Encoder te(t->engineState(), dec);
                            te.build();
                            o << "," << te.encode(tl, tp) << ",\"searches\":[";
                            for (int sd = 0; sd < nseeds; ++sd) {
                                const uint64_t seed = mix(0x5EED0000ull + static_cast<uint64_t>(sd));
                                IsMctsSearch search(seed, std::make_shared<RiftboundHeuristicEvaluator>(), 2.0, sims, true);
                                auto make_world = [&]() {
                                    std::unique_ptr<openspiel::RiftboundState> world(
                                        static_cast<openspiel::RiftboundState*>(t->Clone().release()));
                                    std::mt19937_64& rng = search.rng();
                                    const uint64_t draw_seed = rng();
                                    const uint64_t engine_rng_seed = rng();
                                    world->editHiddenInformation(
                                        [&](GameState& gs) { std::mt19937_64 draw(draw_seed); sampler.sample(gs, dec, draw); },
                                        engine_rng_seed);
                                    return world;
                                };
                                const SearchSummary ss = tl.size() == 1 ? SearchSummary{} : search.step(make_world, tl);
                                o << (sd ? "," : "") << "{\"pi\":" << ss.chosen << ",\"vis\":[";
                                for (size_t j = 0; j < ss.visits.size(); ++j) o << (j ? "," : "") << ss.visits[j];
                                o << "],\"q\":[";
                                for (size_t j = 0; j < ss.value.size(); ++j) o << (j ? "," : "") << fnum(ss.value[j]);
                                o << "]}";
                            }
                            o << "]";
                        }
                        o << "}";
                    } catch (const std::exception& ex) {
                        o << ",\"tgt\":{\"error\":" << jstr(ex.what()) << "}";
                    }
                }
            }
            if (!err.empty()) o << ",\"error\":" << jstr(err);
            o << "}";
        }
        o << "},\"secs\":" << fnum(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()) << "}\n";
        out << o.str();
        out.flush();
    }
}

std::vector<GameSpec> readGames(const std::string& path) {
    std::vector<GameSpec> out;
    std::ifstream in(path);
    GameSpec g;
    while (in >> g.gid >> g.deck_a >> g.deck_b >> g.agent_a >> g.agent_b >> g.seed >> g.swapcand) out.push_back(g);
    return out;
}

}  // namespace dx
}  // namespace probe
}  // namespace riftbound

int main(int argc, char** argv) {
    using namespace riftbound;
    using namespace riftbound::probe;
    using namespace riftbound::probe::dx;
    if (argc < 3) { std::cerr << "usage: see decision_export.cpp\n"; return 2; }
    std::string variant, subjects = "research/card_repr/C_engine/out/subjects.txt";
    int shard = 0, nshards = 1, sims = 100, nseeds = 4;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--variant") variant = argv[++i];
        else if (a == "--shard") { shard = std::stoi(argv[++i]); nshards = std::stoi(argv[++i]); }
        else if (a == "--sims") sims = std::stoi(argv[++i]);
        else if (a == "--tgt") g_tgt = true;
        else if (a == "--seeds") nseeds = std::stoi(argv[++i]);
        else if (a == "--subjects") subjects = argv[++i];
        else pos.push_back(a);
    }
    // One card table for everything: the engine's shared one (the OpenSpiel
    // game and the search agents read it), with the variant added.
    auto& shared = const_cast<openspiel::SharedCards&>(openspiel::sharedCards());
    if (!variant.empty()) {
        if (!findVariant(variant)) { std::cerr << "unknown variant " << variant << "\n"; return 2; }
        registerVariant(shared.registry, variant);
        shared.db.buildFromClasses(shared.registry);
    }
    g_reg = &shared.registry;
    g_db = &shared.db;
    loadKeys(subjects);
    const std::string mode = pos[0];
    if (mode == "play") {
        const auto games = readGames(pos[1]);
        std::cerr << games.size() << " games in " << pos[1] << "\n";
        playMode(games, pos[2], shard, nshards, variant);
        return 0;
    }
    if (mode == "verify") { verifyMode(pos[1]); return 0; }
    if (mode == "swap") {
        if (variant.empty()) { std::cerr << "swap needs --variant\n"; return 2; }
        swapMode(pos[1], pos[2], *findVariant(variant), sims, nseeds);
        return 0;
    }
    if (mode == "validate") {   // validate deck lists given as further arguments
        for (size_t i = 1; i < pos.size(); ++i) std::cout << pos[i] << ": " << validation(pos[i]) << "\n";
        return 0;
    }
    return 2;
}
