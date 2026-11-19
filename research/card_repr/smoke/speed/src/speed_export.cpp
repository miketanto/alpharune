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
#include "rules/deck_validator.h"

#include <random>
#include <array>
#include <cstdlib>
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
    char kind = 'c';        // c card object, b battlefield, k chain item
    std::string key;
    std::vector<long> f;    // situational fields (positional, SCHEMA.txt)
    uint64_t obj = 0;       // object id (c, b) or chain item id (k)
    std::vector<uint64_t> refs;   // objects referenced (attached_to for c; source, targets for k)
    std::string content() const {
        std::string s(1, kind); s += key;
        for (long x : f) { s += ','; s += std::to_string(x); }
        return s;
    }
};

class Encoder {
public:
    Encoder(const GameState& s, PlayerId me) : s_(s), me_(me) {}

    int rel(PlayerId p) const { return p == PlayerId::None ? 0 : (p == me_ ? 1 : 2); }

    Tok cardTok(GameObjectId id) const {
        const auto& o = s_.getObject(id);
        Tok t;
        t.kind = 'c';
        t.obj = id.raw();
        const bool vis = knowledge::identityVisible(s_, id, me_);
        t.key = vis ? cardKey(o.card_def_id, &o) : "hidden";
        const bool board = o.location.has_value();
        t.f = {rel(o.controller), rel(o.owner), static_cast<long>(o.zone.get()), locOf(o),
               board ? o.is_exhausted : 0, board ? o.is_stunned : 0, board ? o.damage_marked : 0,
               board ? o.current_might : 0, board ? o.buff_count : 0, board ? o.temp_might_bonus : 0,
               o.is_token ? 1 : 0, o.is_hidden ? 1 : 0, static_cast<long>(o.combat_designation),
               static_cast<long>(o.attachments.size())};
        if (o.attached_to.has_value()) t.refs.push_back(o.attached_to->raw());
        return t;
    }

    void addObj(GameObjectId id) {
        if (!s_.objectExists(id) || seen_.count(id.raw())) return;
        seen_.insert(id.raw());
        toks_.push_back(cardTok(id));
    }

    void build() {
        // own hand
        for (auto h : s_.player(me_).hand) addObj(h);
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
            const bool vis = !so || so->owner == me_ || knowledge::identityVisible(s_, ci.source, me_) || so->location.has_value();
            t.key = ci.keyword_ability.has_value() ? "keyword_ability" : (vis ? cardKey(ci.card_def_id, so) : "hidden");
            t.f = {rel(ci.controller), pos, resolving ? 1 : 0, ci.is_spell ? 1 : 0, ci.is_ability ? 1 : 0,
                   ci.is_activated_ability ? 1 : 0, static_cast<long>(ci.status), ci.is_equip ? 1 : 0};
            t.refs.push_back(ci.source.valid() ? ci.source.raw() : 0);
            for (auto x : ci.chosen_objects) t.refs.push_back(x.raw());
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
        order(toks_); order(btoks_); order(ktoks_);
        all_.clear();
        for (auto* v : {&btoks_, &toks_, &ktoks_}) for (auto& t : *v) all_.push_back(t);
        reindex();
    }

    void reindex() {
        obj_idx_.clear(); item_idx_.clear();
        for (size_t i = 0; i < all_.size(); ++i) {
            if (all_[i].kind == 'k') item_idx_[all_[i].obj] = static_cast<int>(i);
            else obj_idx_[all_[i].obj] = static_cast<int>(i);
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
        if (i.card.valid()) o << ",\"src\":" << ref(i.card);
        if (i.ability_source.valid()) o << ",\"abs\":" << ref(i.ability_source) << ",\"ai\":" << i.ability_index;
        list("tg", i.targets);
        list("ch", i.chosen_objects);
        list("mv", i.units_to_move);
        list("mu", i.cards_to_mulligan);
        if (!i.damage_assignments.empty()) {
            o << ",\"dm\":[";
            for (size_t j = 0; j < i.damage_assignments.size(); ++j)
                o << (j ? "," : "") << "[" << ref(i.damage_assignments[j].target_unit) << ","
                  << i.damage_assignments[j].damage << "]";
            o << "]";
        }
        const int dst = i.play_location.has_value() ? locCode(i.play_location) : locCode(i.move_destination);
        if (dst >= 0) o << ",\"dst\":" << dst;
        if (i.chosen_value.has_value()) {
            o << ",\"v\":" << *i.chosen_value;
            if (p.kind == PromptKind::ChainItem) {
                const int ci = itemRef(static_cast<uint64_t>(*i.chosen_value));
                if (ci >= 0) o << ",\"ci\":" << ci;
            }
        }
        if (i.chosen_card_def.valid()) o << ",\"nc\":" << jstr(cardKey(i.chosen_card_def, nullptr));
        if (i.chosen_battlefield.valid()) o << ",\"bf\":" << i.chosen_battlefield.raw();
        if (i.play_source != Intent::PlaySource::Hand) o << ",\"ps\":" << static_cast<int>(i.play_source);
        if (i.use_alt_play_cost) o << ",\"alt\":1";
        if (i.pay_location_cost) o << ",\"plc\":1";
        if (i.declines) o << ",\"dec\":1";
        if (i.type == IntentType::PlayFirstDecision) o << ",\"first\":" << (i.choose_first ? 1 : 0);
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
            } else if (t.kind == 'k') {
                auto idx = [&](uint64_t r) { auto it = obj_idx_.find(r); return it == obj_idx_.end() ? -1 : it->second; };
                o << "," << idx(t.refs[0]) << ",[";
                for (size_t j = 1; j < t.refs.size(); ++j) o << (j > 1 ? "," : "") << idx(t.refs[j]);
                o << "]";
            }
            o << "]";
        }
        o << "]";
        return o.str();
    }

    const std::vector<Tok>& all() const { return all_; }

    // Everything the network sees at this decision: global, tokens, actions.
    std::string encode(const std::vector<Intent>& legal, const Prompt& p, int* psrc_out = nullptr) {
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
    std::set<uint64_t> seen_;
    std::vector<Tok> toks_, btoks_, ktoks_, all_;
    std::map<uint64_t, int> obj_idx_, item_idx_, bf_by_obj_;
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
    std::set<std::string> swap_seen;
    long n_multi = 0;
    // speed: kind -> {n, n_searched, t_decide, t_encode, sum_legal}
    std::map<std::string, std::array<double, 5>> tim;
    // replay
    const std::vector<int>* replay = nullptr;
    std::function<bool(int, const GameState&, const std::vector<Intent>&, const Prompt&)> on_seq;  // true: stop
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
            return legal.at((*ctx_.replay)[k]);
        }
        const PlayerId dec = deciderOf(s, legal, p);
        std::string head;
        static const bool norec = std::getenv("SPEED_NOREC") != nullptr;
        static const bool autopay = std::getenv("SPEED_AUTOPAY") != nullptr;
        if (norec) ctx_.record = false;
        const auto te0 = std::chrono::steady_clock::now();
        if (ctx_.record && legal.size() >= 2) {
            Encoder enc(s, dec);
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
        const double t_enc = std::chrono::duration<double>(std::chrono::steady_clock::now() - te0).count();
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
        const bool ap = autopay && legal.size() >= 2 &&
            (p.kind == PromptKind::RuneForEnergy || p.kind == PromptKind::RuneForPower);
        const auto td0 = std::chrono::steady_clock::now();
        const Intent pick = ap ? legal.front() : inner_->decide(s, legal, p);
        const double t_dec = std::chrono::duration<double>(std::chrono::steady_clock::now() - td0).count();
        {
            auto& a = ctx_.tim[std::string(toString(p.kind)) + (legal.size() >= 2 ? "" : "|single")];
            a[0] += 1; a[1] += (ismcts_ && !ap && ismcts_->lastSearch().searched) ? 1 : 0;
            a[2] += t_dec; a[3] += t_enc; a[4] += legal.size();
        }
        const int idx = indexOf(legal, pick);
        if (idx < 0) throw std::runtime_error("agent answered an option not in the legal list");
        ctx_.choices.push_back(idx);
        if (!head.empty()) {
            std::ostringstream o;
            o << head << ",\"pi\":" << idx;
            if (ismcts_ && !ap) {
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
          << ",\"timing\":{";
        { bool ft = true; for (auto& [k, a] : ctx.tim) { o << (ft ? "" : ",") << jstr(k) << ":[" << a[0] << "," << a[1] << ","
              << a[2] << "," << a[3] << "," << a[4] << "]"; ft = false; } }
        o << "},\"autopay\":" << (std::getenv("SPEED_AUTOPAY") ? 1 : 0)
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
            Encoder enc(s, deciderOf(s, legal, p));
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
