// card_probe.cpp -- path C of research/card_repr: engine behaviour probes.
//
// For every subject (a card class of the engine, a renamed copy of one, or a
// test-only synthetic variant defined below) this tool builds a fixed set of
// canonical positions, plays / triggers / activates the card through the real
// engine path (GameEngine::executeIntent + runChain + cleanup, the sequence
// tests/cards/card_test_fixture.h calls playThroughEngine), and records
//   * legality: which actions the engine offers for the card in each window,
//   * the state delta: zone counts per side, Might / damage / status changes of
//     fixed role objects, tokens, XP, score, battlefield control, what was
//     paid from the rune pool (energy, power per domain), events seen.
// The behaviour vector of a subject is the concatenation over probes.
//
// Nothing in the vector is the card's name, id or printing: the card is a
// GameObject whose behaviour is measured by deltas only.
//
// Usage: card_probe <subjects.txt> <raw_out.json>
//   subjects.txt lines: <subject name> <class id> <rename 0|1>
//
// Built only as an EXCLUDE_FROM_ALL target (see card_probe.cmake); test-only
// card classes live in this file, never in src/.

#include "cards/card.h"
#include "cards/card_helpers.h"
#include "cards/card_registry.h"
#include "core/card_db.h"
#include "core/events.h"
#include "core/game_state.h"
#include "core/intent.h"
#include "engine/effect_executor.h"
#include "engine/game_engine.h"
#include "agents/agent_interface.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace riftbound {
namespace probe {

// ─── Fixed cards used to build positions (never the subject) ───────────────
constexpr CardDefId kSkulker = 175;      // Shipyard Skulker, Chaos 3, no text
constexpr CardDefId kDrakehound = 329;   // Eager Drakehound, Fury 3 ("I enter ready": placed, not played)
constexpr CardDefId kGrabber = 99;       // Garbage Grabber, Mind gear
constexpr CardDefId kProgressDay = 114;  // Mind spell (opponent hand)
constexpr CardDefId kMobilize = 134;     // Body spell (opponent hand)
constexpr CardDefId kVoidSeeker = 24;    // Fury spell 3E 1P "Deal 4 to a unit at a battlefield. Draw 1."
constexpr CardDefId kVengeance = 229;    // Order spell 4E 2P "Kill a unit."

// ─── Test-only synthetic card classes ──────────────────────────────────────
// Each copies the engine card's code (src/cards/...) with one parameter
// exposed. A "selfcheck" instance with the original parameters must give the
// same vector as the engine class (checked in the report).

class PunchFirstVar : public SpellCard {
public:
    PunchFirstVar(CardDef d, int amount) : def_(std::move(d)), amount_(amount) {}
    const CardDef& def() const override { return def_; }
    TargetRequirements getTargetRequirements() const override {
        return TargetRequirements{.count = 1, .must_be_unit = true};
    }
    bool needsPlayTimeTarget() const override { return true; }
    void onResolve(CardContext& ctx, const std::vector<GameObjectId>&) override {
        auto legal = enumerateLegalTargets(ctx.state, ctx.controller);
        GameObjectId picked = pickTarget(ctx, "<SELF>: give a unit +N", legal);
        if (picked == kInvalidId) return;
        ctx.executor.giveTemporaryMight(picked, amount_);
    }
private:
    CardDef def_;
    int amount_;
};

class DisciplineVar : public SpellCard {
public:
    DisciplineVar(CardDef d, int draw, bool enemy_only)
        : def_(std::move(d)), draw_(draw), enemy_(enemy_only) {}
    const CardDef& def() const override { return def_; }
    TargetRequirements getTargetRequirements() const override {
        return TargetRequirements{.count = 1, .must_be_unit = true, .must_be_enemy = enemy_};
    }
    bool needsPlayTimeTarget() const override { return true; }
    void onResolve(CardContext& ctx, const std::vector<GameObjectId>& targets) override {
        GameObjectId picked = kInvalidId;
        bool suspended = false;
        if (!targets.empty()) {
            picked = targets[0];
        } else {
            auto legal = enumerateLegalTargets(ctx.state, ctx.controller);
            picked = pickTarget(ctx, "<SELF>: give a unit +2", legal);
            if (picked == kInvalidId && ctx.state.chain.resuming &&
                ctx.state.chain.resuming->resume_point == 7)
                suspended = true;
        }
        if (suspended) return;
        if (picked != kInvalidId && ctx.state.objectExists(picked))
            ctx.executor.giveTemporaryMight(picked, 2);
        ctx.executor.drawCards(ctx.controller, draw_);
    }
private:
    CardDef def_;
    int draw_;
    bool enemy_;
};

class LonelyPoroVar : public UnitCard {
public:
    LonelyPoroVar(CardDef d, int draw) : def_(std::move(d)), draw_(draw) {}
    const CardDef& def() const override { return def_; }
    TriggerType triggerType() const override { return TriggerType::WhenIDie; }
    void onTrigger(CardContext& ctx, const std::vector<GameObjectId>&) override {
        if (!ctx.state.objectExists(ctx.source)) return;
        const auto& lk = asItLeft(ctx, ctx.source);
        if (!lk.valid) return;
        if (!lk.other_friendly_units_here.empty()) return;
        ctx.executor.drawCards(ctx.controller, draw_);
    }
private:
    CardDef def_;
    int draw_;
};

// Defy with its caps exposed; Hush (not in the engine) is the same shape with
// "Power cost [A][A] or more" as its predicate.
class CounterVar : public SpellCard {
public:
    enum class Rule { EnergyAndPowerCap, PowerAtLeast };
    CounterVar(CardDef d, Rule rule, int a, int b) : def_(std::move(d)), rule_(rule), a_(a), b_(b) {}
    const CardDef& def() const override { return def_; }
    unsigned choicesAtFinalization() const override { return kFinalizeResolve; }
    FinalizeChoice mayChooseAtFinalization(const GameState&, PlayerId, GameObjectId) const override {
        return FinalizeChoice::No;
    }
    bool qualifies(const ChainItem& ci, const CardDB& db) const {
        if (!ci.is_spell || ci.card_def_id == kInvalidId) return false;
        const auto& d = db.get(ci.card_def_id);
        if (rule_ == Rule::EnergyAndPowerCap) return d.energy_cost <= a_ && d.power_cost <= b_;
        return d.power_cost >= a_;
    }
    bool hasLegalTargets(const GameState& state, PlayerId, const CardDB& db) const override {
        return anyChainItem(state, [&](const ChainItem& ci) { return qualifies(ci, db); });
    }
    bool hasLegalTargets(const GameState& state, PlayerId) const override {
        return anyChainItem(state, [](const ChainItem& ci) { return ci.is_spell; });
    }
    void onResolve(CardContext& ctx, const std::vector<GameObjectId>&) override {
        const CardDB& db = ctx.executor.cardDB();
        const auto spell = pickChainItem(
            ctx, "<SELF>: counter a spell",
            chainItemChoices(ctx, [&](const ChainItem& ci) { return qualifies(ci, db); }));
        if (!spell.has_value()) return;
        if (finalizingChoices(ctx)) return;
        if (*spell != kInvalidId) counterChosenItem(ctx, *spell);
    }
private:
    CardDef def_;
    Rule rule_;
    int a_, b_;
};

class SabotageVar : public SpellCard {
public:
    SabotageVar(CardDef d, bool unit_only) : def_(std::move(d)), unit_only_(unit_only) {}
    const CardDef& def() const override { return def_; }
    void onResolve(CardContext& ctx, const std::vector<GameObjectId>&) override {
        auto& ri = ctx.state.chain.resuming.value();
        PlayerId opp = opponent(ctx.controller);
        auto& opps = ctx.state.player(opp);
        switch (ri.resume_point) {
        case 0: {
            if (opps.hand.empty()) return;
            for (auto card_id : opps.hand) {
                if (!ctx.state.objectExists(card_id)) continue;
                const auto& obj = ctx.state.getObject(card_id);
                ctx.events.emit(CardRevealedEvent{card_id, obj.card_def_id, obj.owner, true,
                                                  PlayerId::None, ZoneType::Hand});
            }
            std::vector<Intent> choices;
            for (auto card_id : opps.hand) {
                if (!ctx.state.objectExists(card_id)) continue;
                const auto& obj = ctx.state.getObject(card_id);
                if (obj.isUnit() != unit_only_) continue;
                Intent c;
                c.type = IntentType::MakeChoice;
                c.player = ctx.controller;
                c.chosen_objects = {card_id};
                choices.push_back(c);
            }
            if (choices.empty()) return;
            ctx.executor.requestChoice(ctx.controller, std::move(choices),
                                       "recycle 1 card from opponent's hand (<SELF>)",
                                       PromptKind::ChooseObject);
            ri.resume_point = 1;
            return;
        }
        case 1: {
            auto choice = ctx.executor.takeChoice();
            if (!choice || choice->chosen_objects.empty()) return;
            auto card_id = choice->chosen_objects[0];
            if (!ctx.state.objectExists(card_id)) return;
            auto& hand = opps.hand;
            auto it = std::find(hand.begin(), hand.end(), card_id);
            if (it == hand.end()) return;
            hand.erase(it);
            ctx.executor.recycleCards(opp, {card_id}, knowledge::everyone());
            return;
        }
        }
    }
private:
    CardDef def_;
    bool unit_only_;
};

class PitRookieVar : public UnitCard {
public:
    explicit PitRookieVar(CardDef d) : def_(std::move(d)) {}
    const CardDef& def() const override { return def_; }
    unsigned choicesAtFinalization() const override { return kFinalizeTrigger; }
    TriggerType triggerType() const override { return TriggerType::WhenYouPlayMe; }
    void onTrigger(CardContext& ctx, const std::vector<GameObjectId>&) override {
        std::vector<GameObjectId> others;
        for (const auto& [id, obj] : ctx.state.objects.onBoard()) {
            if (id == ctx.source) continue;
            if (!obj.isUnit() || obj.controller != ctx.controller) continue;
            if (!obj.location.has_value()) continue;
            others.push_back(id);
        }
        std::sort(others.begin(), others.end());
        const GameObjectId t = pickTarget(ctx, "<SELF>: buff another friendly unit", others);
        if (finalizingChoices(ctx)) return;
        if (t == kInvalidId) return;
        ctx.executor.buffUnit(t);
    }
private:
    CardDef def_;
};

// ─── Play-changing variants (smoke/pairs) ──────────────────────────────────
// Each keeps the base card's CardDef (cost, domains, keywords, type) and swaps
// the effect for one that changes the right way to play it.
enum class ScoreCond { Always, NoFriendlyUnits, NoOtherFriendlyUnits, OppHand4, FriendlyUnitAtBattlefield, Behind };
inline int friendlyUnitsOnBoard(const GameState& s, PlayerId p, GameObjectId except) {
    int n = 0;
    for (const auto& [id, obj] : s.objects.onBoard()) {
        if (id == except) continue;
        if (obj.isUnit() && obj.controller == p && obj.location.has_value()) ++n;
    }
    return n;
}
inline bool scoreCondHolds(const CardContext& ctx, ScoreCond c, GameObjectId except) {
    switch (c) {
    case ScoreCond::Always: return true;
    case ScoreCond::NoFriendlyUnits:
    case ScoreCond::NoOtherFriendlyUnits: return friendlyUnitsOnBoard(ctx.state, ctx.controller, except) == 0;
    case ScoreCond::OppHand4: return ctx.state.player(opponent(ctx.controller)).hand.size() >= 4;
    case ScoreCond::FriendlyUnitAtBattlefield:
        for (const auto& [id, obj] : ctx.state.objects.onBoard())
            if (obj.isUnit() && obj.controller == ctx.controller && obj.isAtBattlefield()) return true;
        return false;
    case ScoreCond::Behind:
        return ctx.state.player(opponent(ctx.controller)).score > ctx.state.player(ctx.controller).score;
    }
    return false;
}
// Spell: "Score 1 point [if <cond>]." No target.
class ScoreSpellVar : public SpellCard {
public:
    ScoreSpellVar(CardDef d, ScoreCond c) : def_(std::move(d)), c_(c) {}
    const CardDef& def() const override { return def_; }
    void onResolve(CardContext& ctx, const std::vector<GameObjectId>&) override {
        if (scoreCondHolds(ctx, c_, kInvalidId)) ctx.executor.scorePoints(ctx.controller, 1);
    }
private:
    CardDef def_;
    ScoreCond c_;
};
// Punch First's targeting with "Kill a unit." in place of "+5 this turn".
class PunchKillVar : public SpellCard {
public:
    explicit PunchKillVar(CardDef d) : def_(std::move(d)) {}
    const CardDef& def() const override { return def_; }
    TargetRequirements getTargetRequirements() const override {
        return TargetRequirements{.count = 1, .must_be_unit = true};
    }
    bool needsPlayTimeTarget() const override { return true; }
    void onResolve(CardContext& ctx, const std::vector<GameObjectId>&) override {
        auto legal = enumerateLegalTargets(ctx.state, ctx.controller);
        GameObjectId picked = pickTarget(ctx, "<SELF>: kill a unit", legal);
        if (picked == kInvalidId) return;
        ctx.executor.killObject(picked);
    }
private:
    CardDef def_;
};
// Punch First's targeting with "Its controller scores 1 point" (or "Its
// controller's opponent scores 1 point") in place of "+5 this turn": the right
// target side is fixed by the text (friendly vs enemy).
class PunchScoreTargetVar : public SpellCard {
public:
    PunchScoreTargetVar(CardDef d, bool opponent_scores) : def_(std::move(d)), opp_(opponent_scores) {}
    const CardDef& def() const override { return def_; }
    TargetRequirements getTargetRequirements() const override {
        return TargetRequirements{.count = 1, .must_be_unit = true};
    }
    bool needsPlayTimeTarget() const override { return true; }
    void onResolve(CardContext& ctx, const std::vector<GameObjectId>&) override {
        auto legal = enumerateLegalTargets(ctx.state, ctx.controller);
        GameObjectId picked = pickTarget(ctx, "<SELF>: a unit's controller scores", legal);
        if (picked == kInvalidId || !ctx.state.objectExists(picked)) return;
        const PlayerId c = ctx.state.getObject(picked).controller;
        ctx.executor.scorePoints(opp_ ? opponent(c) : c, 1);
    }
private:
    CardDef def_;
    bool opp_;
};
// Unit: "When you play me, score 1 point [if you control no other units]."
class ScoreUnitVar : public UnitCard {
public:
    ScoreUnitVar(CardDef d, ScoreCond c) : def_(std::move(d)), c_(c) {}
    const CardDef& def() const override { return def_; }
    TriggerType triggerType() const override { return TriggerType::WhenYouPlayMe; }
    void onTrigger(CardContext& ctx, const std::vector<GameObjectId>&) override {
        if (scoreCondHolds(ctx, c_, ctx.source)) ctx.executor.scorePoints(ctx.controller, 1);
    }
private:
    CardDef def_;
    ScoreCond c_;
};

CardDef derive(const CardRegistry& reg, CardDefId base, uint32_t new_id, const std::string& tag) {
    CardDef d = *reg.classDef(base);
    d.id = CardDefId::fromRaw(new_id);
    d.def_id = "synthetic-" + tag;
    d.name = "Synthetic " + tag;
    d.public_code.clear();
    return d;
}

struct SyntheticSpec { std::string name; uint32_t id; };
std::vector<SyntheticSpec> registerSynthetics(CardRegistry& reg) {
    std::vector<SyntheticSpec> out;
    auto add = [&](const std::string& name, uint32_t id, std::unique_ptr<Card> c) {
        reg.registerCard(CardDefId::fromRaw(id), std::move(c));
        out.push_back({name, id});
    };
    constexpr CardDefId kPunch = 419, kDisc = 58, kPoro = 777, kDefy = 45, kSab = 156, kRookie = 136;
    // selfchecks: the copied code with the original parameters
    add("selfcheck:punch_first", 1601, std::make_unique<PunchFirstVar>(derive(reg, kPunch, 1601, "a"), 5));
    add("selfcheck:discipline", 1602, std::make_unique<DisciplineVar>(derive(reg, kDisc, 1602, "b"), 1, false));
    add("selfcheck:lonely_poro", 1603, std::make_unique<LonelyPoroVar>(derive(reg, kPoro, 1603, "c"), 1));
    add("selfcheck:defy", 1604, std::make_unique<CounterVar>(derive(reg, kDefy, 1604, "d"), CounterVar::Rule::EnergyAndPowerCap, 4, 1));
    add("selfcheck:sabotage", 1605, std::make_unique<SabotageVar>(derive(reg, kSab, 1605, "e"), false));
    add("selfcheck:pit_rookie", 1606, std::make_unique<PitRookieVar>(derive(reg, kRookie, 1606, "f")));
    // synthetic variants of cards.json
    add("synthetic:punch_first__amount4", 1611, std::make_unique<PunchFirstVar>(derive(reg, kPunch, 1611, "g"), 4));
    {
        auto d = derive(reg, kPunch, 1612, "h"); d.energy_cost = 3;
        add("synthetic:punch_first__energy3", 1612, std::make_unique<PunchFirstVar>(d, 5));
    }
    {
        auto d = derive(reg, kPunch, 1613, "i"); d.power_cost = 0;
        add("synthetic:punch_first__power0", 1613, std::make_unique<PunchFirstVar>(d, 5));
    }
    {
        auto d = derive(reg, kPunch, 1614, "j");
        d.keywords.clear(Keyword::Action); d.keywords.set(Keyword::Reaction);
        add("synthetic:punch_first__reaction", 1614, std::make_unique<PunchFirstVar>(d, 5));
    }
    add("synthetic:discipline__draw2", 1615, std::make_unique<DisciplineVar>(derive(reg, kDisc, 1615, "k"), 2, false));
    add("synthetic:discipline__enemy", 1616, std::make_unique<DisciplineVar>(derive(reg, kDisc, 1616, "l"), 1, true));
    add("synthetic:lonely_poro__draw2", 1617, std::make_unique<LonelyPoroVar>(derive(reg, kPoro, 1617, "m"), 2));
    add("synthetic:defy__cost2", 1618, std::make_unique<CounterVar>(derive(reg, kDefy, 1618, "n"), CounterVar::Rule::EnergyAndPowerCap, 2, 1));
    add("synthetic:sabotage__unit", 1619, std::make_unique<SabotageVar>(derive(reg, kSab, 1619, "o"), true));
    {
        auto d = derive(reg, kRookie, 1620, "p"); d.might = 3;
        add("synthetic:pit_rookie__might3", 1620, std::make_unique<PitRookieVar>(d));
    }
    // Hush (rad-041-167): not implemented by the engine (Radiance). Stand-in
    // with its printed stats, the Defy code shape and Hush's predicate.
    {
        CardDef d = derive(reg, kDefy, 1650, "hush");
        d.domains = {};            // cards.json lists no domain for this printing
        d.energy_cost = 1;
        d.power_cost = 2;
        d.keywords.reset(); d.keywords.set(Keyword::Reaction);
        add("standin:hush", 1650, std::make_unique<CounterVar>(d, CounterVar::Rule::PowerAtLeast, 2, 0));
    }
    return out;
}

// ─── Global context ────────────────────────────────────────────────────────
CardRegistry* g_reg = nullptr;
CardDB* g_db = nullptr;
CardDefId g_subject = kInvalidId;
bool g_place_nothing = false;   // baseline run: the subject is not put in play
bool g_renamed = false;
const char* kRename = "Quorvath Jindle";

constexpr PlayerId P1 = PlayerId::Player1;
constexpr PlayerId P2 = PlayerId::Player2;
// Whose perspective the features are written from ("me"). Pool H: always P1.
// Pool S (situations/sampler.cpp): the player deciding at the situation.
PlayerId g_me = P1;

// ─── Agent: deterministic policy ───────────────────────────────────────────
// Priority / focus: pass. Engine-ordering prompts (trigger order, which staged
// combat begins): first option. A card's prompt (a "may", a target, a mode,
// an X, a cost decision): policy FIRST takes the first option (declines every
// "may": Card::confirmOptional lists "No" first), policy LAST takes the last
// option (accepts "may"s and optional costs, chooses the highest-id object).
// Anything else (combat damage assignment...): first legal option.
class ProbeAgent : public AgentInterface {
public:
    bool last = false;
    int prompts = 0;
    explicit ProbeAgent(bool l) : last(l) {}
    static bool prefixAll(const std::vector<Intent>& legal, const char* p) {
        return std::all_of(legal.begin(), legal.end(),
                           [&](const Intent& i) { return i.choice_label.rfind(p, 0) == 0; });
    }
    Intent selectAction(const GameState&, const std::vector<Intent>& legal) override {
        if (legal.empty()) return Intent{};
        const bool is_choice = std::all_of(legal.begin(), legal.end(),
            [](const Intent& i) { return i.type == IntentType::MakeChoice; });
        if (is_choice) {
            if (prefixAll(legal, "Order \xE2\x80\x94 ") ||
                prefixAll(legal, "Combat begins at \xE2\x80\x94 ") ||
                prefixAll(legal, "Showdown begins at \xE2\x80\x94 "))
                return legal.front();
            ++prompts;
            if (prompts > 200) return legal.front();   // safety against a long "last" loop
            return last ? legal.back() : legal.front();
        }
        for (const auto& i : legal)
            if (i.type == IntentType::PassPriority || i.type == IntentType::PassFocus) return i;
        return legal.front();
    }
};

// ─── Worlds ────────────────────────────────────────────────────────────────
const std::vector<std::string> kRoles = {"F1", "F2", "E1", "E2", "EG", "E3", "H1", "H2", "OU", "OM", "OB", "SELF"};

struct World {
    GameState st;
    std::map<std::string, GameObjectId> roles;
    GameObjectId h1 = kInvalidId, h2 = kInvalidId;   // my two filler units in hand
};

GameObjectId newObj(World& w, PlayerId owner, CardDefId def_id) {
    auto id = w.st.createObject();
    auto& obj = w.st.getObject(id);
    obj.card_def_id = def_id;
    obj.owner = owner;
    obj.controller = owner;
    const auto& def = g_db->get(def_id);
    obj.name = (g_renamed && def_id == g_subject) ? std::string(kRename) : def.name;
    obj.card_type = def.card_type;
    obj.super_type = def.super_type;
    obj.keywords = def.keywords;
    obj.domains = def.domains;
    obj.tags = def.tags;
    obj.base_might = def.might;
    obj.current_might = def.might;
    obj.assault_value = def.assault_value;
    obj.shield_value = def.shield_value;
    obj.deflect_value = def.deflect_value;
    obj.might_bonus = def.might_bonus;
    return id;
}
GameObjectId inHand(World& w, PlayerId p, CardDefId d) {
    auto id = newObj(w, p, d);
    w.st.getObject(id).zone = ZoneType::Hand;
    w.st.player(p).hand.push_back(id);
    return id;
}
GameObjectId inDeck(World& w, PlayerId p, CardDefId d) {
    auto id = newObj(w, p, d);
    w.st.getObject(id).zone = ZoneType::MainDeck;
    w.st.player(p).main_deck.push_back(id);
    return id;
}
GameObjectId inTrash(World& w, PlayerId p, CardDefId d) {
    auto id = newObj(w, p, d);
    w.st.getObject(id).zone = ZoneType::Trash;
    w.st.player(p).trash.push_back(id);
    return id;
}
// loc: -1 base, 0/1 battlefield
GameObjectId unitAt(World& w, PlayerId p, CardDefId d, int loc, int might = -1) {
    auto id = newObj(w, p, d);
    auto& o = w.st.getObject(id);
    o.card_type = CardType::Unit;
    if (might >= 0) { o.base_might = might; o.current_might = might; }
    if (loc >= 0) {
        o.zone = ZoneType::BattlefieldZone;
        o.location = BattlefieldLocation{BattlefieldId::fromRaw(loc)};
    } else {
        o.zone = ZoneType::Base;
        o.location = BaseLocation{p};
    }
    return id;
}
GameObjectId gearAtBase(World& w, PlayerId p, CardDefId d) {
    auto id = newObj(w, p, d);
    auto& o = w.st.getObject(id);
    o.zone = ZoneType::Base;
    o.location = BaseLocation{p};
    return id;
}
GameObjectId legendOf(World& w, PlayerId p, CardDefId d) {
    auto id = newObj(w, p, d);
    auto& o = w.st.getObject(id);
    o.card_type = CardType::Legend;
    o.zone = ZoneType::LegendZone;
    o.location = std::nullopt;   // in the Legend Zone, not on the board
    w.st.player(p).legend_zone = id;
    return id;
}
GameObjectId battlefieldCard(World& w, int bf, CardDefId d) {
    auto id = newObj(w, PlayerId::None, d);
    auto& b = w.st.getObject(id);
    auto& slot = w.st.battlefields[bf];
    b.card_type = CardType::Battlefield;
    b.zone = ZoneType::BattlefieldZone;
    b.owner = b.controller = slot.controller.value_or(PlayerId::None);
    slot.card_object_id = id;
    if (auto* bfc = dynamic_cast<const BattlefieldCard*>(g_reg->get(d))) {
        slot.blocks_move_to_base = bfc->blocksMoveToBase();
        slot.blocks_unit_play = bfc->blocksUnitPlay();
    }
    return id;
}
void hideAt(World& w, GameObjectId card, int bf) {
    auto& o = w.st.getObject(card);
    auto& hand = w.st.player(o.owner).hand;
    hand.erase(std::remove(hand.begin(), hand.end(), card), hand.end());
    o.zone = ZoneType::FacedownZone;
    o.is_hidden = true;
    o.hidden_at = BattlefieldId::fromRaw(bf);
    o.hidden_on_turn = w.st.turn.turn_number - 1;
    w.st.battlefields[bf].facedown.push_back(card);
}
GameObjectId pushSpellOnChain(World& w, PlayerId controller, CardDefId def_id,
                              std::vector<GameObjectId> targets) {
    auto src = newObj(w, controller, def_id);
    auto& obj = w.st.getObject(src);
    obj.zone = ZoneType::Chain;
    ChainItem item;
    item.id = w.st.chain.allocateId();
    item.status = ChainItemStatus::Finalized;
    item.source = src;
    item.controller = controller;
    item.is_spell = true;
    item.card_def_id = def_id;
    item.targets = std::move(targets);
    w.st.chain.items.push_back(item);
    return src;
}
void combatInProgress(World& w, int bf, PlayerId attacker) {
    auto& b = w.st.battlefields[bf];
    b.combat_in_progress = true;
    b.attacker = attacker;
    b.defender = opponent(attacker);
    b.is_contested = true;
    b.contested_by = attacker;
    for (auto& [id, obj] : w.st.objects) {
        if (!obj.isUnit() || obj.battlefieldId() != std::optional<BattlefieldId>{b.id}) continue;
        obj.combat_designation = obj.controller == attacker ? CombatDesignation::Attacker
                                                            : CombatDesignation::Defender;
        obj.card_counters["__attack_defend_checked"] = static_cast<int>(b.id.raw()) + 1;
    }
}

// The shared skeleton: two battlefields, decks, hands, my funded pool.
// units: whether the standard board (F1 F2 E1 E2 EG) is built.
World skeleton(PlayerId turn_player) {
    World w;
    w.st.mode = ModeOfPlay{};
    w.st.players[0].id = P1;
    w.st.players[1].id = P2;
    for (int i = 0; i < 2; ++i) {
        BattlefieldState bf;
        bf.id = BattlefieldId::fromRaw(i);
        w.st.battlefields.push_back(bf);
    }
    w.st.turn.phase = TurnPhase::MainPhase;
    w.st.turn.turn_number = 3;
    w.st.turn.turn_player = turn_player;
    w.st.turn.regular_turn_player = turn_player;
    for (int i = 0; i < 8; ++i) inDeck(w, P1, kSkulker);
    for (int i = 0; i < 8; ++i) inDeck(w, P2, kSkulker);
    w.h1 = inHand(w, P1, kSkulker);
    w.h2 = inHand(w, P1, kSkulker);
    w.roles["H1"] = w.h1;
    w.roles["H2"] = w.h2;
    w.roles["OU"] = inHand(w, P2, kSkulker);       // opponent's unit card in hand
    w.roles["OM"] = inHand(w, P2, kProgressDay);   // opponent's Mind spell in hand
    w.roles["OB"] = inHand(w, P2, kMobilize);      // opponent's Body spell in hand
    auto& pool = w.st.player(P1).rune_pool;
    pool.energy = 14;
    for (int d = 0; d < static_cast<int>(Domain::Count); ++d) pool.power[d] = 3;
    pool.universal_power = 2;
    return w;
}
void standardBoard(World& w, int f2_loc = 0) {
    w.roles["F1"] = unitAt(w, P1, kSkulker, -1, 3);
    w.st.getObject(w.roles["F1"]).is_exhausted = true;   // something to ready
    w.roles["F2"] = unitAt(w, P1, kSkulker, f2_loc, 2);
    w.roles["E1"] = unitAt(w, P2, kDrakehound, 1, 3);
    w.roles["E2"] = unitAt(w, P2, kSkulker, 1, 2);
    w.roles["EG"] = gearAtBase(w, P2, kGrabber);
    if (f2_loc == 0) w.st.battlefields[0].controller = P1;
    w.st.battlefields[1].controller = P2;
}

// Put the subject "in play" for trigger / static probes.
//   unit: at `unit_loc` (-1 base, 0/1 battlefield); gear: my base;
//   legend: my legend zone; battlefield: slot `bf_slot`. Spells / runes: not placed.
GameObjectId placeSelfInPlay(World& w, int unit_loc, int bf_slot) {
    if (g_place_nothing) return kInvalidId;
    const auto& def = g_db->get(g_subject);
    GameObjectId id = kInvalidId;
    switch (def.card_type) {
        case CardType::Unit: id = unitAt(w, P1, g_subject, unit_loc); break;
        case CardType::Gear: id = gearAtBase(w, P1, g_subject); break;
        case CardType::Legend: id = legendOf(w, P1, g_subject); break;
        case CardType::Battlefield: id = battlefieldCard(w, bf_slot, g_subject); break;
        default: return kInvalidId;
    }
    w.roles["SELF"] = id;
    return id;
}

// ─── Features ──────────────────────────────────────────────────────────────
std::vector<std::string> g_feature_names;   // one block's names (filled once)

const std::vector<std::string> kRoleFeat = {"on_board", "at_base", "at_bf0", "at_bf1", "in_hand",
    "in_trash", "in_deck", "in_banish", "facedown", "might", "temp_might", "damage", "exhausted",
    "stunned", "buffs", "controlled_by_me"};
const std::vector<std::string> kSideFeat = {"hand", "deck", "trash", "banish", "units", "tokens",
    "gear", "sum_might", "sum_damage", "stunned", "exhausted_units", "buffs", "score", "xp", "facedown",
    // phase 2: runes on the board (pool S pays costs from runes, not from a funded pool)
    "runes", "runes_exhausted"};
const std::vector<std::string> kEventFeat = {"ev_drawn_me", "ev_drawn_opp", "ev_dmg_to_my_units",
    "ev_dmg_to_opp_units", "ev_died_me", "ev_died_opp", "ev_tokens_me", "ev_tokens_opp",
    "ev_token_might_me", "ev_stun_my_units", "ev_stun_opp_units", "ev_readied_me", "ev_revealed",
    "ev_returned_to_hand", "ev_moved_me", "ev_moved_opp", "ev_spells_resolved_me",
    "ev_spells_resolved_opp", "ev_cards_played_me", "ev_banished", "ev_hidden", "prompts", "error"};

bool contains(const std::vector<GameObjectId>& v, GameObjectId id) {
    return std::find(v.begin(), v.end(), id) != v.end();
}

std::vector<double> stateFeatures(const GameState& s, const std::map<std::string, GameObjectId>& roles) {
    std::vector<double> f;
    for (const auto& r : kRoles) {
        auto it = roles.find(r);
        if (it == roles.end() || !s.objectExists(it->second)) {
            f.insert(f.end(), kRoleFeat.size(), 0.0);
            continue;
        }
        const auto& o = s.getObject(it->second);
        const auto& own = s.player(o.owner);
        const bool on = o.location.has_value() && o.zone != ZoneType::FacedownZone &&
                        o.zone != ZoneType::LegendZone;
        auto bf = o.battlefieldId();
        const bool at_base = on && !bf.has_value();
        f.push_back(on || o.zone == ZoneType::LegendZone);
        f.push_back(at_base);
        f.push_back(on && bf.has_value() && bf->raw() == 0);
        f.push_back(on && bf.has_value() && bf->raw() == 1);
        f.push_back(contains(own.hand, o.id));
        f.push_back(contains(own.trash, o.id));
        f.push_back(contains(own.main_deck, o.id));
        f.push_back(contains(own.banishment, o.id));
        f.push_back(o.zone == ZoneType::FacedownZone);
        f.push_back(on && o.isUnit() ? o.current_might : 0);
        f.push_back(on ? o.temp_might_bonus : 0);   // "+N this turn" before the 0 floor
        f.push_back(on ? o.damage_marked : 0);
        f.push_back(on ? o.is_exhausted : 0);
        f.push_back(on ? o.is_stunned : 0);
        f.push_back(on ? o.buff_count : 0);
        f.push_back(o.controller == g_me);
    }
    for (PlayerId p : {g_me, opponent(g_me)}) {
        const auto& ps = s.player(p);
        double units = 0, tokens = 0, gear = 0, might = 0, dmg = 0, stun = 0, exh = 0, buffs = 0;
        double runes = 0, runes_exh = 0;
        for (const auto& [id, o] : s.objects) {
            if (!o.location.has_value() || o.controller != p) continue;
            if (o.zone == ZoneType::FacedownZone || o.zone == ZoneType::LegendZone) continue;
            if (o.isUnit()) {
                ++units;
                if (o.is_token) ++tokens;
                might += o.current_might;
                dmg += o.damage_marked;
                stun += o.is_stunned;
                exh += o.is_exhausted;
                buffs += o.buff_count;
            } else if (o.isGear()) {
                ++gear;
                if (o.is_token) ++tokens;
            } else if (o.card_type == CardType::Rune) {
                ++runes;
                runes_exh += o.is_exhausted;
            }
        }
        double facedown = 0;
        for (const auto& b : s.battlefields)
            for (auto c : b.facedown)
                if (s.objectExists(c) && s.getObject(c).controller == p) ++facedown;
        f.insert(f.end(), {double(ps.hand.size()), double(ps.main_deck.size()), double(ps.trash.size()),
                           double(ps.banishment.size()), units, tokens, gear, might, dmg, stun, exh,
                           buffs, double(ps.score), double(ps.xp), facedown, runes, runes_exh});
    }
    const auto& pool = s.player(g_me).rune_pool;
    f.push_back(pool.energy);
    for (int d = 0; d < static_cast<int>(Domain::Count); ++d) f.push_back(pool.power[d]);
    f.push_back(pool.universal_power);
    for (int i = 0; i < 2; ++i) {
        if (i >= static_cast<int>(s.battlefields.size())) { f.insert(f.end(), 5, 0.0); continue; }
        const auto& b = s.battlefields[i];
        f.push_back(b.controller == std::optional<PlayerId>{g_me});
        f.push_back(b.controller == std::optional<PlayerId>{opponent(g_me)});
        f.push_back(b.is_contested);
        f.push_back(b.combat_staged || b.combat_in_progress);
        f.push_back(b.showdown_staged || b.showdown_in_progress);
    }
    f.push_back(s.chain.items.size());
    return f;
}

std::vector<std::string> stateFeatureNames() {
    std::vector<std::string> n;
    for (const auto& r : kRoles)
        for (const auto& x : kRoleFeat) n.push_back(r + "." + x);
    for (const char* side : {"me", "opp"})
        for (const auto& x : kSideFeat) n.push_back(std::string(side) + "." + x);
    n.push_back("pool.energy");
    for (int d = 0; d < static_cast<int>(Domain::Count); ++d)
        n.push_back(std::string("pool.power_") + toString(static_cast<Domain>(d)));
    n.push_back("pool.power_any");
    for (int i = 0; i < 2; ++i)
        for (const char* x : {"ctrl_me", "ctrl_opp", "contested", "combat", "showdown"})
            n.push_back("bf" + std::to_string(i) + "." + x);
    n.push_back("chain_size");
    return n;
}

struct EventCounter {
    std::vector<double> c = std::vector<double>(kEventFeat.size(), 0.0);
    std::vector<boost::signals2::connection> conns;
    const GameState* live = nullptr;
    PlayerId ctl(GameObjectId id) const {
        if (live && live->objectExists(id)) return live->getObject(id).controller;
        return PlayerId::None;
    }
    void connect(EventBus& ev) {
        conns.push_back(ev.on_cards_drawn.connect([this](const CardsDrawnEvent& e) {
            c[e.player == g_me ? 0 : 1] += e.count; }));
        conns.push_back(ev.on_damage_dealt.connect([this](const DamageDealtEvent& e) {
            c[ctl(e.target) == g_me ? 2 : 3] += e.amount; }));
        conns.push_back(ev.on_unit_died.connect([this](const UnitDiedEvent& e) {
            c[e.controller == g_me ? 4 : 5] += 1; }));
        conns.push_back(ev.on_token_created.connect([this](const TokenCreatedEvent& e) {
            c[e.controller == g_me ? 6 : 7] += 1;
            if (e.controller == g_me && live && live->objectExists(e.object))
                c[8] += live->getObject(e.object).current_might; }));
        conns.push_back(ev.on_unit_stunned.connect([this](const UnitStunnedEvent& e) {
            c[e.victim_controller == g_me ? 9 : 10] += 1; }));
        conns.push_back(ev.on_unit_readied.connect([this](const UnitReadiedEvent& e) {
            if (e.controller == g_me) c[11] += 1; }));
        conns.push_back(ev.on_card_revealed.connect([this](const CardRevealedEvent&) { c[12] += 1; }));
        conns.push_back(ev.on_unit_returned_to_hand.connect([this](const UnitReturnedToHandEvent&) { c[13] += 1; }));
        conns.push_back(ev.on_unit_moved.connect([this](const UnitMovedEvent& e) {
            c[e.controller == g_me ? 14 : 15] += 1; }));
        conns.push_back(ev.on_spell_resolved.connect([this](const SpellResolvedEvent& e) {
            c[e.controller == g_me ? 16 : 17] += 1; }));
        conns.push_back(ev.on_card_played.connect([this](const CardPlayedEvent& e) {
            if (e.player == g_me) c[18] += 1; }));
        conns.push_back(ev.on_left_board.connect([this](const LeftBoardEvent& e) {
            if (e.destination == ZoneType::Banishment) c[19] += 1; }));
        conns.push_back(ev.on_card_hidden.connect([this](const CardHiddenEvent&) { c[20] += 1; }));
    }
    void disconnect() { for (auto& x : conns) x.disconnect(); conns.clear(); }
};

int g_errors = 0;
std::vector<std::string> g_error_log;

using EngineFn = std::function<void(GameEngine&, World&)>;

void settle(GameEngine& e) {
    e.testHook_runChain();
    e.testHook_cleanup();
    e.testHook_runChain();
}

// Runs `pre` (not recorded), then `act` (recorded): returns the delta block
// (state features after - before, then the event counts).
std::vector<double> runRecorded(const World& w0, const EngineFn& pre, const EngineFn& act,
                                bool policy_last, const std::string& what) {
    World w = w0;
    EventBus events;
    ProbeAgent a1(policy_last), a2(false);
    std::vector<double> before, after;
    EventCounter ec;
    try {
        GameEngine e(*g_db, events, *g_reg);
        e.mutableState() = w.st;
        e.testHook_setAgents(&a1, &a2);
        e.testHook_cleanup();
        if (pre) pre(e, w);
        before = stateFeatures(e.state(), w.roles);
        ec.live = &e.state();
        ec.connect(events);
        act(e, w);
        ec.disconnect();
        after = stateFeatures(e.state(), w.roles);
    } catch (const std::exception& ex) {
        ec.disconnect();
        ++g_errors;
        g_error_log.push_back(what + ": " + ex.what());
        if (before.empty()) before = stateFeatures(w.st, w.roles);
        after = before;
        ec.c.back() = 1;
    }
    std::vector<double> d(before.size());
    for (size_t i = 0; i < d.size(); ++i) d[i] = after[i] - before[i];
    ec.c[kEventFeat.size() - 2] = a1.prompts;
    d.insert(d.end(), ec.c.begin(), ec.c.end());
    return d;
}

size_t deltaDim() { return stateFeatureNames().size() + kEventFeat.size(); }

// ─── Output accumulation ───────────────────────────────────────────────────
struct Block { std::string name; std::vector<std::string> fields; std::vector<double> values; };
struct SubjectOut { std::vector<Block> blocks; std::vector<Block> raw; };

std::vector<std::string> deltaNames() {
    auto n = stateFeatureNames();
    n.insert(n.end(), kEventFeat.begin(), kEventFeat.end());
    return n;
}

// ─── Play windows ──────────────────────────────────────────────────────────
// Slots of actions the engine may offer for the card itself.
const std::vector<std::string> kSlots = {"play_noloc", "play_base", "play_bf0", "play_bf1", "hide", "activate"};

int slotOf(const Intent& i, GameObjectId self) {
    const bool is_play = i.type == IntentType::PlayCard || i.type == IntentType::PlayActionCard ||
                         i.type == IntentType::PlayReaction;
    if (is_play && i.card == self) {
        if (!i.play_location.has_value()) return 0;
        if (std::holds_alternative<BaseLocation>(*i.play_location)) return 1;
        auto bf = std::get<BattlefieldLocation>(*i.play_location).id.raw();
        return bf == 0 ? 2 : 3;
    }
    if (i.type == IntentType::HideCard && i.card == self) return 4;
    const bool is_act = i.type == IntentType::ActivateAbility || i.type == IntentType::ActivateActionAbility ||
                        i.type == IntentType::ActivateReactionAbility;
    if (is_act && i.ability_source == self) return 5;
    return -1;
}

struct Window {
    std::string name;
    std::function<World()> build;   // SELF placed (hand / trash / facedown)
    EngineFn pre;                   // unrecorded setup actions inside the engine
};

// Play h1 (a vanilla unit from my hand) to base through the engine: the
// "a card was already played this turn" pre-action.
void playFiller(GameEngine& e, GameObjectId card, int loc) {
    Intent play;
    play.type = IntentType::PlayCard;
    play.player = P1;
    play.card = card;
    if (loc < 0) play.play_location = LocationId{BaseLocation{P1}};
    else play.play_location = LocationId{BattlefieldLocation{BattlefieldId::fromRaw(loc)}};
    e.testHook_executeIntent(play);
    settle(e);
}

std::vector<Window> playWindows() {
    std::vector<Window> ws;
    ws.push_back({"empty_main", [] {
        World w = skeleton(P1);
        w.roles["SELF"] = inHand(w, P1, g_subject);
        return w; }, nullptr});
    ws.push_back({"board_main", [] {
        World w = skeleton(P1); standardBoard(w);
        w.roles["SELF"] = inHand(w, P1, g_subject);
        return w; }, nullptr});
    ws.push_back({"board_main_after_a_card", [] {
        World w = skeleton(P1); standardBoard(w);
        w.roles["SELF"] = inHand(w, P1, g_subject);
        return w; }, [](GameEngine& e, World& w) { playFiller(e, w.h1, -1); }});
    ws.push_back({"showdown_my_focus", [] {
        World w = skeleton(P1); standardBoard(w);
        w.roles["E3"] = unitAt(w, P2, kDrakehound, 0, 3);
        w.roles["SELF"] = inHand(w, P1, g_subject);
        combatInProgress(w, 0, P1);
        w.st.turn.ns_state = NeutralShowdownState::Showdown;
        w.st.turn.focus_holder = P1;
        return w; }, nullptr});
    ws.push_back({"opp_fury_spell_on_chain", [] {
        World w = skeleton(P2); standardBoard(w);
        w.roles["SELF"] = inHand(w, P1, g_subject);
        pushSpellOnChain(w, P2, kVoidSeeker, {w.roles["F2"]});
        w.st.turn.oc_state = OpenClosedState::Closed;
        w.st.turn.priority_holder = P1;
        return w; }, nullptr});
    ws.push_back({"opp_power2_spell_on_chain", [] {
        World w = skeleton(P2); standardBoard(w);
        w.roles["SELF"] = inHand(w, P1, g_subject);
        pushSpellOnChain(w, P2, kVengeance, {w.roles["F2"]});
        w.st.turn.oc_state = OpenClosedState::Closed;
        w.st.turn.priority_holder = P1;
        return w; }, nullptr});
    ws.push_back({"from_my_trash", [] {
        World w = skeleton(P1); standardBoard(w);
        w.roles["SELF"] = inTrash(w, P1, g_subject);
        return w; }, nullptr});
    ws.push_back({"facedown_opp_spell_on_chain", [] {
        World w = skeleton(P2); standardBoard(w);
        w.roles["SELF"] = inHand(w, P1, g_subject);
        hideAt(w, w.roles["SELF"], 0);
        pushSpellOnChain(w, P2, kVoidSeeker, {w.roles["F2"]});
        w.st.turn.oc_state = OpenClosedState::Closed;
        w.st.turn.priority_holder = P1;
        return w; }, nullptr});
    // Phase 2 (coverage gaps of phase 1):
    // Weak enemies at a battlefield: a damage spell can kill (Alpha Strike's
    // "for each unit this kills, gain 1 XP"; Rampage-style kills).
    ws.push_back({"board_weak_enemies", [] {
        World w = skeleton(P1); standardBoard(w);
        for (const char* r : {"E1", "E2"}) {
            auto& o = w.st.getObject(w.roles[r]);
            o.base_might = 1; o.current_might = 1;
        }
        w.roles["SELF"] = inHand(w, P1, g_subject);
        return w; }, nullptr});
    // My two units together at one battlefield, no friendly unit alone
    // anywhere (En Garde's "if it is the only unit you control there" false).
    ws.push_back({"board_friends_together", [] {
        World w = skeleton(P1); standardBoard(w);
        auto& f1 = w.st.getObject(w.roles["F1"]);
        f1.zone = ZoneType::BattlefieldZone;
        f1.location = BattlefieldLocation{BattlefieldId::fromRaw(0)};
        w.roles["SELF"] = inHand(w, P1, g_subject);
        return w; }, nullptr});
    return ws;
}

void probePlayWindows(SubjectOut& out) {
    const auto dn = deltaNames();
    bool engine_offers_hide = false;   // set by board_main: the facedown window needs a legal Hide
    for (const auto& win : playWindows()) {
        const World w0 = win.build();
        const bool na = win.name == "facedown_opp_spell_on_chain" && !engine_offers_hide;
        // Legality: what the engine offers for the card.
        std::vector<std::vector<Intent>> by_slot(kSlots.size());
        try {
            World w = w0;
            EventBus events;
            ProbeAgent a1(false), a2(false);
            GameEngine e(*g_db, events, *g_reg);
            e.mutableState() = w.st;
            e.testHook_setAgents(&a1, &a2);
            e.testHook_cleanup();
            if (win.pre) win.pre(e, w);
            const auto legal = e.generateLegalActions();
            if (std::getenv("PROBE_DEBUG"))
                std::cerr << "  [" << win.name << "] " << legal.size() << " legal, self="
                          << toString(w.roles["SELF"]) << "\n";
            for (const auto& i : legal) {
                if (std::getenv("PROBE_DEBUG"))
                    std::cerr << "     type=" << int(i.type) << " card=" << toString(i.card)
                              << " src=" << toString(i.ability_source) << "\n";
                int s = slotOf(i, w.roles["SELF"]);
                if (s >= 0 && !na) by_slot[s].push_back(i);
            }
            if (win.name == "board_main" && !by_slot[4].empty()) engine_offers_hide = true;
        } catch (const std::exception& ex) {
            ++g_errors;
            g_error_log.push_back(win.name + " legality: " + ex.what());
        }
        Block leg{"legal/" + win.name, kSlots, {}};
        for (auto& v : by_slot) leg.values.push_back(double(v.size()));
        out.blocks.push_back(leg);
        for (size_t s = 0; s < kSlots.size(); ++s) {
            for (bool last : {false, true}) {
                Block b{"exec/" + win.name + "/" + kSlots[s] + (last ? "/last" : "/first"), dn, {}};
                if (by_slot[s].empty()) {
                    b.values.assign(dn.size(), 0.0);
                } else {
                    // Intents hold object ids; the world is rebuilt identically,
                    // so an offered intent of the slot is valid again.
                    // Phase 2: policy FIRST runs the first offered intent of
                    // the slot, policy LAST the last one (phase 1 ran the
                    // first under both, so a card whose targets are intent
                    // targets, e.g. En Garde, saw one target only).
                    const Intent pick = last ? by_slot[s].back() : by_slot[s].front();
                    auto played = runRecorded(w0, win.pre, [&](GameEngine& e, World&) {
                        e.testHook_executeIntent(pick);
                        settle(e);
                    }, last, b.name);
                    // Baseline: the same window when the player passes instead
                    // (the opponent's spell on the chain still resolves).
                    auto passed = runRecorded(w0, win.pre, [](GameEngine& e, World&) { settle(e); },
                                              last, b.name + "/pass");
                    out.raw.push_back({b.name, dn, played});
                    b.values.resize(played.size());
                    for (size_t k = 0; k < played.size(); ++k) b.values[k] = played[k] - passed[k];
                    b.values[played.size() - 1] = played.back();   // error flag stays raw
                }
                out.blocks.push_back(b);
            }
        }
    }
}

// ─── Trigger probes (the card already in play; an event happens) ───────────
struct TriggerProbe {
    std::string name;
    bool units_only;
    std::function<World()> build;
    EngineFn act;
};

std::vector<TriggerProbe> triggerProbes() {
    std::vector<TriggerProbe> ps;
    auto isUnit = [] { return g_db->get(g_subject).card_type == CardType::Unit; };
    // Death of the card, alone at its battlefield (F2 at base instead).
    ps.push_back({"die_alone", true, [] {
        World w = skeleton(P1); standardBoard(w, -1);
        placeSelfInPlay(w, 0, 0);
        return w; }, [](GameEngine& e, World& w) {
        e.testHook_executor().killObject(w.roles["SELF"]); settle(e); }});
    ps.push_back({"die_with_friend", true, [] {
        World w = skeleton(P1); standardBoard(w);
        placeSelfInPlay(w, 0, 0);
        return w; }, [](GameEngine& e, World& w) {
        e.testHook_executor().killObject(w.roles["SELF"]); settle(e); }});
    ps.push_back({"friendly_unit_dies", false, [] {
        World w = skeleton(P1); standardBoard(w);
        placeSelfInPlay(w, -1, 0);
        return w; }, [](GameEngine& e, World& w) {
        e.testHook_executor().killObject(w.roles["F2"]); settle(e); }});
    // Combat: attack a lone 2-Might enemy at bf1.
    auto attackWorld = [isUnit](bool pair) {
        World w = skeleton(P1);
        // F1 holds bf0 (so "each other battlefield you control" counts 1).
        w.roles["F1"] = unitAt(w, P1, kSkulker, 0, 3);
        w.st.battlefields[0].controller = P1;
        w.roles["E1"] = unitAt(w, P2, kDrakehound, -1, 3);
        w.roles["E2"] = unitAt(w, P2, kSkulker, 1, 2);
        w.roles["EG"] = gearAtBase(w, P2, kGrabber);
        auto& b = w.st.battlefields[1];
        b.controller = P2;
        if (isUnit()) {
            placeSelfInPlay(w, 1, 1);
            w.roles["F2"] = unitAt(w, P1, kSkulker, pair ? 1 : -1, 2);
        } else {
            w.roles["F2"] = unitAt(w, P1, kSkulker, 1, 3);
            if (pair) w.st.getObject(w.roles["F1"]).location = BattlefieldLocation{BattlefieldId::fromRaw(1)},
                      w.st.getObject(w.roles["F1"]).zone = ZoneType::BattlefieldZone;
            placeSelfInPlay(w, -1, 1);
        }
        b.is_contested = true;
        b.contested_by = P1;
        b.combat_staged = true;
        return w;
    };
    ps.push_back({"attack_alone", false, [attackWorld] { return attackWorld(false); },
        [](GameEngine& e, World&) {
            e.testHook_runCombat(BattlefieldId::fromRaw(1)); settle(e); }});
    ps.push_back({"attack_with_friend", false, [attackWorld] { return attackWorld(true); },
        [](GameEngine& e, World&) {
            e.testHook_runCombat(BattlefieldId::fromRaw(1)); settle(e); }});
    ps.push_back({"defend_alone", false, [isUnit] {
        World w = skeleton(P2);
        w.roles["F1"] = unitAt(w, P1, kSkulker, -1, 3);
        w.roles["E1"] = unitAt(w, P2, kDrakehound, 0, 3);
        w.roles["E2"] = unitAt(w, P2, kSkulker, -1, 2);
        w.roles["EG"] = gearAtBase(w, P2, kGrabber);
        auto& b = w.st.battlefields[0];
        b.controller = P1;
        if (isUnit()) {
            w.roles["F2"] = unitAt(w, P1, kSkulker, -1, 2);
            placeSelfInPlay(w, 0, 0);
        } else {
            w.roles["F2"] = unitAt(w, P1, kSkulker, 0, 2);
            placeSelfInPlay(w, -1, 0);
        }
        b.is_contested = true;
        b.contested_by = P2;
        b.combat_staged = true;
        return w; }, [](GameEngine& e, World&) {
        e.testHook_runCombat(BattlefieldId::fromRaw(0)); settle(e); }});
    ps.push_back({"hold_bf0", false, [] {
        World w = skeleton(P1); standardBoard(w);
        placeSelfInPlay(w, 0, 0);
        return w; }, [](GameEngine& e, World&) {
        e.testHook_scoreHold(P1, BattlefieldId::fromRaw(0)); settle(e); }});
    ps.push_back({"i_play_two_units", false, [] {
        World w = skeleton(P1); standardBoard(w);
        placeSelfInPlay(w, -1, 0);
        return w; }, [](GameEngine& e, World& w) {
        playFiller(e, w.h1, 0);
        playFiller(e, w.h2, -1); }});
    auto moveTo = [](int dest) {
        return [dest](GameEngine& e, World& w) {
            for (const auto& i : e.generateLegalActions()) {
                if (i.type != IntentType::StandardMove) continue;
                if (i.units_to_move != std::vector<GameObjectId>{w.roles["SELF"]}) continue;
                if (!i.move_destination || !std::holds_alternative<BattlefieldLocation>(*i.move_destination)) continue;
                if (std::get<BattlefieldLocation>(*i.move_destination).id.raw() != uint32_t(dest)) continue;
                e.testHook_executeIntent(i);
                settle(e);
                return;
            }
        };
    };
    ps.push_back({"move_to_my_bf0", true, [] {
        World w = skeleton(P1); standardBoard(w);
        placeSelfInPlay(w, -1, 0);
        return w; }, moveTo(0)});
    ps.push_back({"move_to_enemy_bf1", true, [] {
        World w = skeleton(P1); standardBoard(w);
        placeSelfInPlay(w, -1, 0);
        return w; }, moveTo(1)});
    // Phase 2: a combat that conquers for sure (a 5-Might friend beside the
    // card against a lone 0-Might enemy), and the enemy also has a token gear
    // (Energy cost 0 for "a gear with Energy cost no more than my Might").
    ps.push_back({"attack_conquer", false, [isUnit] {
        World w = skeleton(P1);
        w.roles["F1"] = unitAt(w, P1, kSkulker, 0, 3);
        w.st.battlefields[0].controller = P1;
        w.roles["E1"] = unitAt(w, P2, kDrakehound, -1, 3);
        // a 0-Might defender: it deals no combat damage, so a 1-Might
        // card beside the friend survives to conquer
        w.roles["E2"] = unitAt(w, P2, kSkulker, 1, 0);
        w.roles["EG"] = gearAtBase(w, P2, kGrabber);
        w.roles["E3"] = gearAtBase(w, P2, kGrabber);
        w.st.getObject(w.roles["E3"]).is_token = true;
        auto& b = w.st.battlefields[1];
        b.controller = P2;
        w.roles["F2"] = unitAt(w, P1, kSkulker, 1, 5);
        if (isUnit()) placeSelfInPlay(w, 1, 1);
        else placeSelfInPlay(w, -1, 1);
        b.is_contested = true;
        b.contested_by = P1;
        b.combat_staged = true;
        return w; }, [](GameEngine& e, World&) {
        e.testHook_runCombat(BattlefieldId::fromRaw(1)); settle(e); }});
    // Phase 2: battlefield-to-battlefield move ([Ganking]); with 6 XP too
    // ([Level 6] grants, e.g. Master Yi, Tempered).
    for (bool xp6 : {false, true}) {
        ps.push_back({xp6 ? "move_bf0_to_bf1_xp6" : "move_bf0_to_bf1", true, [xp6] {
            World w = skeleton(P1); standardBoard(w);
            if (xp6) w.st.player(P1).xp = 6;
            placeSelfInPlay(w, 0, 0);
            return w; }, moveTo(1)});
    }
    ps.push_back({"activate_first_ability", false, [] {
        World w = skeleton(P1); standardBoard(w);
        placeSelfInPlay(w, -1, 0);
        return w; }, [](GameEngine& e, World& w) {
        for (const auto& i : e.generateLegalActions()) {
            if (slotOf(i, w.roles["SELF"]) != 5) continue;
            e.testHook_executeIntent(i);
            settle(e);
            return;
        } }});
    return ps;
}

void probeTriggers(SubjectOut& out) {
    const auto dn = deltaNames();
    const auto t = g_db->get(g_subject).card_type;
    const bool permanent = t == CardType::Unit || t == CardType::Gear || t == CardType::Legend ||
                           t == CardType::Battlefield;
    for (const auto& p : triggerProbes()) {
        for (bool last : {false, true}) {
            Block b{"trigger/" + p.name + (last ? "/last" : "/first"), dn, {}};
            const bool applies = permanent && (!p.units_only || t == CardType::Unit);
            if (!applies) { b.values.assign(dn.size(), 0.0); out.blocks.push_back(b); continue; }
            auto v = runRecorded(p.build(), nullptr, p.act, last, b.name);
            out.raw.push_back({b.name, dn, v});
            // Baseline: the same event with a vanilla 3-Might unit in the
            // card's place (unit subjects) or with nothing in play (others).
            const bool unit = t == CardType::Unit;
            const std::string key = p.name + (last ? "/last" : "/first") + (unit ? "/unit" : "/none");
            static std::map<std::string, std::vector<double>> cache;
            if (!cache.count(key)) {
                const CardDefId saved = g_subject;
                const bool saved_rn = g_renamed;
                g_renamed = false;
                if (unit) g_subject = kSkulker; else g_place_nothing = true;
                cache[key] = runRecorded(p.build(), nullptr, p.act, last, b.name + "/baseline");
                g_subject = saved;
                g_renamed = saved_rn;
                g_place_nothing = false;
            }
            const auto& base = cache[key];
            b.values.resize(v.size());
            for (size_t k = 0; k < v.size(); ++k) b.values[k] = v[k] - base[k];
            b.values[v.size() - 1] = v.back();
            out.blocks.push_back(b);
        }
    }
}

// ─── Static presence: with vs without the card in play, after a Cleanup ────
void probeStatic(SubjectOut& out) {
    struct S { std::string name; std::function<World(bool)> build; };
    std::vector<S> ss;
    ss.push_back({"board", [](bool with) {
        World w = skeleton(P1); standardBoard(w);
        if (with) placeSelfInPlay(w, 0, 1);
        return w; }});
    ss.push_back({"friend_defends_alone", [](bool with) {
        World w = skeleton(P2); standardBoard(w);
        w.roles["E3"] = unitAt(w, P2, kDrakehound, 0, 3);
        if (with) placeSelfInPlay(w, -1, 1);
        combatInProgress(w, 0, P2);
        w.st.turn.ns_state = NeutralShowdownState::Showdown;
        return w; }});
    ss.push_back({"xp6", [](bool with) {
        World w = skeleton(P1); standardBoard(w);
        w.st.player(P1).xp = 6;
        if (with) placeSelfInPlay(w, 0, 1);
        return w; }});
    std::vector<std::string> names = stateFeatureNames();
    std::vector<std::string> kw;
    for (int k = 0; k < static_cast<int>(Keyword::Count); ++k)
        kw.push_back(std::string("SELF.kw_") + toString(static_cast<Keyword>(k)));
    names.insert(names.end(), kw.begin(), kw.end());
    names.push_back("SELF.untargetable_by_enemy");
    names.push_back("error");
    const auto t = g_db->get(g_subject).card_type;
    const bool permanent = t == CardType::Unit || t == CardType::Gear || t == CardType::Legend ||
                           t == CardType::Battlefield;
    for (const auto& s : ss) {
        Block b{"static/" + s.name, names, {}};
        if (!permanent) { b.values.assign(names.size(), 0.0); out.blocks.push_back(b); continue; }
        auto settled = [&](bool with, std::vector<double>& feats, std::vector<double>& extra) {
            World w = s.build(with);
            EventBus events;
            ProbeAgent a1(false), a2(false);
            GameEngine e(*g_db, events, *g_reg);
            e.mutableState() = w.st;
            e.testHook_setAgents(&a1, &a2);
            e.testHook_cleanup();
            feats = stateFeatures(e.state(), w.roles);
            extra.assign(kw.size() + 1, 0.0);
            auto it = w.roles.find("SELF");
            if (it != w.roles.end() && e.state().objectExists(it->second)) {
                const auto& o = e.state().getObject(it->second);
                for (int k = 0; k < static_cast<int>(Keyword::Count); ++k)
                    extra[k] = o.hasKeyword(static_cast<Keyword>(k));
                extra[kw.size()] = o.untargetable_by_enemy || o.untargetable_by_enemy_this_turn;
            }
        };
        try {
            std::vector<double> f0, x0, f1, x1;
            settled(false, f0, x0);
            settled(true, f1, x1);
            for (size_t i = 0; i < f0.size(); ++i) b.values.push_back(f1[i] - f0[i]);
            b.values.insert(b.values.end(), x1.begin(), x1.end());
            b.values.push_back(0);
        } catch (const std::exception& ex) {
            ++g_errors;
            g_error_log.push_back(b.name + ": " + ex.what());
            b.values.assign(names.size(), 0.0);
            b.values.back() = 1;
        }
        out.blocks.push_back(b);
    }
}

// ─── Phase 2: printed facts, always present ────────────────────────────────
// Phase 1 had no block that does not depend on the card being legal or in
// play: a card legal in no probe (synthetic:defy__cost2) became a zero vector
// and lost its own cost. This block is read from the engine's card table (the
// CardDef the engine plays with), never zero for any card (the type one-hot).
std::vector<std::string> staticCardNames() {
    std::vector<std::string> n = {"energy_cost", "power_cost"};
    for (int d = 0; d < static_cast<int>(Domain::Count); ++d)
        n.push_back(std::string("domain_") + toString(static_cast<Domain>(d)));
    n.insert(n.end(), {"n_domains", "might", "might_bonus", "assault", "shield", "deflect"});
    for (const char* t : {"unit", "spell", "gear", "rune", "battlefield", "legend"}) n.push_back(std::string("type_") + t);
    for (const char* t : {"none", "champion", "signature", "token"}) n.push_back(std::string("super_") + t);
    for (int k = 0; k < static_cast<int>(Keyword::Count); ++k)
        n.push_back(std::string("kw_") + toString(static_cast<Keyword>(k)));
    n.insert(n.end(), {"timing_action", "timing_reaction", "timing_main_only", "n_activated_abilities"});
    return n;
}
std::vector<double> staticCardValues(CardDefId id) {
    const auto& d = g_db->get(id);
    std::vector<double> v = {double(d.energy_cost), double(d.power_cost)};
    int nd = 0;
    for (int k = 0; k < static_cast<int>(Domain::Count); ++k) {
        const bool has = std::find(d.domains.begin(), d.domains.end(), static_cast<Domain>(k)) != d.domains.end();
        v.push_back(has);
        nd += has;
    }
    v.insert(v.end(), {double(nd), double(d.might), double(d.might_bonus), double(d.assault_value),
                       double(d.shield_value), double(d.deflect_value)});
    for (int t = 0; t < 6; ++t) v.push_back(static_cast<int>(d.card_type) == t);
    for (int t = 0; t < 4; ++t) v.push_back(static_cast<int>(d.super_type) == t);
    for (int k = 0; k < static_cast<int>(Keyword::Count); ++k) v.push_back(d.keywords.has(static_cast<Keyword>(k)));
    const bool act = d.keywords.has(Keyword::Action), rea = d.keywords.has(Keyword::Reaction);
    v.push_back(act);
    v.push_back(rea);
    v.push_back(d.card_type == CardType::Spell && !act && !rea);
    const Card* c = g_reg->get(id);
    v.push_back(c ? double(c->activatedAbilities().size()) : 0.0);
    return v;
}
void probeStaticCard(SubjectOut& out) {
    out.blocks.push_back({"static/card", staticCardNames(), staticCardValues(g_subject)});
}

// ─── Phase 2: the opponent chooses my card ─────────────────────────────────
// The opponent (P2, its main phase, funded) holds "Deal 4 to a unit at a
// battlefield" and the card is a unit of mine at bf0. Recorded: whether the
// engine lets the opponent choose it, how many choices it had, what the
// opponent paid (Deflect shows as extra Power) and what happened to the card.
// Raw values (no baseline). Variants: my XP 0 and 6 ([Level 6] grants).
class TargetingAgent : public AgentInterface {
public:
    GameObjectId self = kInvalidId;
    int offered = 0, options = 0;
    Intent selectAction(const GameState&, const std::vector<Intent>& legal) override {
        if (legal.empty()) return Intent{};
        const bool is_choice = std::all_of(legal.begin(), legal.end(),
            [](const Intent& i) { return i.type == IntentType::MakeChoice; });
        if (is_choice) {
            for (const auto& i : legal)
                if (contains(i.chosen_objects, self)) { offered = 1; options = int(legal.size()); return i; }
            if (!options) options = int(legal.size());
            return legal.front();
        }
        for (const auto& i : legal)
            if (i.type == IntentType::PassPriority || i.type == IntentType::PassFocus) return i;
        return legal.front();
    }
};
const std::vector<std::string> kTargetedNames = {"spell_offered", "self_offered", "n_options",
    "opp_paid_energy", "opp_paid_power", "self_damage", "self_died", "self_might_after", "error"};

void probeTargeted(SubjectOut& out) {
    for (bool xp6 : {false, true}) {
        Block b{std::string("targeted/opp_spell") + (xp6 ? "_xp6" : ""), kTargetedNames, {}};
        if (g_db->get(g_subject).card_type != CardType::Unit) {
            b.values.assign(kTargetedNames.size(), 0.0);
            out.blocks.push_back(b);
            continue;
        }
        try {
            World w = skeleton(P2); standardBoard(w);
            if (xp6) w.st.player(P1).xp = 6;
            const GameObjectId self = unitAt(w, P1, g_subject, 0);
            w.roles["SELF"] = self;
            const GameObjectId spell = inHand(w, P2, kVoidSeeker);
            auto& pool = w.st.player(P2).rune_pool;
            pool.energy = 14;
            for (int d = 0; d < static_cast<int>(Domain::Count); ++d) pool.power[d] = 3;
            pool.universal_power = 2;
            EventBus events;
            ProbeAgent a1(false);
            TargetingAgent a2;
            a2.self = self;
            GameEngine e(*g_db, events, *g_reg);
            e.mutableState() = w.st;
            e.testHook_setAgents(&a1, &a2);
            e.testHook_cleanup();
            std::vector<Intent> plays;
            for (const auto& i : e.generateLegalActions())
                if (i.card == spell && (i.type == IntentType::PlayCard || i.type == IntentType::PlayActionCard ||
                                        i.type == IntentType::PlayReaction)) plays.push_back(i);
            double spell_offered = !plays.empty(), via_intent = 0;
            Intent pick;
            for (const auto& i : plays) if (contains(i.targets, self)) { pick = i; via_intent = 1; break; }
            if (!via_intent && !plays.empty()) pick = plays.front();
            const auto& p2 = e.state().player(P2).rune_pool;
            double e0 = p2.energy, w0 = p2.universal_power;
            for (int d = 0; d < static_cast<int>(Domain::Count); ++d) w0 += p2.power[d];
            if (spell_offered) {
                e.testHook_executeIntent(pick);
                settle(e);
            }
            const auto& q = e.state().player(P2).rune_pool;
            double w1 = q.universal_power;
            for (int d = 0; d < static_cast<int>(Domain::Count); ++d) w1 += q.power[d];
            const bool alive = e.state().objectExists(self) && e.state().getObject(self).location.has_value();
            const int n_opts = via_intent ? int(plays.size()) : a2.options;
            b.values = {spell_offered, double(via_intent || a2.offered), double(n_opts),
                        e0 - q.energy, w0 - w1,
                        alive ? double(e.state().getObject(self).damage_marked) : 0.0,
                        double(!alive), alive ? double(e.state().getObject(self).current_might) : 0.0, 0};
        } catch (const std::exception& ex) {
            ++g_errors;
            g_error_log.push_back(b.name + ": " + ex.what());
            b.values.assign(kTargetedNames.size(), 0.0);
            b.values.back() = 1;
        }
        out.blocks.push_back(b);
    }
}

// ─── Phase 2: what the engine offers around the card while it is in play ───
// Phase 1 had legality only for cards played from a zone; battlefields and
// legends had none. With the card in play on the standard board (a unit at
// bf0, a gear at my base, a legend in my Legend Zone, a battlefield in slot
// 0, which I control), my main phase: its activations, its own Standard Moves
// by destination, and what units may do at slot 0 (play there, move there,
// move from there to base). Raw counts (no baseline). Variants: XP 0 and 6.
const std::vector<std::string> kInPlayNames = {"activate", "self_move_base", "self_move_bf0",
    "self_move_bf1", "unit_play_to_bf0", "unit_move_to_bf0", "unit_move_bf0_to_base", "error"};

std::vector<double> inPlayLegality(const GameState& st, const std::vector<Intent>& legal, GameObjectId self,
                                   int bfslot = 0) {
    std::vector<double> v(kInPlayNames.size(), 0.0);
    auto bfOf = [](const std::optional<LocationId>& l) -> int {
        if (!l.has_value()) return -2;
        if (std::holds_alternative<BaseLocation>(*l)) return -1;
        return int(std::get<BattlefieldLocation>(*l).id.raw());
    };
    for (const auto& i : legal) {
        if (slotOf(i, self) == 5) v[0] += 1;
        if (i.type == IntentType::StandardMove) {
            const int dest = bfOf(i.move_destination);
            if (contains(i.units_to_move, self)) {
                if (dest == -1) v[1] += 1;
                if (dest == 0) v[2] += 1;   // self moves: by absolute slot
                if (dest == 1) v[3] += 1;
            }
            if (dest == bfslot) v[5] += 1;
            if (dest == -1)
                for (auto u : i.units_to_move)
                    if (st.objectExists(u)) {
                        auto bf = st.getObject(u).battlefieldId();
                        if (bf.has_value() && int(bf->raw()) == bfslot) { v[6] += 1; break; }
                    }
        }
        if ((i.type == IntentType::PlayCard || i.type == IntentType::PlayActionCard ||
             i.type == IntentType::PlayReaction) && bfOf(i.play_location) == bfslot &&
            st.objectExists(i.card) && st.getObject(i.card).isUnit())
            v[4] += 1;
    }
    return v;
}

void probeInPlayLegal(SubjectOut& out) {
    const auto t = g_db->get(g_subject).card_type;
    const bool permanent = t == CardType::Unit || t == CardType::Gear || t == CardType::Legend ||
                           t == CardType::Battlefield;
    for (bool xp6 : {false, true}) {
        Block b{std::string("inplay_legal/board") + (xp6 ? "_xp6" : ""), kInPlayNames, {}};
        if (!permanent) { b.values.assign(kInPlayNames.size(), 0.0); out.blocks.push_back(b); continue; }
        try {
            World w = skeleton(P1); standardBoard(w);
            if (xp6) w.st.player(P1).xp = 6;
            const GameObjectId self = placeSelfInPlay(w, 0, 0);
            EventBus events;
            ProbeAgent a1(false), a2(false);
            GameEngine e(*g_db, events, *g_reg);
            e.mutableState() = w.st;
            e.testHook_setAgents(&a1, &a2);
            e.testHook_cleanup();
            b.values = inPlayLegality(e.state(), e.generateLegalActions(), self);
        } catch (const std::exception& ex) {
            ++g_errors;
            g_error_log.push_back(b.name + ": " + ex.what());
            b.values.assign(kInPlayNames.size(), 0.0);
            b.values.back() = 1;
        }
        out.blocks.push_back(b);
    }
}

// ─── Runes: what one ready rune of this card lets its controller pay ───────
void probeRune(SubjectOut& out) {
    std::vector<std::string> names = {"pays_1_energy"};
    for (int d = 0; d < static_cast<int>(Domain::Count); ++d)
        names.push_back(std::string("pays_1_power_") + toString(static_cast<Domain>(d)));
    names.push_back("pays_2_energy");
    Block b{"rune/pay", names, {}};
    if (g_db->get(g_subject).card_type != CardType::Rune) {
        b.values.assign(names.size(), 0.0);
        out.blocks.push_back(b);
        return;
    }
    World w = skeleton(P1);
    w.st.player(P1).rune_pool = RunePool{};
    auto id = newObj(w, P1, g_subject);
    auto& r = w.st.getObject(id);
    r.card_type = CardType::Rune;
    r.zone = ZoneType::Base;
    r.location = BaseLocation{P1};
    EventBus events;
    ProbeAgent a1(false), a2(false);
    GameEngine e(*g_db, events, *g_reg);
    e.mutableState() = w.st;
    e.testHook_setAgents(&a1, &a2);
    b.values.push_back(e.testHook_canPayAdditionalCost(P1, 1, 0, Domain::Fury));
    for (int d = 0; d < static_cast<int>(Domain::Count); ++d)
        b.values.push_back(e.testHook_canPayAdditionalCost(P1, 0, 1, static_cast<Domain>(d)));
    b.values.push_back(e.testHook_canPayAdditionalCost(P1, 2, 0, Domain::Fury));
    out.blocks.push_back(b);
}

// ─── JSON output ───────────────────────────────────────────────────────────
std::string jstr(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (static_cast<unsigned char>(c) < 0x20) { char buf[8]; std::snprintf(buf, 8, "\\u%04x", c); o += buf; }
        else o += c;
    }
    return o + "\"";
}
std::string jnum(double v) {
    if (v == static_cast<long long>(v)) return std::to_string(static_cast<long long>(v));
    char buf[32]; std::snprintf(buf, 32, "%.6g", v); return buf;
}

}  // namespace probe
}  // namespace riftbound

#ifndef CARD_PROBE_NO_MAIN
int main(int argc, char** argv) {
    using namespace riftbound;
    using namespace riftbound::probe;
    if (argc < 3) { std::cerr << "usage: card_probe <subjects.txt> <raw_out.json>\n"; return 2; }
    const auto t0 = std::chrono::steady_clock::now();
    CardRegistry reg;
    reg.loadAll();
    auto synth = registerSynthetics(reg);
    CardDB db;
    db.buildFromClasses(reg);
    g_reg = &reg;
    g_db = &db;

    struct Subj { std::string name; uint32_t id; bool rename; };
    std::vector<Subj> subjects;
    {
        std::ifstream in(argv[1]);
        std::string name; uint32_t id; int rn;
        while (in >> name >> id >> rn) subjects.push_back({name, id, rn != 0});
    }
    for (const auto& s : synth) subjects.push_back({s.name, s.id, false});

    std::ofstream out(argv[2]);
    out << "{\n\"subjects\": {\n";
    bool first_subject = true;
    std::vector<std::string> layout_names;
    std::vector<std::pair<std::string, std::vector<std::string>>> layout;
    for (const auto& s : subjects) {
        const auto ts = std::chrono::steady_clock::now();
        g_subject = CardDefId::fromRaw(s.id);
        g_renamed = s.rename;
        auto& entry = const_cast<CardDef&>(db.get(g_subject));
        const std::string saved_name = entry.name;
        if (s.rename) entry.name = kRename;
        const int err0 = g_errors;
        SubjectOut so;
        probeStaticCard(so);
        probePlayWindows(so);
        probeTriggers(so);
        probeStatic(so);
        probeRune(so);
        probeTargeted(so);
        probeInPlayLegal(so);
        entry.name = saved_name;
        if (layout.empty())
            for (const auto& b : so.blocks) layout.push_back({b.name, b.fields});
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
        if (!first_subject) out << ",\n";
        first_subject = false;
        out << jstr(s.name) << ": {\"class_id\": " << s.id << ", \"renamed\": " << (s.rename ? "true" : "false")
            << ", \"card_type\": " << jstr(toString(db.get(g_subject).card_type))
            << ", \"errors\": " << (g_errors - err0) << ", \"seconds\": " << jnum(secs) << ", \"blocks\": {";
        bool fb = true;
        for (const auto& b : so.blocks) {
            if (!fb) out << ", ";
            fb = false;
            out << jstr(b.name) << ": [";
            for (size_t i = 0; i < b.values.size(); ++i) out << (i ? "," : "") << jnum(b.values[i]);
            out << "]";
        }
        out << "}, \"raw_blocks\": {";
        fb = true;
        for (const auto& b : so.raw) {
            if (!fb) out << ", ";
            fb = false;
            out << jstr(b.name) << ": [";
            for (size_t i = 0; i < b.values.size(); ++i) out << (i ? "," : "") << jnum(b.values[i]);
            out << "]";
        }
        out << "}}";
        std::cerr << s.name << " (" << s.id << ") " << secs << "s errors=" << (g_errors - err0) << "\n";
    }
    out << "\n},\n\"layout\": [";
    for (size_t i = 0; i < layout.size(); ++i) {
        out << (i ? ",\n" : "\n") << "{\"block\": " << jstr(layout[i].first) << ", \"fields\": [";
        for (size_t j = 0; j < layout[i].second.size(); ++j)
            out << (j ? "," : "") << jstr(layout[i].second[j]);
        out << "]}";
    }
    out << "\n],\n\"errors\": [";
    for (size_t i = 0; i < g_error_log.size(); ++i) out << (i ? ",\n" : "\n") << jstr(g_error_log[i]);
    const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    out << "\n],\n\"total_seconds\": " << jnum(total) << "\n}\n";
    std::cerr << "total " << total << "s, errors " << g_errors << "\n";
    return 0;
}
#endif  // CARD_PROBE_NO_MAIN
