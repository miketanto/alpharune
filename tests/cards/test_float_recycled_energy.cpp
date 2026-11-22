/// @file test_float_recycled_energy.cpp
/// A ready rune recycled for Power is exhausted for its Energy first, and the
/// Energy floats in the pool (CR 164.2.a, 164.2.b, 166.1, 167;
/// resource::floatEnergyBeforeRecycling). Each of the three places that
/// recycle a rune for Power, and that nothing is gained otherwise.

#include "engine/resource_payment.h"
#include "engine/knowledge_tracker.h"
#include "openspiel/action_vocab.h"
#include "rules/deck_validator.h"
#include "tests/cards/card_test_fixture.h"

#include <cstdlib>
#include <random>
#include <set>

namespace riftbound::test::test_float_recycled_energy {  // file-unique: test sources are unity-built
using namespace riftbound;
using namespace riftbound::test;

namespace {

constexpr CardDefId kTCalmPowerUnit = 1981;   // 0 Energy + 1 Calm Power
constexpr CardDefId kTOneOneUnit = 1982;      // 1 Energy + 1 Calm Power
constexpr CardDefId kTOneEnergyUnit = 1983;   // 1 Energy

class TUnit : public UnitCard {
public:
    TUnit(CardDefId id, int energy, int power) {
        def_.id = id;
        def_.name = "Float Test Unit";
        def_.card_type = CardType::Unit;
        def_.domains = {Domain::Calm};
        def_.energy_cost = energy;
        def_.power_cost = power;
        def_.might = 1;
    }
    const CardDef& def() const override { return def_; }
private:
    CardDef def_;
};

class FloatRecycledEnergy : public CardTestFixture {
protected:
    void SetUp() override {
        CardTestFixture::SetUp();
        card_registry.registerCard(kTCalmPowerUnit, std::make_unique<TUnit>(kTCalmPowerUnit, 0, 1));
        card_registry.registerCard(kTOneOneUnit, std::make_unique<TUnit>(kTOneOneUnit, 1, 1));
        card_registry.registerCard(kTOneEnergyUnit, std::make_unique<TUnit>(kTOneEnergyUnit, 1, 0));
        card_db.buildFromClasses(card_registry);
    }
    GameObject& o(GameObjectId id) { return state.getObject(id); }
    bool onBoard(GameObjectId id) { return o(id).location.has_value(); }
    int energy(PlayerId p = P1) { return state.player(p).rune_pool.energy; }
    int power(PlayerId p = P1) {
        const auto& r = state.player(p).rune_pool;
        int n = r.universal_power;
        for (int x : r.power) n += x;
        return n;
    }
    void play(GameObjectId card) {
        playThroughEngine(card, nullptr, std::nullopt, nullptr, /*fund=*/false);
    }
};

// ─── resource::payIn (Hide, Equip, activated abilities, "pay [A]") ──────────

TEST_F(FloatRecycledEnergy, APowerOnlyPaymentFloatsTheReadyRunesEnergy) {
    const auto rune = addRune(P1, Domain::Calm);
    ASSERT_TRUE(resource::pay(state, events, P1, 0, 1, Domain::Calm));
    EXPECT_EQ(o(rune).zone, ZoneType::RuneDeck);
    EXPECT_EQ(energy(), 1) << "CR 164.2: exhausted for [1], then recycled";
    EXPECT_EQ(power(), 0) << "the Power was spent on the cost";
    EXPECT_EQ(energy(P2), 0);
}

TEST_F(FloatRecycledEnergy, AnExhaustedRuneGivesNoEnergy) {
    addRune(P1, Domain::Calm, /*exhausted=*/true);
    ASSERT_TRUE(resource::pay(state, events, P1, 0, 1, Domain::Calm));
    EXPECT_EQ(energy(), 0);
}

TEST_F(FloatRecycledEnergy, AnExhaustedRuneIsStillRecycledBeforeAReadyOne) {
    const auto ready = addRune(P1, Domain::Calm);
    const auto spent = addRune(P1, Domain::Calm, /*exhausted=*/true);
    ASSERT_TRUE(resource::pay(state, events, P1, 0, 1, Domain::Calm));
    EXPECT_EQ(o(spent).zone, ZoneType::RuneDeck);
    EXPECT_FALSE(o(ready).is_exhausted);
    EXPECT_EQ(energy(), 0);
}

TEST_F(FloatRecycledEnergy, OneRunePayingBothIsNotCountedTwice) {
    addRune(P1, Domain::Calm);
    ASSERT_TRUE(resource::pay(state, events, P1, 1, 1, Domain::Calm));
    EXPECT_EQ(energy(), 0) << "its one Energy paid the [1]";
}

TEST_F(FloatRecycledEnergy, EachReadyRuneOfAPaymentFloatsItsOwn) {
    for (int i = 0; i < 3; ++i) addRune(P1, Domain::Calm);
    ASSERT_TRUE(resource::pay(state, events, P1, 0, 2, Domain::Count));
    EXPECT_EQ(energy(), 2);
    EXPECT_EQ(readyRuneCount(P1), 1);
}

TEST_F(FloatRecycledEnergy, TheFloatingEnergyPaysALaterCostBeforeAnyRune) {
    addRune(P1, Domain::Calm);
    const auto kept = addRune(P1, Domain::Calm);
    ASSERT_TRUE(resource::pay(state, events, P1, 0, 1, Domain::Calm));
    ASSERT_EQ(energy(), 1);
    ASSERT_TRUE(resource::pay(state, events, P1, 1, 0, Domain::Count));
    EXPECT_EQ(energy(), 0);
    EXPECT_FALSE(o(kept).is_exhausted) << "paid from the pool";
}

TEST_F(FloatRecycledEnergy, APaymentThatCannotBeMadeFloatsNothing) {
    const auto rune = addRune(P1, Domain::Calm);
    EXPECT_FALSE(resource::pay(state, events, P1, 0, 2, Domain::Calm));
    EXPECT_EQ(energy(), 0);
    EXPECT_FALSE(o(rune).is_exhausted);
    EXPECT_EQ(o(rune).zone, ZoneType::Base);
}

// ─── GameEngine::advanceCostPayment (a card's own cost) ─────────────────────

TEST_F(FloatRecycledEnergy, PlayingACardForPowerAloneFloatsTheEnergy) {
    const auto rune = addRune(P1, Domain::Calm);
    const auto card = addToHand(P1, kTCalmPowerUnit);
    play(card);
    ASSERT_TRUE(onBoard(card));
    EXPECT_EQ(o(rune).zone, ZoneType::RuneDeck);
    EXPECT_EQ(energy(), 1);
}

TEST_F(FloatRecycledEnergy, ACardPlayedWithOneRuneForBothLeavesNothingFloating) {
    addRune(P1, Domain::Calm);
    const auto card = addToHand(P1, kTOneOneUnit);
    play(card);
    ASSERT_TRUE(onBoard(card));
    EXPECT_EQ(energy(), 0);
}

TEST_F(FloatRecycledEnergy, TheNextCardIsPaidWithTheFloatingEnergy) {
    addRune(P1, Domain::Calm);                           // the only rune
    const auto first = addToHand(P1, kTCalmPowerUnit);
    const auto second = addToHand(P1, kTOneEnergyUnit);
    play(first);
    ASSERT_TRUE(onBoard(first));
    play(second);
    EXPECT_TRUE(onBoard(second)) << "there is no rune left: the Energy that floated paid for it";
    EXPECT_EQ(energy(), 0);
}

// Offered, too: the action generator counts the floating Energy.
TEST_F(FloatRecycledEnergy, TheNextCardIsOffered) {
    addRune(P1, Domain::Calm);
    const auto first = addToHand(P1, kTCalmPowerUnit);
    const auto second = addToHand(P1, kTOneEnergyUnit);
    play(first);
    bool offered = false;
    withEngine([&](GameEngine& e) {
        for (const auto& i : e.generateLegalActions()) offered |= i.card == second;
    });
    EXPECT_TRUE(offered);
}

// ─── plainPay (an effect play on an executor with no engine) ────────────────

TEST_F(FloatRecycledEnergy, AnEffectPlayWithoutAnEngineFloatsTheEnergy) {
    const auto rune = addRune(P1, Domain::Calm);
    const auto card = addToHand(P1, kTCalmPowerUnit);
    EffectExecutor exec(state, events, card_db, &card_registry);
    ASSERT_TRUE(exec.playPayingCost(P1, card, LocationId{BaseLocation{P1}}));
    EXPECT_EQ(o(rune).zone, ZoneType::RuneDeck);
    EXPECT_EQ(energy(), 1);
}

// ─── setManualRunes: the runes' own abilities as actions (CR 164.2) ─────────

std::vector<Intent> runeActions(const GameState& s, const std::vector<Intent>& legal) {
    std::vector<Intent> out;
    for (const auto& i : legal)
        if (i.ability_source != kInvalidId && s.getObject(i.ability_source).isRune()) out.push_back(i);
    return out;
}

class ManualRunes : public FloatRecycledEnergy {
protected:
    GameObjectId cardRune(Domain d, CardDefId def, bool exhausted = false) {
        const auto id = addRune(P1, d, exhausted);
        o(id).card_def_id = def;
        return id;
    }
    static constexpr CardDefId kCalmRune = 42, kOrderRune = 214;
};

TEST_F(ManualRunes, OffByDefault) {
    cardRune(Domain::Calm, kCalmRune);
    withEngine([&](GameEngine& e) {
        EXPECT_TRUE(runeActions(e.state(), e.generateLegalActions()).empty());
        e.setManualRunes(P2, true);
        EXPECT_TRUE(runeActions(e.state(), e.generateLegalActions()).empty()) << "the other player's setting";
    });
}

TEST_F(ManualRunes, OneActionPerKindOfRuneAndAbility) {
    cardRune(Domain::Calm, kCalmRune);
    cardRune(Domain::Calm, kCalmRune);
    cardRune(Domain::Order, kOrderRune);
    cardRune(Domain::Order, kOrderRune, /*exhausted=*/true);
    withEngine([&](GameEngine& e) {
        e.setManualRunes(P1, true);
        int exhaust = 0, recycle = 0;
        for (const auto& i : runeActions(e.state(), e.generateLegalActions()))
            (i.ability_index == resource::kRuneExhaustAbility ? exhaust : recycle)++;
        EXPECT_EQ(exhaust, 2) << "a ready Calm, a ready Order";
        EXPECT_EQ(recycle, 3) << "Calm, Order ready, Order exhausted";
    });
}

TEST_F(ManualRunes, ExhaustingOneFloatsEnergyThatPaysForACard) {
    const auto rune = cardRune(Domain::Calm, kCalmRune);
    const auto card = addToHand(P1, kTOneEnergyUnit);
    withEngine([&](GameEngine& e) {
        e.setManualRunes(P1, true);
        auto acts = runeActions(e.state(), e.generateLegalActions());
        ASSERT_EQ(acts.size(), 2u);
        e.testHook_executeIntent(acts[0]);
        settleLikeTheEngine(e);
        EXPECT_TRUE(e.state().chain.items.empty());
        EXPECT_TRUE(e.state().getObject(rune).is_exhausted);
        EXPECT_EQ(e.state().player(P1).rune_pool.energy, 1);
        Intent play;
        play.type = IntentType::PlayCard; play.player = P1; play.card = card;
        play.play_location = LocationId{BaseLocation{P1}};
        e.testHook_executeIntent(play);
        settleLikeTheEngine(e);
    });
    EXPECT_TRUE(onBoard(card));
    EXPECT_EQ(energy(), 0);
}

TEST_F(ManualRunes, RecyclingOneFloatsItsDomainsPowerAndItIsSpentOnlyWhenPaid) {
    const auto rune = cardRune(Domain::Calm, kCalmRune);
    withEngine([&](GameEngine& e) {
        e.setManualRunes(P1, true);
        for (const auto& i : runeActions(e.state(), e.generateLegalActions()))
            if (i.ability_index == resource::kRuneRecycleAbility) { e.testHook_executeIntent(i); break; }
        settleLikeTheEngine(e);
    });
    EXPECT_EQ(o(rune).zone, ZoneType::RuneDeck);
    EXPECT_EQ(state.player(P1).rune_pool.power[static_cast<int>(Domain::Calm)], 1);
    EXPECT_EQ(energy(), 1) << "it was ready: exhausted for [1] first";
    EXPECT_EQ(state.player(P1).power_spent_this_turn, 0);
    ASSERT_TRUE(resource::pay(state, events, P1, 0, 1, Domain::Calm));
    EXPECT_EQ(state.player(P1).power_spent_this_turn, 1);
    EXPECT_EQ(power(), 0);
}

// Whole games with both players offered their runes: nothing throws, every
// legal intent has an action id of its own, the knowledge tracker agrees, and
// the abilities are used in the Main Phase, at Focus and at chain priority.
TEST_F(ManualRunes, RandomGamesOfEveryKind) {
    const char* root = std::getenv("RIFTBOUND_ROOT");
    if (!root) GTEST_SKIP() << "needs RIFTBOUND_ROOT";
    CardRegistry reg;
    registerAllCards(reg);
    CardDB db;
    db.buildFromClasses(reg);
    std::set<IntentType> used;
    int finished = 0;
    for (int g = 0; g < 24; ++g) {
        const char* names[] = {"azir", "rengar", "diana"};
        const auto d1 = DeckValidator::loadFromDeckList(std::string(root) + "/decks/" + names[g % 3] + ".txt", db);
        const auto d2 = DeckValidator::loadFromDeckList(std::string(root) + "/decks/" + names[(g / 3) % 3] + ".txt", db);
        EventBus bus;
        GameEngine e(db, bus, reg);
        e.setManualRunes(P1, true);
        e.setManualRunes(P2, g % 2 == 0);
        std::mt19937_64 rng(9000 + g);
        e.beginGame(d1, d2, 5000 + g);
        int n = 0;
        for (; n < 8000 && !e.isStepDone(); ++n) {
            const auto& step = e.currentStep();
            const auto ids = openspiel::actionIds(step.legal, e.state());
            ASSERT_EQ(std::set<int>(ids.begin(), ids.end()).size(), step.legal.size()) << "game " << g << " decision " << n;
            for (const auto& i : runeActions(e.state(), step.legal))
                EXPECT_TRUE(i.player == P1 || g % 2 == 0);
            ASSERT_TRUE(knowledge::check(e.state()).empty());
            const size_t k = std::uniform_int_distribution<size_t>(0, step.legal.size() - 1)(rng);
            const Intent& c = step.legal[k];
            if (c.ability_source != kInvalidId && e.state().getObject(c.ability_source).isRune()) used.insert(c.type);
            e.applyChoice(static_cast<int>(k));
        }
        finished += e.isStepDone();
    }
    EXPECT_EQ(finished, 24);
    EXPECT_TRUE(used.count(IntentType::ActivateAbility)) << "Main Phase";
    EXPECT_TRUE(used.count(IntentType::ActivateActionAbility)) << "Focus";
    EXPECT_TRUE(used.count(IntentType::ActivateReactionAbility)) << "chain priority";
}

}  // namespace
}  // namespace riftbound::test::test_float_recycled_energy
