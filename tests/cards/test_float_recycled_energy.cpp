/// @file test_float_recycled_energy.cpp
/// A ready rune recycled for Power is exhausted for its Energy first, and the
/// Energy floats in the pool (CR 164.2.a, 164.2.b, 166.1, 167;
/// resource::floatEnergyBeforeRecycling). Each of the three places that
/// recycle a rune for Power, and that nothing is gained otherwise.

#include "engine/resource_payment.h"
#include "tests/cards/card_test_fixture.h"

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

}  // namespace
}  // namespace riftbound::test::test_float_recycled_energy
