// ===========================================================================
// Gate 5 (research/card_repr/gate5): card-swap edits + ISMCTS advice.
// Source of truth: research/card_repr/gate5/rbenv_gate5_ext.cpp; spliced into
// selfplay/rbenv/rbenv.cpp by gate5/splice.py (PART A before the gate-4
// applyEdit, the "swap" op inside applyEdit, PART B appended at the end).
//   swap <obj> <cardkey> : the object becomes a fresh instance of the card
//     (riftbound::becomeCard, as the determinizer rewrites a slot), keeping id,
//     owner, controller, zone, location; exhausted / stunned / damage are
//     carried over on the board so only the card identity changes. Refused for
//     facedown, token, attached or gear-carrying objects, and for a board swap
//     that is not unit -> unit. Resume + Cleanup recompute derived state.
//   rbenv_advise(h, i, sims, seed): ISMCTS-<sims> answer (legal index) at env
//     i's pending decision (a fresh IsMctsAgent on the true state, as the
//     in-env opponent is built).
// ===========================================================================
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
