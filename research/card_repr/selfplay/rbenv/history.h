// history.h -- public action history for rbenv / rbsearch observations (PLAYTEST_PLAN.md step 1, 2026-11-21).
// #included by rbenv.cpp after decision_export.cpp (needs cardKey, vocabOf, locCode, knowledge::*).
//
// Every EXTERNAL decision applied to a game (>= 2 options, not a rune payment: the decisions rbenv / the
// search hand to an agent) is recorded as one HistEv, for both seats. An observation carries the newest
// kHistLen events as a separate tensor hist[kHistLen][kHistF], encoded FOR the observer:
//   0 valid (1)            1 actor (1 self, 2 opponent)      2 intent type        3 prompt kind
//   4 card key of the main object (vocab index; the 'hidden' key when the observer did not see it; 0 none)
//   5 its owner (1 self, 2 opponent, 0 none)                  6 card key of the second object (first
//   target / chosen object / moving unit)   7 its owner       8 destination (-1 none, 0 base, 1+i battlefield i)
//   9 chosen value (public prompt kinds only; -2 a decline; -1 n/a)
//   10 number of objects the action names (targets + chosen + moving units + cards mulliganed)
//   11 the actor's ready runes after the action             12 turns ago (0 = this turn, capped 15)
//   13 actor was the turn player                            14 a showdown or combat was under way
//   15 age (0 = the newest event)
//
// Leak rule: an object's identity is written only if the observer could see it just before or just after
// the action (strict visibility, as the search harness's encoder: an opponent card known by identity but
// not by place is "hidden"). A card seen by BOTH players is stored as its key (public, the same in every
// sampled world). A card seen by one player only is stored by object id and its key is read from the
// state being encoded, so in a sampled world it is the world's card (a world redraws the opponent's
// private cards; the history then agrees with the world, never with the truth). Mulligans: the count only.
#pragma once
#include <deque>

namespace rbe {

constexpr int kHistLen = 48, kHistF = 16;

inline bool histVisible(const GameState& s, GameObjectId id, PlayerId me) {
    if (!id.valid() || !s.objectExists(id)) return false;
    if (!knowledge::identityVisible(s, id, me)) return false;
    const GameObject& o = s.getObject(id);
    if (o.owner == me) return true;
    const ZoneType z = o.zone;
    if (z != ZoneType::Hand && z != ZoneType::FacedownZone && z != ZoneType::MainDeck) return true;
    const CardKnowledge k = knowledge::GameKnowledgeView(s, me).about(id);
    if (z == ZoneType::MainDeck) return k.deck_position != DeckPosition::Unknown && k.deck_rank_span == 1;
    return k.zone_confirmed;
}

struct HRef {
    bool present = false;
    int64_t obj = -1;
    std::string pub;          // key, when both players saw the card
    bool vis[3] = {false, false, false};   // by PlayerId raw (1, 2)
    int owner = 0;            // PlayerId raw
};

struct HistEv {
    int actor = 0, itype = 0, pk = 0, turn = 0, tp = 0, sd = 0, dst = -1, value = -1, nref = 0, runes = 0;
    HRef a, b;
};

inline int pidx(PlayerId p) { return p == PlayerId::Player1 ? 1 : (p == PlayerId::Player2 ? 2 : 0); }

inline bool publicValueKind(PromptKind k) {
    switch (k) {
        case PromptKind::May: case PromptKind::Mode: case PromptKind::XAmount: case PromptKind::Location:
        case PromptKind::Accelerate: case PromptKind::Repeat: case PromptKind::AdditionalCost:
        case PromptKind::CostSplit: case PromptKind::SplitAmount: case PromptKind::NextStaged:
            return true;
        default:
            return false;
    }
}

// Pending event: built before the choice is applied (pre-state visibility), finished after.
struct HistPending {
    HistEv ev;
    GameObjectId ida = kInvalidId, idb = kInvalidId;
    bool active = false;
};

inline void histRefPre(const GameState& s, GameObjectId id, HRef& r) {
    if (!id.valid()) return;
    r.present = true;
    r.obj = static_cast<int64_t>(id.raw());
    if (!s.objectExists(id)) return;
    const GameObject& o = s.getObject(id);
    r.owner = pidx(o.owner);
    for (PlayerId p : {PlayerId::Player1, PlayerId::Player2}) r.vis[pidx(p)] = histVisible(s, id, p);
    if (r.vis[1] && r.vis[2]) r.pub = cardKey(o.card_def_id, &o);
}

inline void histRefPost(const GameState& s, GameObjectId id, HRef& r) {
    if (!r.present || !id.valid() || !s.objectExists(id)) return;
    const GameObject& o = s.getObject(id);
    if (!r.owner) r.owner = pidx(o.owner);
    for (PlayerId p : {PlayerId::Player1, PlayerId::Player2}) r.vis[pidx(p)] = r.vis[pidx(p)] || histVisible(s, id, p);
    if (r.pub.empty() && r.vis[1] && r.vis[2]) r.pub = cardKey(o.card_def_id, &o);
}

// Recorded decisions: >= 2 options and not a rune payment (what rbenv / the search hand to an agent).
// Judged here, so every path that applies choices (live game, replays, simulations) records the same.
inline bool histWorthy(const StepResult& sr) {
    const bool rune = sr.prompt.kind == PromptKind::RuneForEnergy || sr.prompt.kind == PromptKind::RuneForPower;
    return sr.legal.size() >= 2 && !rune;
}

inline HistPending histBegin(const GameState& s, const StepResult& sr, PlayerId dec, int k) {
    HistPending h;
    if (!histWorthy(sr) || k < 0 || k >= static_cast<int>(sr.legal.size())) return h;
    const Intent& in = sr.legal[k];
    HistEv& e = h.ev;
    e.actor = pidx(dec);
    e.itype = static_cast<int>(in.type);
    e.pk = static_cast<int>(sr.prompt.kind);
    e.turn = s.turn.turn_number;
    e.tp = s.turn.turn_player == dec ? 1 : 0;
    e.sd = s.turn.ns_state != NeutralShowdownState::Neutral ? 1 : 0;
    e.dst = in.play_location.has_value() ? locCode(in.play_location)
                                         : (in.move_destination.has_value() ? locCode(in.move_destination) : -1);
    if (e.dst < 0 && in.chosen_battlefield.valid()) e.dst = 1 + static_cast<int>(in.chosen_battlefield.raw());
    if (in.declines) e.value = -2;
    else if (in.chosen_value.has_value() && publicValueKind(sr.prompt.kind)) e.value = std::max(0, std::min(15, *in.chosen_value));
    e.nref = static_cast<int>(in.targets.size() + in.chosen_objects.size() + in.units_to_move.size() +
                              in.cards_to_mulligan.size());
    h.ida = in.card.valid() ? in.card : in.ability_source;
    if (!in.targets.empty()) h.idb = in.targets[0];
    else if (!in.chosen_objects.empty()) h.idb = in.chosen_objects[0];
    else if (!in.units_to_move.empty()) h.idb = in.units_to_move[0];
    if (h.idb == h.ida) h.idb = kInvalidId;
    histRefPre(s, h.ida, e.a);
    histRefPre(s, h.idb, e.b);
    h.active = true;
    return h;
}

inline void histEnd(const GameState& s, HistPending& h, std::vector<HistEv>& out) {
    if (!h.active) return;
    histRefPost(s, h.ida, h.ev.a);
    histRefPost(s, h.idb, h.ev.b);
    int ready = 0;
    const PlayerId actor = h.ev.actor == 1 ? PlayerId::Player1 : PlayerId::Player2;
    for (auto id : s.runesInBase(actor)) if (s.objectExists(id) && !s.getObject(id).is_exhausted) ++ready;
    h.ev.runes = ready;
    out.push_back(std::move(h.ev));
    h.active = false;
}

inline int histKey(const GameState& s, const HRef& r, PlayerId me, Counters& c) {
    if (!r.present) return 0;
    const int m = pidx(me);
    const auto id = GameObjectId::fromRaw(static_cast<uint32_t>(r.obj));
    if (!r.vis[m]) {
        // Not seen around the action, but maybe since (a played card is still in its owner's hand while its
        // cost is being paid, and public once it is on the chain): judged on the state being encoded, by
        // the same strict rule, so a sampled world (which keeps what the observer can see) agrees.
        if (s.objectExists(id) && histVisible(s, id, me)) {
            const GameObject& o = s.getObject(id);
            return vocabOf(cardKey(o.card_def_id, &o), c);
        }
        return vocabOf("hidden", c);
    }
    if (!r.pub.empty()) return vocabOf(r.pub, c);
    if (!s.objectExists(id)) return vocabOf("hidden", c);
    const GameObject& o = s.getObject(id);
    return vocabOf(cardKey(o.card_def_id, &o), c);
}

// Writes hist[kHistLen * kHistF] for observer `me` (newest event first).
inline void encodeHist(const GameState& s, const std::vector<HistEv>& hv, PlayerId me, Counters& c, int32_t* out) {
    std::fill(out, out + kHistLen * kHistF, 0);
    const int n = static_cast<int>(hv.size());
    const int m = pidx(me);
    auto rel = [&](int p) { return p == 0 ? 0 : (p == m ? 1 : 2); };
    for (int j = 0; j < kHistLen && j < n; ++j) {
        const HistEv& e = hv[n - 1 - j];
        int32_t* r = out + j * kHistF;
        r[0] = 1;
        r[1] = rel(e.actor);
        r[2] = e.itype;
        r[3] = e.pk;
        r[4] = histKey(s, e.a, me, c);
        r[5] = e.a.present ? rel(e.a.owner) : 0;
        r[6] = histKey(s, e.b, me, c);
        r[7] = e.b.present ? rel(e.b.owner) : 0;
        r[8] = e.dst;
        r[9] = e.value;
        r[10] = std::min(e.nref, 15);
        r[11] = std::min(e.runes, 15);
        r[12] = std::max(0, std::min(15, s.turn.turn_number - e.turn));
        r[13] = e.tp;
        r[14] = e.sd;
        r[15] = j;
    }
}

}  // namespace rbe
