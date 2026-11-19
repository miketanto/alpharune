"""Assemble results.json from the analysis files. Keep rule: flip clearly above
its noise floor and >= 0.30 at the main budget (sims 200); target-stage pairs
on target flips against the seed-half floor of the target prompt."""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
def load(n):
    p = os.path.join(H, n)
    return json.load(open(p)) if os.path.exists(p) else {}
A = {k: load(f'analysis_raw_s{k}.json') for k in (200, 100, 400)}
T = load('analysis_tgt_raw.json')
RT = load('analysis_raw_tgt.json')  # root stage of the --tgt runs (sims 200, 8 seeds)
PAIRS = [
 ('punch_first', 'punch_first__score1', 'effect kind: "+5 Might this turn to a unit" -> "Score 1 point" (no target)',
  'the buff rarely moves points within the search horizon so the teacher rarely plays it; a point now is always worth playing'),
 ('punch_first', 'punch_first__score_nounits', 'condition: "Score 1 point if you control no units"',
  'play only when the board has no friendly unit; hold otherwise; the network must read the board to know'),
 ('punch_first', 'punch_first__score_bfunit', 'condition: "Score 1 point if you control a unit at a battlefield" (the complement-ish of score_nounits)',
  'play only with a unit at a battlefield; opposite states from score_nounits'),
 ('punch_first', 'punch_first__score_behind', 'condition: "Score 1 point if an opponent has more points than you"',
  'play only when behind on score'),
 ('punch_first', 'punch_first__kill', 'effect kind + target side: "+5 Might" -> "Kill a unit" (same targeting)',
  'buff targets a friendly unit, kill an enemy unit; play when an enemy unit blocks a score'),
 ('punch_first', 'punch_first__minus5', 'effect sign: "+5 Might" -> "-5 Might" this turn',
  'buff a friendly unit vs shrink an enemy unit: target side should flip'),
 ('punch_first', 'punch_first__ctrl_scores', 'effect + target: "Choose a unit. Its controller scores 1 point."',
  'only a FRIENDLY target is right'),
 ('punch_first', 'punch_first__opp_of_ctrl_scores', 'effect + target: "Choose a unit. Its controller\'s opponent scores 1 point."',
  'only an ENEMY target is right (mirror of ctrl_scores)'),
 ('pit_rookie', 'pit_rookie__score1', 'trigger kind: "When you play me, buff another friendly unit" -> "When you play me, score 1 point"',
  'the base play is rarely best for the teacher; a play that scores is'),
 ('pit_rookie', 'pit_rookie__score_alone', 'condition: "When you play me, score 1 point if you control no other units"',
  'play it into an empty board only'),
 ('sabotage', 'sabotage__score1', 'effect kind: "look at opponent hand, recycle a non-unit card" -> "Score 1 point"',
  'disruption does not show in points within the horizon; a point does'),
 ('sabotage', 'sabotage__score_opphand4', 'condition: "Score 1 point if the opponent has 4 or more cards in hand"',
  'play only against a full opponent hand'),
]
VERDICT = {
 'punch_first__score1': ('keep', 'play flip 0.85 (0.90 at sims 400) vs noise 0.00; at sims 100 only 0.42 (budget-sensitive).'),
 'punch_first__score_nounits': ('keep', 'flip 0.23 overall, which is capped by the condition (true in 10 of 48 states): plays in 10/10 states where it holds and 0/38 where it does not. Noise 0.00. Stable at sims 100 and 400.'),
 'punch_first__score_bfunit': ('keep', 'flip 0.31: plays in 16/21 states where the condition holds and 0/27 where it does not. Fragile at sims 100 (0.06).'),
 'punch_first__score_behind': ('keep', 'flip 0.29: plays in 13/14 states where it is behind and 0/34 otherwise. Weaker at sims 100 (0.15).'),
 'punch_first__kill': ('drop', 'root play flip 0.00. Target stage: picks an enemy in 0.81 of two-sided states vs 0.56 for the base, but the side flip (0.44) is near the seed-half noise of the base target (0.25 on all, the base target choice is a near-tie).'),
 'punch_first__minus5': ('drop', 'no play flip; target side flip 0.12, below noise. The teacher does not value a temporary Might loss (score-only leaf).'),
 'punch_first__ctrl_scores': ('keep (target pair)', 'best target is FRIENDLY in 32/32 two-sided states (0% enemy) vs 56% enemy for the base, whose target choice is noise. Root play flip only 0.12 (the root q averages over exploring both target sides).'),
 'punch_first__opp_of_ctrl_scores': ('keep (target pair)', 'best target is ENEMY in 32/32 two-sided states. Against ctrl_scores, the best target side flips in 32/32 states: the cleanest "same card shape, opposite use" pair found.'),
 'pit_rookie__score1': ('keep', 'flip 0.35 (0.58 at sims 400, 0.25 at sims 100) vs noise 0.00; budget-sensitive.'),
 'pit_rookie__score_alone': ('keep', 'flip 0.21: plays in 10/11 states with no other friendly unit and 0/37 otherwise. Stable across budgets.'),
 'sabotage__score1': ('keep', 'flip 0.83 vs noise 0.04; 0.62 at sims 100, 0.81 at sims 400.'),
 'sabotage__score_opphand4': ('keep', 'flip 0.58: plays in 30/33 states where the opponent has 4+ cards and 1/15 otherwise. Noise 0.04.'),
}
out = {'setup': {'decks': 'rengar.txt vs rengar.txt (mirror) only', 'states_per_base': 48,
                 'main_budget': 'sims 200, 12 seeds (root); sims 200, 8 seeds (target stage)',
                 'robustness': 'sims 100 x 4 seeds and sims 400 x 4 seeds (root)'}, 'pairs': []}
for base, v, change, why in PAIRS:
    key = 'synthetic:' + v
    r = A[200].get(key) or RT.get(key); t = T.get(key)
    if key not in A[200] and r: e_src = 'root stage of the --tgt run (sims 200, 8 seeds)'
    else: e_src = 'root run (sims 200, 12 seeds)'
    e = {'base': base, 'variant': key, 'change': change, 'intended_reason': why, 'root_source': e_src}
    if r:
        e.update({'n_states': r['n_states'],
                  'a_play_flip_rate': round(r['play_flip_rate'], 3),
                  'a_flip_to_play': round(r['flip_to_play_rate'], 3), 'a_flip_to_noplay': round(r['flip_to_noplay_rate'], 3),
                  'orig_play_rate': round(r['orig_play_rate'], 3), 'variant_play_rate': round(r['swap_play_rate'], 3),
                  'b_dst_flip_rate_given_both_play': r['dst_flip_rate_given_both_play'],
                  'c_noise_play_flip_seed_halves': round(r['noise_play_flip_rate_seed_halves'], 3),
                  'c_noise_dst_flip_seed_halves': r['noise_dst_flip_rate_seed_halves'],
                  'd_variant_no_legal_play_rate': r['swap_no_legal_play_rate'],
                  'variant_play_rate_when_condition_true': r.get('swap_play_rate_cond_true'),
                  'variant_play_rate_when_condition_false': r.get('swap_play_rate_cond_false'),
                  'condition_counts_true_false': [r['counts'].get('cond_true'), r['counts'].get('cond_false')] if r['counts'].get('cond_true') is not None else None})
        for k in (100, 400):
            rr = A[k].get(key)
            if rr: e[f'a_play_flip_rate_sims{k}'] = round(rr['play_flip_rate'], 3)
    if t:
        e.update({'target_stage': {'n_states': t['n_states'], 'n_multi_target_states': t['n_multi_target'],
                  'b_target_flip_rate': round(t['target_flip_rate'], 3), 'b_target_side_flip_rate': round(t['target_side_flip_rate'], 3),
                  'c_noise_target_flip_seed_halves': round(t['noise_target_flip_rate_seed_halves'], 3),
                  'c_noise_side_flip_seed_halves': round(t['noise_side_flip_rate_seed_halves'], 3),
                  'n_states_both_sides_available': t['n_both_sides_available'],
                  'side_flip_rate_both_sides_available': round(t['side_flip_rate_both_sides_available'], 3),
                  'orig_best_target_enemy_rate': round(t['orig_best_enemy_rate_both_sides'], 3),
                  'variant_best_target_enemy_rate': round(t['variant_best_enemy_rate_both_sides'], 3),
                  'best_target_side_counts': {k: c for k, c in t['counts'].items() if 'best_side' in k}}})
    out['pairs'].append(e)
for e in out['pairs']:
    e['keep'], e['verdict'] = VERDICT[e['variant'].split(':')[1]]
out['target_stage_variant_vs_variant'] = T.get('_variant_vs_variant')
out['orig_disagreement_across_processes_sims200'] = A[200].get('_orig_disagreement_across_processes')
out['variant_vs_variant_play_flip_sims200'] = A[200].get('_variant_vs_variant')
json.dump(out, open(os.path.join(H, 'results.json'), 'w'), indent=1)
for e in out['pairs']:
    print(e['variant'], e.get('a_play_flip_rate'), e.get('a_play_flip_rate_sims100'), e.get('a_play_flip_rate_sims400'), e.get('target_stage', {}).get('b_target_side_flip_rate'))
