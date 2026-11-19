"""Play-rate and target-side rates of the probed cards in REAL games: the policy plays both seats of
the deck's mirror (sampling, as in training). usage: real_stats.py <deck> <n_games> <run> [<run> ...]
Per run: for each probed card in the deck,
  held      main-phase decisions (pk 3) with the card in the decider's hand
  offered   ... of which its play is a legal option (affordable / playable)
  played    ... of which the policy chose to play it
  for a target card: target prompts with a friendly and an enemy option, and the share where the
  policy chose an enemy unit; for a conditional card the play rate split by its condition."""
import os, sys, json, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'learner')); sys.path.insert(0, os.path.join(HERE, '..', 'rbenv'))
from model import Net, load_vocab
from selfplay import act_rows
from rbenv import RBEnv
from make_exp_decks import ALL_VARIANTS
from collect_probes import play_opts, no_units
voc = load_vocab()[0]; V = {k: i for i, k in enumerate(voc)}
CARDS = {'rengar__exp_train': [('synthetic:punch_first__ctrl_scores', 'target'), ('synthetic:sabotage__score_opphand4', 'opphand4'),
                               ('synthetic:pit_rookie__score_alone', 'alone')],
         'rengar__exp_heldout': [('synthetic:punch_first__opp_of_ctrl_scores', 'target'), ('synthetic:sabotage__score_opphand4', 'opphand4'),
                                 ('synthetic:pit_rookie__score_alone', 'alone')],
         'rengar__exp_both': [('synthetic:punch_first__ctrl_scores', 'target'), ('synthetic:punch_first__opp_of_ctrl_scores', 'target'),
                              ('synthetic:pit_rookie__score_alone', 'alone')],
         'rengar__exp_heldout_sab1': [('synthetic:sabotage__score1', 'always')]}
COND = {'opphand4': lambda o, e: o['glob'][e, 10] >= 4, 'alone': no_units, 'always': lambda o, e: True}


def run_stats(run, deck, n_games):
    rep = 'id' if 'fix_' in run else run.split('cx_')[1].split('_s')[0]
    env = RBEnv(64, deck, deck, 777000, variant=ALL_VARIANTS)
    sd = torch.load(os.path.join(HERE, '..', 'runs', run, 'latest.pt'), map_location='cpu')['model']
    net = Net(env.dims, voc, rep, ask_head='W_ask.weight' in sd, card_norm='card_ln.weight' in sd)
    net.load_state_dict(sd)
    net.eval()
    S = {k: dict(held=0, offered=0, played=0, offered_c=0, played_c=0, offered_nc=0, played_nc=0, tgt2=0, enemy=0, friendly=0)
         for k, _ in CARDS[deck]}
    games = wins1 = 0
    while games < n_games:
        o = env.observe()
        rows = np.nonzero(o['need'])[0]
        ch = np.zeros(64, np.int64)
        if len(rows):
            a, _, _ = act_rows(net, o, rows, 'cpu')
            ch[rows] = a
        for e, c in zip(rows.tolist(), (a.tolist() if len(rows) else [])):
            for key, kind in CARDS[deck]:
                s = S[key]
                if o['pk'][e] == 3:
                    t = o['tok'][e][o['tok_mask'][e]]
                    if ((t[:, 1] == V[key]) & (t[:, 4] == 2) & (t[:, 2] == 1)).any():
                        s['held'] += 1
                        pl = play_opts(o, e, key)
                        if pl.any():
                            s['offered'] += 1; s['played'] += bool(pl[c])
                            if kind != 'target':
                                cnd = bool(COND[kind](o, e))
                                s['offered_c' if cnd else 'offered_nc'] += 1
                                s['played_c' if cnd else 'played_nc'] += bool(pl[c])
                if kind == 'target' and o['pk'][e] == 24 and o['psrc'][e] >= 0 and o['tok'][e, o['psrc'][e], 1] == V[key]:
                    ref = o['act'][e, :, 8]; m = o['act_mask'][e]
                    sides = {int(o['tok'][e, ref[i], 2]) for i in np.nonzero(m)[0] if ref[i] >= 0}
                    if {1, 2} <= sides:
                        s['tgt2'] += 1
                        side = o['tok'][e, ref[c], 2]
                        s['enemy'] += side == 2; s['friendly'] += side == 1
        env.step(ch)
        for r in env.results():
            games += 1
    env.close()
    return {k.replace('synthetic:', ''): v for k, v in S.items()}


if __name__ == '__main__':
    torch.set_num_threads(2)
    deck, n = sys.argv[1], int(sys.argv[2])
    out = {r: run_stats(r, deck, n) for r in sys.argv[3:]}
    print(json.dumps(out, default=int))
