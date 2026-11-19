"""How far apart does each trained network put ctrl_scores and opp_of_ctrl_scores (card embedding,
before the transformer), relative to the typical distance between two training-deck cards?"""
import sys, os, json, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'learner'))
from model import Net, load_vocab
voc = load_vocab()[0]; V = {k: i for i, k in enumerate(voc)}
D = dict(G=72, T=64, F=17, A=96, K=46, V=len(voc))
deck = ['noxus_hopeful', 'synthetic:pit_rookie__score_alone', 'synthetic:sabotage__score_opphand4', 'synthetic:punch_first__ctrl_scores',
        'inferna', 'grim_apothecary', 'kinkou_initiate', 'irresistible_faefolk', 'first_mate', 'thrill_of_the_hunt', 'rampage',
        'pakaa_cub', 'ferrous_forerunner', 'kai_sa_survivor', 'nidalee_cat_form', 'darius_trifarian', 'pyke_dockside_butcher']
out = {}
for run in sys.argv[1:]:
    rep = run.split('cx_')[1].split('_s')[0]
    sd = torch.load(os.path.join(HERE, '..', 'runs', run, 'latest.pt'), map_location='cpu')['model']
    net = Net(D, voc, rep, ask_head='W_ask.weight' in sd, card_norm='card_ln.weight' in sd)
    net.load_state_dict(sd)
    with torch.no_grad():
        E = net.card_emb(torch.arange(len(voc)))
    a, b = E[V['synthetic:punch_first__ctrl_scores']], E[V['synthetic:punch_first__opp_of_ctrl_scores']]
    ix = [V[k] for k in deck]
    M = E[ix]; dd = torch.cdist(M, M); typ = dd[dd > 0].median()
    out[run] = dict(ctrl_vs_opp=round(float((a - b).norm() / typ), 3),
                    ctrl_vs_base_punch=round(float((a - E[V['punch_first']]).norm() / typ), 3))
print(json.dumps(out, indent=1))
