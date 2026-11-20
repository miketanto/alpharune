"""Policy/value network for no-search self-play (NETWORK.md design, adapted from
smoke/train/train.py to the rbenv tensor contract, SELFPLAY.md).

Inputs (torch tensors, one row per pending decision):
  glob (B,G) int, tok (B,T,F) int, tok_mask (B,T) bool, act (B,A,K) int, act_mask (B,A) bool, pk (B,) int

Field layout: rbenv/README.txt, schema v2 (F=24, K=49; kinds 1 card, 2 battlefield, 3 chain item,
4 pile). tok[...,2:] situational fields (all embedded categorically per kind + a per-kind linear map
of all). act: 0 t, 1 src, 2 abs, 3 ai, 4..17 tg/ch/mv/mu tokens,
18..29 dm (token, damage) pairs, 30 dst, 31 has_v, 32 v, 33 ci token, 34 nc vocab index, 35 bf, 36 ps,
37 alt, 38 plc, 39 dec, 40 first, 41..45 counts, 46 dst controller (-1 n/a, 0 none, 1 self, 2 opp),
47 would start a combat as attacker, 48 would contest (schema v2). 49..55 optional-cost consequences (v3).
v4: TriggerOrder options carry 1 src = the ordered item's source and 33 ci = that chain item; MoveJoin options
carry 30 dst + 46..48, 56..59 the units already moving (leading unit first), 60 their count, 39 dec = "no more". Optional b['psrc'] (B,) marks the asking token.
"""
import json, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 64
N_PK = 40
N_KIND = 4                                          # token kinds 1..4 (card, bf, chain item, pile)
SCHEMA_VERSION = 2
ACT_SRC = [1, 2]                                    # src, abs token refs (summed)
ACT_REFS = list(range(4, 18)) + list(range(18, 30, 2)) + [33]  # tg ch mv mu, dm tokens, ci (mean-pooled)
ACT_DST = 30

HERE = os.path.dirname(os.path.abspath(__file__))
CR = os.path.normpath(os.path.join(HERE, '..', '..'))   # research/card_repr
REPRS_PT = os.path.join(CR, 'smoke', 'train', 'packed', 'reprs.pt')
VOCAB_JSON = os.path.join(CR, 'selfplay', 'rbenv', 'vocab.json')
CARDS_JSON = os.path.join(CR, 'cards.json')
REPR_SRC = {'B2': os.path.join(CR, 'B2_code', 'out', 'vectors.json'),
            'B': os.path.join(CR, 'B_graph', 'out', 'vectors.json')}


def load_vocab():
    """Card vocab: rbenv/vocab.json if present, else ['<pad>', 'hidden'] + cards.json keys."""
    if os.path.exists(VOCAB_JSON):
        v = json.load(open(VOCAB_JSON))
        if isinstance(v, dict):
            v = v.get('vocab', v.get('keys', v))
            if isinstance(v, dict):  # key -> index
                inv = sorted(v.items(), key=lambda kv: kv[1])
                v = [k for k, _ in inv]
        return list(v), 'rbenv/vocab.json'
    d = json.load(open(CARDS_JSON))
    keys = [c['key'] for c in d['cards']] if isinstance(d, dict) and 'cards' in d else list(d)
    return ['<pad>', 'hidden'] + keys, 'cards.json (fallback)'


def special_index(vocab):
    """0 = ordinary card, else a learned row shared by every condition."""
    out = []
    for k in vocab:
        if k in ('<pad>', 'pad'):
            out.append(1)
        elif k == 'hidden':
            out.append(2)
        elif k.startswith('token:'):
            out.append(3)
        elif k == 'keyword_ability':
            out.append(4)
        elif k.startswith('other') or k == 'none':
            out.append(5)
        else:
            out.append(0)
    return torch.tensor(out, dtype=torch.long)


def card_table(repr_name, vocab, seed=777):
    """Returns (table [V, d_in], trainable, missing keys). B2/B from their source vectors.json."""
    V = len(vocab)
    if repr_name == 'id':
        return None, True, 0
    if repr_name == 'random':
        g = torch.Generator().manual_seed(seed)
        return torch.randn(V, 256, generator=g), False, 0
    # B2 / B from the SOURCE vectors.json (not the gitignored packed file), with smoke/train/reprs.py's
    # pipeline: center on the 51 base cards, scale mean base norm to 1, fixed Gaussian projection
    # (seed 1234) to 256 dims. Covers every vocab key the source has (incl. the score variants).
    src_path = REPR_SRC[repr_name]
    raw = json.load(open(src_path))['vectors']
    cj = json.load(open(CARDS_JSON, encoding='utf-8'))
    base_keys = [c['key'] for c in cj['cards']]
    Mb = torch.tensor([raw[k] for k in base_keys], dtype=torch.float64)
    mu = Mb.mean(0)
    scale = (Mb - mu).norm(dim=1).mean()
    gp = torch.Generator().manual_seed(1234)
    P = torch.randn(Mb.shape[1], 256, generator=gp, dtype=torch.float64)
    tab = torch.zeros(V, 256)
    missing = []
    for i, k in enumerate(vocab):
        if k in raw:
            tab[i] = (((torch.tensor(raw[k], dtype=torch.float64) - mu) / scale) @ P).float()
        elif not (k in ('<pad>', 'hidden', 'keyword_ability', 'none') or k.startswith(('token:', 'other'))):
            missing.append(k)
    return tab, False, missing


def onehot_clip(x, lo, hi):
    return F.one_hot((x.clamp(lo, hi) - lo), hi - lo + 1).float()


def slog(x):
    x = x.float()
    return torch.sign(x) * torch.log1p(x.abs())


class Net(nn.Module):
    def __init__(self, dims, vocab, repr_name='id', ask_head=False, card_norm=True, ask_opt=False, opt_cost=None,
                 opt_refs=None):
        super().__init__()
        # ask_opt (gate 3, 2026-11-19; needs ask_head): the ask head's bilinear term also multiplies the
        # option's OWN features (intent kind, value v, decline flag, destination, move-consequence fields,
        # named card). Without it an option that references no token (yes/no of May / Accelerate /
        # AdditionalCost, Location's destinations, TriggerOrder's values) gets an ask term of exactly 0, and
        # the asking card reaches those choices only through the global token (gate3/REPORT.txt: 100-400x
        # weaker than at Target). Checkpoints without W_ask_opt.* are ask_opt=False nets.
        self.ask_opt = bool(ask_opt and ask_head)
        # card_norm (gate 2, 2026-11-19): the card embedding gets its own LayerNorm and a learned scalar
        # gain (init sqrt(F-2) ~ 4.7, so its norm ~37 matches the ~39 of the summed state embeddings)
        # before it joins the token; without it the card was ~9 of the token's ~40 at init and a card
        # difference was a few % of what the transformer sees. card_norm=False = the pre-gate-2 net
        # (old checkpoints have no card_ln.* keys; verify_eval / probe_eval detect that).
        self.card_norm = card_norm
        # ask_head: a direct path from the ASKING object (psrc token: the card being played / the item's
        # source) to every option's score, as a bilinear term with the option's referenced tokens. Without
        # it the asking card reaches the pointer head only through attention (CARD_EXPERIMENT.md, Checks).
        self.ask_head = ask_head
        self.G, self.T, self.F, self.A, self.K = (int(dims[k]) for k in ('G', 'T', 'F', 'A', 'K'))
        V = len(vocab)
        self.repr_name = repr_name
        tab, trainable, missing = card_table(repr_name, vocab)
        self.missing_keys = missing
        if repr_name == 'id':
            self.E_card = nn.Embedding(V, D)
        else:
            self.register_buffer('R', tab)  # frozen
            self.W_card = nn.Linear(tab.shape[1], D, bias=False)
        self.register_buffer('special_idx', special_index(vocab))
        self.E_special = nn.Embedding(6, D)
        nf = self.F - 2
        if card_norm:
            self.card_ln = nn.LayerNorm(D)
            self.card_scale = nn.Parameter(torch.tensor(math.sqrt(nf)))
        self.ncat = nf
        self.kstride = self.ncat * 12
        self.E_state = nn.Embedding(N_KIND * self.kstride, D)
        self.register_buffer('field_off', torch.arange(self.ncat) * 12)
        self.W_num = nn.Linear(nf, N_KIND * D)
        self.E_kind = nn.Embedding(N_KIND + 2, D)   # 0 pad, 1..4 token kinds, 5 the global token
        self.E_psrc = nn.Parameter(torch.zeros(D))
        self.W_glob = nn.Linear(self.G + 11 + N_PK, D)
        layer = nn.TransformerEncoderLayer(D, 4, 2 * D, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(D)
        # opt_cost (gate 3, schema v3, K >= 56): act[49..55] = optional-cost consequences of an Accelerate /
        # AdditionalCost answer (extra Energy, extra Power, runes recycled, their Domain, own runes ready / total
        # after the payment, unit enters ready), encoded into the option's own features. Default: on when the
        # env has those columns; schema-v2 checkpoints (no W_oc.*) load with opt_cost=False.
        self.opt_cost = (self.K >= 56) if opt_cost is None else bool(opt_cost and self.K >= 56)
        if self.opt_cost:
            self.W_oc = nn.Linear(16, D)
        self.E_akind = nn.Embedding(24, D)
        self.E_dst = nn.Embedding(8, D)
        self.W_v = nn.Linear(17 + 6 + 6 + 5 + 5 + 2 + 4 + 2, D)
        self.val = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 1))
        self.pol = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 1))
        if ask_head:
            self.W_ask = nn.Linear(D, D)
            self.W_ask_src = nn.Linear(D, D, bias=False)
            if self.ask_opt:
                self.W_ask_opt = nn.Linear(D, D, bias=False)
                # and the same bilinear form with the GLOBAL token (resources, pk, turn): without it hg is added
                # equally to every option and a resource fact can only reach a yes/no through the pol MLP's
                # nonlinearity (gate 3: Accelerate yes/no from pool Energy stayed at 0.62 after 300 steps).
                self.W_ctx = nn.Linear(D, D)
                self.W_ctx_opt = nn.Linear(D, D, bias=False)
        # opt_refs (gate 3 completion, schema v4, K >= 61): the tokens an option is ABOUT go into its own
        # features h_opt, so they reach the score directly and through W_ask_opt / W_ctx_opt like W_oc's fields:
        #   W_ci: the chain item an option orders (act 33; TriggerOrder, ChainItem);
        #   W_mg: the units already in the move (act 56..59, mean; 60 their count; MoveJoin).
        # MoveJoin's destination (30) and its consequences (46..48) already enter h_opt through E_dst / W_v.
        # Created last, so a fresh net's other parameters are initialised as before. Default: on when the env
        # has the columns; checkpoints without W_mg.* load with opt_refs=False.
        self.opt_refs = (self.K >= 61) if opt_refs is None else bool(opt_refs and self.K >= 61)
        if self.opt_refs:
            self.W_ci = nn.Linear(D, D)
            self.W_mg = nn.Linear(D + 1, D)

    def card_emb(self, key):
        key = key.clamp(0, self.special_idx.shape[0] - 1)
        if self.repr_name == 'id':
            x = self.E_card(key)
        else:
            x = self.W_card(self.R)[key]
        x = x + self.E_special(self.special_idx[key])
        if self.card_norm:
            x = self.card_ln(x) * self.card_scale
        return x

    def forward(self, b):
        tok = b['tok'].long()
        kind = tok[..., 0].clamp(0, N_KIND)
        fields = tok[..., 2:]
        x_card = self.card_emb(tok[..., 1])
        kc = kind.clamp(1, N_KIND) - 1
        sidx = kc[..., None] * self.kstride + self.field_off + (fields[..., :self.ncat].clamp(-1, 10) + 1)
        st_cat = self.E_state(sidx).sum(-2)
        st = self.W_num(slog(fields)).view(*kind.shape, N_KIND, D)
        st = torch.gather(st, 2, kc[..., None, None].expand(*kind.shape, 1, D)).squeeze(2) + st_cat
        x = x_card + st + self.E_kind(kind)
        if 'psrc' in b:  # the object asking the prompt
            ps = b['psrc'].long()
            hit = (torch.arange(x.shape[1], device=x.device)[None] == ps[:, None]).float()
            x = x + hit[..., None] * self.E_psrc
        g = b['glob'].long()
        Bn = g.shape[0]
        gf = torch.cat([slog(g), onehot_clip(g[:, 1], 0, 10), F.one_hot(b['pk'].long().clamp(0, N_PK - 1), N_PK).float()], -1)
        xg = self.W_glob(gf) + self.E_kind.weight[N_KIND + 1]
        h = torch.cat([xg[:, None, :], x], 1)
        mt = b['tok_mask'].bool()
        pad = torch.cat([torch.zeros(Bn, 1, dtype=torch.bool, device=mt.device), ~mt], 1)  # CUDA fix
        h = self.ln(self.enc(h, src_key_padding_mask=pad))
        hg, ht = h[:, 0], h[:, 1:]
        value = torch.tanh(self.val(hg)).squeeze(-1)

        act = b['act'].long()
        A = act.shape[1]
        T = ht.shape[1]

        def tok_h(ix):  # [B, A, R] -> summed [B, A, D], count [B, A]
            Rr = ix.shape[2]
            ok = (ix >= 0) & (ix < T) & torch.gather(mt, 1, ix.clamp(0, T - 1).view(Bn, -1)).view(Bn, A, Rr)
            hh = torch.gather(ht, 1, ix.clamp(0, T - 1).view(Bn, A * Rr)[..., None].expand(Bn, A * Rr, D)).view(Bn, A, Rr, D)
            okf = ok.float()
            return (hh * okf[..., None]).sum(2), okf.sum(2)

        h_src, _ = tok_h(act[..., list(ACT_SRC)])
        sr, nr = tok_h(act[..., ACT_REFS])
        h_ref = sr / nr.clamp(min=1)[..., None]
        v = torch.where(act[..., 31] > 0, act[..., 32], torch.full_like(act[..., 32], -1))
        dmg = act[..., 19:30:2].clamp(min=0).sum(-1)
        sc = torch.cat([onehot_clip(v, -1, 15), onehot_clip(act[..., 3], -1, 4),
                        onehot_clip(act[..., 36], 0, 5), act[..., 37:41].clamp(0, 1).float(),
                        (act[..., 34] > 0).float()[..., None], onehot_clip(act[..., 35], -1, 3),
                        torch.log1p(dmg.float())[..., None], torch.log1p(nr[..., None]),
                        onehot_clip(act[..., 46], -1, 2), act[..., 47:49].clamp(0, 1).float()], -1)
        nc = act[..., 34].clamp(min=0)
        h_opt = (self.E_akind(act[..., 0].clamp(0, 23)) + self.card_emb(nc) * (nc > 0).float()[..., None]
                 + self.E_dst((act[..., ACT_DST] + 1).clamp(0, 7)) + self.W_v(sc))   # the option's own features
        if self.opt_cost:
            oc = act[..., 49:56]
            valid = (oc[..., 0] >= 0).float()[..., None]
            ocf = torch.cat([oc[..., 0:3].clamp(min=0).float(), onehot_clip(oc[..., 3], -1, 5),
                             oc[..., 4:6].clamp(min=0).float() / 4.0, onehot_clip(oc[..., 6], -1, 1), valid], -1)
            h_opt = h_opt + self.W_oc(ocf) * valid
        if self.opt_refs:
            h_ci, n_ci = tok_h(act[..., 33:34])
            h_opt = h_opt + self.W_ci(h_ci) * (n_ci > 0).float()[..., None]
            s_mg, n_mg = tok_h(act[..., 56:60])
            has = (n_mg > 0).float()[..., None]
            mg = torch.cat([s_mg / n_mg.clamp(min=1)[..., None], torch.log1p(act[..., 60].clamp(min=0).float())[..., None]], -1)
            h_opt = h_opt + self.W_mg(mg) * has
        ha = h_opt + h_src + h_ref + hg[:, None, :]
        if self.ask_head:
            ps = b['psrc'].long() if 'psrc' in b else torch.full((Bn,), -1, device=ht.device, dtype=torch.long)
            okp = ((ps >= 0) & (ps < T)).float()[:, None]
            h_ask = torch.gather(ht, 1, ps.clamp(0, T - 1)[:, None, None].expand(Bn, 1, D)).squeeze(1) * okp
            other = h_ref + self.W_ask_src(h_src)
            if self.ask_opt:
                other = other + self.W_ask_opt(h_opt)
            ha = ha + self.W_ask(h_ask)[:, None, :] * other
            if self.ask_opt:
                ha = ha + self.W_ctx(hg)[:, None, :] * self.W_ctx_opt(h_opt)
        ma = b['act_mask'].bool()
        logits = self.pol(ha).squeeze(-1).masked_fill(~ma, -1e9)
        return F.log_softmax(logits, -1), value
