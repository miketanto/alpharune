"""Policy/value network for no-search self-play (NETWORK.md design, adapted from
smoke/train/train.py to the rbenv tensor contract, SELFPLAY.md).

Inputs (torch tensors, one row per pending decision):
  glob (B,G) int, tok (B,T,F) int, tok_mask (B,T) bool, act (B,A,K) int, act_mask (B,A) bool, pk (B,) int

Field layout: rbenv/README.txt (F=17, K=46). tok[...,2:] situational fields (first 15 embedded
categorically + a per-kind linear map of all). act: 0 t, 1 src, 2 abs, 3 ai, 4..17 tg/ch/mv/mu tokens,
18..29 dm (token, damage) pairs, 30 dst, 31 has_v, 32 v, 33 ci token, 34 nc vocab index, 35 bf, 36 ps,
37 alt, 38 plc, 39 dec, 40 first, 41..45 counts. Optional b['psrc'] (B,) marks the asking token.
"""
import json, os
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 64
N_PK = 40
ACT_SRC = [1, 2]                                    # src, abs token refs (summed)
ACT_REFS = list(range(4, 18)) + list(range(18, 30, 2)) + [33]  # tg ch mv mu, dm tokens, ci (mean-pooled)
ACT_DST = 30

HERE = os.path.dirname(os.path.abspath(__file__))
CR = os.path.normpath(os.path.join(HERE, '..', '..'))   # research/card_repr
REPRS_PT = os.path.join(CR, 'smoke', 'train', 'packed', 'reprs.pt')
VOCAB_JSON = os.path.join(CR, 'selfplay', 'rbenv', 'vocab.json')
CARDS_JSON = os.path.join(CR, 'cards.json')


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
    """Returns (table [V, d_in], trainable, n_missing). B2/B from smoke/train/packed/reprs.pt by key."""
    V = len(vocab)
    if repr_name == 'id':
        return None, True, 0
    if repr_name == 'random':
        g = torch.Generator().manual_seed(seed)
        return torch.randn(V, 256, generator=g), False, 0
    R = torch.load(REPRS_PT)
    src = R['tables'][repr_name]
    idx = {k: i for i, k in enumerate(R['vocab'])}
    tab = torch.zeros(V, src.shape[1])
    missing = []
    for i, k in enumerate(vocab):
        if k in idx:
            tab[i] = src[idx[k]]
        elif not (k in ('<pad>', 'hidden', 'keyword_ability') or k.startswith(('token:', 'other'))):
            missing.append(k)
    return tab, False, missing


def onehot_clip(x, lo, hi):
    return F.one_hot((x.clamp(lo, hi) - lo), hi - lo + 1).float()


def slog(x):
    x = x.float()
    return torch.sign(x) * torch.log1p(x.abs())


class Net(nn.Module):
    def __init__(self, dims, vocab, repr_name='id'):
        super().__init__()
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
        self.ncat = min(15, nf)
        self.E_state = nn.Embedding(3 * 15 * 12, D)
        self.register_buffer('field_off', torch.arange(self.ncat) * 12)
        self.W_num = nn.Linear(nf, 3 * D)
        self.E_kind = nn.Embedding(5, D)
        self.E_psrc = nn.Parameter(torch.zeros(D))
        self.W_glob = nn.Linear(self.G + 11 + N_PK, D)
        layer = nn.TransformerEncoderLayer(D, 4, 2 * D, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(D)
        self.E_akind = nn.Embedding(24, D)
        self.E_dst = nn.Embedding(8, D)
        self.W_v = nn.Linear(17 + 6 + 6 + 5 + 5 + 2, D)
        self.val = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 1))
        self.pol = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 1))

    def card_emb(self, key):
        key = key.clamp(0, self.special_idx.shape[0] - 1)
        if self.repr_name == 'id':
            x = self.E_card(key)
        else:
            x = self.W_card(self.R)[key]
        return x + self.E_special(self.special_idx[key])

    def forward(self, b):
        tok = b['tok'].long()
        kind = tok[..., 0].clamp(0, 3)
        fields = tok[..., 2:]
        x_card = self.card_emb(tok[..., 1])
        kc = kind.clamp(1, 3) - 1
        sidx = kc[..., None] * 180 + self.field_off + (fields[..., :self.ncat].clamp(-1, 10) + 1)
        st_cat = self.E_state(sidx).sum(-2)
        st = self.W_num(slog(fields)).view(*kind.shape, 3, D)
        st = torch.gather(st, 2, kc[..., None, None].expand(*kind.shape, 1, D)).squeeze(2) + st_cat
        x = x_card + st + self.E_kind(kind)
        if 'psrc' in b:  # the object asking the prompt
            ps = b['psrc'].long()
            hit = (torch.arange(x.shape[1], device=x.device)[None] == ps[:, None]).float()
            x = x + hit[..., None] * self.E_psrc
        g = b['glob'].long()
        Bn = g.shape[0]
        gf = torch.cat([slog(g), onehot_clip(g[:, 1], 0, 10), F.one_hot(b['pk'].long().clamp(0, N_PK - 1), N_PK).float()], -1)
        xg = self.W_glob(gf) + self.E_kind.weight[4]
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
                        torch.log1p(dmg.float())[..., None], torch.log1p(nr[..., None])], -1)
        nc = act[..., 34].clamp(min=0)
        ha = (self.E_akind(act[..., 0].clamp(0, 23)) + h_src + h_ref + self.card_emb(nc) * (nc > 0).float()[..., None]
              + self.E_dst((act[..., ACT_DST] + 1).clamp(0, 7)) + self.W_v(sc) + hg[:, None, :])
        ma = b['act_mask'].bool()
        logits = self.pol(ha).squeeze(-1).masked_fill(~ma, -1e9)
        return F.log_softmax(logits, -1), value
