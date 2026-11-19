"""Shared helpers for gate 3 (wiring tests)."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SP = os.path.normpath(os.path.join(HERE, '..', 'selfplay'))
sys.path.insert(0, os.path.join(SP, 'learner'))
sys.path.insert(0, os.path.join(SP, 'rbenv'))
PK_NAMES = ('Unspecified BattlefieldSetup Mulligan MainPhase ChainPriority ShowdownFocus NextStaged CombatDamage '
            'CombatDamageNext RuneForEnergy RuneForPower AddForPayment CostSplit Accelerate Repeat AdditionalCost '
            'DiscardCost TriggerOrder ReplacementOrder NewChoices MoveJoin May Mode XAmount Target TargetFirst '
            'TargetSecond ChooseObject SplitTargets SplitAmount ChainItem Location Discard LookTop DeckOrder Value '
            'NameACard').split()
INTENT = ('PlayCard StandardMove ActivateAbility HideCard EndTurn PlayReaction ActivateReactionAbility PassPriority '
          'PlayActionCard ActivateActionAbility PassFocus AssignCombatDamage MulliganDecision ChooseBattlefield '
          'MakeChoice SideboardSwap PlayFirstDecision PlaceOptionalTrigger DeclineOptionalTrigger PayTriggeredCost '
          'DeclineTriggeredCost Concede').split()
KIND = {0: 'pad', 1: 'card', 2: 'bf', 3: 'chain', 4: 'pile'}
OBS_KEYS = ('glob', 'tok', 'tok_mask', 'act', 'act_mask', 'pk', 'psrc')
RUN_TRAINED = os.path.join(SP, 'runs', 'gate2v2_sanity_B2_ask', 'latest.pt')


def pkn(i):
    return PK_NAMES[i] if 0 <= i < len(PK_NAMES) else f'pk{i}'


def load_net(ck, dims, repr_name='B2', ask_head=True, dev='cpu', **kw):
    import torch
    from model import Net, load_vocab
    vocab = load_vocab()[0]
    sd = None
    if ck and ck != 'untrained':
        sd = torch.load(ck, map_location=dev)
        sd = sd.get('model', sd)
        ask_head = 'W_ask.weight' in sd
        kw['ask_opt'] = 'W_ask_opt.weight' in sd
        kw['opt_cost'] = 'W_oc.weight' in sd
        kw['opt_refs'] = 'W_mg.weight' in sd
    net = Net(dims, vocab, repr_name, ask_head=ask_head, **kw).to(dev)
    if sd is not None:
        missing, unexpected = net.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print('load: missing', missing, 'unexpected', unexpected)
    net.eval()
    return net
