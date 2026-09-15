"""One place that builds a PRETRAINED MIL body, so every entry point agrees.

Why this exists. `main_pre.py:abmilpre` used to do

    pre_model = create_model('abmil.base.uni_v2.pc108-24k', from_pretrained=True, ...)
    model = ABMILPretrained(in_dim=args.in_dim, num_classes=cls_num)

-- building the pretrained model and then discarding it, so the arm trained
from random init. Its multi-scale siblings in `main_rl_ms.py` loaded the state
correctly. The trio therefore disagreed about what "pretrained" meant, silently.

CHECKPOINT PER ENCODER (MASTER 2026-08-27):
    conch -> abmil.base.conch_v15.pc108-24k
    uni   -> abmil.base.uni_v2.pc108-24k

CAVEAT TO STATE IN THE PAPER: our CONCH features are conch_v1 (512-d); the only
released CONCH checkpoint is conch_v1.5 (768-d). `patch_embed` is encoder
specific and is dropped on load, so only the 512-d attention body transfers --
but that body was pretrained on v1.5 embeddings, not v1. UNI2 is an exact match
(uni_v2, 1536-d).
"""
import torch

TAGS = {'conch': 'abmil.base.conch_v15.pc108-24k',
        'uni': 'abmil.base.uni_v2.pc108-24k'}


def encoder_of(args):
    """Infer the encoder from the feature path, not from in_dim.

    in_dim is ambiguous -- uni_v2, gigapath and h-optimus are all 1536.
    """
    for src in (getattr(args, 'data_dir', '') or '', getattr(args, 'data_bb', '') or ''):
        s = str(src)
        if 'conch' in s:
            return 'conch'
        if 'uni' in s:
            return 'uni'
    raise ValueError('cannot infer encoder from --data_dir/--data_bb; '
                     'the pretrained checkpoint depends on it')


def build_pretrained_abmil(in_dim, num_classes, encoder, verbose=True):
    """ABMILPretrained with the pc108-24k ATTENTION BODY loaded.

    `classifier` (task specific) and `patch_embed` (encoder specific) are
    dropped, matching what main_rl_ms.py already did for the CSA/concat arms.
    Raises if nothing transferred -- a silent no-op is what caused the original
    defect.
    """
    from src.builder import create_model          # MIL-Lab; patho2 only
    from models.abmil import ABMILPretrained

    tag = TAGS[encoder]
    pre = create_model(tag, from_pretrained=True, num_classes=num_classes)
    model = ABMILPretrained(in_dim=in_dim, num_classes=num_classes)

    state = {k.removeprefix('model.'): v for k, v in pre.state_dict().items()}
    state = {k: v for k, v in state.items()
             if 'classifier' not in k and 'patch_embed' not in k}
    missing, unexpected = model.load_state_dict(state, strict=False)

    loaded = [k for k in state if k not in unexpected]
    if not loaded:
        raise RuntimeError(f'{tag}: NOTHING transferred into ABMILPretrained '
                           f'(unexpected={unexpected[:5]}) -- refusing to train '
                           f'a model that is pretrained in name only')
    if verbose:
        print(f'[pretrained] {tag} -> {len(loaded)} tensors loaded '
              f'({", ".join(sorted(loaded)[:4])}...), '
              f'{len(unexpected)} unexpected, patch_embed+classifier dropped')
    return model
