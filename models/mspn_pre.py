"""MSPN in front of a PRETRAINED ABMIL body.

Mirrors `models.mspn.ABMIL_MSPN` exactly, except the attention head is a
supplied `models.abmil.ABMILPretrained` carrying pretrained weights. The
CALLER loads that state (see `models.pretrained_mil.build_pretrained_abmil`),
so there is one place responsible for it and no chance of two code paths
disagreeing about what "pretrained" means.

The pretrained checkpoint's `patch_embed` is encoder specific and is dropped
on load, so only the attention body transfers. MSPN therefore sits between our
own front end and the pretrained attention.
"""

import torch.nn as nn

from models.mspn import MSPN, DEFAULT_VIEW_SCALES, _front_end


class ABMILMSPNPretrained(nn.Module):
    def __init__(self, in_channels=512, n_classes=2, rpn_hidden=64,
                 view_scales=DEFAULT_VIEW_SCALES, reduction_size=512,
                 pretrained_head=None):
        super().__init__()
        if pretrained_head is None:
            raise ValueError('ABMILMSPNPretrained needs the pretrained ABMIL '
                             'body; the caller must build and load it')
        self.fc = _front_end(in_channels, reduction_size)
        self.mspn = MSPN(reduction_size, rpn_hidden, view_scales)
        self.pre = pretrained_head

    def forward(self, x, coords, vis_heatmap=False, vis_coarse_map=False):
        if not vis_heatmap:
            x = x.squeeze(0)
        coords = coords.squeeze(0).to(x.device)
        h = self.fc(x)

        h_coarse, coarse_map = self.mspn(h, coords)
        if h_coarse.dim() == 2:              # ABMILPretrained wants [B, M, D]
            h_coarse = h_coarse.unsqueeze(0)
        out = self.pre(h_coarse)
        logits = out[0] if isinstance(out, tuple) else out

        if vis_coarse_map:
            return coarse_map, logits
        return logits, None
