"""MSPN -- Multi-scale Pyramidal Network.

A lightweight, plug-and-play guidance module that sits between a frozen
patch encoder and a standard attention-based MIL head. It gives the head
multi-scale context WITHOUT requiring the slide to be tiled at more than one
magnification: the coarse views are POOLED from the 20x features that the MIL
head already consumes.

Contract
--------
    h_out, coarse_maps = mspn(h, coords)

    h       [N, D]   patch features of one slide
    coords  [N, 2]   patch coordinates in slide pixels
    h_out   [N, D]   modulated features, same shape
    coarse_maps      {field_of_view: [1, N]} scalar saliency per patch, one
                     entry per scale. Used by the heatmap/interpretability
                     path and by the optional regularisers in
                     utils/universal_utils.py.

How one scale works
-------------------
For a field of view of `s` slide pixels the patches are binned into a
ceil(span_x/s) x ceil(span_y/s) lattice, mean-pooled per cell, and passed
through a small convolution that emits one saliency score per cell. The score
is scattered back to every patch in that cell and applied as a gated residual

    h <- h + h * sigmoid(score)

Scales are processed LARGEST FIRST, and each scale pools the tensor the
previous scales already modulated, so guidance accumulates coarse -> fine.
That ordering is what makes the module pyramidal rather than a bank of
independent branches.

Two details that are not cosmetic
--------------------------------
1. OCCUPANCY. WSI tissue is not rectangular, so many lattice cells contain no
   patches at all. Leaving them as zero vectors makes empty space
   indistinguishable from tissue that happens to embed near zero. Occupancy
   (binary) and density (normalised patch count) are therefore fed to the
   block as explicit channels, and the guidance of an empty cell is masked to
   zero.

2. POOL THEN PROJECT. The cell features are reduced to the convolution width
   AFTER pooling, not before. Mean pooling is linear so the result is
   identical, but the cost becomes O(M*D*D') instead of O(N*D*D') -- it stops
   depending on bag size, and M << N on every real slide.

Field of view and magnification
-------------------------------
A 20x tile spans 512 slide-pixel units, so a field of view of `s` units
corresponds to magnification 20 * 512 / s:

    1024 -> true 10x      2048 -> true 5x      3072 -> 3.33x

DEFAULT_VIEW_SCALES is the configuration reported in the paper. Note that
3.33x is NOT an acquirable objective magnification -- it is a pooled footprint,
which is precisely the point: MSPN is not restricted to magnifications the
scanner can produce.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.clam import CLAM_MB_Head, CLAM_SB_Head

# Fields of view in slide pixels, reported in the paper. Coarse -> fine.
DEFAULT_VIEW_SCALES = (3072, 2048, 1024)


def _grid_dims(span: float, fov: int) -> int:
    """Number of lattice cells covering `span` slide pixels at this FOV."""
    return max(1, math.ceil(span / fov))


class CoarseGuidanceBlock(nn.Module):
    """One scale of the pyramid: grid cell features -> one score per cell.

    Deliberately shallow. At realistic WSI lattice sizes (order 1e3 cells) the
    convolutions are numerically trivial and wall-clock is dominated by kernel
    launches, not arithmetic, so extra depth costs latency and buys nothing.

    The stem folds the two occupancy channels in BEFORE any spatial mixing, so
    the block can learn "ignore empty space" rather than having to infer it
    from zero activations.
    """

    def __init__(self, hidden: int = 64, n_groups: int = 8):
        super().__init__()
        # +2 input channels: binary occupancy and normalised patch density
        self.stem = nn.Conv2d(hidden + 2, hidden, kernel_size=1)
        self.spatial = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(n_groups, hidden)
        self.head = nn.Conv2d(hidden, 1, kernel_size=1)
        with torch.no_grad():
            self.head.bias.zero_()

    def forward(self, feats: torch.Tensor, occ: torch.Tensor,
                dens: torch.Tensor) -> torch.Tensor:
        x = self.stem(torch.cat([feats, occ, dens], dim=1))
        x = x + F.gelu(self.norm(self.spatial(x)))
        # cells holding no tissue contribute no guidance
        return self.head(x) * occ


class MSPN(nn.Module):
    """Multi-scale Pyramidal Network.

    Args:
        in_channels:     feature dim D of the incoming patch embeddings.
        hidden_channels: D', width of the lattice convolutions.
        view_scales:     fields of view in slide pixels. Sorted internally and
                         processed coarse -> fine.
        dropout_p:       dropout on the per-patch score during training.
        max_cells:       cap on cells per scale. A very small FOV on a very
                         large slide would otherwise allocate a huge lattice;
                         when the cap trips, that scale's FOV is widened just
                         enough to fit. None disables the guard.
    """

    def __init__(
        self,
        in_channels: int = 512,
        hidden_channels: int = 64,
        view_scales: List[int] = DEFAULT_VIEW_SCALES,
        dropout_p: float = 0.1,
        max_cells: Optional[int] = 65536,
    ):
        super().__init__()
        self.view_scales = sorted(view_scales, reverse=True)   # coarse -> fine
        self.hidden = hidden_channels
        self.dropout_p = dropout_p
        self.max_cells = max_cells

        n_groups = math.gcd(8, hidden_channels) or 1

        # Cell-level projection, weights SHARED across scales -- see the
        # "pool then project" note in the module docstring.
        self.proj = nn.Linear(in_channels, hidden_channels)
        self.blocks = nn.ModuleList([
            CoarseGuidanceBlock(hidden_channels, n_groups)
            for _ in self.view_scales
        ])

    # ---- lattice construction ---------------------------------------------

    def _build_indices(self, coords: torch.Tensor) -> List[Tuple[torch.Tensor, int, int]]:
        """Cell index per scale, using exactly ONE device -> host sync.

        The slide bounds are read back once and reused for every scale. Doing
        the aminmax/.tolist() inside the per-scale loop instead (the obvious
        way) costs one stall per scale and dominates wall-clock at these
        lattice sizes.
        """
        x, y = coords[:, 0], coords[:, 1]
        bounds = torch.stack([*x.aminmax(), *y.aminmax()])
        x_min_f, x_max_f, y_min_f, y_max_f = bounds.tolist()   # single sync
        x_range = x_max_f - x_min_f + 1e-6
        y_range = y_max_f - y_min_f + 1e-6

        x_norm = (x - bounds[0]) / x_range
        y_norm = (y - bounds[2]) / y_range

        out = []
        for fov in self.view_scales:
            gx, gy = _grid_dims(x_range, fov), _grid_dims(y_range, fov)
            if self.max_cells is not None and gx * gy > self.max_cells:
                shrink = math.sqrt(gx * gy / self.max_cells)
                gx, gy = max(1, int(gx / shrink)), max(1, int(gy / shrink))
            u = (x_norm * gx).to(torch.int64).clamp_(0, gx - 1)
            v = (y_norm * gy).to(torch.int64).clamp_(0, gy - 1)
            out.append((v * gx + u, gx, gy))
        return out

    # ---- forward -----------------------------------------------------------

    def forward(self, x: torch.Tensor,
                coords: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        h = x
        N, D = h.shape

        with torch.no_grad():
            index_list = self._build_indices(coords)

        ones = h.new_ones(N)
        coarse_maps: Dict[int, torch.Tensor] = {}

        for i, (fov, (idx, gx, gy)) in enumerate(zip(self.view_scales, index_list)):
            M = gx * gy

            # index_add_ rather than torch.bincount: bincount sizes its output
            # from a device-side max and so forces a sync on every scale.
            counts = h.new_zeros(M).index_add_(0, idx, ones)
            # `h` is the progressively modulated tensor, so finer scales pool
            # features the coarser scales have already re-weighted.
            sums = h.new_zeros(M, D).index_add_(0, idx, h)
            pooled = sums / counts.clamp_min(1.0).unsqueeze(1)

            cell = self.proj(pooled)
            feats = cell.view(gy, gx, self.hidden).permute(2, 0, 1).unsqueeze(0)
            occ = (counts > 0).to(h.dtype).view(1, 1, gy, gx)
            dens = (counts / counts.max().clamp_min(1.0)).view(1, 1, gy, gx)

            score = self.blocks[i](feats, occ, dens)          # [1, 1, gy, gx]
            g = score.view(1, M)[:, idx].transpose(0, 1)      # [N, 1]
            if self.training and self.dropout_p > 0:
                g = F.dropout(g, p=self.dropout_p, training=True)

            gate = torch.sigmoid(g)
            coarse_maps[fov] = gate[:, 0].unsqueeze(0)        # [1, N], as v1
            h = h + h * gate

        return h, coarse_maps


class ABMILHead(nn.Module):
    """Gated-attention MIL pooling (Ilse et al., 2018)."""

    def __init__(self, in_channels=512, mil_hidden_1=512, mil_hidden_2=128,
                 attn_branches=1):
        super().__init__()
        self.hid_dim_1 = mil_hidden_1
        self.hid_dim_2 = mil_hidden_2
        self.attn_branches = attn_branches
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_channels, mil_hidden_1), nn.ReLU())
        self.attention_V = nn.Sequential(
            nn.Linear(mil_hidden_1, mil_hidden_2), nn.Tanh())
        self.attention_U = nn.Sequential(
            nn.Linear(mil_hidden_1, mil_hidden_2), nn.Sigmoid())
        self.attention_w = nn.Linear(mil_hidden_2, attn_branches)

    def forward(self, h):
        h = self.feature_extractor(h)                      # [N, hidden_1]
        A = self.attention_w(self.attention_V(h) * self.attention_U(h))
        A = torch.transpose(A, 1, 0)                       # [branches, N]
        A_raw = A
        A = F.softmax(A, dim=1)
        return torch.mm(A, h), A_raw


def _front_end(in_channels, reduction_size):
    """Linear + ReLU in front of MSPN.

    Applied UNCONDITIONALLY, including when in_channels == reduction_size.
    Without it MSPN would read raw frozen features on a 512-d encoder and a
    purely linear map of them on a 1536-d one -- an architecture asymmetry
    between encoders, and in both cases a module that never sees a nonlinear
    learned representation. This matches the front end used by the reference
    MIL implementations the baselines are taken from.
    """
    return nn.Sequential(nn.Linear(in_channels, reduction_size), nn.ReLU())


class ABMIL_MSPN(nn.Module):
    def __init__(self, in_channels=512, n_classes=2, rpn_hidden=64,
                 mil_hidden_1=512, mil_hidden_2=128, attn_branches=1,
                 view_scales=DEFAULT_VIEW_SCALES, reduction_size=512):
        super().__init__()
        self.fc = _front_end(in_channels, reduction_size)
        self.mspn = MSPN(reduction_size, rpn_hidden, view_scales)
        self.abmil_head = ABMILHead(reduction_size, mil_hidden_1, mil_hidden_2,
                                    attn_branches)
        self.classifier = nn.Sequential(
            nn.Linear(mil_hidden_1 * attn_branches, n_classes))

    def forward(self, x, coords, vis_heatmap=False, vis_coarse_map=False):
        if not vis_heatmap:
            x = x.squeeze(0)
        coords = coords.squeeze(0).to(x.device)
        h = self.fc(x)

        h_coarse, coarse_map = self.mspn(h, coords)
        attn_feats, A_raw = self.abmil_head(h_coarse)
        logits = self.classifier(attn_feats)

        if vis_heatmap:
            return A_raw, logits
        elif vis_coarse_map:
            return coarse_map, logits
        return logits, None


class DSMIL_MSPN(nn.Module):
    """MSPN in front of DSMIL.

    DSMIL is the only backbone here with a hard argmax in its path: the
    instance classifier scores every patch and the bag branch keys on the
    highest-scoring one. A patch's CLASS is a property of its 20x tile, so the
    instance classifier is run on the unmodulated features only -- MSPN informs
    the BAG representation alone. Routing the modulated features through the
    instance branch instead lets the module's per-patch gain reorder the
    critical instance by magnitude rather than by evidence.
    """

    def __init__(self, i_classifier, b_classifier, in_channels, n_classes=2,
                 rpn_hidden=64, view_scales=DEFAULT_VIEW_SCALES, surv=False,
                 reduction_size=512):
        super().__init__()
        self.fc = _front_end(in_channels, reduction_size)
        self.mspn = MSPN(reduction_size, rpn_hidden, view_scales)
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        self.fcc = nn.Conv1d(n_classes, n_classes, kernel_size=reduction_size)
        self.surv = surv

    def forward(self, x, coords, vis_heatmap=False, vis_coarse_map=False):
        if not vis_heatmap:
            x = x.squeeze(0)
        coords = coords.squeeze(0).to(x.device)
        h = self.fc(x)

        h_coarse, coarse_map = self.mspn(h, coords)
        feats, classes = self.i_classifier(h)
        # `self.fc` has already reduced to `reduction_size`, so the instance
        # classifier carries no further projection and h_coarse is already in
        # `feats`' space. The assert pins that: were it to gain one, h_coarse
        # must get its own projection rather than borrow the instance branch's,
        # which would send its gradients back into the instance path.
        assert getattr(self.i_classifier, 'fc_reduction', None) is None, (
            'i_classifier has an fc_reduction; h_coarse must not be routed '
            'through it. Give the wrapper its own projection instead.')
        f_in = feats + h_coarse      # bag level: 20x features PLUS context
        c_in = classes               # instance level: 20x only
        A_raw, B = self.b_classifier(f_in, c_in)
        prediction_bag = self.fcc(B).view(1, -1)

        if vis_heatmap:
            return A_raw, prediction_bag
        elif vis_coarse_map:
            return coarse_map, prediction_bag
        if self.surv:
            return prediction_bag, coarse_map
        return classes, prediction_bag, A_raw, B


class CLAMSB_MSPN(nn.Module):
    def __init__(self, in_channels=512, n_classes=2, rpn_hidden=64,
                 mil_hidden_1=512, mil_hidden_2=128, attn_branches=1,
                 view_scales=DEFAULT_VIEW_SCALES, reduction_size=512):
        super().__init__()
        self.fc = _front_end(in_channels, reduction_size)
        self.mspn = MSPN(reduction_size, rpn_hidden, view_scales)
        self.clamsb_head = CLAM_SB_Head(n_classes=n_classes,
                                        embed_dim=reduction_size)
        self.classifier = nn.Sequential(
            nn.Linear(mil_hidden_1 * attn_branches, n_classes))

    def forward(self, x, coords, label=None, instance_eval=False,
                vis_heatmap=False, vis_coarse_map=False):
        if not vis_heatmap:
            x = x.squeeze(0)
        coords = coords.squeeze(0).to(x.device)
        h = self.fc(x)

        h_coarse, coarse_map = self.mspn(h, coords)
        attn_feats, A_raw, total_inst_loss = self.clamsb_head(
            h_coarse, label=label, instance_eval=instance_eval)
        logits = self.classifier(attn_feats)

        if vis_heatmap:
            return A_raw, logits
        elif vis_coarse_map:
            return coarse_map, logits
        return logits, total_inst_loss


class CLAMMB_MSPN(nn.Module):
    def __init__(self, in_channels=512, n_classes=2, rpn_hidden=64,
                 mil_hidden_1=512, mil_hidden_2=128, attn_branches=1,
                 view_scales=DEFAULT_VIEW_SCALES, reduction_size=512):
        super().__init__()
        self.fc = _front_end(in_channels, reduction_size)
        self.mspn = MSPN(reduction_size, rpn_hidden, view_scales)
        self.clammb_head = CLAM_MB_Head(n_classes=n_classes,
                                        embed_dim=reduction_size)
        self.n_classes = n_classes
        self.classifiers = nn.ModuleList(
            [nn.Linear(mil_hidden_1, 1) for _ in range(n_classes)])

    def forward(self, x, coords, label=None, instance_eval=False,
                vis_heatmap=False, vis_coarse_map=False):
        if not vis_heatmap:
            x = x.squeeze(0)
        coords = coords.squeeze(0).to(x.device)
        h = self.fc(x)

        h_coarse, coarse_map = self.mspn(h, coords)
        attn_feats, A_raw, total_inst_loss = self.clammb_head(
            h_coarse, label=label, instance_eval=instance_eval)
        logits = torch.empty(1, self.n_classes).float().to(attn_feats.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](attn_feats[c])

        if vis_heatmap:
            return A_raw, logits
        elif vis_coarse_map:
            return coarse_map, logits
        return logits, total_inst_loss
