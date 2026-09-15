"""MSPN -- Multi-scale Pyramidal Network.

A guidance module between a frozen patch encoder and an attention-based MIL
head. Coarse views are pooled from the 20x features the head already consumes,
so only one magnification is needed on disk.

    h_out, coarse_maps = mspn(h, coords)

    h       [N, D]   patch features of one slide
    coords  [N, 2]   patch coordinates in slide pixels
    h_out   [N, D]   modulated features
    coarse_maps      {fov: [1, N]} per-patch saliency, one entry per scale

A 20x tile spans 512 coord units, so fov s gives magnification 20 * 512 / s:
1024 = 10x, 2048 = 5x, 3072 = 3.33x. DEFAULT_VIEW_SCALES is the configuration
reported in the paper.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.clam import CLAM_MB_Head, CLAM_SB_Head

DEFAULT_VIEW_SCALES = (3072, 2048, 1024)


def _grid_dims(span: float, fov: int) -> int:
    """Number of lattice cells covering `span` slide pixels at this FOV."""
    return max(1, math.ceil(span / fov))


class CoarseGuidanceBlock(nn.Module):
    def __init__(self, hidden: int = 64, n_groups: int = 8):
        super().__init__()
        # +2 channels: occupancy and normalised patch density
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
        return self.head(x) * occ


class MSPN(nn.Module):
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

        # weights shared across scales
        self.proj = nn.Linear(in_channels, hidden_channels)
        self.blocks = nn.ModuleList([
            CoarseGuidanceBlock(hidden_channels, n_groups)
            for _ in self.view_scales
        ])


    def _build_indices(self, coords: torch.Tensor) -> List[Tuple[torch.Tensor, int, int]]:
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

            # index_add_, not bincount: bincount forces a sync per scale
            counts = h.new_zeros(M).index_add_(0, idx, ones)
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
            coarse_maps[fov] = gate[:, 0].unsqueeze(0)
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

        assert getattr(self.i_classifier, 'fc_reduction', None) is None, (
            'i_classifier has an fc_reduction; h_coarse must not be routed '
            'through it. Give the wrapper its own projection instead.')
        f_in = feats + h_coarse      # bag: 20x plus context
        c_in = classes               # instance: 20x only
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
