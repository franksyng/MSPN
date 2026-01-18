import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.clam import CLAM_MB_Head, CLAM_SB_Head
from utils.universal_utils import initialize_weights


def compute_grid_size(coords, field_of_view):
    x = coords[:, 0]
    y = coords[:, 1]
    x_min, x_max = x.aminmax()
    y_min, y_max = y.aminmax()
    ranges = torch.stack([(x_max - x_min), (y_max - y_min)])
    x_range, y_range = ranges.cpu().tolist()
    grid_x = max(1, math.ceil(x_range / field_of_view))
    grid_y = max(1, math.ceil(y_range / field_of_view))

    return grid_x, grid_y


class CGN(nn.Module):
    def __init__(self, in_channels=512, hidden_channels=64, field_of_view=768):
        super(CGN, self).__init__()
        self.field_of_view = field_of_view
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1)
        )
        
    def _build_grid(self, patch_feats, coords):
        device = patch_feats.device
        N, D = patch_feats.shape

        with torch.no_grad():
            grid_x, grid_y = compute_grid_size(coords, field_of_view=self.field_of_view)

            x, y = coords[:, 0], coords[:, 1]
            x_min, x_max = x.aminmax()
            y_min, y_max = y.aminmax()
            x_norm = (x - x_min) / (x_max - x_min + 1e-6)
            y_norm = (y - y_min) / (y_max - y_min + 1e-6)

            # Map to integer cell indices
            u = (x_norm * grid_x).long().clamp(0, grid_x - 1)   # [N]
            v = (y_norm * grid_y).long().clamp(0, grid_y - 1)   # [N]

            flat_idx = v * grid_x + u
            M = grid_x * grid_y

        # sums
        cell_sums = patch_feats.new_zeros((M, D))
        cell_sums.index_add_(0, flat_idx, patch_feats)

        # counts (fast)
        counts = torch.bincount(flat_idx, minlength=M).to(patch_feats.dtype).unsqueeze(1)
        cell_means = cell_sums / counts.clamp_min_(1)

        # [1, D, H, W]
        grid_feats = cell_means.view(grid_y, grid_x, D).permute(2, 0, 1).unsqueeze(0)

        return grid_feats, flat_idx, grid_x, grid_y

    
    def forward(self, x, coords):
            """
            x: [B, C, H, W]
            returns scores: [B, H, W]
            """
            grid_feats, flat_idx, grid_x, grid_y = self._build_grid(x, coords)
            # print(grid_x, grid_y)
            # print(grid_feats.shape)
            scores = self.conv(grid_feats).squeeze(0,1)
            scores_flat = scores.view(grid_x * grid_y)   # [H*W]
            # print(scores_flat.shape)
            # print(flat_idx.shape)
            # print(flat_idx)
            proposal_map = scores_flat[flat_idx]     # [N]
            # print(proposal_map.shape)
            
            return proposal_map.unsqueeze(1)

class MSPN(nn.Module):
    def __init__(self, in_channels=512, hidden_channels=64, view_scales=[768, 1536, 3072]):
        super(MSPN, self).__init__()
        view_scales.sort(reverse=True)
        self.view_scales = view_scales
        self.blocks = []
        for scale in view_scales:
            self.blocks.append(CGN(in_channels=in_channels, hidden_channels=hidden_channels, field_of_view=scale))
        self.blocks = nn.ModuleList(self.blocks)
        self.dropout = nn.Dropout1d(0.25)
    
    def forward(self, x, coords):
        h = x
        coarse_maps = {}
        for i in range(len(self.view_scales)):
            p_map = self.blocks[i](h, coords)
            coarse_maps[self.view_scales[i]] = p_map.T
            # print(h.shape, p_map.shape)
            h_p = self.dropout((h * torch.sigmoid(p_map)).T).T
            h = h + h_p
        return h, coarse_maps


class ABMILHead(nn.Module):
    def __init__(self, in_channels=768, mil_hidden_1=512, mil_hidden_2=128, attn_branches=1):
        super(ABMILHead, self).__init__()
        self.hid_dim_1 = mil_hidden_1
        self.hid_dim_2 = mil_hidden_2
        self.attn_branches = attn_branches
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_channels, self.hid_dim_1),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.hid_dim_1, self.hid_dim_2), # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.hid_dim_1, self.hid_dim_2), # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.hid_dim_2, self.attn_branches) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

    
    def forward(self, h):
        h = self.feature_extractor(h)  # KxM

        A_V = self.attention_V(h)  # KxL
        A_U = self.attention_U(h)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES

        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over K

        attn_feats = torch.mm(A, h)  # ATTENTION_BRANCHESxM

        # logits = self.classifier(Z)
        return attn_feats, A_raw


class ABMIL_MSPN(nn.Module):
    def __init__(self, in_channels=768, n_classes=2, rpn_hidden=64, mil_hidden_1=512, mil_hidden_2=128, attn_branches=1, view_scales=[1536, 2048, 3072], reduction_size=512):
        super(ABMIL_MSPN, self).__init__()
        if in_channels != reduction_size:
            self.fc = nn.Linear(in_channels, reduction_size)
        else:
            self.fc = None
        self.multiscale_rpn = MSPN(in_channels=reduction_size, hidden_channels=rpn_hidden, view_scales=view_scales)
        self.abmil_head = ABMILHead(in_channels=reduction_size, mil_hidden_1=mil_hidden_1, mil_hidden_2=mil_hidden_2)
        self.classifier = nn.Sequential(
            nn.Linear(mil_hidden_1*attn_branches, n_classes),
        )
        initialize_weights(self)
    
    def forward(self, x, coords, vis_heatmap=False, vis_coarse_map=False):
        if not vis_heatmap:
            x = x.squeeze(0)
        h = x
        device = x.device
        coords = coords.squeeze(0).to(device)

        if self.fc != None:
            h = self.fc(h)

        h_coarse, coarse_map = self.multiscale_rpn(h, coords)
        attn_feats, A_raw = self.abmil_head(h_coarse)
        logits = self.classifier(attn_feats)

        
        if vis_heatmap:
            return A_raw, logits
        elif vis_coarse_map:
            return coarse_map, logits
        else:
            return logits, None


class DSMIL_MSPN(nn.Module):
    def __init__(self, i_classifier, b_classifier, in_channels, n_classes=2, rpn_hidden=64, view_scales=[1536, 2048, 3072], surv=False, reduction_size=512):
        super(DSMIL_MSPN, self).__init__()
        if in_channels != reduction_size:
            self.fc = nn.Linear(in_channels, reduction_size)
        else:
            self.fc = None
        self.multiscale_rpn = MSPN(in_channels=reduction_size, hidden_channels=rpn_hidden, view_scales=view_scales)
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        self.fcc = nn.Conv1d(n_classes, n_classes, kernel_size=reduction_size)
        self.surv = surv
        initialize_weights(self)
        
    def forward(self, x, coords, vis_heatmap=False, vis_coarse_map=False):
        if not vis_heatmap:
            x = x.squeeze(0)
        h = x
        device = x.device
        coords = coords.squeeze(0).to(device)

        if self.fc != None:
            h = self.fc(h)

        h_coarse, coarse_map = self.multiscale_rpn(h, coords)
        # print(h_coarse.shape)
        feats, classes = self.i_classifier(h)
        feats_c, classes_c = self.i_classifier(h_coarse)
        A_raw, B = self.b_classifier(feats+feats_c, classes+classes_c)
        prediction_bag = self.fcc(B).view(1,-1)

        
        if vis_heatmap:
            return A_raw, prediction_bag
        elif vis_coarse_map:
            return coarse_map, prediction_bag
        else:
            if self.surv:
                return prediction_bag, coarse_map
            else:
                return classes, prediction_bag, A_raw, B


class CLAMSB_MSPN(nn.Module):
    def __init__(self, in_channels=768, n_classes=2, rpn_hidden=64, mil_hidden_1=512, mil_hidden_2=128, attn_branches=1, view_scales=[1536, 2048, 3072], reduction_size=512):
        super(CLAMSB_MSPN, self).__init__()
        if in_channels != reduction_size:
            self.fc = nn.Linear(in_channels, reduction_size)
        else:
            self.fc = None
        self.multiscale_rpn = MSPN(in_channels=reduction_size, hidden_channels=rpn_hidden, view_scales=view_scales)
        self.clamsb_head = CLAM_SB_Head(n_classes=n_classes, embed_dim=reduction_size)
        self.classifier = nn.Sequential(
            nn.Linear(mil_hidden_1*attn_branches, n_classes),
        )
        initialize_weights(self)
    
    def forward(self, x, coords, label=None, instance_eval=False, vis_heatmap=False, vis_coarse_map=False):
        if not vis_heatmap:
            x = x.squeeze(0)
        h = x
        device = x.device
        coords = coords.squeeze(0).to(device)

        if self.fc != None:
            h = self.fc(h)

        h_coarse, coarse_map = self.multiscale_rpn(h, coords)

        attn_feats, A_raw, total_inst_loss = self.clamsb_head(h_coarse, label=label, instance_eval=instance_eval)
        logits = self.classifier(attn_feats)

        if vis_heatmap:
            return A_raw, logits
        elif vis_coarse_map:
            return coarse_map, logits
        else:
            return logits, total_inst_loss

class CLAMMB_MSPN(nn.Module):
    def __init__(self, in_channels=768, n_classes=2, rpn_hidden=64, mil_hidden_1=512, mil_hidden_2=128, attn_branches=1, view_scales=[1536, 2048, 3072], reduction_size=512):
        super(CLAMMB_MSPN, self).__init__()
        if in_channels != reduction_size:
            self.fc = nn.Linear(in_channels, reduction_size)
        else:
            self.fc = None
        self.multiscale_rpn = MSPN(in_channels=reduction_size, hidden_channels=rpn_hidden, view_scales=view_scales)
        self.clammb_head = CLAM_MB_Head(n_classes=n_classes, embed_dim=reduction_size)
        self.n_classes = n_classes
        bag_classifiers = [nn.Linear(mil_hidden_1, 1) for i in range(n_classes)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        initialize_weights(self)
    
    def forward(self, x, coords, label=None, instance_eval=False, vis_heatmap=False, vis_coarse_map=False):
        if not vis_heatmap:
            x = x.squeeze(0)
        h = x
        device = x.device
        coords = coords.squeeze(0).to(device)

        if self.fc != None:
            h = self.fc(h)

        h_coarse, coarse_map = self.multiscale_rpn(h, coords)

        attn_feats, A_raw, total_inst_loss = self.clammb_head(h_coarse, label=label, instance_eval=instance_eval)
        logits = torch.empty(1, self.n_classes).float().to(attn_feats.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](attn_feats[c])

        if vis_heatmap:
            return A_raw, logits
        elif vis_coarse_map:
            return coarse_map, logits
        else:
            return logits, total_inst_loss
