import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.clam import CLAM_MB_Head, CLAM_SB_Head


def compute_grid_size(coords: torch.Tensor, field_of_view: int = 1536):
    x = coords[:, 0]
    y = coords[:, 1]

    # x_min, x_max = x.min(), x.max()
    # y_min, y_max = y.min(), y.max()

    # x_range = (x_max - x_min).item()
    # y_range = (y_max - y_min).item()

    # grid_x = max(1, math.ceil(x_range / field_of_view))
    # grid_y = max(1, math.ceil(y_range / field_of_view))
    x_min, x_max = x.aminmax()
    y_min, y_max = y.aminmax()
    ranges = torch.stack([(x_max - x_min), (y_max - y_min)])  # on GPU
    x_range, y_range = ranges.cpu().tolist()  # ONE transfer/sync
    grid_x = max(1, math.ceil(x_range / field_of_view))
    grid_y = max(1, math.ceil(y_range / field_of_view))

    return grid_x, grid_y


class LiteRPN(nn.Module):
    def __init__(self, in_channels=512, hidden_channels=64, field_of_view=768):
        super(LiteRPN, self).__init__()
        self.field_of_view = field_of_view
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1)
        )
        
        # # self.conv_cls = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        # self.cls = nn.Linear(hidden_channels, 1)

    # def _build_grid(self, patch_feats: torch.Tensor, coords: torch.Tensor):
    #     """
    #     Build a coarse grid of cell-level features by aggregating patch features.

    #     patch_feats: [N, D]
    #     coords:      [N, 2] (absolute x, y)
    #     fov: int default 1536 (256 * 6, hoping to include maximum of 36 patches per grid)
    #     Returns:
    #       cell_feats: [1, D, H, W]   (for convs)
    #       flat_idx:  [N]             (which cell each patch belongs to, flattened index)
    #     """
    #     # print(patch_feats.shape, coords.shape)
    #     device = patch_feats.device
    #     _, D = patch_feats.shape
    #     grid_x, grid_y = compute_grid_size(coords, field_of_view=self.field_of_view)
    #     # H, W = self.grid_h, self.grid_w

    #     # Normalize coordinates to [0, 1] (preserving aspect ratio)
    #     x = coords[:, 0]
    #     y = coords[:, 1]

    #     x_min, x_max = x.min(), x.max()
    #     y_min, y_max = y.min(), y.max()
    #     # H, W = 

    #     x_norm = (x - x_min) / (x_max - x_min + 1e-6)
    #     y_norm = (y - y_min) / (y_max - y_min + 1e-6)

    #     # Map to integer cell indices
    #     u = (x_norm * grid_x).long().clamp(0, grid_x - 1)   # [N]
    #     v = (y_norm * grid_y).long().clamp(0, grid_y - 1)   # [N]

    #     # Flatten cell index: idx = v * W + u
    #     flat_idx = v * grid_x + u                      # [N]

    #     # Aggregate features per cell using scatter-add
    #     cell_feats_flat = torch.zeros(grid_x * grid_y, D, device=device)
    #     cell_counts_flat = torch.zeros(grid_x * grid_y, 1, device=device)

    #     cell_feats_flat.index_add_(0, flat_idx, patch_feats)
    #     cell_counts_flat.index_add_(0, flat_idx, torch.ones_like(cell_counts_flat[flat_idx]))

    #     # Avoid division by zero (empty cells)
    #     cell_counts_flat = cell_counts_flat.clamp_min(1.0)

    #     cell_feats_flat = cell_feats_flat / cell_counts_flat  # [H*W, D]

    #     # Reshape to [1, D, H, W] for conv
    #     cell_feats = cell_feats_flat.view(grid_y, grid_x, D).permute(2, 0, 1)

    #     return cell_feats, flat_idx, grid_x, grid_y
    def _build_grid(self, patch_feats: torch.Tensor, coords: torch.Tensor):
        # print(patch_feats.shape)
        device = patch_feats.device
        N, D = patch_feats.shape

        # indices don't need gradients
        with torch.no_grad():
            grid_x, grid_y = compute_grid_size(coords, field_of_view=self.field_of_view)

            x, y = coords[:, 0], coords[:, 1]
            x_min, x_max = x.aminmax()
            y_min, y_max = y.aminmax()

            # inv_dx = grid_x / (x_max - x_min + 1e-6)
            # inv_dy = grid_y / (y_max - y_min + 1e-6)

            # u = ((x - x_min) * inv_dx).to(torch.int64).clamp_(0, grid_x - 1)
            # v = ((y - y_min) * inv_dy).to(torch.int64).clamp_(0, grid_y - 1)
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
            # 2) RPN scoring on grid
            # print(grid_x, grid_y)
            # print(grid_feats.shape)
            scores = self.conv(grid_feats).squeeze(0,1)
            # 3) Map cell scores back to per-patch prior
            scores_flat = scores.view(grid_x * grid_y)   # [H*W]
            # print(scores_flat.shape)
            # print(flat_idx.shape)
            # print(flat_idx)
            proposal_map = scores_flat[flat_idx]     # [N]
            # print(proposal_map.shape)
            
            return proposal_map.unsqueeze(1)

# class MultiScaleRPN(nn.Module):
#     def __init__(self, in_channels=512, hidden_channels=64, view_scales=[768, 1536, 3072]):
#         super(MultiScaleRPN, self).__init__()
#         view_scales.sort(reverse=True)
#         self.blocks = []
#         for scale in view_scales:
#             self.blocks.append(LiteRPN(in_channels=in_channels, hidden_channels=hidden_channels, field_of_view=scale))
#         self.blocks = nn.ModuleList(self.blocks)

#     def forward(self, x, coords):
#         maps = [block(x, coords) for block in self.blocks]
#         # maps = [(each_map - each_map.mean()) / (each_map.std() + 1e-6)for each_map in maps] # normalized
#         maps = [torch.sigmoid(curr_map) for curr_map in maps]
#         # print(torch.stack(maps, dim=0).sum(dim=0).shape)
#         # return torch.stack(maps, dim=0).sum(dim=0)
#         return torch.prod(torch.stack(maps, dim=0), dim=0)

class MultiScaleRPN(nn.Module):
    def __init__(self, in_channels=512, hidden_channels=64, view_scales=[768, 1536, 3072]):
        super(MultiScaleRPN, self).__init__()
        view_scales.sort(reverse=True)
        self.view_scales = view_scales
        self.blocks = []
        for scale in view_scales:
            self.blocks.append(LiteRPN(in_channels=in_channels, hidden_channels=hidden_channels, field_of_view=scale))
        self.blocks = nn.ModuleList(self.blocks)
        # self.dropout = nn.Dropout(0.25)
        self.dropout = nn.Dropout1d(0.25)
    
    def forward(self, x, coords):
        h = x
        coarse_maps = {}
        for i in range(len(self.view_scales)):
            p_map = self.blocks[i](h, coords)
            # print(p_map.shape)
            coarse_maps[self.view_scales[i]] = p_map.T
            # print(h.shape, p_map.shape)
            h_p = self.dropout((h * torch.sigmoid(p_map)).T).T
            h = h + h_p
            # h = h * torch.sigmoid(p_map)
            # print(h.shape)
        # h = self.dropout(h)
        # logits = torch.transpose(p_map, 1, 0)
        # A_raw = A
        # p_map = F.softmax(p_map, dim=1)  # softmax over K

        # logits = torch.mm(p_map, x)  # ATTENTION_BRANCHESxM
        return h, coarse_maps

# class MultiScaleRPN(nn.Module):
#     def __init__(self, in_channels=512, hidden_channels=64, view_scales=[768, 1536, 3072]):
#         super(MultiScaleRPN, self).__init__()
#         view_scales.sort(reverse=True)
#         self.view_scales = view_scales
#         self.blocks = []
#         for scale in view_scales:
#             self.blocks.append(LiteRPN(in_channels=in_channels, hidden_channels=hidden_channels, field_of_view=scale))
#         self.blocks = nn.ModuleList(self.blocks)
#         # self.dropout = nn.Dropout(0.25)
#         self.raw_alpha = nn.Parameter(torch.zeros(len(view_scales)))
    
#     def forward(self, h, coords):
#         coarse_maps = {}
#         proposals = []
#         for i in range(len(self.view_scales)):
#             p_map = self.blocks[i](h, coords)
#             # print(p_map.shape)
#             coarse_maps[self.view_scales[i]] = p_map.T
#             proposals.append(p_map)
#             # h = h * (1 + torch.sigmoid(p_map))
#             # h = h * torch.sigmoid(p_map)
#             # print(h.shape)
#         # h = self.dropout(h)
#         proposals_stack = torch.stack(proposals, dim=-1)  # [N, K]

#         # softmax over scales → convex combination
#         alpha = F.softmax(self.raw_alpha, dim=0)          # [K]
#         ms_proposal = (proposals_stack * alpha.unsqueeze(0)).sum(dim=-1)  # [N]

#         return ms_proposal, coarse_maps


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

        # self.classifier = nn.Sequential(
        #     nn.Linear(self.hid_dim_1*self.attn_branches, n_classes),
        #     # nn.Sigmoid()
        # )
        # Learnable scaling for the prior
        # if use_proposal:
        #     self.gamma = nn.Parameter(torch.tensor(0.0))
        # self.use_proposal = use_proposal
    
    def forward(self, h):
        h = self.feature_extractor(h)  # KxM

        A_V = self.attention_V(h)  # KxL
        A_U = self.attention_U(h)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        # print(A.shape)
        # if self.use_proposal and proposal is not None:
        #     # Optionally normalize prior
        #     # prior_norm = (prior - prior.mean()) / (prior.std() + 1e-6)
        #     # print('prior:', prior_norm.shape)
        #     # print('gamma+pri:', (self.gamma * prior_norm).shape)
        #     # A = A + self.gamma * prior_norm.unsqueeze(-1)
        #     # A = A + prior_norm.unsqueeze(-1)
        #     A += proposal
        #     # print(A.shape)
        # if self.use_proposal and ms_proposal is not None:
        #     ms_proposal = ms_proposal.view(-1, 1)             # [N,1]
        #     # normalize per slide
        #     # proposal_norm = (ms_proposal - ms_proposal.mean()) / (ms_proposal.std() + 1e-6)  # [N,1]
        #     # gamma = F.softplus(self.raw_gamma)
        #     A = A + F.softplus(self.gamma) * ms_proposal

        # print(x.shape, A.shape)
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over K

        attn_feats = torch.mm(A, h)  # ATTENTION_BRANCHESxM

        # logits = self.classifier(Z)
        return attn_feats, A_raw


class ABMIL_RPN(nn.Module):
    def __init__(self, in_channels=768, n_classes=2, rpn_hidden=64, mil_hidden_1=512, mil_hidden_2=128, attn_branches=1, view_scales=[1536, 2048, 3072], reduction_size=512):
        super(ABMIL_RPN, self).__init__()
        # self.surv = surv
        # self.field_of_view = field_of_view
        # self.rpn_head = LiteRPN(in_channels=in_channels, hidden_channels=rpn_hidden)
        # print(view_scales)
        if in_channels != reduction_size:
            self.fc = nn.Linear(in_channels, reduction_size)
        else:
            self.fc = None
        self.multiscale_rpn = MultiScaleRPN(in_channels=reduction_size, hidden_channels=rpn_hidden, view_scales=view_scales)
        self.abmil_head = ABMILHead(in_channels=reduction_size, mil_hidden_1=mil_hidden_1, mil_hidden_2=mil_hidden_2)
        self.classifier = nn.Sequential(
            nn.Linear(mil_hidden_1*attn_branches, n_classes),
        )
        # self.alpha = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x, coords, vis_heatmap=False, vis_coarse_map=False):
        if not vis_heatmap:
            x = x.squeeze(0)
        h = x
        device = x.device
        coords = coords.squeeze(0).to(device)

        if self.fc != None:
            h = self.fc(h)
        # build RPN grid
        # cell_feats, flat_idx, grid_x, grid_y = self._build_grid(h, coords)
        # print(cell_feats.shape)

        # 2) RPN scoring on grid
        # cell_scores = self.rpn_head(cell_feats)   # [1, H, W]
        # cell_scores = cell_scores.squeeze(0)      # [H, W]

        # 3) Map cell scores back to per-patch prior
        # cell_scores_flat = cell_scores.view(grid_x * grid_y)   # [H*W]
        # proposal_map = cell_scores_flat[flat_idx]     # [N]
        # patch_prior_gated = torch.sigmoid(patch_prior)
        h_coarse, coarse_map = self.multiscale_rpn(h, coords)
        # print('h out', h.shape)
        # proposal_gated = torch.sigmoid(proposal_map)
        # print(h.shape, proposal_gated.shape)
        # h_gated = h * (1 + torch.sigmoid(proposal_map))
        

        # 4) ABMIL with prior
        attn_feats, A_raw = self.abmil_head(h_coarse)
        logits = self.classifier(attn_feats)
        # print(A_raw.squeeze(0).shape, patch_prior.shape)
        # cos_sim = F.cosine_similarity(A_raw, patch_prior.unsqueeze(0))

        # if vis_heatmap:
        #     return A_raw, logits
        # else:
        #     if self.surv:
        #         hazards = torch.sigmoid(logits)
        #         S = torch.cumprod(1 - hazards, dim=1)
        #         return hazards, S, (A_raw, proposal_map.unsqueeze(0)) # match output dim
        #     else:
        #         return logits, (A_raw, proposal_map.unsqueeze(0))

        
        if vis_heatmap:
            return A_raw, logits
        elif vis_coarse_map:
            return coarse_map, logits
        else:
            # if self.surv:
                # hazards = torch.sigmoid(logits)
                # S = torch.cumprod(1 - hazards, dim=1)
                # return hazards, S, None # match output dim
            # else:
            return logits, None


class DSMIL_RPN(nn.Module):
    def __init__(self, i_classifier, b_classifier, in_channels, n_classes=2, rpn_hidden=64, view_scales=[1536, 2048, 3072], surv=False, reduction_size=512):
        super(DSMIL_RPN, self).__init__()
        if in_channels != reduction_size:
            self.fc = nn.Linear(in_channels, reduction_size)
        else:
            self.fc = None
        self.multiscale_rpn = MultiScaleRPN(in_channels=reduction_size, hidden_channels=rpn_hidden, view_scales=view_scales)
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        self.fcc = nn.Conv1d(n_classes, n_classes, kernel_size=reduction_size)
        self.surv = surv
        
    def forward(self, x, coords, vis_heatmap=False, vis_coarse_map=False):
        if not vis_heatmap:
            x = x.squeeze(0)
        h = x
        device = x.device
        coords = coords.squeeze(0).to(device)

        if self.fc != None:
            h = self.fc(h)

        # build RPN grid
        # cell_feats, flat_idx, grid_x, grid_y = self._build_grid(h, coords)
        # print(cell_feats.shape)

        # 2) RPN scoring on grid
        # cell_scores = self.rpn_head(cell_feats)   # [1, H, W]
        # cell_scores = cell_scores.squeeze(0)      # [H, W]

        # 3) Map cell scores back to per-patch prior
        # cell_scores_flat = cell_scores.view(grid_x * grid_y)   # [H*W]
        # proposal_map = cell_scores_flat[flat_idx]     # [N]
        # patch_prior_gated = torch.sigmoid(patch_prior)
        h_coarse, coarse_map = self.multiscale_rpn(h, coords)
        # print(h_coarse.shape)
        feats, classes = self.i_classifier(h)
        feats_c, classes_c = self.i_classifier(h_coarse)
        # print(feats.shape)
        # A_raw, B = self.b_classifier(feats, classes)
        # print(classes)
        A_raw, B = self.b_classifier(feats+feats_c, classes+classes_c)
        # A_raw_c, B_c = self.b_classifier(feats_c, classes_c)
        # print(B.shape)
        prediction_bag = self.fcc(B).view(1,-1)

        
        if vis_heatmap:
            return A_raw, prediction_bag
        elif vis_coarse_map:
            return coarse_map, prediction_bag
        else:
            if self.surv:
                return prediction_bag, coarse_map # match output dim
            else:
                return classes, prediction_bag, A_raw, B


class CLAMSB_RPN(nn.Module):
    def __init__(self, in_channels=768, n_classes=2, rpn_hidden=64, mil_hidden_1=512, mil_hidden_2=128, attn_branches=1, view_scales=[1536, 2048, 3072], reduction_size=512):
        super(CLAMSB_RPN, self).__init__()
        if in_channels != reduction_size:
            self.fc = nn.Linear(in_channels, reduction_size)
        else:
            self.fc = None
        self.multiscale_rpn = MultiScaleRPN(in_channels=reduction_size, hidden_channels=rpn_hidden, view_scales=view_scales)
        self.clamsb_head = CLAM_SB_Head(n_classes=n_classes, embed_dim=reduction_size)
        self.classifier = nn.Sequential(
            nn.Linear(mil_hidden_1*attn_branches, n_classes),
        )
    
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
            # if self.surv:
                # hazards = torch.sigmoid(logits)
                # S = torch.cumprod(1 - hazards, dim=1)
                # return hazards, S, None # match output dim
            # else:
            return logits, total_inst_loss

class CLAMMB_RPN(nn.Module):
    def __init__(self, in_channels=768, n_classes=2, rpn_hidden=64, mil_hidden_1=512, mil_hidden_2=128, attn_branches=1, view_scales=[1536, 2048, 3072], reduction_size=512):
        super(CLAMMB_RPN, self).__init__()
        if in_channels != reduction_size:
            self.fc = nn.Linear(in_channels, reduction_size)
        else:
            self.fc = None
        self.multiscale_rpn = MultiScaleRPN(in_channels=reduction_size, hidden_channels=rpn_hidden, view_scales=view_scales)
        self.clammb_head = CLAM_MB_Head(n_classes=n_classes, embed_dim=reduction_size)
        self.n_classes = n_classes
        bag_classifiers = [nn.Linear(mil_hidden_1, 1) for i in range(n_classes)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
    
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
            # if self.surv:
                # hazards = torch.sigmoid(logits)
                # S = torch.cumprod(1 - hazards, dim=1)
                # return hazards, S, None # match output dim
            # else:
            return logits, total_inst_loss