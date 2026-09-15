import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mspn import ABMILHead
from models.layers import GlobalAttention, GlobalGatedAttention, create_mlp
from utils.universal_utils import initialize_weights

# class ABMIL(nn.Module):
#     def __init__(self, in_dim=768, n_classes=2, surv=False):
#         super(ABMIL, self).__init__()
#         self.M = in_dim
#         self.L = 128
#         self.ATTENTION_BRANCHES = 1
#         self.surv = surv

#         # self.feature_extractor_part1 = nn.Sequential(
#         #     nn.Conv2d(1, 20, kernel_size=5),
#         #     nn.ReLU(),
#         #     nn.MaxPool2d(2, stride=2),
#         #     nn.Conv2d(20, 50, kernel_size=5),
#         #     nn.ReLU(),
#         #     nn.MaxPool2d(2, stride=2)
#         # )

#         # self.feature_extractor_part2 = nn.Sequential(
#         #     nn.Linear(50 * 4 * 4, self.M),
#         #     nn.ReLU(),
#         # )

#         self.attention_V = nn.Sequential(
#             nn.Linear(self.M, self.L), # matrix V
#             nn.Tanh()
#         )

#         self.attention_U = nn.Sequential(
#             nn.Linear(self.M, self.L), # matrix U
#             nn.Sigmoid()
#         )

#         self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

#         self.classifier = nn.Sequential(
#             nn.Linear(self.M*self.ATTENTION_BRANCHES, n_classes),
#             # nn.Sigmoid()
#         )
    
#     def forward(self, x, vis_heatmap=False):
#         # if not vis_heatmap:
#         x = x.squeeze(0)
#         H = x
#         # H = self.feature_extractor_part1(x)
#         # H = H.view(-1, 50 * 4 * 4)
#         # H = self.feature_extractor_part2(H)  # KxM

#         A_V = self.attention_V(H)  # KxL
#         A_U = self.attention_U(H)  # KxL
#         A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
#         A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
#         A_raw = A
#         A = F.softmax(A, dim=1)  # softmax over K

#         Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

#         logits = self.classifier(Z)

#         if vis_heatmap:
#             return A_raw, logits
#         else:
#             if self.surv:
#                 hazards = torch.sigmoid(logits)
#                 S = torch.cumprod(1 - hazards, dim=1)
#                 return hazards, S, None # match output dim
#             else:
#                 return logits, None

class ABMIL(nn.Module):
    def __init__(self, in_channels=768, n_classes=2, mil_hidden_1=512, mil_hidden_2=128, attn_branches=1):
        super(ABMIL, self).__init__()
        self.abmil_head = ABMILHead(in_channels=in_channels, mil_hidden_1=mil_hidden_1, mil_hidden_2=mil_hidden_2, attn_branches=attn_branches)
        self.classifier = nn.Sequential(
            nn.Linear(mil_hidden_1*attn_branches, n_classes),
        )
    
    def forward(self, x, vis_heatmap=False):
        # if not vis_heatmap:
        x = x.squeeze(0)
        attn_feats, A_raw = self.abmil_head(x)
        logits = self.classifier(attn_feats)

        if vis_heatmap:
            return A_raw, logits
        else:
            # if self.surv:
                # hazards = torch.sigmoid(logits)
                # S = torch.cumprod(1 - hazards, dim=1)
                # return hazards, S, None # match output dim
            # else:
            return logits, None

class ABMILMSCat(nn.Module):
    def __init__(self, in_channels=768, n_classes=2, mil_hidden_1=512, mil_hidden_2=64, attn_branches=1):
        super(ABMILMSCat, self).__init__()
        self.abmil_head_5x = ABMILHead(in_channels=in_channels, mil_hidden_1=mil_hidden_1, mil_hidden_2=mil_hidden_2, attn_branches=attn_branches)
        self.abmil_head_10x = ABMILHead(in_channels=in_channels, mil_hidden_1=mil_hidden_1, mil_hidden_2=mil_hidden_2, attn_branches=attn_branches)
        self.abmil_head_20x = ABMILHead(in_channels=in_channels, mil_hidden_1=mil_hidden_1, mil_hidden_2=mil_hidden_2, attn_branches=attn_branches)
        self.ms_encoder = nn.Sequential(
            nn.Linear(mil_hidden_1, mil_hidden_2),
        )
        self.cs_attn = nn.Sequential(
            nn.Conv2d(mil_hidden_2, int(mil_hidden_2/2), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(mil_hidden_2/2), 1, kernel_size=3, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(int(mil_hidden_2*3), n_classes),
        )
    
    def forward(self, x, vis_heatmap=False):
        # if not vis_heatmap:
        # x = x.squeeze(0)
        x_5x, x_10x, x_20x = x
        x_5x = x_5x.squeeze(0)
        x_10x = x_10x.squeeze(0)
        x_20x = x_20x.squeeze(0)

        attn_feats_5x, A_raw_5x = self.abmil_head_5x(x_5x)
        attn_feats_10x, A_raw_10x = self.abmil_head_10x(x_10x)
        attn_feats_20x, A_raw_20x = self.abmil_head_20x(x_20x)

        ms_feats = torch.concat((self.ms_encoder(attn_feats_5x), self.ms_encoder(attn_feats_10x), self.ms_encoder(attn_feats_20x)), dim=1)
        # print(ms_feats.shape)
        # cs_attn = self.cs_attn(ms_feats.view(1, ms_feats.shape[0], ms_feats.shape[1], 1))
        # print(cs_attn.shape)
        # cs_attn = F.softmax(cs_attn.view(1, 3), dim=1)
        # cs_attn_feats = torch.mm(cs_attn, ms_feats.T)

        logits = self.classifier(ms_feats)

        if vis_heatmap:
            return (A_raw_5x, A_raw_10x, A_raw_20x), logits
        else:
            # if self.surv:
                # hazards = torch.sigmoid(logits)
                # S = torch.cumprod(1 - hazards, dim=1)
                # return hazards, S, None # match output dim
            # else:
            return logits, None

class ABMILMS(nn.Module):
    def __init__(self, in_channels=768, n_classes=2, mil_hidden_1=512, mil_hidden_2=64, attn_branches=1):
        super(ABMILMS, self).__init__()
        self.abmil_head_5x = ABMILHead(in_channels=in_channels, mil_hidden_1=mil_hidden_1, mil_hidden_2=mil_hidden_2, attn_branches=attn_branches)
        self.abmil_head_10x = ABMILHead(in_channels=in_channels, mil_hidden_1=mil_hidden_1, mil_hidden_2=mil_hidden_2, attn_branches=attn_branches)
        self.abmil_head_20x = ABMILHead(in_channels=in_channels, mil_hidden_1=mil_hidden_1, mil_hidden_2=mil_hidden_2, attn_branches=attn_branches)
        self.ms_encoder = nn.Sequential(
            nn.Linear(mil_hidden_1, mil_hidden_2),
        )
        self.cs_attn = nn.Sequential(
            nn.Conv2d(mil_hidden_2, int(mil_hidden_2/2), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(mil_hidden_2/2), 1, kernel_size=3, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(mil_hidden_2, n_classes),
        )
    
    def forward(self, x, vis_heatmap=False):
        # if not vis_heatmap:
        # x = x.squeeze(0)
        x_5x, x_10x, x_20x = x
        x_5x = x_5x.squeeze(0)
        x_10x = x_10x.squeeze(0)
        x_20x = x_20x.squeeze(0)

        attn_feats_5x, A_raw_5x = self.abmil_head_5x(x_5x)
        attn_feats_10x, A_raw_10x = self.abmil_head_10x(x_10x)
        attn_feats_20x, A_raw_20x = self.abmil_head_20x(x_20x)

        ms_feats = torch.concat((self.ms_encoder(attn_feats_5x), self.ms_encoder(attn_feats_10x), self.ms_encoder(attn_feats_20x)), dim=0).T
        # print(ms_feats.shape)
        cs_attn = self.cs_attn(ms_feats.view(1, ms_feats.shape[0], ms_feats.shape[1], 1))
        # print(cs_attn.shape)
        cs_attn = F.softmax(cs_attn.view(1, 3), dim=1)
        cs_attn_feats = torch.mm(cs_attn, ms_feats.T)

        logits = self.classifier(cs_attn_feats)

        if vis_heatmap:
            return (A_raw_5x, A_raw_10x, A_raw_20x), logits
        else:
            # if self.surv:
                # hazards = torch.sigmoid(logits)
                # S = torch.cumprod(1 - hazards, dim=1)
                # return hazards, S, None # match output dim
            # else:
            return logits, None
        

class ABMILPretrained(nn.Module):
    """
    ABMIL (Attention-based Multiple Instance Learning) model.

    This class implements the core ABMIL architecture, which uses a patch embedding MLP,
    followed by a global attention or gated attention mechanism, and an optional classification head.

    Args:
        in_dim (int): Input feature dimension for each instance (default: 1024).
        embed_dim (int): Embedding dimension after patch embedding (default: 512).
        num_fc_layers (int): Number of fully connected layers in the patch embedding MLP (default: 1).
        dropout (float): Dropout rate applied in the MLP and attention layers (default: 0.25).
        attn_dim (int): Dimension of the attention mechanism (default: 384).
        gate (int): Whether to use gated attention (True) or standard attention (False) (default: True).
        num_classes (int): Number of output classes for the classification head (default: 2).
    """

    def __init__(
            self,
            in_dim: int = 1024,
            embed_dim: int = 512,
            num_fc_layers: int = 1,
            dropout: float = 0.25,
            attn_dim: int = 384,
            gate: int = True,
            num_classes: int = 2,
    ):
        super(ABMILPretrained, self).__init__()
        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] *
                     (num_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False
        )

        attn_func = GlobalGatedAttention if gate else GlobalAttention
        self.global_attn = attn_func(
            L=embed_dim,
            D=attn_dim,
            dropout=dropout,
            num_classes=1
        )

        if num_classes > 0:
            self.classifier = nn.Linear(embed_dim, num_classes)
        # self.initialize_weights()
        initialize_weights(self)

    def forward_attention(self, h: torch.Tensor, attn_mask=None, attn_only=True) -> torch.Tensor:
        """
        Compute the attention scores (and optionally the embedded features) for the input instances.

        Args:
            h (torch.Tensor): Input tensor of shape [B, M, D], where B is the batch size,
                M is the number of instances (patches), and D is the input feature dimension.
            attn_mask (torch.Tensor, optional): Optional attention mask of shape [B, M], where 1 indicates
                valid positions and 0 indicates masked positions. If provided, masked positions are set to
                a very large negative value before softmax.
            attn_only (bool, optional): If True, return only the attention scores (A).
                If False, return a tuple (h, A) where h is the embedded features and A is the attention scores.

        Returns:
            torch.Tensor: If attn_only is True, returns the attention scores tensor of shape [B, K, M],
                where K is the number of attention heads (usually 1). If attn_only is False, returns a tuple
                (h, A) where h is the embedded features of shape [B, M, D'] and A is the attention scores.
        """
        h = self.patch_embed(h)
        A = self.global_attn(h)  # B x M x K
        A = torch.transpose(A, -2, -1)  # B x K x M
        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(dim=1) * torch.finfo(A.dtype).min

        if attn_only:
            return A
        return h, A

    def forward_features(self, h: torch.Tensor, attn_mask=None, return_attention: bool = True) -> torch.Tensor:
        """
        Compute bag-level features using attention pooling.

        Args:
            h (torch.Tensor): [B, M, D] input features.
            attn_mask (torch.Tensor, optional): Attention mask.

        Returns:
            Tuple[torch.Tensor, dict]: Bag features [B, D] and attention weights.
        """
        h, A_base = self.forward_attention(h, attn_mask=attn_mask, attn_only=False)  # A == B x K x M
        A = F.softmax(A_base, dim=-1)  # softmax over N
        h = torch.bmm(A, h).squeeze(dim=1)  # B x K x C --> B x C
        log_dict = {'attention': A_base if return_attention else None}
        return h, log_dict

    def forward_head(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B x D]-dim torch.Tensor.

        Returns:
            logits: [B x num_classes]-dim torch.Tensor.
        """
        logits = self.classifier(h)
        return logits

    def forward(self, h: torch.Tensor,
                loss_fn: nn.Module = None,
                label: torch.LongTensor = None,
                attn_mask=None,
                return_attention: bool = False,
                return_slide_feats: bool = False) -> torch.Tensor:
        """
        Forward pass for ABMIL.

        Args:
            h: [B, M, D] input features.
            loss_fn: Optional loss function.
            label: Optional labels.
            attn_mask: Optional attention mask.

        Returns:
            Tuple of (results_dict, log_dict) with logits and loss.
        """
        wsi_feats, log_dict = self.forward_features(h, attn_mask=attn_mask, return_attention=return_attention)
        logits = self.forward_head(wsi_feats)
        # cls_loss = MIL.compute_loss(loss_fn, logits, label)
        # results_dict = {'logits': logits, 'loss': cls_loss}
        # log_dict['loss'] = cls_loss.item() if cls_loss is not None else -1
        # if return_slide_feats:
            # log_dict['slide_feats'] = wsi_feats
        # return results_dict, log_dict
        return logits, None

class ABMILPreMS(nn.Module):
    def __init__(self, in_channels=768, n_classes=2, abmil_head=None,  mil_hidden_1=512, mil_hidden_2=64):
        super(ABMILPreMS, self).__init__()
        self.abmil_head_5x = abmil_head
        self.abmil_head_10x = abmil_head
        self.abmil_head_20x = abmil_head
        self.ms_encoder = nn.Sequential(
            nn.Linear(mil_hidden_1, mil_hidden_2),
        )
        self.cs_attn = nn.Sequential(
            nn.Conv2d(mil_hidden_2, int(mil_hidden_2/2), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(mil_hidden_2/2), 1, kernel_size=3, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(mil_hidden_2, n_classes),
        )
    
    def forward(self, x, vis_heatmap=False):
        # if not vis_heatmap:
        # x = x.squeeze(0)
        x_5x, x_10x, x_20x = x
        # x_5x = x_5x.squeeze(0)
        # x_10x = x_10x.squeeze(0)
        # x_20x = x_20x.squeeze(0)
        # forward_features(h, attn_mask=attn_mask, return_attention=return_attention)
        attn_feats_5x, _ = self.abmil_head_5x.forward_features(x_5x)
        attn_feats_10x, _ = self.abmil_head_10x.forward_features(x_10x)
        attn_feats_20x, _ = self.abmil_head_20x.forward_features(x_20x)

        ms_feats = torch.concat((self.ms_encoder(attn_feats_5x), self.ms_encoder(attn_feats_10x), self.ms_encoder(attn_feats_20x)), dim=0).T
        # print(ms_feats.shape)
        cs_attn = self.cs_attn(ms_feats.view(1, ms_feats.shape[0], ms_feats.shape[1], 1))
        # print(cs_attn.shape)
        cs_attn = F.softmax(cs_attn.view(1, 3), dim=1)
        cs_attn_feats = torch.mm(cs_attn, ms_feats.T)

        logits = self.classifier(cs_attn_feats)

        if vis_heatmap:
            return _, logits
        else:
            # if self.surv:
                # hazards = torch.sigmoid(logits)
                # S = torch.cumprod(1 - hazards, dim=1)
                # return hazards, S, None # match output dim
            # else:
            return logits, None

class ABMILPreMSCat(nn.Module):
    def __init__(self, in_channels=768, n_classes=2, abmil_head=None, mil_hidden_1=512, mil_hidden_2=64):
        super(ABMILPreMSCat, self).__init__()
        self.abmil_head_5x = abmil_head
        self.abmil_head_10x = abmil_head
        self.abmil_head_20x = abmil_head
        self.ms_encoder = nn.Sequential(
            nn.Linear(mil_hidden_1, mil_hidden_2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(int(mil_hidden_2*3), n_classes),
        )
    
    def forward(self, x, vis_heatmap=False):
        # if not vis_heatmap:
        # x = x.squeeze(0)
        x_5x, x_10x, x_20x = x
        # x_5x = x_5x.squeeze(0)
        # x_10x = x_10x.squeeze(0)
        # x_20x = x_20x.squeeze(0)

        attn_feats_5x, _ = self.abmil_head_5x.forward_features(x_5x)
        attn_feats_10x, _ = self.abmil_head_10x.forward_features(x_10x)
        attn_feats_20x, _ = self.abmil_head_20x.forward_features(x_20x)

        ms_feats = torch.concat((self.ms_encoder(attn_feats_5x), self.ms_encoder(attn_feats_10x), self.ms_encoder(attn_feats_20x)), dim=1)
        # print(ms_feats.shape)
        # cs_attn = self.cs_attn(ms_feats.view(1, ms_feats.shape[0], ms_feats.shape[1], 1))
        # print(cs_attn.shape)
        # cs_attn = F.softmax(cs_attn.view(1, 3), dim=1)
        # cs_attn_feats = torch.mm(cs_attn, ms_feats.T)

        logits = self.classifier(ms_feats)

        if vis_heatmap:
            return _, logits
        else:
            # if self.surv:
                # hazards = torch.sigmoid(logits)
                # S = torch.cumprod(1 - hazards, dim=1)
                # return hazards, S, None # match output dim
            # else:
            return logits, None