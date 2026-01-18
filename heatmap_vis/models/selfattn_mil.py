import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

from torch import einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
from new_amil.utils.universal_utils import positional_encoding, positional_encoding2d


# class AttnGate(nn.Module):
#     def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
#         super(AttnGate, self).__init__()
#         self.attention_a = [
#             nn.Linear(L, D),
#             # nn.ReLU()]
#             nn.Tanh()]

#         self.attention_b = [nn.Linear(L, D),
#                             nn.Sigmoid()]
#         if dropout:
#             self.attention_a.append(nn.Dropout(0.25))
#             self.attention_b.append(nn.Dropout(0.25))

#         self.attention_a = nn.Sequential(*self.attention_a)
#         self.attention_b = nn.Sequential(*self.attention_b)

#         self.attention_c = nn.Linear(D, n_classes)

#     def forward(self, x):
#         a = self.attention_a(x)
#         b = self.attention_b(x)
#         A = a.mul(b)
#         A = self.attention_c(A)  # N x n_classes
#         return A, x

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias != None:
                m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class SelfAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_k = nn.Linear(dim, inner_dim , bias=False)
        self.to_v = nn.Linear(dim, inner_dim , bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x_qkv):
        b, n, _, h = *x_qkv.shape, self.heads

        k = self.to_k(x_qkv)
        k = rearrange(k, 'b n (h d) -> b h n d', h = h)

        v = self.to_v(x_qkv)
        v = rearrange(v, 'b n (h d) -> b h n d', h = h)

        q = self.to_q(x_qkv[:, 0].unsqueeze(1))
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)


        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out, attn

class AttnLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, emb_dim=768):
        super().__init__()
        self.norm_layer = norm_layer(emb_dim)
        self.attn = SelfAttention(
            dim = emb_dim,
            heads = 8,
            dim_head = emb_dim//8,
            dropout = 0.1,
        )

    def forward(self, h):
        out, A = self.attn(self.norm_layer(h))
        return h + out, A


class SelfAttnMIL(nn.Module):
    def __init__(self, in_dim=512, pos_enc_2d=True, n_classes=2):
        super(SelfAttnMIL, self).__init__()
        size = [in_dim, 512, 256]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25)]
        self.fc = nn.Sequential(*fc)
        self.attn_layer1 = AttnLayer(emb_dim=size[1])
        self.attn_layer2 = AttnLayer(emb_dim=size[1])
        self.norm_layer = nn.LayerNorm(size[1])
        # attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=0.25, n_classes=1)
        # fc.append(attention_net)
        # self.attention_net = nn.Sequential(*fc)
        self.classifier = nn.Linear(size[1], n_classes)
        self.cls_token = nn.Parameter(torch.randn(1, 1, size[1]))
        self.pos_enc_2d = pos_enc_2d
        initialize_weights(self)

    def relocate(self, device):
        print('[model] Num. of GPU:', torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attn_layer1 = nn.DataParallel(self.attn_layer1, device_ids=device_ids).to(device)
            self.attn_layer2 = nn.DataParallel(self.attn_layer2, device_ids=device_ids).to(device)
            self.norm_layer = nn.DataParallel(self.norm_layer, device_ids=device_ids).to(device)
        else:
            self.attn_layer1 = self.attn_layer1.to(device)
            self.attn_layer2 = self.attn_layer2.to(device)
            self.norm_layer = self.norm_layer.to(device)
        self.fc = self.fc.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self, h, coords, vis_heatmap=False):
        device = h.device
        if vis_heatmap:
            h = h.unsqueeze(0)
            coords = coords.unsqueeze(0)
        x, y = coords[:,:,0].unsqueeze(1), coords[:,:,1].unsqueeze(1) # [1,1,N], h:[1, N, D]
        # cls token
        h = self.fc(h)
        B = h.shape[0]
        # print(h.shape)
        cls_tokens = self.cls_token.expand(B, -1, -1).to(device)
        h = torch.cat((cls_tokens, h), dim=1)
        # print(h.shape)
    
        # attn 1
        h, _ = self.attn_layer1(h)
        # print(h.shape)
        # pos enc
        if self.pos_enc_2d:
            h = positional_encoding2d(h, h.shape[2], x, y, device)
        else:
            h = positional_encoding(h, h.shape[2], h.shape[1], device)
        # attn 2
        h, A = self.attn_layer2(h)  # A: [1, 1, N]
        h = self.norm_layer(h)[:,0]
        h = self.classifier(h)
        if vis_heatmap:
            return A.squeeze(0)[:,:,1:], h
        else:
            return h

