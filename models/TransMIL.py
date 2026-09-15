import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention

from torch import einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )
        # self.attn = CrossAttention(
        #     dim = dim,
        #     heads = 8,
        #     dim_head = dim//8,
        #     dropout = 0.1,
        # )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        # print('1', x.shape)
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        # print('2', x.shape)
        # print('fla', x.flatten(2).shape)
        x = x.flatten(2).transpose(1, 2)
        # print('3', x.shape)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        # print('4', x.shape)
        return x


class TransMIL_Large(nn.Module):
    def __init__(self, in_dim=1536, n_classes=2):
        super(TransMIL_Large, self).__init__()
        self.pos_layer = PPEG(dim=768)
        self._fc1 = nn.Sequential(nn.Linear(in_dim, 768), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 768))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=768)
        self.layer2 = TransLayer(dim=768)
        self.norm = nn.LayerNorm(768)
        self._fc2 = nn.Linear(768, self.n_classes)
        # self.surv = surv

    def forward(self, x):

        # h = kwargs['data'].float() #[B, n, 1024]
        h = x.float()

        h = self._fc1(h)  # [B, n, 512]
        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer xx1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token
        h = self.norm(h)[:, 0]

        # ---->predict
        logits = self._fc2(h)  # [B, n_classes]
        # Y_hat = torch.argmax(logits, dim=1)
        # Y_prob = F.softmax(logits, dim = 1)
        # results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        # return results_dict
        return logits, None


class TransMIL(nn.Module):
    def __init__(self, in_dim=1536, n_classes=2, surv=False):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(in_dim, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)
        self.surv = surv


    def forward(self, x):

        # h = kwargs['data'].float() #[B, n, 1024]
        h = x.float()
        
        h = self._fc1(h) #[B, n, 512]
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer xx1
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]

        #---->cls_token
        h = self.norm(h)[:,0]

        #---->predict
        logits = self._fc2(h) #[B, n_classes]
        # Y_hat = torch.argmax(logits, dim=1)
        # Y_prob = F.softmax(logits, dim = 1)
        # results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        # return results_dict
        if self.surv:
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, None # match output dim
        else:
            return logits, None

class CrossAttention(nn.Module):
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
        return out