import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from torch import einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
from utils.universal_utils import positional_encoding2d, positional_encoding, get_pos_emb, norm_coords, PE2D

""" In this project, we use AMIL from PORPOISE to be the baseline"""


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes (experimental usage for multiclass MIL)
"""


class AttnGate(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(AttnGate, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x
    

class SelfAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_k = nn.Linear(dim, inner_dim , bias=False)
        self.to_v = nn.Linear(dim, inner_dim , bias=False)
        self.to_q = nn.Linear(dim, inner_dim , bias=False)

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

class SelfAttnLayer(nn.Module):
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
        out = self.attn(self.norm_layer(h))
        return h + out


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias != None:
                m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class DBAMIL(nn.Module):
    def __init__(self, in_dim=512, n_classes=2, surv=False):
        super(DBAMIL, self).__init__()
        if in_dim > 512:
            size = [in_dim, 512, 256]
        else:
            size = [in_dim, 256, 128]
        self.n_classes = n_classes
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25)]
        self.fc = nn.Sequential(*fc)
        if n_classes == 2:
            self.attention_net = AttnGate(L=size[1], D=size[2], dropout=0.25, n_classes=1)
        else:
            self.attention_net = AttnGate(L=size[1], D=size[2], dropout=0.25, n_classes=n_classes)
        if n_classes != 2:
            self.projector = nn.Linear(1, n_classes)
        self.classifier = nn.Linear(size[1], n_classes)
        self.surv = surv
        initialize_weights(self)

    def relocate(self, device):
        print('[model] Num. of GPU:', torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to(device)
            # if self.n_classes != 2:
                # self.projector = nn.DataParallel(self.projector, device_ids=device_ids).to(device)
            self.fc = nn.DataParallel(self.fc, device_ids=device_ids).to(device)
        else:
            self.attention_net = self.attention_net.to(device)
            # if self.n_classes != 2:
                # self.projector = self.projector.to(device)
            self.fc = self.fc.to(device)

        self.classifier = self.classifier.to(device)

    def forward(self, h, patch_level=None, vis_heatmap=False):
        if not vis_heatmap and not patch_level:
            h = h.squeeze(dim=0)
        # print(x.shape)
        # print(h.shape)
        h = self.fc(h)

        # slide-lvl
        A, h = self.attention_net(h)
        # if patch_level:
        #     if self.n_classes == 2:
        #         return A
        #     else:
        #         A_logit = self.projector(A)
        #         return A_logit
        # if self.n_classes != 2:
        #     A = torch.mean(A, dim=1, keepdim=True)
        if patch_level:
            # print('in mdl', A.shape)
            return A
        # if not patch-level, mean multi-class for bag-level
        if self.n_classes != 2:
            A = torch.mean(A, dim=1, keepdim=True)
        
        A = torch.transpose(A, 1, 0)
        # print(A.shape)
        A_raw = A
        A = F.softmax(A, dim=1)
        M = torch.mm(A, h)
        h = self.classifier(M)
        if self.surv:
            hazards = torch.sigmoid(h)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, A_raw
        else:
            return h, A_raw

class PathMIL(nn.Module):
    def __init__(self, in_dim=512, n_classes=2, surv=False):
        super(PathMIL, self).__init__()
        size = [in_dim, 512, 256]
        self.n_classes = n_classes
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25)]
        self.fc = nn.Sequential(*fc)
        if n_classes == 2:
            self.attntion_net = AttnGate(L=size[1], D=size[2], dropout=0.25, n_classes=1)
        else:
            self.attntion_net = AttnGate(L=size[1], D=size[2], dropout=0.25, n_classes=n_classes)
        # if n_classes != 2:
        #     self.projector = nn.Linear(1, n_classes)
        self.translayer1 = SelfAttnLayer(emb_dim=size[1])
        self.translayer2 = SelfAttnLayer(emb_dim=size[1])
        # self.pos_enc = CRPE()
        self.norm_layer = nn.LayerNorm(size[1])
        self.classifier = nn.Linear(2 * size[1], n_classes)
        self.cls_token = nn.Parameter(torch.randn(1, 1, size[1]))
        self.surv = surv
        initialize_weights(self)

    def relocate(self, device):
        print('[model] Num. of GPU:', torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attntion_net = nn.DataParallel(self.attntion_net, device_ids=device_ids).to(device)
            # if self.n_classes != 2:
                # self.projector = nn.DataParallel(self.projector, device_ids=device_ids).to(device)
            self.translayer1 = nn.DataParallel(self.translayer1, device_ids=device_ids).to(device)
            self.translayer2 = nn.DataParallel(self.translayer2, device_ids=device_ids).to(device)
            # self.pos_enc = nn.DataParallel(self.pos_enc, device_ids=device_ids).to(device)
            self.norm_layer = nn.DataParallel(self.norm_layer, device_ids=device_ids).to(device)
        else:
            self.attntion_net = self.attntion_net.to(device)
            # if self.n_classes != 2:
                # self.projector = self.projector.to(device)
            self.translayer1 = self.translayer1.to(device)
            self.translayer2 = self.translayer2.to(device)
            # self.pos_enc = self.pos_enc.to(device)
            self.norm_layer = self.norm_layer.to(device)
        self.fc = self.fc.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self, h, coords, patch_level=None, vis_heatmap=False):
        device = h.device
        # adjust dimension for visualization script
        if vis_heatmap:
            h = h.unsqueeze(0)
            coords = coords.unsqueeze(0)

        # x/y in: [1, N, 2]
        x, y = coords[:,:,0].unsqueeze(1), coords[:,:,1].unsqueeze(1) # [1,1,N] to match h
        # cls token
        h = self.fc(h)
        B = h.shape[0]

        # A for instance-level learning and visualization
        A, h = self.attntion_net(h)
        # if patch_level:
        #     if self.n_classes == 2:
        #         return A
        #     else:
        #         A_logit = self.projector(A)
        #         return A_logit
        # print(A.shape)
        # A shape [1, n, n_classes]
        A = A.squeeze(0) # [n, n_classes]
        if patch_level:
            # print('in mdl', A.shape)
            return A
        if self.n_classes != 2:
            A = torch.mean(A, dim=1, keepdim=True)
        
        A = torch.transpose(A, 1, 0)
        # print(A.shape)
        A_raw = A
        A = F.softmax(A, dim=1)
        # print(A.shape, h.shape)
        M = torch.mm(A, h.squeeze(0))
        # print(A.shape, h.shape)

        # CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1).to(device)
        h = torch.cat((cls_tokens, h), dim=1)
    
        # attn 1
        h = self.translayer1(h)

        # pos enc
        h = positional_encoding2d(h, h.shape[2], x, y, device)
        # h = self.pos_enc(h, h.shape[2], x, y, device)
        # h = CRPE(h, h.shape[2], x, y, device)
        
        # attn 2
        h = self.translayer2(h)  # A: [1, 1, N] h: [1, N, D]
        
        h = self.norm_layer(h)[:,0]
        h = self.classifier(torch.cat((h, M), dim=1))

        # return h, A_raw
        if vis_heatmap:
            return A_raw, h
        else:
            if self.surv:
                hazards = torch.sigmoid(h)
                S = torch.cumprod(1 - hazards, dim=1)
                return hazards, S, A_raw
            else:
                return h, A_raw

class XMIL(nn.Module):
    def __init__(self, in_dim=512, n_classes=2, pos_enc='ceg', surv=False):
        super(XMIL, self).__init__()
        size = [in_dim, 512, 256]
        self.n_classes = n_classes
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25)]
        self.fc = nn.Sequential(*fc)
        # if n_classes == 2:
            # self.attntion_net = AttnGate(L=size[1], D=size[2], dropout=0.25, n_classes=1)
        # else:
        self.attntion_net = AttnGate(L=size[1], D=size[2], dropout=0.25, n_classes=n_classes)
        # if n_classes != 2:
        #     self.projector = nn.Linear(1, n_classes)
        self.translayer1 = SelfAttnLayer(emb_dim=size[1])
        self.translayer2 = SelfAttnLayer(emb_dim=size[1])
        # self.pos_enc = CRPE()
        self.norm_layer = nn.LayerNorm(size[1])
        self.classifier = nn.Linear(size[1], n_classes)
        self.cls_token = nn.Parameter(torch.randn(1, 1, size[1]))
        self.surv = surv
        self.pos_enc = pos_enc
        if pos_enc == 'ceg':
            self.pos_layer = CEG()
        elif pos_enc == '2d':
            self.pos_layer = PE2D()
        initialize_weights(self)

    def relocate(self, device):
        print('[model] Num. of GPU:', torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attntion_net = nn.DataParallel(self.attntion_net, device_ids=device_ids).to(device)
            # if self.n_classes != 2:
                # self.projector = nn.DataParallel(self.projector, device_ids=device_ids).to(device)
            self.translayer1 = nn.DataParallel(self.translayer1, device_ids=device_ids).to(device)
            self.translayer2 = nn.DataParallel(self.translayer2, device_ids=device_ids).to(device)
            # self.pos_enc = nn.DataParallel(self.pos_enc, device_ids=device_ids).to(device)
            self.norm_layer = nn.DataParallel(self.norm_layer, device_ids=device_ids).to(device)
        else:
            self.attntion_net = self.attntion_net.to(device)
            # if self.n_classes != 2:
                # self.projector = self.projector.to(device)
            self.translayer1 = self.translayer1.to(device)
            self.translayer2 = self.translayer2.to(device)
            # self.pos_enc = self.pos_enc.to(device)
            self.norm_layer = self.norm_layer.to(device)
        if self.pos_enc == 'ceg' or self.pos_enc == '2d':
            self.pos_layer = self.pos_layer.to(device)
        self.fc = self.fc.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self, h, coords, patch_level=None, vis_heatmap=False):
        device = h.device
        # adjust dimension for visualization script
        if vis_heatmap:
            h = h.unsqueeze(0)
            coords = coords.unsqueeze(0)

        # x/y in: [1, N, 2]
        
        # =============================================
        # for random ablation, remove in normal mode
        # print(coords)
        # print(coords.shape)
        # coords = coords[torch.randperm(coords.size(0))]
        # coords = torch.rand(1, coords.shape[1], 2)
        # print(coords)
        # =============================================

        # x, y = coords[:,:,0].unsqueeze(1), coords[:,:,1].unsqueeze(1) # [1,1,N] to match h
        # print(x.shape, y.shape)
        # cls token
        h = self.fc(h)
        B = h.shape[0]

        # A for instance-level learning and visualization
        A, h = self.attntion_net(h)
        
        # if patch_level:
        #     if self.n_classes == 2:
        #         return A
        #     else:
        #         A_logit = self.projector(A)
        #         return A_logit
        # print(A.shape)
        # A shape [1, n, n_classes]
        A = A.squeeze(0) # [n, n_classes]
        if patch_level:
            # print('in mdl', A.shape)
            return A
        
        # if self.n_classes != 2:
        A, _ = torch.max(A, dim=1, keepdim=True)
            # print(A.shape)
        
        A = torch.transpose(A, 1, 0)
        # print(A.shape)
        A_raw = A
        A = F.softmax(A, dim=1)
        # A = F.tanh(A)
        # print(A.shape, h.shape)
        # M = torch.mm(A, h.squeeze(0))
        # print(A.shape, M.shape)
        # print(A.shape, h.shape)

        # CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1).to(device)
        h = torch.cat((cls_tokens, h), dim=1)
        # h = torch.cat((M, h), dim=1)

        # attn 1
        h = self.translayer1(h)

        # pos enc
        # h = positional_encoding2d(h, h.shape[2], x, y, device)
        # h = self.pos_enc(h, h.shape[2], x, y, A_raw, device)
        if self.pos_enc == 'ceg':
            # h = CRPE_scale(h, h.shape[2], x, y, A, device)
            h = self.pos_layer(h, coords, A, device)
        elif self.pos_enc == '2d':
            # x, y = coords[:,:,0].unsqueeze(1), coords[:,:,1].unsqueeze(1) # [1,1,N] to match h
            # h = positional_encoding2d(h, h.shape[2], x, y, device)
            h = self.pos_layer(h, coords, device)
        elif self.pos_enc == '1d':
            h = positional_encoding(h, h.shape[2], h.shape[1], device)
        else:
            print('Unkown PE method.')
            raise NotImplementedError
        # print('in mdl h shape:', h.shape)
        # h = CRPE_recon(h, h.shape[2], x, y, A_raw, device)
        
        # attn 2
        h = self.translayer2(h)  # A: [1, 1, N] h: [1, N, D]
        
        h = self.norm_layer(h)[:,0]
        # print(h.shape, M.shape)
        h = self.classifier(h)

        # return h, A_raw
        if vis_heatmap:
            return A_raw, h
        else:
            if self.surv:
                hazards = torch.sigmoid(h)
                S = torch.cumprod(1 - hazards, dim=1)
                return hazards, S, A_raw
            else:
                return h, A_raw

class CEG(nn.Module):
    def __init__(self, dim=768):
        super(CEG, self).__init__()
        # if dim % 6 != 0:
        #     raise ValueError("Embed dimension must be divisible by 6.")
        self.dim = dim
        # self.div_term = torch.exp(torch.arange(0., dim // 4, 2) * -(math.log(10000) / (dim // 4)))
        self.div_term = torch.exp(torch.arange(0., dim // 3, 2) * -(math.log(10000) / (dim // 3)))
        # self.div_term = torch.exp(torch.arange(0., dim // 2, 2) * -(math.log(10000) / (dim // 2)))
        # self.div_term = torch.exp(torch.arange(0., dim, 2) * -(math.log(10000) / dim))
        # self.proj = nn.Conv2d(1, dim, 3, 1, 3//2)
        self.proj = nn.Linear(dim, 512)
        self.dropout = nn.Dropout(0.1)

    def forward(self, h, coords, A, device):
        """
        coords_logits: (batch_size, num_points, 3) -> [x, y, logit]
        Returns: (batch_size, num_points, embed_dim)
        """
        cls_token, feat_token = h[:, 0], h[:, 1:]
        self.div_term = self.div_term.to(device)
        # print(cls_token.shape, feat_token.shape)

        x, y = norm_coords(coords)
        A = A.squeeze(0)
        
        # A = A.unsqueeze(-1)
        # # x, y = contextual_recon(coords)
        # # print(x, y)
        # B, H, C = A.shape
        # # print(A.shape)
        # # print(h.shape)
        # _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        # add_length = _H * _W - H
        # # print(add_length)
        # A = torch.cat([A, A[:,:add_length,:]],dim = 1) #[B, N, 512]
        # A = A.transpose(1, 2).view(B, C, _H, _W)
        # A = self.proj(A)
        # A = A.flatten(2).transpose(1, 2)[:, :H, :]

        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        # coords = torch.stack([x, y], dim=-1)
        # print('coords', coords.shape)
        # dist = pairwise_distances(coords).to(device, dtype=torch.float32)
        # print('dist', dist.shape)
        # dist = self.proj(dist).view(logits.shape[1], logits.shape[1], 1)
        # dist = torch.mean(dist, dim=0).squeeze()


        # print('normed dist', dist.shape)
        # x_emb = get_pos_emb(x, self.div_term)
        # print('xemb', x_emb.shape)
        encoded = torch.cat([
            get_pos_emb(x, self.div_term),  # x encoding
            get_pos_emb(y, self.div_term),  # y encoding
            # get_pos_emb(dist, self.div_term),
            get_pos_emb(A, self.div_term),  # logit encoding
        ], dim=-1)
        # print(feat_token.shape)
        # encoded_feat = feat_token + encoded
        encoded_feat = feat_token + self.proj(encoded)
        h = torch.cat((cls_token.unsqueeze(1), encoded_feat), dim=1)
        # print(h.shape)
        return self.dropout(h)  # Shape: (B, N, D)
    
class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)
        self.dropout = nn.Dropout(0.1)

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

class XMILPPEG(nn.Module):
    def __init__(self, in_dim=512, n_classes=2, surv=False):
        super(XMILPPEG, self).__init__()
        size = [in_dim, 512, 256]
        self.n_classes = n_classes
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25)]
        self.fc = nn.Sequential(*fc)
        if n_classes == 2:
            self.attntion_net = AttnGate(L=size[1], D=size[2], dropout=0.25, n_classes=1)
        else:
            self.attntion_net = AttnGate(L=size[1], D=size[2], dropout=0.25, n_classes=n_classes)
        # if n_classes != 2:
        #     self.projector = nn.Linear(1, n_classes)
        self.translayer1 = SelfAttnLayer(emb_dim=size[1])
        self.translayer2 = SelfAttnLayer(emb_dim=size[1])
        # self.pos_enc = CRPE()
        self.pos_layer = PPEG(dim=512)
        self.norm_layer = nn.LayerNorm(size[1])
        self.classifier = nn.Linear(size[1], n_classes)
        self.cls_token = nn.Parameter(torch.randn(1, 1, size[1]))
        self.surv = surv
        initialize_weights(self)

    def relocate(self, device):
        print('[model] Num. of GPU:', torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attntion_net = nn.DataParallel(self.attntion_net, device_ids=device_ids).to(device)
            # if self.n_classes != 2:
                # self.projector = nn.DataParallel(self.projector, device_ids=device_ids).to(device)
            self.translayer1 = nn.DataParallel(self.translayer1, device_ids=device_ids).to(device)
            self.translayer2 = nn.DataParallel(self.translayer2, device_ids=device_ids).to(device)
            self.pos_layer = nn.DataParallel(self.pos_layer, device_ids=device_ids).to(device)
            self.norm_layer = nn.DataParallel(self.norm_layer, device_ids=device_ids).to(device)
        else:
            self.attntion_net = self.attntion_net.to(device)
            # if self.n_classes != 2:
                # self.projector = self.projector.to(device)
            self.translayer1 = self.translayer1.to(device)
            self.translayer2 = self.translayer2.to(device)
            self.pos_layer = self.pos_layer.to(device)
            self.norm_layer = self.norm_layer.to(device)
        self.fc = self.fc.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self, h, coords, patch_level=None, vis_heatmap=False):
        device = h.device
        # adjust dimension for visualization script
        if vis_heatmap:
            h = h.unsqueeze(0)
            coords = coords.unsqueeze(0)

        # x/y in: [1, N, 2]
        x, y = coords[:,:,0].unsqueeze(1), coords[:,:,1].unsqueeze(1) # [1,1,N] to match h
        # cls token
        h = self.fc(h)
        B = h.shape[0]

        # A for instance-level learning and visualization
        A, h = self.attntion_net(h)
        
        # if patch_level:
        #     if self.n_classes == 2:
        #         return A
        #     else:
        #         A_logit = self.projector(A)
        #         return A_logit
        # print(A.shape)
        # A shape [1, n, n_classes]
        A = A.squeeze(0) # [n, n_classes]
        if patch_level:
            # print('in mdl', A.shape)
            return A
        if self.n_classes != 2:
            A, _ = torch.max(A, dim=1, keepdim=True)
        
        A = torch.transpose(A, 1, 0)
        # print(A.shape)
        A_raw = A
        # A = F.softmax(A, dim=1)
        # print(A.shape, h.shape)
        # M = torch.mm(A, h.squeeze(0))
        # print(A.shape, h.shape)

        # CLS token
        B = h.shape[0]
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(device)
        # h = torch.cat((cls_tokens, h), dim=1)
        h = torch.cat((cls_tokens, h), dim=1)
    
        # attn 1
        h = self.translayer1(h)

        # pos enc
        # h = positional_encoding2d(h, h.shape[2], x, y, device)
        # h = self.pos_enc(h, h.shape[2], x, y, A_raw, device)
        # h = CRPE_scale(h, h.shape[2], x, y, A, device)
        h = self.pos_layer(h, _H, _W)
        # h = CRPE_recon(h, h.shape[2], x, y, A_raw, device)
        
        # attn 2
        h = self.translayer2(h)  # A: [1, 1, N] h: [1, N, D]
        
        h = self.norm_layer(h)[:,0]
        h = self.classifier(h)

        # return h, A_raw
        if vis_heatmap:
            return A_raw, h
        else:
            if self.surv:
                hazards = torch.sigmoid(h)
                S = torch.cumprod(1 - hazards, dim=1)
                return hazards, S, A_raw
            else:
                return h, A_raw

class PathMILPPEG(nn.Module):
    def __init__(self, in_dim=512, n_classes=2, surv=False):
        super(PathMILPPEG, self).__init__()
        size = [in_dim, 512, 256]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25)]
        self.fc = nn.Sequential(*fc)
        self.attntion_net = AttnGate(L=size[1], D=size[2], dropout=0.25, n_classes=1)
        self.pos_layer = PPEG(dim=512)
        self.translayer1 = SelfAttnLayer(emb_dim=size[1])
        self.translayer2 = SelfAttnLayer(emb_dim=size[1])
        self.norm_layer = nn.LayerNorm(size[1])
        self.classifier = nn.Linear(2 * size[1], n_classes)
        self.cls_token = nn.Parameter(torch.randn(1, 1, size[1]))
        self.surv = surv
        initialize_weights(self)

    def relocate(self, device):
        print('[model] Num. of GPU:', torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attntion_net = nn.DataParallel(self.attntion_net, device_ids=device_ids).to(device)
            self.pos_layer = nn.DataParallel(self.pos_layer, device_ids=device_ids).to(device)
            self.translayer1 = nn.DataParallel(self.translayer1, device_ids=device_ids).to(device)
            self.translayer2 = nn.DataParallel(self.translayer2, device_ids=device_ids).to(device)
            self.norm_layer = nn.DataParallel(self.norm_layer, device_ids=device_ids).to(device)
        else:
            self.attntion_net = self.attntion_net.to(device)
            self.pos_layer = self.pos_layer.to(device)
            self.translayer1 = self.translayer1.to(device)
            self.translayer2 = self.translayer2.to(device)
            self.norm_layer = self.norm_layer.to(device)
        self.fc = self.fc.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self, h, coords, patch_level=None, vis_heatmap=False):
        device = h.device
        # adjust dimension for visualization script
        if vis_heatmap:
            h = h.unsqueeze(0)
            coords = coords.unsqueeze(0)

        # cls token
        h = self.fc(h)

        # A for instance-level learning and visualization
        A, h = self.attntion_net(h)
        A = A.squeeze(0)
        if patch_level:
            return A
        A = torch.transpose(A, 1, 0)
        A_raw = A
        A = F.softmax(A, dim=1)
        # print(A.shape, h.shape)
        M = torch.mm(A, h.squeeze(0))
        # print(A.shape, h.shape)

        # CLS token
        B = h.shape[0]
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(device)
        h = torch.cat((cls_tokens, h), dim=1)
    
        # attn 1
        h = self.translayer1(h)
        
        # pos enc
        # h = positional_encoding2d(h, h.shape[2], x, y, device)
        h = self.pos_layer(h, _H, _W)

        # attn 2
        h = self.translayer2(h)  # A: [1, 1, N] h: [1, N, D]
        
        h = self.norm_layer(h)[:,0]
        h = self.classifier(torch.cat((h, M), dim=1))

        # return h, A_raw
        if vis_heatmap:
            return A_raw, h
        else:
            if self.surv:
                hazards = torch.sigmoid(h)
                S = torch.cumprod(1 - hazards, dim=1)
                return hazards, S, A_raw
            else:
                return h, A_raw