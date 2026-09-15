"""
HIPT adapted for [N, D] pre-extracted features.

Instead of processing raw images through ViT-256, this module accepts patch
features already extracted by any backbone (shape [N, D]) and feeds them
directly into the ViT-4K regional aggregator.

The VisionTransformer4K (vit4k) module and all its pretrained parameters are
preserved exactly as in the original HIPT implementation.

Reference: https://github.com/mahmoodlab/HIPT
"""

import math
import warnings
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attn: bool = False):
        B, N, C = x.shape
        qkv = (self.qkv(x)
                   .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                   .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        if return_attn:
            # Materialise the full attention map only when explicitly requested
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        else:
            # Memory-efficient flash attention — O(N) memory, no N×N matrix
            dropout_p = self.attn_drop.p if self.training else 0.0
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=dropout_p,
                scale=self.scale,
            ).transpose(1, 2).reshape(B, N, C)
            attn = None

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer, drop=drop,
        )

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer4K(nn.Module):

    def __init__(
        self,
        num_classes: int = 0,
        img_size: list = [224],
        input_embed_dim: int = 384,
        output_embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        qk_scale=None,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        norm_layer=nn.LayerNorm,
        use_checkpoint: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        embed_dim = output_embed_dim
        self.num_features = self.embed_dim = embed_dim

        # Project input_embed_dim → output_embed_dim
        self.phi = nn.Sequential(
            nn.Linear(input_embed_dim, output_embed_dim),
            nn.GELU(),
            nn.Dropout(p=drop_rate),
        )

        num_patches = int(img_size[0] // 16) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        # Force Python ints so scale_factor stays a float (tracing-safe)
        w0 = int(w) + 0.1
        h0 = int(h) + 0.1
        patch_pos_embed = F.interpolate(
            patch_pos_embed
                .reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim)
                .permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        # x: [B, input_embed_dim, w, h]
        B, embed_dim, w, h = x.shape
        x = x.flatten(2, 3).transpose(1, 2)   # [B, w*h, input_embed_dim]
        x = self.phi(x)                         # [B, w*h, output_embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1 + w*h, output_embed_dim]
        x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # CLS token

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def vit4k_xs(use_checkpoint: bool = False, **kwargs):
    """Lightweight vit4k: 6 blocks, 6 heads, embed_dim=192."""
    return VisionTransformer4K(
        input_embed_dim=384,
        output_embed_dim=192,
        depth=6,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint=use_checkpoint,
        **kwargs,
    )


class HIPT(nn.Module):

    def __init__(
        self,
        input_dim: int = 384,
        vit4k_input_embed_dim: int = 384,
        vit4k_output_embed_dim: int = 192,
        vit4k_depth: int = 6,
        vit4k_num_heads: int = 6,
        n_classes: int = 0,
        pretrained_vit4k: Optional[str] = None,
        freeze_vit4k: bool = False,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.freeze_vit4k = freeze_vit4k

        # Adapter: align upstream feature dim with vit4k's phi input dim.
        # When dims already match we use Identity to keep the graph clean.
        self.adapter = (
            nn.Linear(input_dim, vit4k_input_embed_dim, bias=False)
            if input_dim != vit4k_input_embed_dim
            else nn.Identity()
        )

        # Regional aggregator — parameters match the original vit4k_xs config
        self.vit4k = VisionTransformer4K(
            input_embed_dim=vit4k_input_embed_dim,
            output_embed_dim=vit4k_output_embed_dim,
            depth=vit4k_depth,
            num_heads=vit4k_num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_checkpoint=use_checkpoint,
        )

        self.head = (
            nn.Linear(vit4k_output_embed_dim, n_classes)
            if n_classes > 0
            else nn.Identity()
        )

        if isinstance(pretrained_vit4k, str):
            self._load_pretrained_vit4k(pretrained_vit4k)

        if freeze_vit4k:
            for p in self.adapter.parameters():
                p.requires_grad = False
            for p in self.vit4k.parameters():
                p.requires_grad = False


    def _load_pretrained_vit4k(self, path: str):
        state = torch.load(path, map_location='cpu', weights_only=False)
        # Support DINO-style teacher checkpoints
        if 'teacher' in state:
            state = {
                k.replace('backbone.', ''): v
                for k, v in state['teacher'].items()
            }
        msg = self.vit4k.load_state_dict(state, strict=False)
        print(f"[HIPT] Loaded vit4k weights from '{path}' — {msg}")


    def forward(
        self,
        x: torch.Tensor,
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:

        if x.dim() == 2:
            x = x.unsqueeze(0)          # [1, N, D]

        B, N, _ = x.shape

        # 1. Adapt feature dimension if needed
        if self.freeze_vit4k:
            with torch.no_grad():
                x = self.adapter(x)     # [B, N, vit4k_input_embed_dim]
        else:
            x = self.adapter(x)         # [B, N, vit4k_input_embed_dim]

        # 2. Determine spatial grid dimensions
        if grid_size is not None:
            w, h = grid_size
            if w * h != N:
                raise ValueError(
                    f"grid_size {grid_size} implies {w * h} patches but got N={N}."
                )
        else:
            w = h = math.isqrt(N)
            if w * h != N:
                # Pad to the next perfect square with zero tokens
                w = h = w + 1
                pad = w * h - N
                x = F.pad(x, (0, 0, 0, pad))   # [B, w*h, C]

        # 3. Reshape to channel-first spatial grid expected by vit4k
        #    [B, N, C] → [B, C, w, h]
        x = x.transpose(1, 2).reshape(B, -1, w, h)

        # 4. Regional aggregation → CLS token [B, output_embed_dim]
        if self.freeze_vit4k:
            with torch.no_grad():
                out = self.vit4k(x)
        else:
            out = self.vit4k(x)

        return self.head(out), None