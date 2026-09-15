"""
CAMIL: Context-Aware Multiple Instance Learning
https://github.com/olgarithmics/ICLR_CAMIL
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoorePenrosePinv(nn.Module):
    def __init__(self, iters: int = 6):
        super().__init__()
        self.iters = iters

    def forward(self, A):
        # A: [..., n, m]
        abs_A = A.abs()
        z = A.transpose(-1, -2) / (abs_A.sum(-2).max() * abs_A.sum(-1).max() + 1e-8)
        I = torch.eye(z.shape[-1], device=A.device, dtype=A.dtype)
        for _ in range(self.iters):
            Az = A @ z
            z = 0.25 * z @ (13 * I - Az @ (15 * I - Az @ (7 * I - Az)))
        return z


class NystromAttention(nn.Module):
    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8,
                 num_landmarks: int = 256, pinv_iterations: int = 6,
                 residual: bool = True, residual_conv_kernel: int = 33,
                 dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.num_landmarks = num_landmarks
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.pinv = MoorePenrosePinv(iters=pinv_iterations)

        self.residual = residual
        if residual:
            # depthwise conv along the sequence dimension, one filter group per head
            self.res_conv = nn.Conv1d(
                inner_dim, inner_dim,
                kernel_size=residual_conv_kernel,
                padding=residual_conv_kernel // 2,
                groups=heads,
                bias=False,
            )

    def forward(self, x):
        # x: [B, N, D]
        B, N, _ = x.shape
        H, M, dh = self.heads, self.num_landmarks, self.dim_head

        # pad sequence so N is a multiple of num_landmarks
        pad = (M - N % M) % M
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
        N_pad = x.shape[1]

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # [B, N_pad, H*dh] → [B, H, N_pad, dh]
        q, k, v = (t.reshape(B, N_pad, H, dh).permute(0, 2, 1, 3) for t in (q, k, v))

        q = q * self.scale

        # landmark pooling along sequence dim
        def pool_landmarks(t):
            # t: [B, H, N_pad, dh] → pool N_pad → M
            t_flat = t.reshape(B * H, N_pad, dh).transpose(1, 2)  # [B*H, dh, N_pad]
            pooled = F.adaptive_avg_pool1d(t_flat, M)              # [B*H, dh, M]
            return pooled.transpose(1, 2).reshape(B, H, M, dh)     # [B, H, M, dh]

        q_land = pool_landmarks(q)   # [B, H, M, dh]
        k_land = pool_landmarks(k)   # [B, H, M, dh]

        # three similarity matrices
        sim1 = torch.einsum('bhid,bhjd->bhij', q,      k_land)   # [B, H, N_pad, M]
        sim2 = torch.einsum('bhid,bhjd->bhij', q_land, k_land)   # [B, H, M, M]
        sim3 = torch.einsum('bhid,bhjd->bhij', q_land, k)        # [B, H, M, N_pad]

        attn1 = sim1.softmax(dim=-1)
        attn2 = sim2.softmax(dim=-1)
        attn3 = sim3.softmax(dim=-1)

        attn2_inv = self.pinv(attn2)
        out = (attn1 @ attn2_inv) @ (attn3 @ v)   # [B, H, N_pad, dh]

        if self.residual:
            # depthwise conv along sequence for each head
            v_flat = v.permute(0, 1, 3, 2).reshape(B, H * dh, N_pad)  # [B, H*dh, N_pad]
            res = self.res_conv(v_flat)                                  # [B, H*dh, N_pad]
            res = res.reshape(B, H, dh, N_pad).permute(0, 1, 3, 2)     # [B, H, N_pad, dh]
            out = out + res

        # merge heads, project, trim padding
        out = out.permute(0, 2, 1, 3).reshape(B, N_pad, H * dh)   # [B, N_pad, inner_dim]
        out = self.to_out(out)
        return out[:, :N]    # [B, N, D]

class CustomAttention(nn.Module):
    """Scaled dot-product Q·K^T without softmax (scores fed to NeighborAggregator)."""
    def __init__(self, in_dim: int, proj_dim: int = 256):
        super().__init__()
        self.wq = nn.Linear(in_dim, proj_dim, bias=False)
        self.wk = nn.Linear(in_dim, proj_dim, bias=False)
        self.scale = proj_dim ** -0.5

    def forward(self, x):
        # x: [B, N, D]
        q = self.wq(x)   # [B, N, proj_dim]
        k = self.wk(x)   # [B, N, proj_dim]
        return (q @ k.transpose(-1, -2)) * self.scale   # [B, N, N]

class NeighborAggregator(nn.Module):
    """
    For each instance i: score_i = Σ_j adj[i,j] · attn_matrix[i,j]
    then softmax over i → normalized attention weights.
    """
    def forward(self, attn_matrix, adj):
        # attn_matrix: [B, N, N], adj: [B, N, N]
        alpha_raw = (attn_matrix * adj).sum(dim=-1)        # [B, N]
        alpha = alpha_raw.softmax(dim=-1)                   # [B, N]
        return alpha, alpha_raw

class MILAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, use_gated: bool = True):
        super().__init__()
        self.V = nn.Linear(in_dim, hidden_dim, bias=False)
        self.W = nn.Linear(hidden_dim, 1, bias=False)
        self.use_gated = use_gated
        if use_gated:
            self.U = nn.Linear(in_dim, hidden_dim, bias=False)

    def forward(self, x):
        # x: [B, N, D]
        h = torch.tanh(self.V(x))               # [B, N, hidden_dim]
        if self.use_gated:
            h = h * torch.sigmoid(self.U(x))
        scores = self.W(h)                        # [B, N, 1]
        return scores.softmax(dim=1)             # [B, N, 1]

class CAMIL(nn.Module):
    def __init__(self, in_dim: int = 512, n_classes: int = 2,
                 hidden_dim: int = 256, use_gated: bool = True,
                 nystrom_heads: int = 8, nystrom_dim_head: int = 64,
                 num_landmarks: int = 256, pinv_iterations: int = 6,
                 dropout: float = 0.0, k_neighbors: int = 8,
                 sparse: bool = True):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.sparse = sparse

        # --- encoder components ---
        self.nyst_att = NystromAttention(
            dim=in_dim, dim_head=nystrom_dim_head, heads=nystrom_heads,
            num_landmarks=num_landmarks, pinv_iterations=pinv_iterations,
            dropout=dropout,
        )
        self.custom_att = CustomAttention(in_dim=in_dim, proj_dim=hidden_dim)
        self.neigh = NeighborAggregator()
        self.wv = nn.Linear(in_dim, in_dim)

        # --- bag-level aggregation ---
        self.mil_attn = MILAttentionLayer(in_dim=in_dim, hidden_dim=hidden_dim,
                                          use_gated=use_gated)
        self.classifier = nn.Linear(in_dim, n_classes)

    # ------------------------------------------------------------------
    def build_edges(self, coords, chunk=1024):
        c = coords[0]
        N = c.shape[0]
        k = min(self.k_neighbors, N - 1)
        if k <= 0:
            z = torch.zeros(0, dtype=torch.long, device=c.device)
            return z, z
        ar = torch.arange(N, device=c.device)
        src_l, dst_l = [], []
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            d = (c[s:e].unsqueeze(1) - c.unsqueeze(0)).norm(dim=-1)   # [c, N]
            # dense masked the diagonal with +2*max, which can never enter the
            # bottom-k; +inf selects exactly the same k neighbours
            d[ar[:e - s], ar[s:e]] = float('inf')
            src_l.append(ar[s:e].unsqueeze(1).expand(-1, k).reshape(-1))
            dst_l.append(d.topk(k, dim=-1, largest=False).indices.reshape(-1))
        src, dst = torch.cat(src_l), torch.cat(dst_l)
        lin = torch.unique(torch.cat([src, dst]) * N + torch.cat([dst, src]))
        return lin // N, lin % N

    def build_adj(self, coords):
        B, N, _ = coords.shape
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)          # [B, N, N, 2]
        dist = diff.norm(dim=-1)                                   # [B, N, N]
        # mask self-distance so it's not selected as a neighbor
        dist_no_self = dist + torch.eye(N, device=coords.device) * dist.max() * 2
        k = min(self.k_neighbors, N - 1)
        knn_idx = dist_no_self.topk(k, dim=-1, largest=False).indices  # [B, N, k]
        adj = torch.zeros(B, N, N, device=coords.device, dtype=coords.dtype)
        adj.scatter_(-1, knn_idx, 1.0)
        adj = (adj + adj.transpose(-1, -2)).clamp(max=1.0)        # symmetrize
        return adj   # [B, N, N]

    def _edge_scores(self, enc, src, dst, chunk=1 << 18):
        """alpha_raw[i] = sum over the k-NN of i of (q_i . k_j) * scale."""
        q = self.custom_att.wq(enc)                          # [B, N, P]
        k = self.custom_att.wk(enc)                          # [B, N, P]
        out = q.new_zeros(q.shape[0], q.shape[1])            # [B, N]
        for s in range(0, src.numel(), chunk):
            e = min(s + chunk, src.numel())
            v = (q[:, src[s:e]] * k[:, dst[s:e]]).sum(-1) * self.custom_att.scale
            out = out.index_add(1, src[s:e], v)
        return out

    def forward(self, x, coords=None, adj=None):
        # --- Nystrom feature refinement + residual ---
        enc = self.nyst_att(x) + x          # [B, N, D]

        if adj is None and not self.sparse:
            assert coords is not None, "Provide either coords or adj"
            adj = self.build_adj(coords).to(x.device)    # [B, N, N]

        if adj is not None:
            # dense path, kept for callers that hand in a precomputed adjacency
            attn_matrix = self.custom_att(enc)               # [B, N, N]
            norm_alpha, alpha_raw = self.neigh(attn_matrix, adj)  # [B, N]
        else:
            assert coords is not None, "Provide either coords or adj"
            src, dst = self.build_edges(coords.to(x.device))
            alpha_raw = self._edge_scores(enc, src, dst)     # [B, N]
            norm_alpha = alpha_raw.softmax(dim=-1)           # [B, N]

        # --- neighbor-weighted values ---
        value = self.wv(x)                               # [B, N, D]
        xl = norm_alpha.unsqueeze(-1) * value            # [B, N, D]

        # --- gating fusion between xl and enc ---
        wei = torch.sigmoid(-xl)      # [B, N, D]
        wei2 = wei ** 2
        xo = xl * 2 * wei2 + 2 * enc * (1 - wei2)       # [B, N, D]

        # --- ABMIL bag aggregation ---
        alpha = self.mil_attn(xo)                        # [B, N, 1]
        bag = (alpha * xo).sum(dim=1)                    # [B, D]

        return self.classifier(bag), alpha_raw


def build_camil(in_dim: int, cls_num: int, sparse: bool = True) -> CAMIL:
    return CAMIL(
        sparse=sparse,
        in_dim=in_dim,
        n_classes=cls_num,
        hidden_dim=256,
        use_gated=True,
        nystrom_heads=8,
        nystrom_dim_head=64,
        num_landmarks=256,
        pinv_iterations=6,
        dropout=0.0,
        k_neighbors=8,
    )
