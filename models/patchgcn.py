"""
PatchGCN: Dense PyTorch re-implementation of
https://github.com/mahmoodlab/Patch-GCN

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseGENConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, t: float = 1.0):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(t).log())   # learn log(t) for stability

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x, adj):
        # x:   [B, N, in_dim]
        # adj: [B, N, N]  binary adjacency (0/1)
        t = self.log_t.exp().clamp(min=1e-4)

        # masked softmax over neighbors: weight_ij = exp(1/t) if adj[i,j]=1, else 0
        scores = adj / t                                    # [B, N, N]
        scores = scores.masked_fill(adj == 0, -1e9)
        weights = scores.softmax(dim=-1)                    # [B, N, N]
        agg = weights @ x                                   # [B, N, in_dim]

        return self.mlp(x + agg)                            # [B, N, out_dim]


class DenseDeepGCNLayer(nn.Module):
    """
    Residual GCN layer: x = act(norm(conv(x, adj))) + x, with optional dropout.
    Matches DeepGCNLayer(block='res').
    """
    def __init__(self, conv: nn.Module, dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv = conv
        self.norm = nn.LayerNorm(dim)
        self.act  = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj):
        # x: [B, N, D]
        out = self.conv(x, adj)
        out = self.norm(out)
        out = self.act(out)
        out = self.drop(out)
        return out + x      # residual


class AttnNetGated(nn.Module):
    """
    Gated ABMIL attention.
    Returns (A [B, N, n_classes], x [B, N, L]) — x is passed through unchanged
    so the caller can use it for the weighted sum.
    """
    def __init__(self, L: int, D: int, dropout: float = 0.25, n_classes: int = 1):
        super().__init__()
        self.attn_a = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.attn_b = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        if dropout > 0:
            self.attn_a = nn.Sequential(nn.Linear(L, D), nn.Tanh(),    nn.Dropout(dropout))
            self.attn_b = nn.Sequential(nn.Linear(L, D), nn.Sigmoid(), nn.Dropout(dropout))
        self.attn_c = nn.Linear(D, n_classes)

    def forward(self, x):
        # x: [B, N, L]
        a = self.attn_a(x)              # [B, N, D]
        b = self.attn_b(x)              # [B, N, D]
        A = self.attn_c(a * b)          # [B, N, n_classes]
        return A, x


class PatchGCN(nn.Module):
    def __init__(self, in_dim: int = 1024, num_layers: int = 4,
                 hidden_dim: int = 128, dropout: float = 0.25,
                 n_classes: int = 2, k_neighbors: int = 8,
                 max_nodes: int | None = None):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.max_nodes = max_nodes
        n_gcn = num_layers - 1                         # number of GCN layers
        cat_dim = num_layers * hidden_dim              # dim of concatenated features

        # --- input projection ---
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # --- GCN layers ---
        # layers[0].conv is called directly (no residual on first pass)
        # layers[1:] are full DenseDeepGCNLayers with residual
        self.convs = nn.ModuleList()
        self.deep_layers = nn.ModuleList()
        for i in range(n_gcn):
            conv = DenseGENConv(hidden_dim, hidden_dim, t=1.0)
            self.convs.append(conv)
            if i > 0:
                self.deep_layers.append(
                    DenseDeepGCNLayer(conv, dim=hidden_dim, dropout=0.1)
                )

        # --- bag aggregation ---
        self.path_phi = nn.Sequential(
            nn.Linear(cat_dim, cat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.path_attention_head = AttnNetGated(L=cat_dim, D=cat_dim,
                                                dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(
            nn.Linear(cat_dim, cat_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(cat_dim, n_classes)

    def build_adj(self, coords):
        """
        k-NN adjacency from patch pixel coordinates.
        coords: [B, N, 2]  →  adj: [B, N, N]  (symmetric binary)
        """
        B, N, _ = coords.shape
        diff  = coords.unsqueeze(2) - coords.unsqueeze(1)      # [B, N, N, 2]
        dist  = diff.norm(dim=-1)                               # [B, N, N]
        k = min(self.k_neighbors, N - 1)
        # exclude self by inflating diagonal
        dist_no_self = dist + torch.eye(N, device=coords.device) * dist.max() * 2
        knn_idx = dist_no_self.topk(k, dim=-1, largest=False).indices  # [B, N, k]
        adj = torch.zeros(B, N, N, device=coords.device)
        adj.scatter_(-1, knn_idx, 1.0)
        return (adj + adj.transpose(-1, -2)).clamp(max=1.0)    # symmetrize

    def forward(self, x, coords=None, adj=None):
        """
        x:      [B, N, D]
        coords: [B, N, 2]  pixel coordinates (builds adj when adj is None)
        adj:    [B, N, N]  precomputed adjacency (optional)

        Returns:
            logits:   [B, n_classes]
            attn:     [B, N]  attention weights (0 for any patches dropped by max_nodes)
        """
        B, N_orig, _ = x.shape
        sub_idx = None
        if self.max_nodes is not None and N_orig > self.max_nodes:
            sub_idx = torch.randperm(N_orig)[:self.max_nodes].sort().values  # CPU
            x = x[:, sub_idx.to(x.device)]
            if coords is not None:
                coords = coords[:, sub_idx.to(coords.device)]
            if adj is not None:
                sidx = sub_idx.to(adj.device)
                adj = adj[:, sidx][:, :, sidx]

        if adj is None:
            assert coords is not None, "Provide either coords or adj"
            adj = self.build_adj(coords).to(x.device)   # [B, N, N]

        # --- input projection ---
        h = self.fc(x)          # [B, N, hidden_dim]
        h_cat = h               # running concat  [B, N, hidden_dim]

        # --- first conv (no residual) ---
        h = self.convs[0](h, adj)              # [B, N, hidden_dim]
        h_cat = torch.cat([h_cat, h], dim=-1)  # [B, N, 2*hidden_dim]

        # --- remaining layers with residual ---
        for layer in self.deep_layers:
            h = layer(h, adj)                          # [B, N, hidden_dim]
            h_cat = torch.cat([h_cat, h], dim=-1)      # grows by hidden_dim each step

        # --- bag-level aggregation ---
        h_path = self.path_phi(h_cat)                  # [B, N, cat_dim]

        A, h_path = self.path_attention_head(h_path)   # A: [B, N, 1], h_path: [B, N, cat_dim]
        A = A.transpose(-1, -2)                         # [B, 1, N]
        h_path = F.softmax(A, dim=-1) @ h_path         # [B, 1, cat_dim]
        h_path = h_path.squeeze(1)                      # [B, cat_dim]

        h = self.path_rho(h_path)                       # [B, cat_dim]

        attn = A.squeeze(1)                             # [B, N_sub]
        if sub_idx is not None:
            full_attn = torch.zeros(B, N_orig, device=x.device, dtype=attn.dtype)
            full_attn[:, sub_idx] = attn
            attn = full_attn

        return self.classifier(h), attn                 # [B, n_classes], [B, N_orig]


def build_patchgcn(in_dim: int, cls_num: int, max_nodes: int | None = 10000) -> PatchGCN:
    return PatchGCN(
        in_dim=in_dim,
        num_layers=4,
        hidden_dim=128,
        dropout=0.25,
        n_classes=cls_num,
        k_neighbors=8,
        max_nodes=max_nodes,
    )
