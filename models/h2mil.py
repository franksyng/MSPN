"""
H2MIL: Exploring Hierarchical Representation with Heterogeneous MIL for WSI Analysis.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _scatter_sum(src: torch.Tensor, idx: torch.Tensor, dim_size: int) -> torch.Tensor:
    """src [E, ...], idx [E]  →  [dim_size, ...]"""
    shape = (dim_size,) + src.shape[1:]
    out = src.new_zeros(shape)
    expand = idx.view(-1, *([1] * (src.dim() - 1))).expand_as(src).to(src.device)
    return out.scatter_add_(0, expand, src)


def _scatter_mean(src: torch.Tensor, idx: torch.Tensor, dim_size: int) -> torch.Tensor:
    out  = _scatter_sum(src, idx, dim_size)
    cnt  = _scatter_sum(torch.ones(idx.size(0), 1, device=src.device, dtype=src.dtype),
                        idx, dim_size)
    return out / cnt.clamp(min=1)


def _edge_softmax(scores: torch.Tensor, idx: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Softmax over edges grouped by destination node. scores [E, H], idx [E]."""
    exp_s  = scores.exp()
    denom  = _scatter_sum(exp_s, idx, dim_size)[idx]
    return exp_s / denom.clamp(min=1e-16)


def _knn_edges(coords: torch.Tensor, k: int, chunk: int = 2048) -> torch.Tensor:
    """k-NN graph from [N, 2] coords. Returns edge_index [2, E]."""
    N = coords.size(0)
    k = min(k, N - 1)
    if k == 0:
        return coords.new_zeros(2, 0, dtype=torch.long)

    ar = torch.arange(N, device=coords.device)
    src_l, dst_l = [], []
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        d = (coords[s:e].unsqueeze(1) - coords.unsqueeze(0)).pow(2).sum(-1)
        d[ar[:e - s], ar[s:e]] = float('inf')                  # self-distance
        src_l.append(ar[s:e].unsqueeze(1).expand(-1, k).reshape(-1))
        dst_l.append(d.topk(k, largest=False).indices.reshape(-1))
    return torch.stack([torch.cat(src_l), torch.cat(dst_l)], dim=0)


def _kmeans(pts: torch.Tensor, K: int, n_iter: int = 20) -> torch.Tensor:
    """Simple k-means on CPU. pts [N, d] → cluster assignments [N]."""
    N = pts.size(0)
    K = min(K, N)
    centers = pts[torch.randperm(N)[:K]].clone()
    assign  = torch.zeros(N, dtype=torch.long)
    for _ in range(n_iter):
        dists  = torch.cdist(pts, centers)        # [N, K]
        assign = dists.argmin(dim=1)              # [N]
        for c in range(K):
            mask = assign == c
            if mask.any():
                centers[c] = pts[mask].mean(0)
    return assign


def build_hetero_graph(
    feats: torch.Tensor,           # [N, D]
    coords: torch.Tensor | None,   # [N, 2] or None
    n_super: int | None = None,    # K level-1 super-patches; default ceil(sqrt(N))
    k_neighbors: int = 8,
):

    N, D = feats.shape
    device = feats.device
    K = n_super if n_super is not None else max(1, math.ceil(math.sqrt(N)))

    if coords is not None:
        pts = coords.float().to(device)
    else:
        # Random 2-D proxy when coords unavailable
        pts = torch.rand(N, 2, device=device)

    # k-means on CPU (no gradients through clustering)
    assign = _kmeans(pts.detach().cpu(), K).to(device)    # [N]
    K = int(assign.max().item()) + 1                       # actual clusters (≤ K)

    # --- node features ---
    x_global = feats.new_zeros(1, D)
    x_super  = _scatter_mean(feats, assign, K)             # [K, D]
    x        = torch.cat([x_global, x_super, feats], dim=0)  # [1+K+N, D]

    # --- node types ---
    node_type = torch.zeros(1 + K + N, dtype=torch.long, device=device)
    node_type[1:1+K] = 1
    node_type[1+K:]  = 2

    # --- parent pointers ---
    tree = torch.full((1 + K + N,), -1, dtype=torch.long, device=device)
    tree[1:1+K] = 0                     # level-1 → global node
    tree[1+K:]  = 1 + assign            # level-2 → its super-patch

    # --- coordinates ---
    xy_global = pts.new_zeros(1, 2)
    xy_super  = _scatter_mean(pts, assign, K)             # [K, 2]
    x_y_index = torch.cat([xy_global, xy_super, pts], dim=0)  # [1+K+N, 2]

    # --- edges ---
    # within level-2: k-NN
    e_l2 = _knn_edges(pts, k_neighbors) + (1 + K)

    # within level-1: k-NN on centroids
    e_l1 = _knn_edges(xy_super, min(k_neighbors, K - 1)) + 1

    # cross level-2 ↔ level-1
    patch_g  = torch.arange(N, device=device) + 1 + K     # [N]
    super_g  = 1 + assign                                  # [N]
    e_cross_21 = torch.stack([patch_g, super_g], dim=0)
    e_cross_12 = torch.stack([super_g, patch_g], dim=0)

    # cross level-1 ↔ level-0
    sv = torch.arange(K, device=device) + 1
    gv = torch.zeros(K, dtype=torch.long, device=device)
    e_cross_10 = torch.stack([sv, gv], dim=0)
    e_cross_01 = torch.stack([gv, sv], dim=0)
    # print(e_l2.device, e_l1.device, e_cross_21.device, e_cross_12.device, e_cross_10.device, e_cross_01.device)
    edge_index = torch.cat([e_l2, e_l1, e_cross_21, e_cross_12, e_cross_10, e_cross_01], dim=1)

    return x, node_type, tree, x_y_index, edge_index


def _glorot(t: torch.Tensor) -> None:
    stdv = math.sqrt(6.0 / (t.size(-2) + t.size(-1)))
    t.data.uniform_(-stdv, stdv)


class RAConv(nn.Module):
    """
    Resolution-Aware Graph Convolution (Lin et al., AAAI 2022).
    Sparse edge-list port of the original PyG-based implementation.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 heads: int = 1, dropout: float = 0.0,
                 negative_slope: float = 0.2):
        super().__init__()
        H, C = heads, out_channels
        self.heads, self.out_channels = H, C
        self.dropout = dropout
        self.negative_slope = negative_slope

        # Shared src/dst projection (original sets lin_r = lin_l)
        self.lin   = nn.Linear(in_channels, H * C, bias=False)
        self.t_lin = nn.Linear(in_channels, H * C, bias=False)

        self.att_l   = nn.Parameter(torch.empty(1, H, C))
        self.att_r   = nn.Parameter(torch.empty(1, H, C))
        self.t_att_l = nn.Parameter(torch.empty(1, H, C))
        self.t_att_r = nn.Parameter(torch.empty(1, H, C))
        self.bias    = nn.Parameter(torch.zeros(H * C))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        _glorot(self.lin.weight)
        _glorot(self.t_lin.weight)
        _glorot(self.att_l)
        _glorot(self.att_r)
        _glorot(self.t_att_l)
        _glorot(self.t_att_r)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                node_type: torch.Tensor) -> torch.Tensor:
        """
        x          [N, D]
        edge_index [2, E]   directed src → dst
        node_type  [N]      in {0, 1, 2}
        Returns    [N, H*C]
        """
        H, C = self.heads, self.out_channels
        N = x.size(0)
        if edge_index.size(1) == 0:
            return self.lin(x) + self.bias

        src, dst = edge_index[0], edge_index[1]

        # --- node-level projections ---
        xl = xr = self.lin(x).view(N, H, C)           # [N, H, C]
        alpha_l = (xl * self.att_l).sum(-1)            # [N, H]
        alpha_r = (xr * self.att_r).sum(-1)

        # --- virtual resolution nodes ---
        # One virtual node per unique (node_type[src], dst) pair.
        src_type = node_type[src]                      # [E]
        res_key  = src_type + dst * 3                  # [E]  in [0, 3*N)

        num_res = int(res_key.max().item()) + 1
        v_x     = _scatter_mean(x[src], res_key, num_res)     # [num_res, D]

        t_x  = torch.cat([x, v_x], dim=0)                     # [N+num_res, D]
        t_xl = t_xr = self.t_lin(t_x).view(-1, H, C)
        t_al = (t_xl * self.t_att_l).sum(-1)                  # [N+num_res, H]
        t_ar = (t_xr * self.t_att_r).sum(-1)

        # Resolution-level edges: virtual node k → original node (k // 3)
        uniq_keys = torch.unique(res_key)
        rv_src = uniq_keys + N                                 # indices into t_x
        rv_dst = uniq_keys // 3

        t_alpha = t_al[rv_src] + t_ar[rv_dst]                 # [M, H]
        t_alpha = F.leaky_relu(t_alpha, self.negative_slope)
        t_beta  = _edge_softmax(t_alpha, rv_dst, N)           # [M, H]  resolution attn

        # Map each edge → its resolution-level attention
        edge_to_res = torch.bucketize(res_key, uniq_keys)     # [E]

        # --- node-level attention (softmax within each (dst, res_type) group) ---
        alpha_e = alpha_l[src] + alpha_r[dst]                 # [E, H]
        alpha_e = F.leaky_relu(alpha_e, self.negative_slope)
        alpha_e = _edge_softmax(alpha_e, res_key, num_res)    # [E, H]

        # Final attention = node-level × resolution-level
        alpha_final = alpha_e * t_beta[edge_to_res]           # [E, H]
        if self.training and self.dropout > 0:
            alpha_final = F.dropout(alpha_final, p=self.dropout)

        # --- aggregate ---
        msg = (xl[src] * alpha_final.unsqueeze(-1)).view(-1, H * C)  # [E, H*C]
        out = _scatter_sum(msg, dst, N)                               # [N, H*C]
        return out + self.bias


def _euclidean_dist_xyf(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a [M, 3], b [N, 3]  (x, y, fitness) → distances [M, N].
    Matches the original: dist_xy + dist_fitness.
    """
    a_xy, a_f = a[:, :2].float(), a[:, 2:3].float()
    b_xy, b_f = b[:, :2].float(), b[:, 2:3].float()
    return torch.cdist(a_xy, b_xy) + torch.cdist(a_f, b_f)


def _select_reps(N: int, ratio: float) -> list[int]:
    """Indices into a length-N sorted array to serve as representatives."""
    if ratio < 1:
        step = max(1, math.ceil(1.0 / ratio))
    else:
        if N < ratio:
            step = max(1, math.ceil(float(N) / N))
        else:
            step = max(1, math.ceil(N / ratio))
    return list(range(0, N, step))


class IHPool(nn.Module):
    """
    Intelligent Hierarchical Pooling (Lin et al., AAAI 2022).
    Selects representative nodes by fitness and clusters the rest.
    """
    def __init__(self, in_channels: int, ratio: float = 0.1):
        super().__init__()
        self.ratio = ratio
        bound = 1.0 / math.sqrt(in_channels)
        self.weight_1 = nn.Parameter(torch.empty(1, in_channels).uniform_(-bound, bound))
        self.weight_2 = nn.Parameter(torch.empty(1, in_channels).uniform_(-bound, bound))

    def forward(self, x, edge_index, node_type, tree, x_y_index):
        device = x.device
        N = x.size(0)

        idx_1 = (node_type == 1).nonzero(as_tuple=True)[0]   # level-1 global indices
        idx_2 = (node_type == 2).nonzero(as_tuple=True)[0]   # level-2 global indices
        N_1, N_2 = len(idx_1), len(idx_2)

        # ----- level-1 fitness & clustering -----
        fit1 = torch.tanh((x[idx_1] * self.weight_1).sum(-1)
                          / self.weight_1.norm(p=2, dim=-1).clamp(min=1e-8))  # [N_1]
        xyf1 = torch.cat([x_y_index[idx_1].float(), fit1.unsqueeze(1)], dim=-1)  # [N_1, 3]

        _, sort1 = fit1.sort()
        rep_local1 = torch.tensor(_select_reps(N_1, self.ratio),
                                  device=device, dtype=torch.long)
        reps1 = sort1[rep_local1]                                    # [K1] local indices
        K1 = len(reps1)

        dist1 = _euclidean_dist_xyf(xyf1[reps1], xyf1)              # [K1, N_1]
        cluster1 = dist1.argmin(dim=0)                               # [N_1] → 0..K1-1
        new_xy1  = _scatter_mean(x_y_index[idx_1].float(), cluster1, K1)

        # ----- level-2 fitness & per-cluster sub-clustering -----
        fit2 = torch.tanh((x[idx_2] * self.weight_2).sum(-1)
                          / self.weight_2.norm(p=2, dim=-1).clamp(min=1e-8))  # [N_2]

        # map each level-2 node to its level-1 cluster
        global_to_local1 = torch.zeros(N, dtype=torch.long, device=device)
        global_to_local1[idx_1] = torch.arange(N_1, device=device)
        parent_local1 = global_to_local1[tree[idx_2]]   # [N_2] local level-1 index
        parent_cluster1 = cluster1[parent_local1]        # [N_2] cluster of parent

        all_cluster2 = torch.zeros(N_2, dtype=torch.long, device=device)
        new_xy_parts: list[torch.Tensor] = []
        tree2_parts:  list[torch.Tensor] = []
        offset2 = 0

        for k in range(K1):
            in_k = (parent_cluster1 == k).nonzero(as_tuple=True)[0]
            N_k  = len(in_k)
            if N_k == 0:
                continue

            xyf2_k = torch.cat([x_y_index[idx_2[in_k]].float(),
                                 fit2[in_k].unsqueeze(1)], dim=-1)   # [N_k, 3]
            _, sort2_k = fit2[in_k].sort()

            # Select 2 representatives (or 1 when N_k == 1), matching original
            if N_k == 1:
                reps2_k = sort2_k[[0]]
            else:
                reps2_k = sort2_k[[0, N_k - 1]]
            K2_k = len(reps2_k)

            dist2_k   = _euclidean_dist_xyf(xyf2_k[reps2_k], xyf2_k)  # [K2_k, N_k]
            cluster2_k = dist2_k.argmin(dim=0)                          # [N_k]
            all_cluster2[in_k] = cluster2_k + offset2

            new_xy_parts.append(
                _scatter_mean(x_y_index[idx_2[in_k]].float(), cluster2_k, K2_k))
            tree2_parts.append(
                torch.full((K2_k,), k + 1, dtype=torch.long, device=device))
            offset2 += K2_k

        total_K2 = offset2

        # ----- assemble coarsened graph -----
        # new numbering: 0=global, 1..K1=level-1, K1+1..K1+K2=level-2
        N_new = 1 + K1 + total_K2

        cluster_map = torch.zeros(N, dtype=torch.long, device=device)
        cluster_map[idx_1] = cluster1 + 1
        cluster_map[idx_2] = all_cluster2 + K1 + 1

        x_new = _scatter_mean(x, cluster_map, N_new)

        # coarsen edges (deduplicate)
        s_new = cluster_map[edge_index[0]]
        d_new = cluster_map[edge_index[1]]
        eids  = s_new * N_new + d_new
        eids_u = torch.unique(eids)
        s_new  = eids_u // N_new
        d_new  = eids_u % N_new
        edge_new = torch.stack([s_new, d_new], dim=0)

        nt_new = torch.zeros(N_new, dtype=torch.long, device=device)
        nt_new[1:1+K1] = 1
        nt_new[1+K1:]  = 2

        tree_new = torch.full((N_new,), -1, dtype=torch.long, device=device)
        tree_new[1:1+K1] = 0
        if tree2_parts:
            tree_new[1+K1:] = torch.cat(tree2_parts)

        new_xy2 = torch.cat(new_xy_parts, dim=0) if new_xy_parts else x.new_zeros(0, 2)
        xy_new  = torch.cat([x_y_index[:1].float(), new_xy1, new_xy2], dim=0)

        fitness   = torch.cat([x.new_zeros(1), fit1, fit2])
        batch_new = torch.zeros(N_new, dtype=torch.long, device=device)
        return x_new, edge_new, None, batch_new, cluster_map, nt_new, tree_new, fitness, xy_new

class H2MIL(nn.Module):
    """
    Hierarchical Heterogeneous MIL (Lin et al., AAAI 2022).
    """
    def __init__(
        self,
        in_dim:      int   = 1024,
        hidden_dim:  int   = 256,
        n_classes:   int   = 2,
        pool1_ratio: float = 0.1,
        pool2_ratio: float = 4.0,
        dropout:     float = 0.3,
        k_neighbors: int   = 8,
        n_super:     int | None = None,
        pool_method: str   = 'mean',
    ):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.n_super      = n_super

        self.norm_in  = nn.LayerNorm(in_dim)
        self.conv1    = RAConv(in_dim,     hidden_dim)
        self.norm1    = nn.LayerNorm(hidden_dim)
        self.drop1    = nn.Dropout(dropout)
        self.pool1    = IHPool(hidden_dim, ratio=pool1_ratio)

        self.conv2    = RAConv(hidden_dim, hidden_dim)
        self.norm2    = nn.LayerNorm(hidden_dim)
        self.drop2    = nn.Dropout(dropout)
        self.pool2    = IHPool(hidden_dim, ratio=pool2_ratio)

        self.lin1     = nn.Linear(hidden_dim, hidden_dim // 2)
        self.norm3    = nn.LayerNorm(hidden_dim // 2)
        self.drop3    = nn.Dropout(dropout)
        self.lin2     = nn.Linear(hidden_dim // 2, n_classes)

        assert pool_method in ('mean', 'max')
        self._pool_fn = self._mean_pool if pool_method == 'mean' else self._max_pool

    @staticmethod
    def _mean_pool(x):
        return x.mean(0, keepdim=True)   # [1, D]

    @staticmethod
    def _max_pool(x):
        return x.max(0, keepdim=True).values

    def forward(
        self,
        x:      torch.Tensor,           # [N, D]
        coords: torch.Tensor | None = None,  # [N, 2] optional spatial coords
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits  [1, n_classes]
        attn    [N]  per-patch fitness proxy (from first pool layer, for vis)
        """
        if x.dim() == 3:
            x = x.squeeze(0)            # tolerate [1, N, D] input
        if coords is not None and coords.dim() == 3:
            coords = coords.squeeze(0)

        # --- build heterogeneous graph ---
        x_g, node_type, tree, xy, edge_index = build_hetero_graph(
            x, coords, n_super=self.n_super, k_neighbors=self.k_neighbors)
        mask_l2 = node_type == 2        # [1+K+N] boolean, True for original patches

        # --- layer 1 ---
        x_g = self.norm_in(x_g)
        x_g = self.conv1(x_g, edge_index, node_type)
        x_g = F.relu(x_g)
        x_g = self.norm1(x_g)
        x_g = self.drop1(x_g)

        # --- pool 1 ---
        x_g, ei1, _, _, _, nt1, tr1, fit1, xy1 = self.pool1(
            x_g, edge_index, node_type, tree, xy)
        x1 = self._pool_fn(x_g)         # [1, D]

        # --- layer 2 ---
        x_g = self.conv2(x_g, ei1, nt1)
        x_g = F.relu(x_g)
        x_g = self.norm2(x_g)
        x_g = self.drop2(x_g)

        # --- pool 2 ---
        x_g, ei2, _, _, _, nt2, tr2, fit2, xy2 = self.pool2(
            x_g, ei1, nt1, tr1, xy1)
        x2 = self._pool_fn(x_g)         # [1, D]

        # --- classifier ---
        h = x1 + x2                     # [1, D]
        h = F.relu(self.lin1(h))
        h = self.norm3(h)
        h = self.drop3(h)
        logits = self.lin2(h)           # [1, n_classes]

        # fitness proxy: scores of the original N level-2 (patch) nodes from pool1
        attn = fit1[mask_l2]            # [N]
        return logits, attn


def build_h2mil(in_dim: int, cls_num: int) -> H2MIL:
    return H2MIL(
        in_dim=in_dim,
        hidden_dim=256,
        n_classes=cls_num,
        pool1_ratio=0.1,
        pool2_ratio=4.0,
        dropout=0.3,
        k_neighbors=8,
    )
