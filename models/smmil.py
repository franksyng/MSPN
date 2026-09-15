import torch
import torch.nn as nn
import torch.nn.functional as F


class ApproxSm(nn.Module):

    def __init__(self, alpha: float = 0.5, num_steps: int = 10):
        super().__init__()
        if alpha == "trainable":
            self.coef = nn.Parameter(torch.tensor(1.0))
        else:
            self.coef = 1.0 / (1.0 - alpha) - 1.0
        self.num_steps = num_steps

    def forward(self, f: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # f: [B, N, D], A: [B, N, N] row-normalised adjacency
        A = A.to(f.device)
        alpha = 1.0 / (1.0 + self.coef)
        g = f
        for _ in range(self.num_steps):
            g = (1.0 - alpha) * f + alpha * torch.bmm(A, g)
        return g


class ExactSm(nn.Module):

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        if alpha == "trainable":
            self.coef = nn.Parameter(torch.tensor(1.0))
        else:
            self.coef = 1.0 / (1.0 - alpha) - 1.0

    def forward(self, f: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # f: [B, N, D], A: [B, N, N] row-normalised adjacency
        A = A.to(f.device)
        B, N, _ = f.shape
        I = torch.eye(N, device=f.device, dtype=f.dtype).unsqueeze(0).expand(B, -1, -1)
        M = (1.0 + self.coef) * I - self.coef * A
        return torch.linalg.solve(M, f)


def build_knn_adjacency(x: torch.Tensor, k: int) -> torch.Tensor:

    B, N, _ = x.shape
    k = min(k, N - 1)

    x_norm = F.normalize(x, dim=-1)
    sim = torch.bmm(x_norm, x_norm.transpose(1, 2))  # [B, N, N]

    # Mask self-similarity
    diag_mask = torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0)
    sim = sim.masked_fill(diag_mask, float("-inf"))

    _, topk_idx = sim.topk(k, dim=-1)  # [B, N, k]
    A = torch.zeros(B, N, N, device=x.device, dtype=x.dtype)
    A.scatter_(-1, topk_idx, 1.0 / k)
    return A


def build_knn_adjacency_from_coords(coords: torch.Tensor, k: int) -> torch.Tensor:

    B, N, _ = coords.shape
    k = min(k, N - 1)

    # Pairwise squared Euclidean distances: [B, N, N]
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)       # [B, N, N, C]
    dist = (diff * diff).sum(dim=-1)                        # [B, N, N]

    diag_mask = torch.eye(N, device=coords.device, dtype=torch.bool).unsqueeze(0)
    dist = dist.masked_fill(diag_mask, float("inf"))

    _, topk_idx = dist.topk(k, dim=-1, largest=False)      # [B, N, k]
    A = torch.zeros(B, N, N, device=coords.device, dtype=coords.dtype)
    A.scatter_(-1, topk_idx, 1.0 / k)
    return A


class SmMIL(nn.Module):
    """
    "Sm: enhanced localization in Multiple Instance Learning for medical imaging classification", NeurIPS 2024.
    https://github.com/Franblueee/SmMIL
    """

    def __init__(
        self,
        in_dim: int,
        emb_dim: int = 512,
        att_dim: int = 128,
        num_classes: int = 1,
        sm_alpha: float = 0.5,
        sm_mode: str = "approx",
        sm_steps: int = 10,
        sm_where: str = "early",
        knn_k: int = 8,
        max_nodes: int | None = 10000,
    ):
        super().__init__()
        assert sm_where in ("early", "mid", "late"), "sm_where must be 'early', 'mid', or 'late'"

        self.sm_where = sm_where
        self.knn_k = knn_k
        self.max_nodes = max_nodes

        if sm_mode == "approx":
            self.sm = ApproxSm(alpha=sm_alpha, num_steps=sm_steps)
        elif sm_mode == "exact":
            self.sm = ExactSm(alpha=sm_alpha)
        else:
            raise ValueError("sm_mode must be 'approx' or 'exact'")

        self.fc_proj = nn.Sequential(nn.Linear(in_dim, emb_dim), nn.ReLU())

        # Gated attention (ABMIL)
        self.att_V = nn.Sequential(nn.Linear(emb_dim, att_dim), nn.Tanh())
        self.att_U = nn.Sequential(nn.Linear(emb_dim, att_dim), nn.Sigmoid())
        self.att_w = nn.Linear(att_dim, 1, bias=False)

        self.classifier = nn.Linear(emb_dim, num_classes)

    def _adjacency(self, x: torch.Tensor, adj_mat) -> torch.Tensor:
        if adj_mat is None:
            return build_knn_adjacency(x, self.knn_k)
        if adj_mat.dim() == 3 and adj_mat.shape[1] == adj_mat.shape[2]:
            return adj_mat  # already [B, N, N]
        # [N, C] coords — add batch dim, build, then expand to match x
        if adj_mat.dim() == 2:
            adj_mat = adj_mat.unsqueeze(0)  # [1, N, C]
        A = build_knn_adjacency_from_coords(adj_mat, self.knn_k)  # [1, N, N] or [B, N, N]
        return A.expand(x.shape[0], -1, -1)  # broadcast over batch

    def forward(
        self,
        x: torch.Tensor,
        adj_mat: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:       [B, N, D] instance features.
            adj_mat: [B, N, N] row-normalised adjacency, or [B, N, C] spatial
                     coordinates. Built automatically via kNN on x when not provided.
            mask:    [B, N] bool tensor, True = valid instance (optional).
        Returns:
            logits:  [B, num_classes]
        """
        _, N_orig, _ = x.shape
        if self.max_nodes is not None and N_orig > self.max_nodes:
            sub_idx = torch.randperm(N_orig)[:self.max_nodes].sort().values  # CPU
            x = x[:, sub_idx.to(x.device)]
            if mask is not None:
                mask = mask[:, sub_idx.to(mask.device)]
            if adj_mat is not None:
                sidx = sub_idx.to(adj_mat.device)
                if adj_mat.dim() == 3 and adj_mat.shape[-2] == N_orig and adj_mat.shape[-1] == N_orig:
                    adj_mat = adj_mat[:, sidx][:, :, sidx]   # [B, N, N] → subsample both axes
                else:
                    adj_mat = adj_mat[:, sidx]                # [B, N, C] coords → subsample rows

        A = self._adjacency(x, adj_mat)

        if self.sm_where == "early":
            x = self.sm(x, A)

        H = self.fc_proj(x)  # [B, N, emb_dim]

        if self.sm_where == "mid":
            H = self.sm(H, A)

        att = self.att_w(self.att_V(H) * self.att_U(H))  # [B, N, 1]

        if self.sm_where == "late":
            att = self.sm(att, A)

        if mask is not None:
            att = att.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        att = torch.softmax(att, dim=1)  # [B, N, 1]
        z = (att * H).sum(dim=1)        # [B, emb_dim]

        return self.classifier(z), None       # [B, num_classes]
