import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from nystrom_attention import NystromAttention


class FeedForwardNetwork(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class GatedAttention(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Handle both batched and unbatched inputs
        squeeze = False
        if h.dim() == 2:
            h = h.unsqueeze(0)
            squeeze = True
            
        # Gated attention mechanism
        A_V = self.attention_V(h)  # [B, N, hidden]
        A_U = self.attention_U(h)  # [B, N, hidden]
        A = self.attention_w(A_V * A_U)  # [B, N, 1]
        A = self.dropout(A)
        
        # Normalize attention scores
        A_normalized = F.softmax(A, dim=1)  # [B, N, 1]
        
        # Aggregate: bag_repr = sum(a_i * h_i)
        bag_repr = torch.bmm(A_normalized.transpose(1, 2), h)  # [B, 1, D]
        
        if squeeze:
            A = A.squeeze(0)
            bag_repr = bag_repr.squeeze(0)
            
        return A, bag_repr


class IntegratedAttentionModule(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 8,
        num_landmarks: int = 64,
        attn_dropout: float = 0.3,
        agg_dropout: float = 0.25,
        use_official_nystrom: bool = True
    ):
        super().__init__()
        
        # FNN for dimension transformation
        self.fnn = FeedForwardNetwork(in_dim, out_dim)
        
        # Nyström attention for efficiency (use official package if available)
        self.attention = NystromAttention(
            dim = out_dim,
            dim_head = out_dim//8,
            heads = 8,
            num_landmarks = out_dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )
        
        # Post-attention normalization
        self.norm = nn.LayerNorm(out_dim)
        
        # Gated attention for bag aggregation
        self.aggregation = GatedAttention(out_dim, hidden_dim=out_dim // 2, dropout=agg_dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
            
        # Transform features
        h = self.fnn(x)
        
        # Self-attention for context
        h = self.attention(h)
        h = self.norm(h)
        
        # Get attention scores and bag representation
        attn_scores, bag_repr = self.aggregation(h)
        
        if squeeze:
            h = h.squeeze(0)
            
        return h, attn_scores, bag_repr


class IntegratedAttentionTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int = 1024,
        hidden_dims: List[int] = [1024, 1536, 512, 1024],
        n_classes: int = 2,
        num_heads: int = 8,
        num_landmarks: int = 64,
        attn_dropout: float = 0.3,
        agg_dropout: float = 0.25,
        use_official_nystrom: bool = True
    ):
        super().__init__()
        
        self.n_classes = n_classes
        self.num_modules = len(hidden_dims)
        
        # Build IAM modules
        dims = [in_dim] + hidden_dims
        self.iam_modules = nn.ModuleList([
            IntegratedAttentionModule(
                in_dim=dims[i],
                out_dim=dims[i + 1],
                num_heads=num_heads,
                num_landmarks=num_landmarks,
                attn_dropout=attn_dropout,
                agg_dropout=agg_dropout,
                use_official_nystrom=use_official_nystrom
            )
            for i in range(self.num_modules)
        ])
        
        # Learnable weights for fusing bag representations
        self.bag_fusion_weights = nn.Parameter(torch.ones(self.num_modules) / self.num_modules)

        _fd = hidden_dims[-1]
        self.fusion_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(d, _fd), nn.LayerNorm(_fd), nn.GELU())
            for d in hidden_dims
        ])
        
        # Classifier for bag-level prediction
        final_dim = hidden_dims[-1]
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(final_dim // 2, n_classes)
        )
        
        # Instance-level classifier for patch loss
        self.instance_classifier = nn.Linear(final_dim, n_classes)
        
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
            
        B = x.shape[0]
        
        bag_representations = []
        h = x
        final_attn = None
        
        # Pass through IAM modules
        for i, iam in enumerate(self.iam_modules):
            h, attn_scores, bag_repr = iam(h)
            bag_representations.append(bag_repr)
            
            # Keep attention from last layer
            if i == self.num_modules - 1:
                final_attn = attn_scores
        
        # Fuse bag representations with learned weights
        weights = F.softmax(self.bag_fusion_weights, dim=0)
        
        # Stack and weight bag representations
        # Each bag_repr is [B, 1, D], need to project to same dimension first
        final_dim = bag_representations[-1].shape[-1]
        fused_bag = torch.zeros(B, 1, final_dim, device=x.device)
        
        for i, bag_repr in enumerate(bag_representations):
            # registered in __init__ -- trained, saved, deterministic
            bag_repr = self.fusion_proj[i](bag_repr)
            fused_bag = fused_bag + weights[i] * bag_repr
            
        fused_bag = fused_bag.squeeze(1)  # [B, D]
        
        # Classification
        logits = self.classifier(fused_bag)
        Y_prob = F.softmax(logits, dim=-1)
        Y_hat = torch.argmax(Y_prob, dim=-1)
        
        # Get instance predictions for top-k instances (for patch loss)
        instance_logits = self.instance_classifier(h)  # [B, N, n_classes]
        
        result = {
            'logits': logits,
            'Y_prob': Y_prob,
            'Y_hat': Y_hat,
            'attention': final_attn.squeeze(-1) if final_attn is not None else None,
            'instance_logits': instance_logits,
        }
        
        if return_features:
            result['bag_representations'] = bag_representations
            result['fused_bag'] = fused_bag
            result['instance_features'] = h
            
        if squeeze:
            for k in result:
                if result[k] is not None and isinstance(result[k], torch.Tensor):
                    if result[k].shape[0] == 1:
                        result[k] = result[k].squeeze(0)
                        
        return result


class HAGMIL(nn.Module):
    
    def __init__(
        self,
        in_dim: int = 1024,
        hidden_dims: List[int] = [1024, 1536, 512, 1024],
        n_classes: int = 2,
        num_levels: int = 3,
        k_per_level: Optional[List[int]] = None,
        num_heads: int = 8,
        num_landmarks: int = 64,
        attn_dropout: float = 0.3,
        agg_dropout: float = 0.25,
        label_smoothing: float = 0.1,
        patch_loss_weight: float = 1.0,
        use_official_nystrom: bool = True
    ):
        """
        Args:
            in_dim: Input feature dimension (from pretrained encoder)
            hidden_dims: List of hidden dimensions for IAT modules
            n_classes: Number of output classes
            num_levels: Number of resolution levels (default: 3 for levels 0, 1, 2)
            k_per_level: Number of patches to keep at each level [k2, k1]
                        (not needed for level 0 as we use all selected patches)
            num_heads: Number of attention heads in transformer
            num_landmarks: Number of landmarks for Nyström attention
            attn_dropout: Dropout for attention layers
            agg_dropout: Dropout for aggregation layers
            label_smoothing: Label smoothing for lower resolution losses
            patch_loss_weight: Weight for patch-level loss (lambda in paper)
            use_official_nystrom: Use official nystrom_attention package if available
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.num_levels = num_levels
        self.k_per_level = k_per_level or [1000, 4000]  # Default: keep top 1000 at level 2, 4000 at level 1
        self.label_smoothing = label_smoothing
        self.patch_loss_weight = patch_loss_weight
        
        # Create IAT for each resolution level
        self.level_models = nn.ModuleDict({
            f'level_{i}': IntegratedAttentionTransformer(
                in_dim=in_dim,
                hidden_dims=hidden_dims,
                n_classes=n_classes,
                num_heads=num_heads,
                num_landmarks=num_landmarks,
                attn_dropout=attn_dropout,
                agg_dropout=agg_dropout,
                use_official_nystrom=use_official_nystrom
            )
            for i in range(num_levels)
        })
        
    def expand_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Expand indices from lower resolution to higher resolution.
        Each patch at level L corresponds to 4 patches at level L-1 (quadtree).
        
        Args:
            indices: Patch indices at lower resolution [K]
            
        Returns:
            Expanded indices at higher resolution [4*K]
        """
        # For each index i, the corresponding 4 sub-patches are: 4i, 4i+1, 4i+2, 4i+3
        base = indices * 4
        expanded = torch.stack([base, base + 1, base + 2, base + 3], dim=-1)
        return expanded.flatten()
    
    def get_top_k_indices(
        self,
        attention: torch.Tensor,
        k: int
    ) -> torch.Tensor:
        """
        Get indices of top-k patches by attention score.
        
        Args:
            attention: Attention scores [N] or [B, N]
            k: Number of patches to select
            
        Returns:
            indices: Top-k indices sorted by attention [K]
        """
        if attention.dim() == 1:
            k = min(k, attention.shape[0])
            _, indices = torch.topk(attention, k, sorted=True)
        else:
            k = min(k, attention.shape[1])
            _, indices = torch.topk(attention, k, dim=1, sorted=True)
        return indices
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        label: torch.Tensor,
        instance_logits: Optional[torch.Tensor] = None,
        attention: Optional[torch.Tensor] = None,
        k_patch: int = 8,
        use_label_smoothing: bool = False
    ) -> torch.Tensor:
        """
        Compute combined bag and patch loss.
        
        Args:
            logits: Bag-level logits [B, n_classes] or [n_classes]
            label: Ground truth label [B] or scalar
            instance_logits: Instance-level logits [B, N, n_classes]
            attention: Attention scores [B, N] or [N]
            k_patch: Number of top patches for patch loss
            use_label_smoothing: Whether to use label smoothing
            
        Returns:
            Total loss
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if isinstance(label, int) or label.dim() == 0:
            label = torch.tensor([label], device=logits.device)
            
        # Bag-level cross entropy loss
        if use_label_smoothing and self.label_smoothing > 0:
            bag_loss = F.cross_entropy(
                logits, label, 
                label_smoothing=self.label_smoothing
            )
        else:
            bag_loss = F.cross_entropy(logits, label)
        
        # Patch-level loss on top-k attention patches
        patch_loss = torch.tensor(0.0, device=logits.device)
        if instance_logits is not None and attention is not None:
            if instance_logits.dim() == 2:
                instance_logits = instance_logits.unsqueeze(0)
            if attention.dim() == 1:
                attention = attention.unsqueeze(0)
                
            B = instance_logits.shape[0]
            
            for b in range(B):
                top_k_idx = self.get_top_k_indices(attention[b], k_patch)
                top_k_logits = instance_logits[b, top_k_idx]  # [k, n_classes]
                
                # All top-k patches should have the same label as the bag
                patch_labels = label[b].expand(len(top_k_idx))
                
                # Use smooth SVM loss as in paper (approximate with hinge loss)
                # For simplicity, using cross entropy here
                patch_loss = patch_loss + F.cross_entropy(top_k_logits, patch_labels)
                
            patch_loss = patch_loss / B
            
        total_loss = bag_loss + self.patch_loss_weight * patch_loss
        return total_loss
    
    def forward_single_level(
        self,
        x: torch.Tensor,
        level: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for single resolution level.
        
        Args:
            x: Features [N, D] or [B, N, D]
            level: Resolution level index
            
        Returns:
            Model outputs
        """
        model = self.level_models[f'level_{level}']
        return model(x, return_features=True)
    
    def forward(
        self,
        x,
        label: Optional[torch.Tensor] = None,
        return_all_levels: bool = False,
        inference_k_multiplier: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Hierarchical forward pass through all resolution levels.
        
        Args:
            features_dict: Dict mapping level names to feature tensors
                         {'level_2': [N2, D], 'level_1': [N1, D], 'level_0': [N0, D]}
            x: Alternative single-level input [N, D] (uses level_0 model only)
            label: Ground truth label for computing loss
            return_all_levels: Whether to return outputs from all levels
            inference_k_multiplier: Multiplier for k during inference (can increase for better results)
            
        Returns:
            Dict containing final predictions and optionally per-level outputs
        """
        # # Handle single-level input
        # if features_dict is None and x is not None:
        #     return self.forward_single_level(x, level=0)
        
        # if features_dict is None:
        #     raise ValueError("Must provide either features_dict or x")
        x_5x, x_10x, x_20x = x
        features_dict = {
        'level_2': x_5x,   # Lowest resolution
        'level_1': x_10x,  # Medium resolution
        'level_0': x_20x,  # Highest resolution
        }

        results = {
            'attention_per_level': {},
            'selected_indices_per_level': {},
            'outputs_per_level': {} if return_all_levels else None,
            'losses_per_level': {} if label is not None else None
        }
        
        # Start from lowest resolution (level 2) and work up to highest (level 0)
        current_indices = None
        
        for level in range(self.num_levels - 1, -1, -1):
            level_key = f'level_{level}'
            
            # Get features for this level
            if level_key not in features_dict:
                continue
                
            features = features_dict[level_key]
            
            # If we have indices from previous level, select corresponding patches
            if current_indices is not None:
                # Filter indices to valid range
                valid_mask = current_indices < features.shape[0]
                current_indices = current_indices[valid_mask]
                
                # Handle edge case: if no valid indices remain, use all features
                if len(current_indices) == 0:
                    import warnings
                    warnings.warn(
                        f"No valid indices for {level_key} after expansion. "
                        f"Using all {features.shape[0]} patches. "
                        "Consider increasing k_per_level values."
                    )
                    # Don't filter - use all features
                else:
                    features = features[current_indices]
            
            # Ensure we have at least 1 patch (safety check)
            if features.shape[0] == 0:
                raise ValueError(
                    f"Empty feature tensor at {level_key}. "
                    "Check your data or increase k_per_level values."
                )
            
            # Forward through level model
            outputs = self.forward_single_level(features, level=level)
            
            # Store attention scores
            results['attention_per_level'][level_key] = outputs['attention']
            
            if return_all_levels:
                results['outputs_per_level'][level_key] = outputs
                
            # Compute loss if label provided
            if label is not None:
                use_smoothing = level > 0  # Use label smoothing for lower resolutions
                loss = self.compute_loss(
                    outputs['logits'],
                    label,
                    outputs['instance_logits'],
                    outputs['attention'],
                    use_label_smoothing=use_smoothing
                )
                results['losses_per_level'][level_key] = loss
            
            # Select top-k patches for next level (if not at highest resolution)
            if level > 0:
                k = int(self.k_per_level[level - 1] * inference_k_multiplier)
                top_k_indices = self.get_top_k_indices(outputs['attention'], k)
                
                # Expand to higher resolution indices
                expanded_indices = self.expand_indices(top_k_indices)
                
                # Store for next iteration (clipping happens at start of next level)
                current_indices = expanded_indices
                results['selected_indices_per_level'][level_key] = top_k_indices
        
        # Final outputs come from highest resolution (level 0)
        final_level = 'level_0' if 'level_0' in features_dict else list(features_dict.keys())[-1]
        final_outputs = results['outputs_per_level'][final_level] if return_all_levels else outputs
        
        results['logits'] = final_outputs['logits']
        results['Y_prob'] = final_outputs['Y_prob']
        results['Y_hat'] = final_outputs['Y_hat']
        results['attention'] = final_outputs['attention']
        
        # Combined loss (sum of all level losses)
        if label is not None:
            results['loss'] = sum(results['losses_per_level'].values())
            
        if label is not None:
            return results['logits'], results['loss']
        else:
            return results['logits']