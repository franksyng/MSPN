import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
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
        squeeze = False
        if h.dim() == 2:
            h = h.unsqueeze(0)
            squeeze = True
            
        A_V = self.attention_V(h)  # b n h
        A_U = self.attention_U(h)  # b n h
        A = self.attention_w(A_V * A_U)  # b n 1
        A = self.dropout(A)
        
        A_normalized = F.softmax(A, dim=1)
        
        bag_repr = torch.bmm(A_normalized.transpose(1, 2), h)
        
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
        
        self.fnn = FeedForwardNetwork(in_dim, out_dim)
        
        self.attention = NystromAttention(
            dim = out_dim,
            dim_head = out_dim//8,
            heads = 8,
            num_landmarks = out_dim//2,
            pinv_iterations = 6,
            residual = True,
            dropout=0.1
        )
        
        self.norm = nn.LayerNorm(out_dim)
        
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
        
        self.bag_fusion_weights = nn.Parameter(torch.ones(self.num_modules) / self.num_modules)
        
        final_dim = hidden_dims[-1]
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(final_dim // 2, n_classes)
        )
        
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
        
        for i, iam in enumerate(self.iam_modules):
            h, attn_scores, bag_repr = iam(h)
            bag_representations.append(bag_repr)
            
            if i == self.num_modules - 1:
                final_attn = attn_scores
        
        weights = F.softmax(self.bag_fusion_weights, dim=0)
        
        final_dim = bag_representations[-1].shape[-1]
        fused_bag = torch.zeros(B, 1, final_dim, device=x.device)
        
        for i, bag_repr in enumerate(bag_representations):
            if bag_repr.shape[-1] != final_dim:
                proj = nn.Linear(bag_repr.shape[-1], final_dim).to(x.device)
                bag_repr = proj(bag_repr)
            fused_bag = fused_bag + weights[i] * bag_repr
            
        fused_bag = fused_bag.squeeze(1)
        
        logits = self.classifier(fused_bag)
        Y_prob = F.softmax(logits, dim=-1)
        Y_hat = torch.argmax(Y_prob, dim=-1)
        
        instance_logits = self.instance_classifier(h)
        
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
        super().__init__()
        
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.num_levels = num_levels
        self.k_per_level = k_per_level
        self.label_smoothing = label_smoothing
        self.patch_loss_weight = patch_loss_weight
        
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
        base = indices * 4
        expanded = torch.stack([base, base + 1, base + 2, base + 3], dim=-1)
        return expanded.flatten()
    
    def get_top_k_indices(
        self,
        attention: torch.Tensor,
        k: int
    ) -> torch.Tensor:
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
        
        patch_loss = torch.tensor(0.0, device=logits.device)
        if instance_logits is not None and attention is not None:
            if instance_logits.dim() == 2:
                instance_logits = instance_logits.unsqueeze(0)
            if attention.dim() == 1:
                attention = attention.unsqueeze(0)
                
            B = instance_logits.shape[0]
            
            for b in range(B):
                top_k_idx = self.get_top_k_indices(attention[b], k_patch)
                top_k_logits = instance_logits[b, top_k_idx]
                
                patch_labels = label[b].expand(len(top_k_idx))
                patch_loss = patch_loss + F.cross_entropy(top_k_logits, patch_labels)
                
            patch_loss = patch_loss / B
            
        total_loss = bag_loss + self.patch_loss_weight * patch_loss
        return total_loss
    
    def forward_single_level(
        self,
        x: torch.Tensor,
        level: int = 0
    ) -> Dict[str, torch.Tensor]:
        model = self.level_models[f'level_{level}']
        return model(x, return_features=True)
    
    def forward(
        self,
        x,
        label: Optional[torch.Tensor] = None,
        return_all_levels: bool = False,
        inference_k_multiplier: float = 1.0
    ) -> Dict[str, torch.Tensor]:
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
        
        current_indices = None
        
        for level in range(self.num_levels - 1, -1, -1):
            level_key = f'level_{level}'
            
            if level_key not in features_dict:
                continue
                
            features = features_dict[level_key]
            
            if current_indices is not None:
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
                else:
                    features = features[current_indices]
            
            if features.shape[0] == 0:
                raise ValueError(
                    f"Empty feature tensor at {level_key}. "
                    "Check your data or increase k_per_level values."
                )
            
            outputs = self.forward_single_level(features, level=level)
            
            results['attention_per_level'][level_key] = outputs['attention']
            
            if return_all_levels:
                results['outputs_per_level'][level_key] = outputs
                
            if label is not None:
                use_smoothing = level > 0
                loss = self.compute_loss(
                    outputs['logits'],
                    label,
                    outputs['instance_logits'],
                    outputs['attention'],
                    use_label_smoothing=use_smoothing
                )
                results['losses_per_level'][level_key] = loss
            
            if level > 0:
                k = int(self.k_per_level[level - 1] * inference_k_multiplier)
                top_k_indices = self.get_top_k_indices(outputs['attention'], k)
                
                expanded_indices = self.expand_indices(top_k_indices)
                
                current_indices = expanded_indices
                results['selected_indices_per_level'][level_key] = top_k_indices
        
        final_level = 'level_0' if 'level_0' in features_dict else list(features_dict.keys())[-1]
        final_outputs = results['outputs_per_level'][final_level] if return_all_levels else outputs
        
        results['logits'] = final_outputs['logits']
        results['Y_prob'] = final_outputs['Y_prob']
        results['Y_hat'] = final_outputs['Y_hat']
        results['attention'] = final_outputs['attention']
        
        if label is not None:
            results['loss'] = sum(results['losses_per_level'].values())
            
            
        return results['logits'], results['loss']