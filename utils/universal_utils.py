import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from collections import defaultdict, OrderedDict
import random
import os
import numpy as np
import math

def setup_seed(seed, device):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def setup_seed_surv(seed, device):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(False)

def create_dir(dir_path):
    """
    Check folder exist or not. If not, create one.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


class L1Reg:
    def __init__(self):
        self.w_reg = None

    def l1_reg(self, model):
        w_reg = None
        for w in model.parameters():
            if w_reg is None:
                w_reg = torch.sum(torch.abs(w))
            else:
                w_reg = w_reg + torch.sum(torch.abs(w))
        return w_reg

    def apply_reg(self, model):
        self.w_reg = 0
        self.w_reg += self.l1_reg(model)
        return self.w_reg


class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.defaults.update(defaults)
        self.state = defaultdict(dict)
        # manually add our defaults to the param groups
        for name, default in defaults.items():
            for group in self.param_groups:
                group.setdefault(name, default)
        
        # add hooks for scheduler update
        self._optimizer_step_pre_hooks = OrderedDict()
        self._optimizer_step_post_hooks = OrderedDict()
        self._optimizer_state_dict_pre_hooks = OrderedDict()
        self._optimizer_state_dict_post_hooks = OrderedDict()
        self._optimizer_load_state_dict_pre_hooks = OrderedDict()
        self._optimizer_load_state_dict_post_hooks = OrderedDict()

    def update_slow(self, group):
        for fast_p in group["params"]:
            if fast_p.grad is None:
                continue
            param_state = self.state[fast_p]
            if 'slow_buffer' not in param_state:
                param_state['slow_buffer'] = torch.empty_like(fast_p.data)
                param_state['slow_buffer'].copy_(fast_p.data)
            slow = param_state['slow_buffer']
            # slow.add_(group['lookahead_alpha'], fast_p.data - slow)
            slow.add_(fast_p.data - slow, alpha=group['lookahead_alpha'])
            fast_p.data.copy_(slow)

    def sync_lookahead(self):
        for group in self.param_groups:
            self.update_slow(group)

    def step(self, closure=None):
        #assert id(self.param_groups) == id(self.base_optimizer.param_groups)
        loss = self.base_optimizer.step(closure)
        for group in self.param_groups:
            group['lookahead_step'] += 1
            if group['lookahead_step'] % group['lookahead_k'] == 0:
                self.update_slow(group)
        return loss

    def state_dict(self):
        fast_state_dict = self.base_optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict['state']
        param_groups = fast_state_dict['param_groups']
        return {
            'state': fast_state,
            'slow_state': slow_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        fast_state_dict = {
            'state': state_dict['state'],
            'param_groups': state_dict['param_groups'],
        }
        self.base_optimizer.load_state_dict(fast_state_dict)

        # We want to restore the slow state, but share param_groups reference
        # with base_optimizer. This is a bit redundant but least code
        slow_state_new = False
        if 'slow_state' not in state_dict:
            print('Loading state_dict from optimizer without Lookahead applied.')
            state_dict['slow_state'] = defaultdict(dict)
            slow_state_new = True
        slow_state_dict = {
            'state': state_dict['slow_state'],
            'param_groups': state_dict['param_groups'],  # this is pointless but saves code
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.param_groups = self.base_optimizer.param_groups  # make both ref same container
        if slow_state_new:
            # reapply defaults to catch missing lookahead specific ones
            for name, default in self.defaults.items():
                for group in self.param_groups:
                    group.setdefault(name, default)

def mspn_constrain_loss(coarse_maps):
    scales = list(coarse_maps.keys())
    proposal_loss = 0.0
    for i in range(len(scales) - 1):
        t_logits = coarse_maps[scales[i]].detach()      # teacher
        s_logits = coarse_maps[scales[i + 1]]           # student

        P_t = F.softmax(t_logits, dim=1)          # [1,N]
        P_s = F.log_softmax(s_logits, dim=1)          # [1,N]

        # KL(teacher || student)
        proposal_loss = proposal_loss + F.kl_div(P_s, P_t, reduction="batchmean")
    return proposal_loss

def mspn_contain_loss(coarse_map_dict, top_p=0.3):
    scales = sorted(coarse_map_dict.keys(), reverse=True)  # e.g. [3072, 2048, 1536]
    if len(scales) < 2:
        return torch.tensor(0.0, device=next(iter(coarse_map_dict.values())).device)

    loss = []
    for i in range(len(scales) - 1):
        z_c = coarse_map_dict[scales[i]]       # [1, N]
        z_f = coarse_map_dict[scales[i + 1]]   # [1, N]
        loss.append(contain_loss_from_logits(z_c, z_f, top_p=top_p))
    return torch.stack(loss).mean()

def probs_from_logits(logits_1xN, tau=1.0, eps=1e-12):
    """
    logits_1xN: [1, N] (or [B, N])
    returns probs: [1, N] summing to 1 per row
    """
    return F.softmax(logits_1xN / max(tau, eps), dim=1)

def contain_loss_from_logits(
    coarse_logits,              # [1, N] teacher (coarser FOV)
    fine_logits,                # [1, N] student (finer FOV)
    top_p=0.3,                  # keep a generous support (0.2~0.6 typical)
    detach_teacher=True,
):
    """
    L_contain = sum_{i not in support(coarse)} P_fine(i)

    support(coarse) chosen by top_p fraction of coarse probabilities.
    """
    # Convert to probabilities
    if detach_teacher:
        P_coarse = probs_from_logits(coarse_logits.detach())
    else:
        P_coarse = probs_from_logits(coarse_logits)

    P_fine = probs_from_logits(fine_logits)

    B, N = P_coarse.shape
    k = max(1, int(torch.ceil(torch.tensor(top_p * N)).item()))

    # top-k support mask per sample
    _, idx = torch.topk(P_coarse, k=k, dim=1, largest=True, sorted=False)  # [B, k]
    mask = torch.zeros_like(P_coarse, dtype=torch.bool)                    # [B, N]
    mask.scatter_(1, idx, True)

    # mass outside coarse support
    outside_mass = (P_fine * (~mask).to(P_fine.dtype)).sum(dim=1)          # [B]
    return outside_mass.mean()

def find_coords_scale(x, y, eps=1e-8):
    # max_xy = np.maximum(np.max(x), np.max(y))
    # min_xy = np.minimum(np.min(x), np.min(y))
    scale_x = np.maximum(np.max(x) - np.min(x), eps)
    scale_y = np.maximum(np.max(y) - np.min(y), eps)
    scale = np.maximum(scale_x, scale_y)
    return scale

def contextual_recon(coords, patch_scale=512):
    x = coords[..., 0]
    y = coords[..., 1]
    x_min = torch.min(x)
    # x_max = torch.max(x)
    y_min = torch.min(y)
    # y_max = torch.max(y)
    x_norm = ((x - x_min)/patch_scale).int()
    y_norm = ((y - y_min)/patch_scale).int()
    # max_xy = np.maximum(np.max(x), np.max(y))
    # min_xy = np.minimum(np.min(x), np.min(y))
    # scale_x = np.maximum(np.max(x) - np.min(x), eps)
    # scale_y = np.maximum(np.max(y) - np.min(y), eps)
    # scale = np.maximum(scale_x, scale_y)
    return x_norm.view(-1), y_norm.view(-1)

def min_max_scaler(coord, scale):
    return (coord-np.min(coord))/scale

def positional_encoding(h, d_model, max_len, device):
    position = torch.arange(max_len).unsqueeze(1)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
    pe = torch.zeros(max_len, 1, d_model).to(device)
    pe[:, 0, 0::2] = torch.sin(position * inv_freq)
    pe[:, 0, 1::2] = torch.cos(position * inv_freq)
    # return F.dropout(h + pe[:h.size(0)], p=0.1)
    return F.dropout(h + pe[:h.size(0)], p=0.1)

class PE2D(nn.Module):
    def __init__(self, dim=512):
        super(PE2D, self).__init__()
        # if dim % 6 != 0:
        #     raise ValueError("Embed dimension must be divisible by 6.")
        self.dim = dim
        # self.div_term = torch.exp(torch.arange(0., dim // 4, 2) * -(math.log(10000) / (dim // 4)))
        # self.div_term = torch.exp(torch.arange(0., dim // 3, 2) * -(math.log(10000) / (dim // 3)))
        self.div_term = torch.exp(torch.arange(0., dim // 2, 2) * -(math.log(10000) / (dim // 2)))
        self.dropout = nn.Dropout(0.1)
        # self.div_term = torch.exp(torch.arange(0., dim, 2) * -(math.log(10000) / dim))
        # self.proj = nn.Conv2d(1, dim, 3, 1, 3//2)
        # self.proj = nn.Linear(dim, 512)

    def forward(self, h, coords, device):
        """
        coords_logits: (batch_size, num_points, 3) -> [x, y, logit]
        Returns: (batch_size, num_points, embed_dim)
        """
        cls_token, feat_token = h[:, 0], h[:, 1:]
        self.div_term = self.div_term.to(device)
        # print(cls_token.shape, feat_token.shape)

        x, y = norm_coords(coords)
        # A = A.squeeze(0)
        
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
            # get_pos_emb(A, self.div_term),  # logit encoding
        ], dim=-1)
        # print(feat_token.shape)
        # encoded_feat = feat_token + encoded
        encoded_feat = feat_token + encoded
        h = torch.cat((cls_token.unsqueeze(1), encoded_feat), dim=1)
        # print(h.shape)
        return self.dropout(h)  # Shape: (B, N, D)
    
def positional_encoding2d(h, d_model, coords, device):
    x, y = coords[:,:,0].unsqueeze(1), coords[:,:,1].unsqueeze(1) # [1,1,N], h:[1, N, D]
    cls_token, feat_token = h[:, 0], h[:, 1:] # cls [1, D]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model)).to(device, dtype=torch.float32)
    # print(inv_freq.dtype)
    # print(x.shape)
    # pos_x = torch.tensor(scaler.fit_transform(x.view(-1,1))).to(device, dtype=inv_freq.dtype).view(-1)
    # pos_y = torch.tensor(scaler.fit_transform(y.view(-1,1))).to(device, dtype=inv_freq.dtype).view(-1)\
    # print(x.shape, y.shape)
    x = np.array(x)
    y = np.array(y)
    scale = find_coords_scale(x, y)
    pos_x = torch.tensor(min_max_scaler(x, scale)).to(device, dtype=inv_freq.dtype).view(-1)
    pos_y = torch.tensor(min_max_scaler(y, scale)).to(device, dtype=inv_freq.dtype).view(-1)
    # print(torch.arange(h.shape[1]).shape)
    # print(pos_x.shape)
    # print(pos_x)
    # print(inv_freq.shape)
    sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
    sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq)
    # print(sin_inp_x.shape)
    emb_x = get_emb(sin_inp_x) # [1, N, D]
    emb_y = get_emb(sin_inp_y)
    # print(emb_x.shape)
    # print(cls_token.shape)
    # print(h.shape)
    # print(cls_token.shape, emb_x.shape)
    # emb_x = torch.cat((cls_token.unsqueeze(1), emb_x), dim=1)
    # emb_y = torch.cat((cls_token.unsqueeze(1), emb_y), dim=1)
    encoded_feat = feat_token + emb_x + emb_y
    # emb_x = torch.cat((cls_token.unsqueeze(1), emb_x), dim=1)
    # emb_y = torch.cat((cls_token.unsqueeze(1), emb_y), dim=1)
    # emb_a = torch.cat((cls_token.unsqueeze(1), emb_a), dim=1)
    h = torch.cat((cls_token.unsqueeze(1), encoded_feat), dim=1)
    return h

# def CRPE_scale(h, d_model, x, y, A, device):
#     cls_token, feat_token = h[:, 0], h[:, 1:] # cls [1, D]
#     inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model)).to(device, dtype=torch.float32)
#     x = np.array(x)
#     y = np.array(y)
#     # print(A.shape)
#     scale = find_coords_scale(x, y)
#     pos_x = torch.tensor(min_max_scaler(x, scale)).to(device, dtype=inv_freq.dtype).view(-1)
#     pos_y = torch.tensor(min_max_scaler(y, scale)).to(device, dtype=inv_freq.dtype).view(-1)
#     print('>>>>>', pos_x.shape, inv_freq.shape)
#     sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
#     sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq)
#     a_score = torch.einsum("i,j->ij", A.squeeze(), inv_freq)
#     emb_x = get_emb(sin_inp_x) # [1, N, D]
#     emb_y = get_emb(sin_inp_y)
#     emb_a = get_emb(a_score)
#     print(emb_x.shape)
#     print(emb_a.shape)
#     # print(emb_a.shape)
#     # encoded_feat = feat_token + emb_x + emb_y
#     encoded_feat = feat_token + emb_x + emb_y + emb_a
#     # encoded_cls = cls_token.unsqueeze(1) + emb_a
#     h = torch.cat((cls_token.unsqueeze(1), encoded_feat), dim=1)
#     # h = torch.cat((encoded_cls, encoded_feat), dim=1)
#     return h

def norm_coords(coords):
    x = coords[..., 0]
    y = coords[..., 1]
    # print('norm coords', x.shape, y.shape)
    x = np.array(x)
    y = np.array(y)
    scale = find_coords_scale(x, y)
    pos_x = torch.tensor(min_max_scaler(x, scale)).view(-1)
    pos_y = torch.tensor(min_max_scaler(y, scale)).view(-1)
    return pos_x, pos_y

def pairwise_distances(coords):
    """
    coords: (N, 2) tensor of (x, y) coordinates
    returns: (N, N) matrix of Euclidean distances
    """
    x = coords.unsqueeze(1)  # (N, 1, 2)
    y = coords.unsqueeze(0)  # (1, N, 2)
    # print(x.shape, y.shape)
    diff = x - y             # (N, N, 2)
    # print(diff.shape)

    # Use the new torch.linalg.norm (vector norm across dim=-1)
    dist = torch.linalg.norm(diff, ord=2, dim=-1)  # (N, N)
    return dist

def CRPE_scale(h, d_model, x, y, A, device):
    cls_token, feat_token = h[:, 0], h[:, 1:] # cls [1, D]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model)).to(device, dtype=torch.float32)
    x = np.array(x)
    y = np.array(y)
    # print(A.shape)
    scale = find_coords_scale(x, y)
    pos_x = torch.tensor(min_max_scaler(x, scale)).to(device, dtype=inv_freq.dtype).view(-1)
    pos_y = torch.tensor(min_max_scaler(y, scale)).to(device, dtype=inv_freq.dtype).view(-1)
    print('>>>>>', pos_x.shape, inv_freq.shape)
    sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
    sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq)
    a_score = torch.einsum("i,j->ij", A.squeeze(), inv_freq)
    # print('<<<<<', sin_inp_x.shape)
    emb_x = get_emb(sin_inp_x) # [1, N, D]
    emb_y = get_emb(sin_inp_y)
    emb_a = get_emb(a_score)
    # print(emb_x.shape)
    # print(emb_a.shape)
    # print(emb_a.shape)
    # encoded_feat = feat_token + emb_x + emb_y
    encoded_feat = feat_token + emb_x + emb_y + emb_a
    # encoded_cls = cls_token.unsqueeze(1) + emb_a
    h = torch.cat((cls_token.unsqueeze(1), encoded_feat), dim=1)
    # h = torch.cat((encoded_cls, encoded_feat), dim=1)
    return h

def get_pos_emb(pos, div_term):
    # pos = pos.unsqueeze(0)
    # angles = pos * div_term
    # print('!!!!!', pos.shape, div_term.shape)
    angles = torch.einsum("i,j->ij", pos, div_term)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).unsqueeze(0)

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1).unsqueeze(0)


def CRPE_recon(h, d_model, x, y, A, device):
    cls_token, feat_token = h[:, 0], h[:, 1:] # cls [1, D]
    x = x.to(device, dtype=torch.float32)
    y = y.to(device, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model)).to(device, dtype=torch.float32)
    # print(inv_freq.dtype)
    # print(x.shape)
    # pos_x = torch.tensor(scaler.fit_transform(x.view(-1,1))).to(device, dtype=inv_freq.dtype).view(-1)
    # pos_y = torch.tensor(scaler.fit_transform(y.view(-1,1))).to(device, dtype=inv_freq.dtype).view(-1)\
    # print(x.shape, y.shape)
    # x = np.array(x)
    # y = np.array(x)
    # scale = find_coords_scale(x, y)
    # pos_x = torch.tensor(min_max_scaler(x, scale)).to(device, dtype=inv_freq.dtype).view(-1)
    # pos_y = torch.tensor(min_max_scaler(y, scale)).to(device, dtype=inv_freq.dtype).view(-1)
    # print(torch.arange(h.shape[1]).shape)
    # print(pos_x.shape)
    # print(pos_x)
    # print(inv_freq.shape)
    pos_x, pos_y = contextual_recon(x, y)
    # print(pos_x.shape)
    # print(A.shape)
    sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
    sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq)
    a_score = torch.einsum("i,j->ij", A.squeeze(), inv_freq)
    # print(sin_inp_x.shape)
    # print(sin_inp_x.shape)
    # print(a_score.shape)
    emb_x = get_emb(sin_inp_x) # [1, N, D]
    emb_y = get_emb(sin_inp_y)
    emb_a = get_emb(a_score)
    # print(emb_x.shape)
    # print(cls_token.shape)
    # print(h.shape)
    # print(cls_token.shape, emb_x.shape)
    # print(emb_x.shape)
    encoded_feat = feat_token + emb_x + emb_y + emb_a
    # emb_x = torch.cat((cls_token.unsqueeze(1), emb_x), dim=1)
    # emb_y = torch.cat((cls_token.unsqueeze(1), emb_y), dim=1)
    # emb_a = torch.cat((cls_token.unsqueeze(1), emb_a), dim=1)
    h = torch.cat((cls_token.unsqueeze(1), encoded_feat), dim=1)
    return h

def auto_thresholding(A_ranked, th=0.9):
    while True:
        filtered_pidx = torch.squeeze(torch.nonzero(A_ranked >= th), dim=1)
        if filtered_pidx.shape[0] > 0:
            break
        else:
            th -= 0.01
        if th <= 0:
            print('No intersection found in patch-level.')
            raise RuntimeError
    return filtered_pidx

def get_patches(data, A, device, target, th, coords=None, binary=True):
    # print('>>>>>', A.shape)
    A_ranked = torch.argsort(torch.squeeze(A, dim=0), dim=0, descending=True, stable=True)/A.shape[1]
    # print(A_ranked)
    A_ranked = A_ranked.detach().cpu()
    # print('<<<< a rank', A_ranked.shape)
    filtered_pidx = auto_thresholding(A_ranked, th)
    # print('<<<<', filtered_pidx.shape)
    if binary:
    # neg bag label
        if coords != None:
            return data[:, filtered_pidx,:], coords[:, filtered_pidx,:], torch.unsqueeze(A_ranked[filtered_pidx], dim=0)
        else:
            return data[:, filtered_pidx,:], torch.unsqueeze(A_ranked[filtered_pidx], dim=0)
        # if int(target) == 0:
        #     # neg_patch_idx = torch.squeeze(torch.nonzero(A_ranked >= th), dim=1).tolist()
        #     pseudo_targets = torch.unsqueeze(torch.zeros(len(filtered_pidx)), dim=0).to(device)
        #     if coords != None:
        #         return data[:, filtered_pidx,:], coords[:, filtered_pidx,:], A[:, A_ranked, :]
        #     else:
        #         return data[:, filtered_pidx,:], pseudo_targets
        # elif int(target) == 1:
        #     # pos_patch_idx = torch.squeeze(torch.nonzero(A_ranked >= th), dim=1).tolist()
        #     pseudo_targets = torch.unsqueeze(torch.ones(len(filtered_pidx)), dim=0).to(device)
        #     if coords != None:
        #         return data[:, filtered_pidx,:], coords[:, filtered_pidx,:], pseudo_targets
        #     else:
        #         return data[:, filtered_pidx,:], pseudo_targets
        # else:
        #     print('Only binary classification supported.')
        #     NotImplementedError
    else:
        if int(target) == 0:
            # neg_patch_idx = torch.squeeze(torch.nonzero(A_ranked >= th), dim=1).tolist()
            pseudo_targets = torch.unsqueeze(torch.zeros(len(filtered_pidx)), dim=0).to(device)
            if coords != None:
                return data[:, filtered_pidx,:], coords[:, filtered_pidx,:], pseudo_targets
            else:
                return data[:, filtered_pidx,:], pseudo_targets
        elif int(target) == 1:
            # pos_patch_idx = torch.squeeze(torch.nonzero(A_ranked >= th), dim=1).tolist()
            pseudo_targets = torch.unsqueeze(torch.ones(len(filtered_pidx)), dim=0).to(device)
            if coords != None:
                return data[:, filtered_pidx,:], coords[:, filtered_pidx,:], pseudo_targets
            else:
                return data[:, filtered_pidx,:], pseudo_targets
        else:
            pseudo_targets = torch.unsqueeze(torch.Tensor(len(filtered_pidx) * [int(target)]), dim=0).to(device)
            if coords != None:
                return data[:, filtered_pidx,:], coords[:, filtered_pidx,:], pseudo_targets
            else:
                return data[:, filtered_pidx,:], pseudo_targets

# def get_patches(data, A, device, target, th, coords=None, binary=True):
#     # print('>>>>>', A.shape)
#     A_ranked = torch.argsort(torch.squeeze(A, dim=0), dim=0, descending=True, stable=True)/A.shape[1]
#     # print(A_ranked)
#     A_ranked = A_ranked.detach().cpu()
#     # print('<<<< a rank', A_ranked.shape)
#     filtered_pidx = auto_thresholding(A_ranked, th)
#     # print('<<<<', filtered_pidx.shape)
#     if binary:
#     # neg bag label
#         if int(target) == 0:
#             # neg_patch_idx = torch.squeeze(torch.nonzero(A_ranked >= th), dim=1).tolist()
#             pseudo_targets = torch.unsqueeze(torch.zeros(len(filtered_pidx)), dim=0).to(device)
#             if coords != None:
#                 return data[:, filtered_pidx,:], coords[:, filtered_pidx,:], pseudo_targets
#             else:
#                 return data[:, filtered_pidx,:], pseudo_targets
#         elif int(target) == 1:
#             # pos_patch_idx = torch.squeeze(torch.nonzero(A_ranked >= th), dim=1).tolist()
#             pseudo_targets = torch.unsqueeze(torch.ones(len(filtered_pidx)), dim=0).to(device)
#             if coords != None:
#                 return data[:, filtered_pidx,:], coords[:, filtered_pidx,:], pseudo_targets
#             else:
#                 return data[:, filtered_pidx,:], pseudo_targets
#         else:
#             print('Only binary classification supported.')
#             NotImplementedError
#     else:
#         if int(target) == 0:
#             # neg_patch_idx = torch.squeeze(torch.nonzero(A_ranked >= th), dim=1).tolist()
#             pseudo_targets = torch.unsqueeze(torch.zeros(len(filtered_pidx)), dim=0).to(device)
#             if coords != None:
#                 return data[:, filtered_pidx,:], coords[:, filtered_pidx,:], pseudo_targets
#             else:
#                 return data[:, filtered_pidx,:], pseudo_targets
#         elif int(target) == 1:
#             # pos_patch_idx = torch.squeeze(torch.nonzero(A_ranked >= th), dim=1).tolist()
#             pseudo_targets = torch.unsqueeze(torch.ones(len(filtered_pidx)), dim=0).to(device)
#             if coords != None:
#                 return data[:, filtered_pidx,:], coords[:, filtered_pidx,:], pseudo_targets
#             else:
#                 return data[:, filtered_pidx,:], pseudo_targets
#         else:
#             pseudo_targets = torch.unsqueeze(torch.Tensor(len(filtered_pidx) * [int(target)]), dim=0).to(device)
#             if coords != None:
#                 return data[:, filtered_pidx,:], coords[:, filtered_pidx,:], pseudo_targets
#             else:
#                 return data[:, filtered_pidx,:], pseudo_targets


def load_summary_logs(binary, surv=False):
    if surv:
        logs = {
            'epoch': 0,
            'ckpt': 0,
            'train_cindex': 0,
            'val_cindex': 0,
            'test_cindex': 0,
            'loss': 1000,
            'weight': None,
            'p_lvl_loss': 1000,
            'p_lvl_epoch':0,
            'train_pred_data': None,
            'val_pred_data': None,
            'test_pred_data': None,
        }
        return logs
    if binary:
        logs = {
            'epoch': 0,
            'ckpt': 0,
            'train_auc': 0,
            'train_roc': 0,
            'train_f1': 0,
            'train_ci': None,
            'train_se': 0,
            'train_sp': 0,
            'train_recall':0,
            'train_pred_data': None,
            'train_cnf_matrix': None,
            'val_auc': 0,
            'val_roc': 0,
            'val_f1': 0,
            'val_ci': (0, 0),
            'val_se': 0,
            'val_sp': 0,
            'val_recall':0,
            'val_pred_data': None,
            'val_cnf_matrix': None,
            'test_auc': 0,
            'test_roc': 0,
            'test_f1': 0,
            'test_ci': (0, 0),
            'test_se': 0,
            'test_sp': 0,   
            'test_recall':0,
            'test_pred_data': None,
            'test_cnf_matrix': None,
            'loss': 1000,
            'weight': None,
            'p_lvl_loss': 1000,
            'p_lvl_epoch':0,
        }
    else:
        logs = {
            'epoch': 0,
            'ckpt': 0,
            'weight': None,
            'train_auc': 0,
            'train_f1': 0,
            'val_ci': (0, 0),
            'train_pred_data': None,
            'train_cnf_matrix': None,
            'val_auc': 0,
            'val_f1': 0,
            'val_ci': (0, 0),
            'val_pred_data': None,
            'val_cnf_matrix': None,
            'test_auc': 0,
            'test_f1': 0,
            'test_ci': (0, 0),
            'test_pred_data': None,
            'test_cnf_matrix': None,
            'loss': 1000,
            'weight': None,
            'p_lvl_loss': 1000,
            'p_lvl_epoch':0,
        }
    return logs

def load_loop_logs(binary, phase, surv=False):
    if surv:
        logs = {
            phase + '_cindex': 0,
        }
        return logs
    if binary:
       logs = {
           phase + '_auc': 0,
           phase + '_roc': 0,
           phase + '_ci': 0,
           phase + '_cnf_matrix': 0,
           phase + '_f1': 0,
           phase + '_se': 0,
           phase + '_sp': 0,
           phase + '_recall': 0,
           phase + '_loss': 0,
           phase + '_pred_data': 0,
       }
    else:
        logs = {
           phase + '_auc': 0,
           phase + '_ci': 0,
           phase + '_cnf_matrix': 0,
           phase + '_f1': 0,
           phase + '_loss': 0,
           phase + '_pred_data': 0,
       }
    return logs

def logging_epoch(summary_logs, log_list):
    for logs in log_list:
        for i, key in enumerate(logs):
            summary_logs[key] = logs[key]
    return summary_logs

def print_epoch_summary(best_logs, binary):
    val_lower, val_upper = best_logs['val_ci']
    test_lower, test_upper = best_logs['test_ci']
    if binary:
        print(f"[core] Best epoch {best_logs['epoch']} -"
              f" val/test auc: {best_logs['val_auc']:.4f} (CI {val_lower:.4f}-{val_upper:.4f})/{best_logs['test_auc']:.4f} (CI {test_lower:.4f}-{test_upper:.4f}),"
              f" val/test f1: {best_logs['val_f1']:.4f}/{best_logs['test_f1']:.4f},"
              f" val/test se: {best_logs['val_se']:.4f}/{best_logs['test_se']:.4f},"
              f" val/test sp: {best_logs['val_sp']:.4f}/{best_logs['test_sp']:.4f},"
              f" val/test recall: {best_logs['val_recall']:.4f}/{best_logs['test_recall']:.4f}")
    else:
        print(f"[core] Best epoch {best_logs['epoch']} -"
              f" val/test auc: {best_logs['val_auc']:.4f} (CI {val_lower:.4f}-{val_upper:.4f})/{best_logs['test_auc']:.4f} (CI {test_lower:.4f}-{test_upper:.4f}),"
              f" val/test f1: {best_logs['val_f1']:.4f}/{best_logs['test_f1']:.4f}")
        

def get_gpu_num(man_specified):
    if man_specified == -1:
        n_gpu = torch.cuda.device_count()
    else:
        n_gpu = man_specified
    return n_gpu

def initialize_weights(module):
    """
    Initialize the weights of the model with kaiming he for linear layers, and xavier for all others
    """
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.LayerNorm):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.BatchNorm1d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)