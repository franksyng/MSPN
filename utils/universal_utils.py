import torch
from torch.optim.optimizer import Optimizer
from collections import defaultdict, OrderedDict
import random
import os
import numpy as np

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
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
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
        
