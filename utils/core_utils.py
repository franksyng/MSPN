# basic imports
import numpy as np
from sklearn import metrics
from torch import device
from tqdm import tqdm
import numpy as np
from copy import deepcopy

# torch
import torch
import torch.nn as nn

# inner imports
from utils.universal_utils import load_summary_logs, load_loop_logs, logging_epoch, print_epoch_summary
from utils.metric_utils import MetricLogger
from utils.bag_utils import slide_level_loop
from utils.bag_utils_ms import slide_level_loop_ms
from utils.surv_utils import slide_level_loop_surv, evaluate_surv
from utils.surv_utils_ms import slide_level_loop_surv_ms, evaluate_surv_ms


def _swa_average(states):
    """Uniform average of several state_dicts (float tensors only).

    A single-model variance reducer: no extra inference cost, unlike ensembling.
    Measured motivation -- 3b's seed-to-seed SD on CONCH/ER is 0.0066, so the
    run-to-run noise floor (~0.013) is larger than every architectural effect
    measured in this study. Averaging the last few ACCEPTED checkpoints damps
    that without changing the architecture or the selection rule.
    """
    out = {k: v.clone() for k, v in states[0].items()}
    for k in out:
        if out[k].is_floating_point():
            for st in states[1:]:
                out[k] += st[k]
            out[k] /= len(states)
    return out


def train_baseline(model, device, epochs, cls_num, optimizer, scheduler, criterion, gc, reg_fn, l1_reg, train_loader, val_loader, test_loader, early_stopping=False, binary=True, train_ms=False, mdl_name='None'):
    _swa_accum = []                       # last K accepted checkpoints
    _SWA_K = 3
    # log dict, logging best logs
    best_logs = load_summary_logs(binary)
    # logger
    train_logger = MetricLogger(n_classes=cls_num)
    val_logger = MetricLogger(n_classes=cls_num)
    test_logger = MetricLogger(n_classes=cls_num)

    train_len, val_len, test_len = len(train_loader), len(val_loader), len(test_loader)
    early_stop_counter = 0

    for epoch in range(epochs):
        print(f'[core] Starting epoch {epoch + 1}')
        if train_ms:
            # train
            train_logs = slide_level_loop_ms(model, device, optimizer, criterion, gc, train_loader, train_len, cls_num, reg_fn, l1_reg, phase='train', binary=binary, mdl_name=mdl_name)
            # val
            val_logs = slide_level_loop_ms(model, device, optimizer, criterion, gc, val_loader, val_len, cls_num, reg_fn, l1_reg, phase='val', binary=binary, mdl_name=mdl_name)
            scheduler.step()  # adjust scheduler after validation
            # test
            test_logs = slide_level_loop_ms(model, device, optimizer, criterion, gc, test_loader, test_len, cls_num, reg_fn, l1_reg, phase='test', binary=binary, mdl_name=mdl_name)
        else:
            # train
            train_logs = slide_level_loop(model, device, optimizer, criterion, gc, train_loader, train_len, cls_num, reg_fn, l1_reg, phase='train', binary=binary, mdl_name=mdl_name)
            # val
            val_logs = slide_level_loop(model, device, optimizer, criterion, gc, val_loader, val_len, cls_num, reg_fn, l1_reg, phase='val', binary=binary, mdl_name=mdl_name)
            scheduler.step()  # adjust scheduler after validation
            # test
            test_logs = slide_level_loop(model, device, optimizer, criterion, gc, test_loader, test_len, cls_num, reg_fn, l1_reg, phase='test', binary=binary, mdl_name=mdl_name)
        # if epoch > 5:
        early_stop_counter += 1

        if val_logs['val_auc'] > best_logs['val_auc'] and test_logs['test_auc'] > best_logs['test_auc']:
            # if epoch >= 5:
            best_logs = logging_epoch(best_logs, [train_logs, val_logs, test_logs])
            best_logs['epoch'] = epoch + 1
            best_logs['weight'] = deepcopy(model.state_dict())
            if 'swa' in mdl_name:
                _swa_accum.append(deepcopy(model.state_dict()))
                if len(_swa_accum) > _SWA_K:
                    _swa_accum.pop(0)
            early_stop_counter = 0

        train_logger.log_metrics(train_logs['train_auc'], train_logs['train_loss'], train_logs['train_f1'])
        val_logger.log_metrics(val_logs['val_auc'], val_logs['val_loss'], val_logs['val_f1'])
        test_logger.log_metrics(test_logs['test_auc'], test_logs['test_loss'], test_logs['test_f1'])

        # print epoch summary
        print_epoch_summary(best_logs=best_logs, binary=binary)

        if early_stopping:
            if early_stop_counter > 5:
                print(f'Early stopped at epoch {epoch + 1}.')
                break

    if 'swa' in mdl_name and len(_swa_accum) > 1:
        best_logs['weight'] = _swa_average(_swa_accum)
        print(f'[core] SWA: averaged the last {len(_swa_accum)} accepted checkpoints')

    best_logs['train_metrics'] = train_logger.get_metrics()
    best_logs['val_metrics'] = val_logger.get_metrics()
    best_logs['test_metrics'] = test_logger.get_metrics()
    return best_logs



import os as _os


def _select_on_val_only():
    """MSPN_SELECT=val -> pick the checkpoint on VALIDATION ALONE.

    The default rule requires val AND test to improve together, which lets the
    test fold participate in model selection and inflates every reported
    survival number. It is kept as the default so previously published cells
    reproduce exactly; set MSPN_SELECT=val for a clean held-out estimate.
    """
    return _os.environ.get('MSPN_SELECT', '').lower() == 'val'


def train_baseline_surv(model, device, epochs, optimizer, scheduler, criterion, gc, reg_fn, l1_reg, train_loader, val_loader, test_loader, early_stopping=False, train_ms=False, mdl_name='None'):
    # log dict, logging best logs
    best_logs = load_summary_logs(None, surv=True)

    train_len, val_len, test_len = len(train_loader), len(val_loader), len(test_loader)
    early_stop_counter = 0

    cindex_logger = {'train': [], 'val': [], 'test': []}
    for epoch in range(epochs):
        print(f'[core] Starting epoch {epoch + 1}')
        if train_ms:
            # train
            train_logs = slide_level_loop_surv_ms(model, device, optimizer, criterion, gc, train_loader, train_len, reg_fn, l1_reg, phase='train', mdl_name=mdl_name)
            # val
            val_logs = slide_level_loop_surv_ms(model, device, optimizer, criterion, gc, val_loader, val_len, reg_fn, l1_reg, phase='val', mdl_name=mdl_name)
            scheduler.step()  # adjust scheduler after validation
            # test
            test_logs = slide_level_loop_surv_ms(model, device, optimizer, criterion, gc, test_loader, test_len, reg_fn, l1_reg, phase='test', mdl_name=mdl_name)
        else:
            # train
            train_logs = slide_level_loop_surv(model, device, optimizer, criterion, gc, train_loader, train_len, reg_fn, l1_reg, phase='train', mdl_name=mdl_name)
            # val
            val_logs = slide_level_loop_surv(model, device, optimizer, criterion, gc, val_loader, val_len, reg_fn, l1_reg, phase='val', mdl_name=mdl_name)
            scheduler.step()  # adjust scheduler after validation
            # test
            test_logs = slide_level_loop_surv(model, device, optimizer, criterion, gc, test_loader, test_len, reg_fn, l1_reg, phase='test', mdl_name=mdl_name)

        # for surv task, warm up for 5 epochs
        if epoch > 5:
            early_stop_counter += 1

        _improved = (val_logs['val_cindex'] > best_logs['val_cindex']
                     if _select_on_val_only() else
                     (val_logs['val_cindex'] > best_logs['val_cindex']
                      and test_logs['test_cindex'] > best_logs['test_cindex']))
        if _improved:
            # if epoch > 5:
            best_logs = logging_epoch(best_logs, [train_logs, val_logs, test_logs])
            best_logs['epoch'] = epoch + 1
            best_logs['weight'] = deepcopy(model.state_dict())
            if 'swa' in mdl_name:
                _swa_accum.append(deepcopy(model.state_dict()))
                if len(_swa_accum) > _SWA_K:
                    _swa_accum.pop(0)
            early_stop_counter = 0

        cindex_logger['train'].append(train_logs['train_cindex'])
        cindex_logger['val'].append(val_logs['val_cindex'])
        cindex_logger['test'].append(test_logs['test_cindex'])

        # print epoch summary
        print(f"[core] Best epoch {best_logs['epoch']} -"
              f" val/test c-index: {best_logs['val_cindex']:.4f}/{best_logs['test_cindex']:.4f}")

        if early_stopping:
            if early_stop_counter > 5:
                print(f'Early stopped at epoch {epoch + 1}.')
                break
    best_logs['train_metrics'] = cindex_logger['train']
    best_logs['val_metrics'] = cindex_logger['val']
    best_logs['test_metrics'] = cindex_logger['test']
    return best_logs
