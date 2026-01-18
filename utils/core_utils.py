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


def train_baseline(model, device, epochs, cls_num, optimizer, scheduler, criterion, gc, reg_fn, l1_reg, train_loader, val_loader, test_loader, early_stopping=False, binary=True, train_ms=False, mdl_name='None'):
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

        if val_logs['val_auc'] > best_logs['val_auc']:
            best_logs = logging_epoch(best_logs, [train_logs, val_logs, test_logs])
            best_logs['epoch'] = epoch + 1
            best_logs['weight'] = deepcopy(model.state_dict())
            early_stop_counter = 0

        train_logger.log_metrics(train_logs['train_auc'], train_logs['train_loss'], train_logs['train_f1'])
        val_logger.log_metrics(val_logs['val_auc'], val_logs['val_loss'], val_logs['val_f1'])
        test_logger.log_metrics(test_logs['test_auc'], test_logs['test_loss'], test_logs['test_f1'])

        # print epoch summary
        print_epoch_summary(best_logs=best_logs, binary=binary)

        if early_stopping:
            if early_stop_counter > 10:
                print(f'Early stopped at epoch {epoch + 1}.')
                break

    best_logs['train_metrics'] = train_logger.get_metrics()
    best_logs['val_metrics'] = val_logger.get_metrics()
    best_logs['test_metrics'] = test_logger.get_metrics()
    return best_logs



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

        if val_logs['val_cindex'] > best_logs['val_cindex']:
            # if epoch > 5:
            best_logs = logging_epoch(best_logs, [train_logs, val_logs, test_logs])
            best_logs['epoch'] = epoch + 1
            best_logs['weight'] = deepcopy(model.state_dict())
            early_stop_counter = 0

        cindex_logger['train'].append(train_logs['train_cindex'])
        cindex_logger['val'].append(val_logs['val_cindex'])
        cindex_logger['test'].append(test_logs['test_cindex'])

        # print epoch summary
        print(f"[core] Best epoch {best_logs['epoch']} -"
              f" val/test c-index: {best_logs['val_cindex']:.4f}/{best_logs['test_cindex']:.4f}")

        if early_stopping:
            if early_stop_counter > 10:
                print(f'Early stopped at epoch {epoch + 1}.')
                break
    best_logs['train_metrics'] = cindex_logger['train']
    best_logs['val_metrics'] = cindex_logger['val']
    best_logs['test_metrics'] = cindex_logger['test']
    return best_logs
