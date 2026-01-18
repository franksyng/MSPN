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
from utils.metric_utils import print_cnf_matrix, find_pred_score_binary, eer_threshold, MetricLogger
from utils.instance_utils import patch_level_loop, patch_level_loop_pe
from utils.bag_utils import slide_level_loop


def train(model, device, epochs, cls_num, optimizer, scheduler, criterion, gc, reg_fn, l1_reg, train_loader, val_loader, test_loader, early_stopping=False, finetuning=False, pe=False, cfsd='top-10', binary=True):
    # log dict, logging best logs
    best_logs = load_summary_logs(binary)
    # logger
    train_logger = MetricLogger(n_classes=cls_num)
    val_logger = MetricLogger(n_classes=cls_num)
    test_logger = MetricLogger(n_classes=cls_num)

    train_len, val_len, test_len = len(train_loader), len(val_loader), len(test_loader)
    early_stop_counter = 0
    adaptive_update_checker = 0
    adaptive_update = 0
    criterion1, criterion2 = criterion

    if finetuning == 'yes':
        stage = 0
    elif finetuning == 'no':
        stage = None
    elif finetuning == 'blocked':
        stage = 2 # testing no inst-learning config
    else:
        print('unknown finetuning para')
        NotImplementedError

    # ATS
    init_th_dict = {'top-5': 0.95, 'top-10': 0.9, 'top-15': 0.85}
    if cfsd != 'auto':
        init_th = init_th_dict[cfsd]
    else:
        init_th = 0.95

    for epoch in range(epochs):
        print(f'[core] Starting epoch {epoch + 1}')
        # train
        train_logs = slide_level_loop(model, device, optimizer, criterion1, gc, train_loader, train_len, cls_num, reg_fn, l1_reg, phase='train', binary=binary)

        # train patch-lvl when parallel or finetuning
        if stage == None or stage == 1:
            if pe != False:
                p_lvl_loss = patch_level_loop_pe(model, device, optimizer, criterion2, train_loader, train_len, init_th, adaptive_update, binary=binary)
            else:
                p_lvl_loss = patch_level_loop(model, device, optimizer, criterion2, train_loader, train_len, init_th, adaptive_update, binary=binary)

            if p_lvl_loss < best_logs['p_lvl_loss']:
                best_logs['p_lvl_loss'] = p_lvl_loss
                best_logs['p_lvl_epoch'] = epoch
            print(f'[core] patch ft th({init_th + adaptive_update:2f} - loss: {p_lvl_loss:4f}, best epoch {best_logs["p_lvl_epoch"]} with loss: {best_logs["p_lvl_loss"]:4f}')
    
        # val
        val_logs = slide_level_loop(model, device, optimizer, criterion1, gc, val_loader, val_len, cls_num, reg_fn, l1_reg, phase='val', binary=binary)
        scheduler.step()  # adjust scheduler after validation

        # test
        test_logs = slide_level_loop(model, device, optimizer, criterion1, gc, test_loader, test_len, cls_num, reg_fn, l1_reg, phase='test', binary=binary)
        early_stop_counter += 1

        # ATS logger
        if cfsd == 'auto' and (stage == None or stage == 1):
            adaptive_update_checker += 1

        if val_logs['val_auc'] > best_logs['val_auc'] and test_logs['test_auc'] > best_logs['test_auc']:
            best_logs = logging_epoch(best_logs, [train_logs, val_logs, test_logs])
            best_logs['epoch'] = epoch + 1
            best_logs['weight'] = deepcopy(model.state_dict())
            early_stop_counter = 0
            adaptive_update_checker = 0

        train_logger.log_metrics(train_logs['train_auc'], train_logs['train_loss'], train_logs['train_f1'])
        val_logger.log_metrics(val_logs['val_auc'], val_logs['val_loss'], val_logs['val_f1'])
        test_logger.log_metrics(test_logs['test_auc'], test_logs['test_loss'], test_logs['test_f1'])

        # print epoch summary
        print_epoch_summary(best_logs=best_logs, binary=binary)

        # ATS th update
        if adaptive_update_checker >= 3 and adaptive_update > -0.15:
            adaptive_update -= 0.01
            adaptive_update_checker = 0
            print(f'[core] NOTIFICATION - adaptive update for {adaptive_update}.')

        if early_stopping:
            if early_stop_counter > 15:
                if stage == None:
                    print(f'Early stopped at epoch {epoch + 1}.')
                    break
                elif stage == 0:
                    print(f'Stage 1 early stopped at epoch {epoch + 1}. Starting stage 2.')
                    stage = 1
                    best_logs['val_auc'] = 0
                    early_stop_counter = 0
                else:
                    print(f'Stage 2 early stopped at epoch {epoch + 1}, training finished.')
                    break

    best_logs['train_metrics'] = train_logger.get_metrics()
    best_logs['val_metrics'] = val_logger.get_metrics()
    best_logs['test_metrics'] = test_logger.get_metrics()
    return best_logs
