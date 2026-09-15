
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
import torch.nn.functional as F

from utils.universal_utils import load_loop_logs
from utils.metric_utils import print_cnf_matrix, find_pred_score_binary, eer_threshold, find_best_threshold_youden, MetricLogger

def slide_level_loop_ms(model, device, optimizer, criterion, gc, loader, case_len, cls_num=None, reg_fn=None, l1_reg=None, phase=None, binary=False, mdl_name='None'):
    loop_logger = load_loop_logs(binary, phase)
    metrics_logger = MetricLogger(n_classes=cls_num)
    assert phase is not None  # either train, val or test should be chosen

    if phase == 'train':
        model.train()
    else:
        model.eval()

    # use slide-lvl feature data
    with tqdm(total=case_len, desc=phase + '\t', unit=' slide', ncols=100) as pbar:
        interval_loss = []
        for idx, (data, target, case_id) in enumerate(loader):

            target = target.to(device, dtype=torch.long)
            if len(data) == 2:
                data, coords = data
                data = [each.to(device, dtype=torch.float32) for each in data]
                # data = data.to(device, dtype=torch.float32)
                with torch.torch.set_grad_enabled(phase == 'train'):
                    if 'clam' in mdl_name or 'scl' in mdl_name:
                        mdl_out = model(data, coords, label=target, instance_eval=True)
                        output, inst_loss = mdl_out
                        loss = 0.5*criterion(output, target) + 0.5*inst_loss
                    elif 'dsmil' in mdl_name:
                        mdl_out = model(data, coords)
                        classes, output, _, _ = mdl_out
                        max_prediction, index = torch.max(classes, 0)
                        loss_bag = criterion(output, target)
                        loss_max = criterion(max_prediction.view(1, -1), target)
                        loss = 0.5*loss_bag + 0.5*loss_max
                    else:
                        mdl_out = model(data, coords)
                        output, _ = mdl_out
                        loss = criterion(output, target)
            else:
                data = [each.to(device, dtype=torch.float32) for each in data]
                with torch.torch.set_grad_enabled(phase == 'train'):
                    if 'clam' in mdl_name or 'scl' in mdl_name:
                        mdl_out = model(data, label=target, instance_eval=True)
                        output, inst_loss = mdl_out
                        loss = 0.5*criterion(output, target) + 0.5*inst_loss
                    elif 'hag' in mdl_name:
                        mdl_out = model(data, label=target)
                        output, loss = mdl_out
                    else:
                        mdl_out = model(data)
                        output, _ = mdl_out
                        loss = criterion(output, target)

            # loss = criterion(output, target)
            lr_1 = optimizer.param_groups[0]['lr']
            loss_value = loss.item()
            # A_raw, last_p_map = maps
            # L_align = F.kl_div(
            #         F.log_softmax(last_p_map, dim=-1),
            #         F.softmax(A_raw, dim=-1),
            #         reduction="batchmean"
            #     )

            if gc > 1:
                loss_reg = reg_fn.apply_reg(model) * l1_reg
                # loss = (loss - cos_sim) / gc + loss_reg
                loss = loss / gc + loss_reg
                # loss = (loss + L_align) / gc + loss_reg


            if phase == 'train':
                loss.backward()
                if (idx + 1) % gc == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            pbar.update(len(data[0]))
            interval_loss.append(loss_value)
            pbar.set_postfix(**{'loss (batch)': np.mean(interval_loss), 'lr': lr_1})

            # process mdl out
            out_probs = torch.softmax(output, dim=1)
            if binary:
                y_hat = find_pred_score_binary(out_probs.tolist())  # y_hat here are pred scores
                metrics_logger.log_batch(y_hat, target.detach().cpu().tolist(), list(case_id))
            else:
                curr_disc = torch.argmax(out_probs, dim=1)
                y_hat = out_probs[0][curr_disc]
                y_proba = out_probs[0].detach().cpu().tolist()
                metrics_logger.log_batch_mul(y_hat.detach().cpu().tolist(), target.detach().cpu().tolist(), curr_disc.detach().cpu().tolist(), y_proba, list(case_id))

    epoch_loss = np.mean(interval_loss)
    metrics_logger.generate_y_metrics()  # process y_true, y_pred, y_disc from this batch into format for metrics computation
    metrics_logger.get_correctness()
    f1 = metrics_logger.get_f1()

    if binary:
        auc, roc, ci = metrics_logger.get_auc()
        cnf_matrix, se, sp, recall = metrics_logger.get_cnf_matrix()
        pred_data = metrics_logger.data_all
        print(f'[core] {phase} - loss: {epoch_loss:4f}, auc: {auc:4f}, f1: {f1:4f}, se: {se:4f}, sp: {sp:}, recall: {recall:4f}')
        print_cnf_matrix(cnf_matrix)
        print_cnf_matrix(cnf_matrix, normalize=True)
        loop_logger[phase + '_auc'] = auc
        loop_logger[phase + '_roc'] = roc
        loop_logger[phase + '_ci'] = ci
        loop_logger[phase + '_cnf_matrix'] = cnf_matrix
        loop_logger[phase + '_f1'] = f1
        loop_logger[phase + '_se'] = se
        loop_logger[phase + '_sp'] = sp
        loop_logger[phase + '_recall'] = recall
        loop_logger[phase + '_loss'] = epoch_loss
        loop_logger[phase + '_pred_data'] = pred_data
        return loop_logger
    else:
        auc, ci = metrics_logger.get_auc_mul()
        cnf_matrix = metrics_logger.get_cnf_matrix()
        pred_data = metrics_logger.data_all
        print(f'[core] {phase} - loss: {epoch_loss:4f}, auc: {auc:4f}, f1: {f1:4f}')
        print_cnf_matrix(cnf_matrix)
        print_cnf_matrix(cnf_matrix, normalize=True)
        loop_logger[phase + '_auc'] = auc
        loop_logger[phase + '_ci'] = ci
        loop_logger[phase + '_cnf_matrix'] = cnf_matrix
        loop_logger[phase + '_f1'] = f1
        loop_logger[phase + '_loss'] = epoch_loss
        loop_logger[phase + '_pred_data'] = pred_data
        return loop_logger


def evaluate_ms(model, device, criterion, test_loader, cls_num, binary, mdl_name='None'):
    eval_logger = MetricLogger(n_classes=cls_num)
    if binary:
        logs = {'eval_auc': 0,
                'eval_roc': 0,
                'eval_f1': 0,
                'eval_cnf_matrix': 0,
                'eval_se': 0,
                'eval_sp': 0,
                'eval_recall': 0,
                'opt_th': 0.5,
                'attn_scores':None,
            }
    else:
        logs = {'eval_auc': 0,
                'eval_f1': 0,
                'eval_cnf_matrix': 0
            }
    
    with tqdm(total=len(test_loader), desc='eval', unit=' slide', ncols=100) as pbar:
        interval_loss = []
        model.eval()
        attn_scores = []
        for idx, (data, target, case_id) in enumerate(test_loader):
            target = target.to(device, dtype=torch.long)
            if len(data) == 2:
                data, coords = data
                data = [each.to(device, dtype=torch.float32) for each in data]
                # data = data.to(device, dtype=torch.float32)
                # coords = coords.to(device, dtype=torch.float32)
                with torch.no_grad():
                    if 'clam' in mdl_name or 'scl' in mdl_name:
                        mdl_out = model(data, coords, label=target, instance_eval=True)
                        output, inst_loss = mdl_out
                        loss = 0.5*criterion(output, target) + 0.5*inst_loss
                    elif 'dsmil' in mdl_name:
                        mdl_out = model(data, coords)
                        classes, output, _, _ = mdl_out
                        max_prediction, index = torch.max(classes, 0)
                        loss_bag = criterion(output, target)
                        loss_max = criterion(max_prediction.view(1, -1), target)
                        loss = 0.5*loss_bag + 0.5*loss_max
                    else:
                        mdl_out = model(data, coords)
                        output, _ = mdl_out
                        loss = criterion(output, target)
            else:
                data = [each.to(device, dtype=torch.float32) for each in data]
                with torch.no_grad():
                    if 'clam' in mdl_name or 'scl' in mdl_name:
                        mdl_out = model(data, label=target, instance_eval=True)
                        output, inst_loss = mdl_out
                        loss = 0.5*criterion(output, target) + 0.5*inst_loss
                    elif 'hag' in mdl_name:
                        mdl_out = model(data, label=target)
                        output, loss = mdl_out
                    # elif 'dsmil' in mdl_name:
                    #     mdl_out = model(data)
                    #     classes, output, _, _ = mdl_out
                    #     max_prediction, index = torch.max(classes, 0)
                    #     loss_bag = criterion(output, target)
                    #     loss_max = criterion(max_prediction.view(1, -1), target)
                    #     loss = 0.5*loss_bag + 0.5*loss_max
                    else:
                        mdl_out = model(data)
                        output, _ = mdl_out
                        loss = criterion(output, target)

            loss_value = loss.item()
            pbar.update(len(data[0]))
            interval_loss.append(loss_value)
            pbar.set_postfix(**{'loss (batch)': np.mean(interval_loss)})
            out_probs = torch.softmax(output, dim=1)

            if binary:
                pred_score = find_pred_score_binary(out_probs.tolist())
                eval_logger.log_batch(pred_score, target.detach().cpu().tolist(), list(case_id))
            else:
                curr_disc = torch.argmax(out_probs, dim=1)
                y_hat = out_probs[0][curr_disc]
                y_proba = out_probs[0].detach().cpu().tolist()
                eval_logger.log_batch_mul(y_hat.detach().cpu().tolist(), target.detach().cpu().tolist(), curr_disc.detach().cpu().tolist(), y_proba, list(case_id))
    
    eval_logger.generate_y_metrics()
    if binary:
        logs['eval_auc'], logs['eval_roc'], logs['eval_ci'] = eval_logger.get_auc()
        lower, upper = logs['eval_ci']
        fpr, tpr, th = logs['eval_roc']
        # opt_th = eer_threshold(fpr, tpr, th)
        opt_th = find_best_threshold_youden(fpr, tpr, th)
        logs['opt_th'] = opt_th
        print('Evaluated optimal threshold:', opt_th)
        eval_logger.relog_pred(opt_th)
        # do discrete classification stuff only after setting optimal threshold
        eval_logger.get_correctness()
        logs['eval_pred_data'] = eval_logger.data_all
        logs['eval_cnf_matrix'], logs['eval_se'], logs['eval_sp'], logs['eval_recall'] = eval_logger.get_cnf_matrix()
        logs['eval_f1'] = eval_logger.get_f1()
        logs['attn_scores'] = attn_scores
        print(f"[core] eval - auc: {logs['eval_auc']:.4f} 95% CI ({lower:.4f}-{upper:.4f}), f1: {logs['eval_f1']:.4f}, se: {logs['eval_se']:.4f}, sp: {logs['eval_sp']:.4f}, recall: {logs['eval_recall']:4f}")
        return logs
    else:
        logs['eval_auc'], logs['eval_ci'] = eval_logger.get_auc_mul()
        lower, upper = logs['eval_ci']
        # do discrete classification stuff only after setting optimal threshold
        eval_logger.get_correctness()
        logs['eval_pred_data'] = eval_logger.data_all
        logs['eval_pred_data']['y_pred'] = eval_logger.y_probas
        logs['eval_cnf_matrix'] = eval_logger.get_cnf_matrix()
        logs['eval_f1'] = eval_logger.get_f1()
        # logs['eval_recall'] = eval_logger.get_recall()
        # precision = eval_logger.get_precision()
        print(f"[core] eval - auc: {logs['eval_auc']:.4f} 95% CI ({lower:.4f}-{upper:.4f}), f1: {logs['eval_f1']:.4f}")
        return logs
