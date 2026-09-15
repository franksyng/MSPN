
# basic imports
import json
import numpy as np
from sklearn import metrics
from torch import device
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from sksurv.metrics import concordance_index_censored

# torch
import torch
import torch.nn as nn

from utils.universal_utils import load_loop_logs
from utils.metric_utils import print_cnf_matrix, find_pred_score_binary, eer_threshold, MetricLogger

def slide_level_loop_surv_ms(model, device, optimizer, criterion, gc, loader, case_len, reg_fn=None, l1_reg=None, phase=None, mdl_name='None'):
    loop_logger = load_loop_logs(None, phase)
    # metrics_logger = MetricLogger(n_classes=cls_num)
    assert phase is not None  # either train, val or test should be chosen

    if phase == 'train':
        model.train()
    else:
        model.eval()
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    # use slide-lvl feature data
    with tqdm(total=case_len, desc=phase + '\t', unit=' slide', ncols=100) as pbar:
        interval_loss = []
        for idx, (data, target, case_id, surv_month, censorship) in enumerate(loader):
            target = target.to(device, dtype=torch.long)
            surv_month = surv_month.to(device, dtype=torch.float32)
            censorship = censorship.to(device, dtype=torch.float32)
            if len(data) == 2:
                data, coords = data
                # data = data.to(device, dtype=torch.float32)
                data = [each.to(device, dtype=torch.float32) for each in data]
                with torch.torch.set_grad_enabled(phase == 'train'):
                    if 'clam' in mdl_name or 'scl' in mdl_name:
                        mdl_out = model(data, coords, label=target, instance_eval=True)
                        output, inst_loss = mdl_out
                        loss = 0.5*criterion(logits=output, y=target, c=censorship) + 0.5*inst_loss
                    elif 'dsmil' in mdl_name:
                        mdl_out = model(data, coords)
                        # DSMIL emits 4 values (classes, bag_pred, A, B) in its
                        # classification form, but the MSPNTrue wrapper built
                        # with surv=True returns just (pred, coarse_map) --
                        # `if self.surv: return pred, cm` in DSMIL_MSPNTrue.
                        # Assuming 4 crashed every dsmil_mspntrue survival run.
                        # Branch on the actual arity rather than the name.
                        if len(mdl_out) == 4:
                            classes, output, _, _ = mdl_out
                            max_prediction, _ = torch.max(classes, 0)
                            loss = (0.5 * criterion(logits=output, y=target, c=censorship)
                                    + 0.5 * criterion(max_prediction.view(1, -1), target))
                        else:
                            # surv-mode wrapper: no instance branch to combine,
                            # so the bag NLL is the whole loss
                            output = mdl_out[0]
                            loss = criterion(logits=output, y=target, c=censorship)
                    else:
                        mdl_out = model(data, coords)
                        output, _ = mdl_out
                        loss = criterion(logits=output, y=target, c=censorship)
            else:
                # data = data.to(device, dtype=torch.float32)
                data = [each.to(device, dtype=torch.float32) for each in data]
                with torch.torch.set_grad_enabled(phase == 'train'):
                    if 'clam' in mdl_name or 'scl' in mdl_name:
                        mdl_out = model(data, label=target, instance_eval=True)
                        output, inst_loss = mdl_out
                        loss = 0.5*criterion(logits=output, y=target, c=censorship) + 0.5*inst_loss
                    elif 'hag' in mdl_name:
                        # HAG-MIL returns a DICT and computes its own loss with
                        # F.cross_entropy, which is undefined for censored data.
                        # For survival we take the bag logits and apply the NLL
                        # criterion, DROPPING its auxiliary hierarchical and
                        # patch-level CE losses. That is a necessary adaptation,
                        # not the published method -- state it when reporting.
                        # The classification sibling (utils/bag_utils_ms.py) has
                        # a 'hag' branch; this one was never mirrored, so every
                        # HAG-MIL survival run failed on `output, _ = mdl_out`.
                        mdl_out = model(data)
                        output = mdl_out['logits'] if isinstance(mdl_out, dict) else mdl_out[0]
                        if output.dim() == 1:      # HAG-MIL emits [n_classes],
                            output = output.unsqueeze(0)   # the NLL wants [B, n_classes]
                        loss = criterion(logits=output, y=target, c=censorship)
                    else:
                        mdl_out = model(data)
                        output, _ = mdl_out
                        loss = criterion(logits=output, y=target, c=censorship)

            lr_1 = optimizer.param_groups[0]['lr']
            loss_value = loss.item()

            hazards = torch.sigmoid(output)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
            all_risk_scores[idx] = risk
            all_censorships[idx] = censorship.detach().cpu().numpy()
            all_event_times[idx] = surv_month
            if gc > 1:
                loss_reg = reg_fn.apply_reg(model) * l1_reg
                loss = loss / gc + loss_reg

            if phase == 'train':
                loss.backward()
                if (idx + 1) % gc == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            pbar.update(len(data[0]))
            interval_loss.append(loss_value)
            pbar.set_postfix(**{'loss (batch)': np.mean(interval_loss), 'lr': lr_1})

    epoch_loss = np.mean(interval_loss)
    c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    print(f'[core] {phase} - loss: {epoch_loss:4f}, c-index: {c_index:4f}')
    loop_logger[phase + '_cindex'] = c_index
    return loop_logger

def evaluate_surv_ms(model, device, criterion, test_loader, mdl_name='None'):
    logs = {'eval_cindex': 0}
    eval_logger = MetricLogger(n_classes=4)
    all_risk_scores = np.zeros((len(test_loader)))
    all_logits = []                      # RAW logits, one 4-vector per slide
    all_censorships = np.zeros((len(test_loader)))
    all_event_times = np.zeros((len(test_loader)))
    with tqdm(total=len(test_loader), desc='eval', unit=' slide', ncols=100) as pbar:
        interval_loss = []
        model.eval()
        for idx, (data, target, case_id, surv_month, censorship) in enumerate(test_loader):
            target = target.to(device, dtype=torch.long)
            surv_month = surv_month.to(device, dtype=torch.float32)
            censorship = censorship.to(device, dtype=torch.float32)
            if len(data) == 2:
                data, coords = data
                # data = data.to(device, dtype=torch.float32)
                data = [each.to(device, dtype=torch.float32) for each in data]
                # coords = coords.to(device, dtype=torch.float32)
                with torch.no_grad():
                    if 'clam' in mdl_name or 'scl' in mdl_name:
                        mdl_out = model(data, coords, label=target, instance_eval=True)
                        output, inst_loss = mdl_out
                        loss = 0.5*criterion(logits=output, y=target, c=censorship) + 0.5*inst_loss
                    elif 'dsmil' in mdl_name:
                        mdl_out = model(data, coords)
                        # DSMIL emits 4 values (classes, bag_pred, A, B) in its
                        # classification form, but the MSPNTrue wrapper built
                        # with surv=True returns just (pred, coarse_map) --
                        # `if self.surv: return pred, cm` in DSMIL_MSPNTrue.
                        # Assuming 4 crashed every dsmil_mspntrue survival run.
                        # Branch on the actual arity rather than the name.
                        if len(mdl_out) == 4:
                            classes, output, _, _ = mdl_out
                            max_prediction, _ = torch.max(classes, 0)
                            loss = (0.5 * criterion(logits=output, y=target, c=censorship)
                                    + 0.5 * criterion(max_prediction.view(1, -1), target))
                        else:
                            # surv-mode wrapper: no instance branch to combine,
                            # so the bag NLL is the whole loss
                            output = mdl_out[0]
                            loss = criterion(logits=output, y=target, c=censorship)
                    else:
                        mdl_out = model(data, coords)
                        output, _ = mdl_out
                        loss = criterion(logits=output, y=target, c=censorship)
            else:
                # data = data.to(device, dtype=torch.float32)
                data = [each.to(device, dtype=torch.float32) for each in data]
                with torch.no_grad():
                    if 'clam' in mdl_name or 'scl' in mdl_name:
                        mdl_out = model(data, label=target, instance_eval=True)
                        output, inst_loss = mdl_out
                        loss = 0.5*criterion(logits=output, y=target, c=censorship) + 0.5*inst_loss
                    elif 'hag' in mdl_name:
                        # HAG-MIL returns a DICT and computes its own loss with
                        # F.cross_entropy, which is undefined for censored data.
                        # For survival we take the bag logits and apply the NLL
                        # criterion, DROPPING its auxiliary hierarchical and
                        # patch-level CE losses. That is a necessary adaptation,
                        # not the published method -- state it when reporting.
                        # The classification sibling (utils/bag_utils_ms.py) has
                        # a 'hag' branch; this one was never mirrored, so every
                        # HAG-MIL survival run failed on `output, _ = mdl_out`.
                        mdl_out = model(data)
                        output = mdl_out['logits'] if isinstance(mdl_out, dict) else mdl_out[0]
                        if output.dim() == 1:      # HAG-MIL emits [n_classes],
                            output = output.unsqueeze(0)   # the NLL wants [B, n_classes]
                        loss = criterion(logits=output, y=target, c=censorship)
                    else:
                        mdl_out = model(data)
                        output, _ = mdl_out
                        loss = criterion(logits=output, y=target, c=censorship)
                        
            loss_value = loss.item()
            pbar.update(len(data[0]))
            interval_loss.append(loss_value)
            pbar.set_postfix(**{'loss (batch)': np.mean(interval_loss)})
            out_probs = torch.softmax(output, dim=1)
            # risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            # all_risk_scores[idx] = risk
            # all_censorships[idx] = censorship.item()
            # all_event_times[idx] = surv_month
            all_logits.append([float(v) for v in output.detach().cpu().view(-1)])
            hazards = torch.sigmoid(output)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
            all_risk_scores[idx] = risk
            all_censorships[idx] = censorship.detach().cpu().numpy()
            all_event_times[idx] = surv_month
            curr_disc = torch.argmax(out_probs, dim=1)
            y_hat = out_probs[0][curr_disc]
            y_proba = out_probs[0].detach().cpu().tolist()
            eval_logger.log_batch_mul(y_hat.detach().cpu().tolist(), target.detach().cpu().tolist(), curr_disc.detach().cpu().tolist(), y_proba, list(case_id))

    c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    logs['eval_cindex'] = c_index
    eval_logger.get_correctness()
    logs['eval_pred_data'] = eval_logger.data_all
    logs['eval_pred_data']['y_pred'] = eval_logger.y_probas
    # The c-index tooling reads these four columns; softmax(y_pred) discards
    # the per-slide additive constant, so risk is NOT recoverable from it.
    logs['eval_pred_data']['risk'] = list(map(float, all_risk_scores))
    logs['eval_pred_data']['event_time'] = list(map(float, all_event_times))
    logs['eval_pred_data']['censorship'] = list(map(float, all_censorships))
    logs['eval_pred_data']['logits'] = [json.dumps(v) for v in all_logits]
    print(f"[core] eval - c-index: {logs['eval_cindex']:.4f}")
    return logs

class NLLSurvLoss(object):
    def __init__(self, alpha=0.0, eps=1e-7):
        self.alpha = alpha
        self.eps = eps

    def __call__(self, logits, y, c):
        return nll_loss(logits=logits, y=y.unsqueeze(dim=1), c=c.unsqueeze(dim=1), alpha=self.alpha, eps=self.eps)
        
# def nll_loss(logits, y, c, alpha=0.0, eps=1e-7):
#     batch_size = len(Y)
#     Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
#     c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
#     if S is None:
#         S = torch.cumprod(1 - hazards, dim=1)  # surival is cumulative product of 1 - hazards
#     # without padding, S(0) = S[0], h(0) = h[0]
#     S_padded = torch.cat([torch.ones_like(c), S], 1)  # S(-1) = 0, all patients are alive from (-inf, 0) by definition
#     # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
#     # h[y] = h(1)
#     # S[1] = S(1)
#     uncensored_loss = -(1 - c) * (
#         torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
#     )
#     censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
#     neg_l = censored_loss + uncensored_loss
#     loss = (1 - alpha) * neg_l + alpha * uncensored_loss
#     loss = loss.mean()
#     return loss

def nll_loss(logits, y, c, alpha=0.0, eps=1e-7):
    y = y.type(torch.int64)
    c = c.type(torch.int64)

    hazards = torch.sigmoid(logits)
    # print("hazards shape", hazards.shape)

    S = torch.cumprod(1 - hazards, dim=1)
    # print("S.shape", S.shape, S)

    S_padded = torch.cat([torch.ones_like(c), S], 1)
    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)
    # print('s_prev.s_prev', s_prev.shape, s_prev)
    # print('h_this.shape', h_this.shape, h_this)
    # print('s_this.shape', s_this.shape, s_this)

    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - c * torch.log(s_this)
    total_loss = censored_loss + uncensored_loss
    loss = ((1 - alpha) * total_loss + alpha * uncensored_loss).mean()
    return loss
    
