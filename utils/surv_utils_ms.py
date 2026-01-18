
# basic imports
from tqdm import tqdm
import numpy as np
from sksurv.metrics import concordance_index_censored

# torch
import torch

from utils.universal_utils import load_loop_logs
from utils.metric_utils import MetricLogger

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
                data = [each.to(device, dtype=torch.float32) for each in data]
                with torch.torch.set_grad_enabled(phase == 'train'):
                    if 'clam' in mdl_name:
                        mdl_out = model(data, coords, label=target, instance_eval=True)
                        output, inst_loss = mdl_out
                        loss = 0.5*criterion(logits=output, y=target, c=censorship) + 0.5*inst_loss
                    elif 'dsmil' in mdl_name:
                        mdl_out = model(data, coords)
                        classes, output, _, _ = mdl_out
                        max_prediction, index = torch.max(classes, 0)
                        loss_bag = criterion(logits=output, y=target, c=censorship)
                        loss_max = criterion(max_prediction.view(1, -1), target)
                        loss = 0.5*loss_bag + 0.5*loss_max
                    else:
                        mdl_out = model(data, coords)
                        output, _ = mdl_out
                        loss = criterion(logits=output, y=target, c=censorship)
            else:
                data = [each.to(device, dtype=torch.float32) for each in data]
                with torch.torch.set_grad_enabled(phase == 'train'):
                    if 'clam' in mdl_name:
                        mdl_out = model(data, label=target, instance_eval=True)
                        output, inst_loss = mdl_out
                        loss = 0.5*criterion(logits=output, y=target, c=censorship) + 0.5*inst_loss
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
                data = [each.to(device, dtype=torch.float32) for each in data]
                with torch.no_grad():
                    if 'clam' in mdl_name:
                        mdl_out = model(data, coords, label=target, instance_eval=True)
                        output, inst_loss = mdl_out
                        loss = 0.5*criterion(logits=output, y=target, c=censorship) + 0.5*inst_loss
                    elif 'dsmil' in mdl_name:
                        mdl_out = model(data, coords)
                        classes, output, _, _ = mdl_out
                        max_prediction, index = torch.max(classes, 0)
                        loss_bag = criterion(logits=output, y=target, c=censorship)
                        loss_max = criterion(max_prediction.view(1, -1), target)
                        loss = 0.5*loss_bag + 0.5*loss_max
                    else:
                        mdl_out = model(data, coords)
                        output, _ = mdl_out
                        loss = criterion(logits=output, y=target, c=censorship)
            else:
                data = [each.to(device, dtype=torch.float32) for each in data]
                with torch.no_grad():
                    if 'clam' in mdl_name:
                        mdl_out = model(data, label=target, instance_eval=True)
                        output, inst_loss = mdl_out
                        loss = 0.5*criterion(logits=output, y=target, c=censorship) + 0.5*inst_loss
                    else:
                        mdl_out = model(data)
                        output, _ = mdl_out
                        loss = criterion(logits=output, y=target, c=censorship)
                        
            loss_value = loss.item()
            pbar.update(len(data[0]))
            interval_loss.append(loss_value)
            pbar.set_postfix(**{'loss (batch)': np.mean(interval_loss)})
            out_probs = torch.softmax(output, dim=1)
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
    print(f"[core] eval - c-index: {logs['eval_cindex']:.4f}")
    return logs

class NLLSurvLoss(object):
    def __init__(self, alpha=0.0, eps=1e-7):
        self.alpha = alpha
        self.eps = eps

    def __call__(self, logits, y, c):
        return nll_loss(logits=logits, y=y.unsqueeze(dim=1), c=c.unsqueeze(dim=1), alpha=self.alpha, eps=self.eps)

def nll_loss(logits, y, c, alpha=0.0, eps=1e-7):
    y = y.type(torch.int64)
    c = c.type(torch.int64)

    hazards = torch.sigmoid(logits)
    S = torch.cumprod(1 - hazards, dim=1)

    S_padded = torch.cat([torch.ones_like(c), S], 1)
    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)

    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - c * torch.log(s_this)
    total_loss = censored_loss + uncensored_loss
    loss = ((1 - alpha) * total_loss + alpha * uncensored_loss).mean()
    return loss
    
