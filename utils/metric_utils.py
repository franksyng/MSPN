import numpy as np
import torch
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.utils import resample
import itertools
import os
from math import sqrt
from scipy.stats import norm
from sklearn.metrics import roc_curve, roc_auc_score


def print_cnf_matrix(cnf_matrix, normalize=False):
    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')
    print(cnf_matrix)


def plot_cnf_matrix(cnf_matrix, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    th = cnf_matrix.max() / 2
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt), horizontalalignment="center", color="white" if cnf_matrix[i, j] > th else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.gcf().subplots_adjust(bottom=0.25)


def eer_threshold(fpr, tpr, th):
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    min_idx = np.argmin(abs_diffs)
    # eer = np.mean((fpr[min_idx], fnr[min_idx]))  # we currently return opt threshold only
    return th[min_idx]

def find_best_threshold_youden(fpr, tpr, th):
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    return th[best_idx]

def th_linear_projection(y_pred, curr_th, tgt_th=0.5):
    """
    function to project y_pred from the scale of one optimal threshold to the target optimal threshold
    specifically here 0.5
    """
    projected_pred = []
    tgt_range = tgt_upper = tgt_lower = tgt_th
    for pred in y_pred:
        residual = pred - curr_th  # if positive, 1; else, 0
        if residual > 0:
            # consider label 1
            upper_range = 1 - curr_th
            curr_percentage = residual / upper_range
        else:
            # consider label 0
            lower_range = curr_th
            curr_percentage = residual / lower_range
        projected_pred.append(curr_percentage * tgt_range + tgt_th)  # curr_percentage * tgt_range -> projected residual
    return projected_pred


def compare_metrics(all_metrics: dict, save_dir):
    """
    metrics should be a dictionary, e.g.: {'auc': {'train': [], 'val': [], 'test': []}, 'f1': {'train': [], 'val': [], 'test': []}}
    """
    for _, curr_metric in enumerate(all_metrics.keys()):
        # initialize fig and ax
        fig, ax = plt.subplots()
        fig.set_size_inches((10, 6))
        set_types = list(all_metrics[curr_metric].keys())
        # print(set_types)
        xaxis = np.arange(len(all_metrics[curr_metric][set_types[0]]))  # choose whatever data points to get length, since all lengths should be identical
        lns = None  # initialize
        for i in range(len(set_types)):
            curr_ln = ax.plot(xaxis, all_metrics[curr_metric][set_types[i]], label='%s %s' % (set_types[i], curr_metric))
            if i == 0:
                lns = curr_ln
            else:
                lns += curr_ln
        ax.set_xlabel('Epoch number', fontsize=12)
        ax.set_ylabel('Metric', fontsize=12)
        ax.set_title(f'Metrics comparison ({curr_metric})', fontsize=12)
        labs = [ln.get_label() for ln in lns]
        ax.legend(lns, labs, loc='upper left')
        fig.savefig(os.path.join(save_dir, f'compare_metrics_{curr_metric}'))
        plt.close(fig)


def plot_roc_curves(roc_curves, save_dir, ensemble=False):
    """
    roc_curves should be a dictionary,
    e.g.: {'set_type_1':
            {
            'split_0': {'roc': [fpr, tpr, th], 'ci': (lower, upper)},
            'split_1': {'roc': [fpr, tpr, th], 'ci': (lower, upper)},...
            },
           'set_type_2': {}...
          }
    """
    # fig = plt.figure()
    for _, curr_set in enumerate(roc_curves.keys()):
        fig = plt.figure()
        fig.set_size_inches((6, 6))
        splits = sorted(roc_curves[curr_set].keys())  # ensure from 0 to n
        aucs = []
        for i in range(len(splits)):
            fpr, tpr, th = roc_curves[curr_set][splits[i]]['roc']
            opt_th = eer_threshold(fpr, tpr, th)
            auc = metrics.auc(fpr, tpr)
            aucs.append(auc)
            lower, upper = roc_curves[curr_set][splits[i]]['ci']
            if ensemble:
                plt.plot(fpr, tpr, label='%s (area = %0.3f (%0.3f-%0.3f))' % ('Ensemble ', auc, lower, upper))
            else:
                plt.plot(fpr, tpr, label='%s (area = %0.3f (%0.3f-%0.3f))' % ('Split ' + str(i), auc, lower, upper))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        # plt.title('ROC (opt th: %.4f' % opt_th, fontsize=12)
        plt.title(f'ROC ({curr_set}) - avg. {np.mean(aucs):.4f}', fontsize=12)
        plt.legend(loc='lower right')
        fig.savefig(os.path.join(save_dir, f'roc_curves_{curr_set}'))
        plt.close(fig)

def roc_with_ci(preds, ensemble_pred, y, save_dir):
    # ---- 1) Per-model AUCs and ensemble AUC ----
    # auc_each = [roc_auc_score(y, p) for p in preds]
    # auc_mean_models = float(np.mean(auc_each))
    # auc_std_models  = float(np.std(auc_each, ddof=1))  # model-to-model variability (unbiased)

    # Ensemble prediction by averaging probabilities across the 5 models
    # pred_ensemble = preds.mean(axis=0)
    auc_ensemble = roc_auc_score(y, ensemble_pred)
    fpr_grid = np.linspace(0.0, 1.0, 1001)
    tpr_matrix = []

    # for p in preds:
    #     fpr_i, tpr_i, _ = roc_curve(y, p)
    #     # Ensure strictly increasing FPR for interpolation safety
    #     uniq_fpr, uniq_idx = np.unique(fpr_i, return_index=True)
    #     tpr_i = tpr_i[uniq_idx]
    #     # Interpolate TPR at the common grid
    #     tpr_interp = np.interp(fpr_grid, uniq_fpr, tpr_i)
    #     tpr_interp[0] = 0.0
    #     tpr_interp[-1] = 1.0
    #     tpr_matrix.append(tpr_interp)

    # tpr_matrix = np.vstack(tpr_matrix)  # (5, len(fpr_grid))
    # tpr_mean = tpr_matrix.mean(axis=0)
    # tpr_std  = tpr_matrix.std(axis=0, ddof=1)

    # # Also compute ROC for the ensemble prediction for the main curve
    # fpr_e, tpr_e, _ = roc_curve(y, ensemble_pred)

    # # ---- 3) Plot: single curve with shaded model-variability band ----
    # plt.figure(figsize=(6, 6))

    # # Shaded band: mean ± 1 std across models at fixed FPR
    # tpr_upper = np.clip(tpr_mean + tpr_std, 0, 1)
    # tpr_lower = np.clip(tpr_mean - tpr_std, 0, 1)
    # plt.fill_between(fpr_grid, tpr_lower, tpr_upper, alpha=0.2, label='Model STD band')

    # # Ensemble ROC curve (single representative curve)
    # plt.plot(fpr_e, tpr_e, linewidth=2, label=f'Ensemble ROC (AUC={auc_ensemble:.3f})')

    # # Chance line
    # plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1)

    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('External ROC: 5-fold models → one curve with model STD band')
    # plt.legend()
    # plt.tight_layout()

    # --- Bootstrap sampling ---
    # print(y)
    y = np.array(y)
    ensemble_pred = np.array(ensemble_pred)
    # ---- 1) Base ROC from the full external set ----
    fpr_base, tpr_base, _ = roc_curve(y, ensemble_pred)
    auc_base = roc_auc_score(y, ensemble_pred)

    # ---- 2) Bootstrap ROC curves to form a CI ribbon ----
    B = 2000                    # number of bootstrap replicates (adjust as you like)
    rng = np.random.default_rng(42)
    fpr_grid = np.linspace(0, 1, 1001)  # common FPR grid for interpolation
    tpr_boot = []               # will become (B_eff, len(fpr_grid))

    for _ in range(B):
        # resample indices with replacement
        idx = rng.integers(0, len(y), len(y))
        y_b = y[idx]
        p_b = ensemble_pred[idx]

        # skip draws with a single class (ROC undefined)
        if np.unique(y_b).size < 2:
            continue

        fpr_b, tpr_b, _ = roc_curve(y_b, p_b)
        # ensure strictly increasing FPR before interpolation
        ufpr, uidx = np.unique(fpr_b, return_index=True)
        utpr = tpr_b[uidx]
        # interpolate to common grid
        tpr_interp = np.interp(fpr_grid, ufpr, utpr)
        tpr_interp[0] = 0.0
        tpr_interp[-1] = 1.0
        tpr_boot.append(tpr_interp)

    tpr_boot = np.vstack(tpr_boot)  # shape: (B_eff, len(fpr_grid))

    # pointwise 95% CI band
    tpr_lo = np.percentile(tpr_boot, 2.5, axis=0)
    tpr_hi = np.percentile(tpr_boot, 97.5, axis=0)

    # (optional) 95% CI for AUC as a caption
    auc_boot = []
    for tpr_interp in tpr_boot:
        # approximate AUC for each bootstrap curve via trapezoid rule on the grid
        auc_boot.append(np.trapz(tpr_interp, fpr_grid))
    auc_boot = np.asarray(auc_boot)
    auc_ci_l, auc_ci_u = np.percentile(auc_boot, [2.5, 97.5])

    # ---- 3) Plot: ROC with 95% CI ribbon ----
    plt.figure(figsize=(6, 6))

    # ribbon
    plt.fill_between(fpr_grid, tpr_lo, tpr_hi, alpha=0.2, label='95% CI')

    # base ROC (from full data)
    plt.plot(fpr_base, tpr_base, linewidth=2,
            label=f'Ensemble ROC (AUC={auc_base:.3f}; 95% CI [{auc_ci_l:.3f}, {auc_ci_u:.3f}])')

    # chance
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC (95% CI)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'roc_curves_ci'))
    return auc_base, auc_ci_l, auc_ci_u



def save_cnf_matrix(cnf_matrix, classes, save_dir):
    plt.figure()
    plot_cnf_matrix(cnf_matrix, classes=classes, title='Confusion matrix without normalization')
    plt.savefig(os.path.join(save_dir, 'cnf_matrix'))
    plt.close()

    plt.figure()
    plot_cnf_matrix(cnf_matrix, normalize=True, classes=classes, title='Confusion matrix with normalization')
    plt.savefig(os.path.join(save_dir, 'cnf_matrix_normalized'))
    plt.close()


def find_pred_score_binary(out_probs):
    """
    Find prediction scores for PyTorch CELoss output
    """
    pred_scores = []
    for i in range(len(out_probs)):
        curr_probs = out_probs[i][1]
        pred_scores.append(curr_probs)
    return pred_scores


def get_auc_ci(y_true, y_pred, pos=1, alpha=0.05):
    auc = metrics.roc_auc_score(y_true, y_pred)
    n1 = sum(y_true == pos)
    n2 = sum(y_true != pos)
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    se_auc = sqrt((auc*(1 - auc) + (n1 - 1)*(q1 - auc**2) + (n2 - 1)*(q2 - auc**2)) / (n1*n2))
    confident_level = 1 - alpha
    z_lower, z_upper = norm.interval(confident_level)
    lower = auc + z_lower*se_auc
    upper = auc + z_upper*se_auc
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return lower, upper

def get_auc_ci_mul(y_true, y_proba, n_bootstrap=1000, alpha=0.05):
    confidence_level = 1 - alpha
    aucs = []
    # auc = metrics.roc_auc_score(y_true, y_proba, average='macro', multi_class='ovr')
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = resample(range(len(y_true)), replace=True)
        y_true_resampled = np.array(y_true)[indices]
        y_probs_resampled = np.array(y_proba)[indices]

        try:
            auc_score = metrics.roc_auc_score(y_true_resampled, y_probs_resampled, multi_class="ovr")
            aucs.append(auc_score)
        except:
            continue  # Skip in case of errors (e.g., single-class resample)
    # Compute confidence interval percentiles
    lower = np.percentile(aucs, (1 - confidence_level) / 2 * 100)
    upper = np.percentile(aucs, (1 + confidence_level) / 2 * 100)
    return lower, upper

class MetricLogger:
    def __init__(self, n_classes, threshold=None):
        super(MetricLogger, self).__init__()
        if threshold == None:
            threshold = 0.5
        self.n_classes = n_classes
        self.threshold = threshold
        # storing metrics
        self.data = [{'count': 0, 'correct': 0} for i in range(self.n_classes)]
        self.data_all = {'y_true': [], 'y_pred': [], 'y_disc': [], 'case_id': [], 'correctness': []}
        if n_classes > 2:
            self.y_probas = []
        self.metrics = {'auc': [], 'loss': [], 'f1': []}
        self.y_true = None
        self.y_pred = None
        self.y_disc = None

    def relog_pred(self, opt_th):
        self.data_all['y_disc'] = [1 if each > opt_th else 0 for each in self.data_all['y_pred']]
        self.y_disc = np.asarray(self.data_all['y_disc']).reshape([-1,])

    def log_batch(self, y_hat, y_true, case_id, y_disc=None):
        if y_disc == None:
            y_disc = [1 if each > self.threshold else 0 for each in y_hat]
        self.data_all['y_pred'].extend(y_hat)
        self.data_all['y_true'].extend(y_true)
        self.data_all['y_disc'].extend(y_disc)
        self.data_all['case_id'].extend(case_id)
    
    def log_batch_mul(self, y_pred, y_true, y_disc, y_proba, case_id):
        self.data_all['y_true'].extend(y_true)
        self.data_all['y_disc'].extend(y_disc)
        self.data_all['y_pred'].extend(y_pred)
        self.y_probas.append(y_proba)  # for multi-class auc
        self.data_all['case_id'].extend(case_id)

    def log_metrics(self, auc, loss, f1):
        self.metrics['auc'].append(auc)
        self.metrics['loss'].append(loss)
        self.metrics['f1'].append(f1)
    
    # def log_metrics_mul(self, auc, f1, loss):
    #     self.metrics['auc'].append(auc)
    #     self.metrics['f1'].append(f1)
    #     self.metrics['loss'].append(loss)

    def get_f1(self):
        # if self.n_classes == 2:
        return metrics.f1_score(self.y_true, self.y_disc, average='macro')
        # else:
        #     return metrics.f1_score(self.y_true, self.y_disc, average='micro')

    def get_auc_mul(self):
        # auc = metrics.roc_auc_score(self.y_true, self.y_probas, average='micro', multi_class='ovr')
        auc = metrics.roc_auc_score(self.y_true, self.y_probas, average='macro', multi_class='ovr')
        # fpr, tpr, th = metrics.roc_curve(self.y_true, self.y_pred, pos_label=1)
        lower, upper = get_auc_ci_mul(self.y_true, self.y_probas)
        return auc, (lower, upper)

    def get_auc(self):
        fpr, tpr, th = metrics.roc_curve(self.y_true, self.y_pred, pos_label=1)
        lower, upper = get_auc_ci(self.y_true, self.y_pred)
        return metrics.roc_auc_score(self.y_true, self.y_pred), (fpr, tpr, th), (lower, upper)

    def get_prc(self):
        precision, recall, th = metrics.precision_recall_curve(self.y_true, self.y_pred, pos_label=1)
        return (precision, recall, th)

    def get_cnf_matrix(self):
        cnf_matrix = metrics.confusion_matrix(self.y_true, self.y_disc)
        if self.n_classes == 2:
            tn, fp, fn, tp = cnf_matrix.ravel()
            tpr = tp / (tp + fn) # tpr is precision
            tnr = tn / (tn + fp)
            recall = tp / (tp + fn)
            return cnf_matrix, tpr, tnr, recall
        else:
            return cnf_matrix

    def get_recall(self):
        return metrics.recall_score(self.y_true, self.y_disc, average='micro')

    def get_precision(self):
        return metrics.precision_score(self.y_true, self.y_disc, average='micro')

    def get_metrics(self):
        return self.metrics

    def generate_y_metrics(self):
        self.y_true = np.asarray(self.data_all['y_true']).reshape([-1,])
        self.y_pred = np.asarray(self.data_all['y_pred']).reshape([-1,])
        self.y_disc = np.asarray(self.data_all['y_disc']).reshape([-1,])
    
    def get_correctness(self):
        self.data_all['correctness'] = np.equal(self.data_all['y_true'], self.data_all['y_disc'])