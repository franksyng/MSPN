import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.utils import resample
import itertools
import os
from math import sqrt
from scipy.stats import norm


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

    def get_metrics(self):
        return self.metrics

    def generate_y_metrics(self):
        self.y_true = np.asarray(self.data_all['y_true']).reshape([-1,])
        self.y_pred = np.asarray(self.data_all['y_pred']).reshape([-1,])
        self.y_disc = np.asarray(self.data_all['y_disc']).reshape([-1,])
    
    def get_correctness(self):
        self.data_all['correctness'] = np.equal(self.data_all['y_true'], self.data_all['y_disc'])