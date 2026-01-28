import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import numpy as np
from utils.universal_utils import setup_seed, create_dir, L1Reg, Lookahead
from utils.metric_utils import plot_roc_curves, save_cnf_matrix, compare_metrics
from utils.core_utils import train_baseline
from utils.bag_utils import evaluate
from datasets import SlideDataset
from torch.utils.data import DataLoader
# from models.dbamil import DBAMIL
from models.abmil import ABMILPretrained
from models.mspn_pre import ABMILMSPNPretrained
from sklearn.utils.class_weight import compute_class_weight
from src.builder import create_model

parser = argparse.ArgumentParser('pretrained_MIL')
parser.add_argument('--arch', default='abmil', help='select model architecture.')
parser.add_argument('--n_gpu', type=int, default=-1, help='Manually give gpu number')
parser.add_argument('--in_dim', type=int, default=1024, help='input dim of embedding.')
parser.add_argument('--n_classes', type=int, default=2, help='num of classes.')
# Path and dataset params
parser.add_argument('--ann', type=str, default='/path/to', help='Annotation file.')
parser.add_argument('--res_root', default='results', help='Result directory.')
parser.add_argument('--split_dir', type=str, default='/path/to', help='Split directory')
parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for dataloader')
parser.add_argument('--data_dir', type=str, default='/path/to', help='Data directory.')
parser.add_argument('--task', type=str, choices=['er','pr', 'her2'], help='Benchmarking task name')

# Optimiser params
parser.add_argument('--opt', type=str, choices=['adam', 'adamw', 'sgd'], default='adamw')
parser.add_argument('--scheduler', choices=['CALR', 'CAWR'], default='CALR')
parser.add_argument('--loss_func', choices=['ce', 'bce'], default='ce')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, help='Expected training epoch number.')
parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 reg for optimizer.')
parser.add_argument('--pos_enc', default=False, action='store_true', help='use positional encoding')
parser.add_argument('--pos_enc_2d', default=False, action='store_true', help='use 2d positional encoding')
parser.add_argument('--early_stopping', default=True, action='store_true', help='early stopping')
# gradient accumulation
parser.add_argument('--gc', type=int, default=1, help='Number of epoch for cumulative gradient. Set to 1 to disable l1 reg.')
parser.add_argument('--l1_reg', type=float, default=1e-5, help='L1 reg for cumulative gradient.')
parser.add_argument('--seed', type=int, default=2024, help='select a seed for reproducibility')
parser.add_argument('--debug', default=False, action='store_true', help='debug mode')

args = parser.parse_args()
# setup CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
arch = args.arch
pos_enc = args.pos_enc
# setup seed
setup_seed(args.seed, device)
generator = torch.Generator()
generator.manual_seed(args.seed)

# annotations and splits
ann_path = args.ann
split_dir = args.split_dir
data_dir = args.data_dir
task = args.task
cls_num = args.n_classes
if cls_num == 2:
    binary = True
else:
    binary = False

if task == 'er':
    classes = ['ER-', 'ER+']
    label_col = 'labels_er_2cls'
elif task == 'pr':
    classes = ['PR-', 'PR+']
    label_col = 'labels_pr_2cls'
elif task == 'her2':
    classes = ['Negative', 'Positive']
    label_col = 'labels_her2_2cls'
else:
    print(f'Unsupported task: {task}.')
    raise NotImplementedError

# automatic cross-validation
splits = sorted(os.listdir(split_dir))  # sorted beginning from 0 to n split
splits = [each for each in splits if each != '.DS_Store']  # partial dev env is macOS, remove influences
# prepare result directory
if arch == 'selfattn':
    if args.pos_enc_2d:
        res_name = (f"{arch}-2dpe_{task}_b{args.batch_size}_gc{args.gc}_"
                            f"{args.opt}_e{args.epochs}_lr{args.lr}_"
                            f"{args.loss_func}_{args.scheduler}")
    else:
        res_name = (f"{arch}-1dpe_{task}_b{args.batch_size}_gc{args.gc}_"
                            f"{args.opt}_e{args.epochs}_lr{args.lr}_"
                            f"{args.loss_func}_{args.scheduler}")
else:
    res_name = (f"{arch}_{task}_b{args.batch_size}_gc{args.gc}_"
                        f"{args.opt}_e{args.epochs}_lr{args.lr}_"
                        f"{args.loss_func}_{args.scheduler}")
res_dir = os.path.join(args.res_root, res_name)
create_dir(args.res_root)  # if not exist, create one
if args.debug == False:
    if os.path.exists(res_dir):
        print(f'Path exists: {res_dir}')
        exit(0)
# save results
roc_curves = {'val': {}, 'eval': {}}
f1_metrics = {'train': [], 'val': [], 'test': [], 'eval': []}
auc_metrics = {'val': [], 'test': [], 'eval': [], 'val_ci': [], 'eval_ci':[]}

# ensure reproducibility
# cpu_state = hash(torch.get_rng_state().numpy().tobytes())
# cuda_state = hash(torch.cuda.get_rng_state().cpu().numpy().tobytes()) if device.type == 'cuda' else None

# dataset and dataloader
def get_data(curr_split):
    train_set = SlideDataset(annotations, data_dir, curr_split, label_col=label_col, set_type='train', pos_enc=pos_enc, eval_mode=False)
    val_set = SlideDataset(annotations, data_dir, curr_split, label_col=label_col, set_type='val', pos_enc=pos_enc, eval_mode=False)
    test_set = SlideDataset(annotations, data_dir, curr_split, label_col=label_col, set_type='test', pos_enc=pos_enc, eval_mode=False)
    return train_set, val_set, test_set


if __name__ == '__main__':
    print(f'[main] SELECTED NUM. OF WORKER {args.num_workers}')
    # start n-fold cross-cv
    for i in range(len(splits)):
        print(f'[main] BEGINNING SPLIT {i}')
        split_path = os.path.join(split_dir, splits[i])
        split_res_dir = os.path.join(res_dir, f'split_{i}')
        val_res_dir = os.path.join(split_res_dir, 'val')
        test_res_dir = os.path.join(split_res_dir, 'test')
        ckpt_dir = os.path.join(split_res_dir, 'ckpt')
        create_dir(split_res_dir)  # if not exist, create one
        create_dir(val_res_dir)  # if not exist, create one
        create_dir(test_res_dir)  # if not exist, create one
        create_dir(ckpt_dir)  # if not exist, create one

        # prepare model
        if arch == 'abmilpre':
            pre_model = create_model('abmil.base.uni_v2.pc108-24k', from_pretrained=True, num_classes=cls_num)
            model = ABMILPretrained(in_dim=args.in_dim, num_classes=cls_num)
        elif arch == 'abmil_mspn_pre' and pos_enc != False:
            pre_model = create_model('abmil.base.uni_v2.pc108-24k', from_pretrained=True, num_classes=cls_num)
            model = ABMILMSPNPretrained(in_dim=args.in_dim, num_classes=cls_num, view_scales=[1536, 2048, 3072]) # 1536, 2304, 3072
        else:
            raise NotImplementedError
        
        state = pre_model.state_dict()
        for k,v in state.items():
            print(k)
        state = {k.removeprefix("model."): v for k, v in state.items()}
        state = {k: v for k, v in state.items() if 'classifier' not in k and 'patch_embed' not in k}
        missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)
        # put model on one or multiple GPUs
        if hasattr(model, 'relocate'):
            model.relocate(device, args.n_gpu)
        else:
            model = model.to(device)

        if args.gc > 1:
            reg_fn = L1Reg()
        else:
            reg_fn = None

        # prepare optimizer
        if args.opt == 'adam':
            print('[pipeline] optimizer: adam')
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.opt == 'adamw':
            print('[pipeline] optimizer: adamw')
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.opt == 'sgd':
            print('[pipeline] optimizer: sgd')
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
        else:
            raise NotImplementedError
        if arch == 'transmil':
            optimizer = Lookahead(base_optimizer=optimizer)

        # prepare scheduler
        if args.scheduler == 'CALR':
            warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
            calr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs-5)
            scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, calr_scheduler], milestones=[5])
        elif args.scheduler == 'CAWR':
            warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
            cawr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
            scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cawr_scheduler], milestones=[5])
        else:
            raise NotImplementedError

        # load data
        annotations = pd.read_csv(ann_path, dtype={'case_id': str})
        curr_split = pd.read_csv(split_path, dtype={'case_id': str})
        train_set, val_set, test_set = get_data(curr_split)
        curr_labels = train_set.get_label_list()
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(curr_labels), y=np.array(curr_labels))
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        if args.loss_func == 'ce':
            # if arch == 'dsmil':
            #     counter = [i for i in curr_labels if i == 1]
            #     pos_weight = (len(curr_labels) - len(counter)) / len(counter)
            #     pos_weight = torch.tensor(pos_weight)
            #     criterion = nn.BCEWithLogitsLoss(pos_weight)
            # else:
            criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
        else:
            raise NotImplementedError
        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, generator=generator)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

        # start receptor pred
        logs = train_baseline(model, device, args.epochs, cls_num, optimizer, scheduler, criterion, args.gc, reg_fn, args.l1_reg, train_loader, val_loader, test_loader, args.early_stopping, binary=binary, mdl_name=arch)


        # eval
        best_epoch = logs['epoch']
        best_weight = logs['weight']
        model.load_state_dict(best_weight)

        eval_logs = evaluate(model, device, criterion, test_loader, cls_num, binary, mdl_name=arch)
        
        # output
        all_metrics = {
            'auc': {
                'train': logs['train_metrics']['auc'],
                'val': logs['val_metrics']['auc'],
                'test': logs['test_metrics']['auc'],    
            },
            'loss': {
                'train': logs['train_metrics']['loss'],
                'val': logs['val_metrics']['loss'],
                'test': logs['test_metrics']['loss'],
            },
            'f1': {
                'train': logs['train_metrics']['f1'],
                'val': logs['val_metrics']['f1'],
                'test': logs['test_metrics']['f1'],
            }
        }
        cnf_matrices = [logs['val_cnf_matrix'], eval_logs['eval_cnf_matrix']]
        split_cis = [logs['val_ci'], eval_logs['eval_ci']]
        pred_details = [logs['train_pred_data'], logs['val_pred_data'], logs['test_pred_data'], eval_logs['eval_pred_data']]
        # all_metrics, split_roc, split_cis, best_weight, cnf_matrices, pred_details, opt_th

        compare_metrics(all_metrics, split_res_dir)
        train_details_df = pd.DataFrame(pred_details[0])
        val_details_df = pd.DataFrame(pred_details[1])
        test_details_df = pd.DataFrame(pred_details[2])
        eval_details_df = pd.DataFrame(pred_details[3])
        train_details_df.to_csv(os.path.join(split_res_dir, 'train_details.csv'))
        val_details_df.to_csv(os.path.join(split_res_dir, 'val_details.csv'))
        test_details_df.to_csv(os.path.join(split_res_dir, 'test_details.csv'))

        
        split_metics_auc_df = pd.DataFrame(dict({'train': all_metrics['auc']['train'], 'val': all_metrics['auc']['val'], 'test': all_metrics['auc']['test']}))
        split_metics_f1_df = pd.DataFrame(dict({'train': all_metrics['f1']['train'], 'val': all_metrics['f1']['val'], 'test': all_metrics['f1']['test']}))
        split_metics_ci_df = pd.DataFrame(dict({'val': logs['val_ci'], 'test': eval_logs['eval_ci']}))
        split_metics_auc_df.to_csv(os.path.join(split_res_dir, 'split_auc_metrics.csv'))
        split_metics_f1_df.to_csv(os.path.join(split_res_dir, 'split_f1_metrics.csv'))
        split_metics_ci_df.to_csv(os.path.join(split_res_dir, 'split_ci_metrics.csv'))



        best_val_ci, eval_ci = split_cis

        if cls_num == 2:
            split_roc = [logs['val_roc'], eval_logs['eval_roc']]
            best_val_roc, eval_roc = split_roc
            roc_curves['val'][f'split_{i}'] = {'roc': best_val_roc, 'ci': best_val_ci}
            roc_curves['eval'][f'split_{i}'] = {'roc': eval_roc, 'ci': eval_ci}
            eval_details_df.to_csv(os.path.join(split_res_dir, f'eval_details_{eval_logs["opt_th"]:.4f}.csv'))
        else:
            eval_details_df.to_csv(os.path.join(split_res_dir, f'eval_details.csv'))

        f1_metrics['train'].append(logs['train_f1'])
        f1_metrics['val'].append(logs['val_f1'])
        f1_metrics['test'].append(logs['test_f1'])
        f1_metrics['eval'].append(eval_logs['eval_f1'])

        auc_metrics['val'].append(logs['val_auc'])
        auc_metrics['test'].append(logs['test_auc'])
        auc_metrics['eval'].append(eval_logs['eval_auc'])
        auc_metrics['val_ci'].append(best_val_ci)
        auc_metrics['eval_ci'].append(eval_ci)

        torch.save(best_weight, os.path.join(ckpt_dir, f'best_val.pt'))

        best_val_cnf_matrix, eval_cnf_matrix = cnf_matrices
        save_cnf_matrix(best_val_cnf_matrix, classes=classes, save_dir=val_res_dir)
        save_cnf_matrix(eval_cnf_matrix, classes=classes, save_dir=test_res_dir)

    if cls_num == 2:
        plot_roc_curves(roc_curves, res_dir)

    # compute mean
    f1_metrics['train'].append(np.mean(f1_metrics['train']))
    f1_metrics['val'].append(np.mean(f1_metrics['val']))
    f1_metrics['test'].append(np.mean(f1_metrics['test']))
    f1_metrics['eval'].append(np.mean(f1_metrics['eval']))
    
    f1_metrics_df = pd.DataFrame(f1_metrics)
    f1_metrics_df.to_csv(os.path.join(res_dir, 'f1_metrics.csv'))
    auc_metrics_df = pd.DataFrame(auc_metrics)
    auc_metrics_df.to_csv(os.path.join(res_dir, 'auc_metrics.csv'))
