import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import numpy as np
from utils.universal_utils import setup_seed_surv, create_dir, L1Reg, Lookahead
from utils.metric_utils import compare_metrics
from utils.core_utils import train_baseline_surv
from utils.surv_utils_ms import evaluate_surv_ms
from datasets import SlideSurvDatasetMS
from torch.utils.data import DataLoader
from models.clam import CLAM_MB_MS, CLAM_SB_MS, CLAM_MB_MSCat, CLAM_SB_MSCat
from models.abmil import ABMILMS, ABMILMSCat, ABMILPreMS, ABMILPreMSCat, ABMILPretrained
from models.dsmil import FCLayer, BClassifier, DSMILMS, DSMILMSCat
from models.hag_mil import HAGMIL
from utils.surv_utils import NLLSurvLoss
from src.builder import create_model

parser = argparse.ArgumentParser('receptor prediction')
parser.add_argument('--arch', default='amil', help='select model architecture.')
parser.add_argument('--n_gpu', type=int, default=-1, help='Manually give gpu number')
parser.add_argument('--in_dim', type=int, default=1024, help='input dim of embedding.')
parser.add_argument('--n_classes', type=int, default=4, help='num of classes.')
# Path and dataset params
parser.add_argument('--ann', type=str, default='../annotations/annotations_luad_surv.csv', help='Annotation file.')
parser.add_argument('--res_root', default='results', help='Result directory.')
parser.add_argument('--split_dir', type=str, default='../annotations/5fold_splits_surv/', help='Split directory')
parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for dataloader')

parser.add_argument('--data_bb', type=str, default='conch', help='backbone for the data')
parser.add_argument('--task', type=str, default='luad_surv', choices=['surgen_surv'], help='Benchmarking task name')

# Optimiser params
parser.add_argument('--opt', type=str, choices=['adam', 'adamw', 'sgd'], default='adamw')
parser.add_argument('--scheduler', choices=['CALR', 'CAWR'], default='CALR')
parser.add_argument('--loss_func', choices=['nll_surv'], default='nll_surv')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=150, help='Expected training epoch number.')
parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 reg for optimizer.')
parser.add_argument('--pos_enc', default=False, action='store_true', help='include coords')
parser.add_argument('--early_stopping', default=True, action='store_true', help='early stopping')
# gradient accumulation
parser.add_argument('--gc', type=int, default=32, help='Number of epoch for cumulative gradient. Set to 1 to disable l1 reg.')
parser.add_argument('--l1_reg', type=float, default=1e-5, help='L1 reg for cumulative gradient.')
parser.add_argument('--seed', type=int, default=2024, help='select a seed for reproducibility')
parser.add_argument('--debug', default=False, action='store_true', help='debug mode')

args = parser.parse_args()
# setup CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
arch = args.arch
pos_enc = args.pos_enc
# setup seed
setup_seed_surv(args.seed, device)

# annotations and splits
ann_path = args.ann
split_dir = args.split_dir
# data_dir = args.data_dir
task = args.task
cls_num = args.n_classes
if task == 'surgen_surv':
    classes = ['0', '1', '2', '3']
    label_col = 'label'
    data_5x = f'/path/to/{args.data_bb}_feats/surgen_256_5x_feats/'
    data_10x = f'/path/to/{args.data_bb}_feats/surgen_256_10x_feats/'
    data_20x = f'/path/to/{args.data_bb}_feats/surgen_256_20x_feats/'
else:
    print(f'Unsupported task: {task}.')
    raise NotImplementedError

data_dir = (data_5x, data_10x, data_20x)

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
cindex_metrics = {'train': [], 'val': [], 'test': [], 'eval': []}

# dataset and dataloader
def get_data(curr_split):
    train_set = SlideSurvDatasetMS(annotations, data_dir, curr_split, label_col=label_col, set_type='train', pos_enc=pos_enc, eval_mode=False)
    val_set = SlideSurvDatasetMS(annotations, data_dir, curr_split, label_col=label_col, set_type='val', pos_enc=pos_enc, eval_mode=False)
    test_set = SlideSurvDatasetMS(annotations, data_dir, curr_split, label_col=label_col, set_type='test', pos_enc=pos_enc, eval_mode=False)
    return train_set, val_set, test_set


if __name__ == '__main__':
    print(f'[main] SELECTED NUM. OF WORKER {args.num_workers}')
    # start n-fold cross-cv
    for i in range(len(splits)):
        # split result directory
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
        if arch == 'clammbms':
            model = CLAM_MB_MS(n_classes=cls_num, embed_dim=args.in_dim, mil_hidden_2=64)
        elif arch == 'clamsbms':
            model = CLAM_SB_MS(n_classes=cls_num, embed_dim=args.in_dim, mil_hidden_2=64)
        elif arch == 'clamsbmscat':
            model = CLAM_SB_MSCat(n_classes=cls_num, embed_dim=args.in_dim, mil_hidden_2=64)
        elif arch == 'clammbmscat':
            model = CLAM_MB_MSCat(n_classes=cls_num, embed_dim=args.in_dim, mil_hidden_2=64)
        elif arch == 'abmilms':
            model = ABMILMS(in_channels=args.in_dim, n_classes=cls_num)
        elif arch == 'abmilmscat':
            model = ABMILMSCat(in_channels=args.in_dim, n_classes=cls_num)
        elif arch == 'abmilprems':
            pre_model = create_model('abmil.base.uni_v2.pc108-24k', from_pretrained=True, num_classes=cls_num)
            scratch_model = ABMILPretrained(in_dim=args.in_dim, num_classes=cls_num)
            state = pre_model.state_dict()
            for k,v in state.items():
                print(k)
            state = {k.removeprefix("model."): v for k, v in state.items()}
            state = {k: v for k, v in state.items() if 'classifier' not in k and 'patch_embed' not in k}
            missing_keys, unexpected_keys = scratch_model.load_state_dict(state, strict=False)
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)
            model = ABMILPreMS(in_channels=args.in_dim, n_classes=cls_num, abmil_head=scratch_model)
        elif arch == 'abmilpremscat':
            pre_model = create_model('abmil.base.uni_v2.pc108-24k', from_pretrained=True, num_classes=cls_num)
            scratch_model = ABMILPretrained(in_dim=args.in_dim, num_classes=cls_num)
            state = pre_model.state_dict()
            for k,v in state.items():
                print(k)
            state = {k.removeprefix("model."): v for k, v in state.items()}
            state = {k: v for k, v in state.items() if 'classifier' not in k and 'patch_embed' not in k}
            missing_keys, unexpected_keys = scratch_model.load_state_dict(state, strict=False)
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)
            model = ABMILPreMSCat(in_channels=args.in_dim, n_classes=cls_num, abmil_head=scratch_model)
        elif arch == 'hagmil':
            model = HAGMIL(in_dim=args.in_dim,
                    hidden_dims=[1024, 1536, 512, 1024],  # Bottleneck structure from paper
                    n_classes=cls_num,
                    num_levels=3,
                    k_per_level=[200,800])
        elif arch == 'dsmilms':
            i_classifier = FCLayer(args.in_dim, 512, cls_num)
            b_classifier = BClassifier(input_size=512)
            model = DSMILMS(n_classes=cls_num, i_classifier=i_classifier, b_classifier=b_classifier, mil_hidden_1=512)
        elif arch == 'dsmilmscat':
            i_classifier = FCLayer(args.in_dim, 512, cls_num)
            b_classifier = BClassifier(input_size=512)
            model = DSMILMSCat(n_classes=cls_num, i_classifier=i_classifier, b_classifier=b_classifier, mil_hidden_1=512)
        else:
            raise NotImplementedError

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

        if args.loss_func == 'nll_surv':
            criterion = NLLSurvLoss(alpha=0.15)
        else:
            raise NotImplementedError

        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

        # start receptor pred
        logs = train_baseline_surv(model, device, args.epochs, optimizer, scheduler, criterion, args.gc, reg_fn, args.l1_reg, train_loader, val_loader, test_loader, args.early_stopping, train_ms=True, mdl_name=arch)

        # eval
        best_epoch = logs['epoch']
        best_weight = logs['weight']
        model.load_state_dict(best_weight)

        eval_logs = evaluate_surv_ms(model, device, criterion, test_loader, mdl_name=arch)
        # output
        all_metrics = {
            'cindex': {
                'train': logs['train_metrics'],
                'val': logs['val_metrics'],
                'test': logs['test_metrics'],    
            }
        }
        # pred_details = [logs['train_pred_data'], logs['val_pred_data'], logs['test_pred_data'], eval_logs['eval_pred_data']]
        # all_metrics, split_roc, split_cis, best_weight, cnf_matrices, pred_details, opt_th
        pred_details = eval_logs['eval_pred_data']
        case_ids = eval_logs['eval_pred_data']['case_id']

        # compare_metrics(all_metrics, split_res_dir)
        eval_details_df = pd.DataFrame(pred_details)


        # TO DO: for getting prediciton score only.
        eval_details_df.to_csv(os.path.join(split_res_dir, f'test_details.csv'))

        compare_metrics(all_metrics, split_res_dir)
        split_metics_df = pd.DataFrame({'val': logs['val_metrics'], 'test': logs['test_metrics']})
        split_metics_df.to_csv(os.path.join(split_res_dir, 'split_cindex.csv'))

        cindex_metrics['train'].append(logs['train_cindex'])
        cindex_metrics['val'].append(logs['val_cindex'])
        cindex_metrics['test'].append(logs['test_cindex'])
        cindex_metrics['eval'].append(eval_logs['eval_cindex'])

        torch.save(best_weight, os.path.join(ckpt_dir, f'best_val.pt'))

    # compute mean
    cindex_metrics['train'].append(np.mean(cindex_metrics['train']))
    cindex_metrics['val'].append(np.mean(cindex_metrics['val']))
    cindex_metrics['test'].append(np.mean(cindex_metrics['test']))
    cindex_metrics['eval'].append(np.mean(cindex_metrics['eval']))

    cindex_metrics_df = pd.DataFrame(cindex_metrics)
    cindex_metrics_df.to_csv(os.path.join(res_dir, 'cindex_metrics.csv'))
