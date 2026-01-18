#!/bin/sh

# Pretrained ABMIL
# On gigapath2

# PR
python ../main_pre.py --arch abmilpre --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_pre.py --arch abmil_rpn_pre --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR --pos_enc

# ER
python ../main_pre.py --arch abmilpre --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_pre.py --arch abmil_rpn_pre --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR --pos_enc

# HER2
python ../main_pre.py --arch abmilpre --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_pre.py --arch abmil_rpn_pre --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR --pos_enc

# Surgen Surv
python ../main_pre_surv.py --arch abmilpre --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_surgen_surv.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_surgen_surv/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/surgen_256_20x_feats/ --res_root ../results/gigapath/surv  --task surgen_surv --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_pre_surv.py --arch abmil_rpn_pre --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_surgen_surv.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_surgen_surv/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/surgen_256_20x_feats/ --res_root ../results/gigapath/surv  --task surgen_surv --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR --pos_enc