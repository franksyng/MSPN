#!/bin/sh

# --- gigapath --- #
# Breast Cancer
python ../main_rl.py --arch abmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR --pos_enc

python ../main_rl.py --arch abmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl.py --arch abmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR --pos_enc

python ../main_rl.py --arch abmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl.py --arch abmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR --pos_enc

python ../main_rl.py --arch abmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

# SteatoSITE
python ../main_rl.py --arch abmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_dahep.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_dahep/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/steatosite_512_40x_feats/ --res_root ../results/gigapath/dahep  --task dahep --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR  --pos_enc

python ../main_rl.py --arch abmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_dahep.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_dahep/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/steatosite_512_40x_feats/ --res_root ../results/gigapath/dahep  --task dahep --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR 

python ../main_rl.py --arch abmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_thrb.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_thrb/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/steatosite_512_40x_feats/ --res_root ../results/gigapath/thrb  --task thrb --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR  --pos_enc

python ../main_rl.py --arch abmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_thrb.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_thrb/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/steatosite_512_40x_feats/ --res_root ../results/gigapath/thrb  --task thrb --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR 


# # TCGA-LUAD Surv
# python ../main_rl_surv.py --arch abmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_luad_surv.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_surv/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/nsclc_256_20x_feats/ --res_root ../results/gigapath/surv  --task luad_surv --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR   --pos_enc

# # python ../main_rl_surv.py --arch abmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_luad_surv.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_surv/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/nsclc_256_20x_feats/ --res_root ../results/gigapath/surv  --task luad_surv --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

