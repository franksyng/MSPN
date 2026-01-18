#!/bin/sh

# --- CONCH --- #
# Breast Cancer
python ../main_rl.py --arch abmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_dir /media/frank/FlashVol/all_data/clean/conch_feats/leica_1st_he_256_20x_feats/ --res_root ../results/conch/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CALR --pos_enc

# python ../main_rl.py --arch abmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_dir /media/frank/FlashVol/all_data/clean/conch_feats/leica_1st_he_256_20x_feats/ --res_root ../results/conch/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CALR

python ../main_rl.py --arch abmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_dir /media/frank/FlashVol/all_data/clean/conch_feats/leica_1st_he_256_20x_feats/ --res_root ../results/conch/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CALR --pos_enc

# python ../main_rl.py --arch abmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_dir /media/frank/FlashVol/all_data/clean/conch_feats/leica_1st_he_256_20x_feats/ --res_root ../results/conch/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CALR

python ../main_rl.py --arch abmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_dir /media/frank/FlashVol/all_data/clean/conch_feats/leica_1st_he_256_20x_feats/ --res_root ../results/conch/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CALR --pos_enc

# python ../main_rl.py --arch abmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_dir /media/frank/FlashVol/all_data/clean/conch_feats/leica_1st_he_256_20x_feats/ --res_root ../results/conch/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CALR

# SteatoSITE
python ../main_rl.py --arch abmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_dahep.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_dahep/ --data_dir /media/frank/FlashVol/all_data/clean/conch_feats/steatosite_512_40x_feats/ --res_root ../results/conch/dahep  --task dahep --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CALR  --pos_enc

# python ../main_rl.py --arch abmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_dahep.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_dahep/ --data_dir /media/frank/FlashVol/all_data/clean/conch_feats/steatosite_512_40x_feats/ --res_root ../results/conch/dahep  --task dahep --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CALR 

python ../main_rl.py --arch abmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_thrb.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_thrb/ --data_dir /media/frank/FlashVol/all_data/clean/conch_feats/steatosite_512_40x_feats/ --res_root ../results/conch/thrb  --task thrb --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CALR  --pos_enc

# python ../main_rl.py --arch abmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_thrb.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_thrb/ --data_dir /media/frank/FlashVol/all_data/clean/conch_feats/steatosite_512_40x_feats/ --res_root ../results/conch/thrb  --task thrb --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CALR 


# TCGA-LUAD Surv
python ../main_rl_surv.py --arch abmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_luad_surv.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_surv/ --data_dir /media/frank/FlashVol/all_data/clean/conch_feats/nsclc_256_20x_feats/ --res_root ../results/conch/surv  --task luad_surv --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CALR   --pos_enc

# python ../main_rl_surv.py --arch abmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_luad_surv.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_surv/ --data_dir /media/frank/FlashVol/all_data/clean/conch_feats/nsclc_256_20x_feats/ --res_root ../results/conch/surv  --task luad_surv --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CALR


# --- Virchow2 --- #
# Breast Cancer
python ../main_rl.py --arch abmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_dir /media/frank/FlashVol/all_data/clean/virchow2_feats/leica_1st_he_256_20x_feats/ --res_root ../results/virchow2/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 2560 --scheduler CALR --pos_enc

# python ../main_rl.py --arch abmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_dir /media/frank/FlashVol/all_data/clean/virchow2_feats/leica_1st_he_256_20x_feats/ --res_root ../results/virchow2/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 2560 --scheduler CALR

python ../main_rl.py --arch abmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_dir /media/frank/FlashVol/all_data/clean/virchow2_feats/leica_1st_he_256_20x_feats/ --res_root ../results/virchow2/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 2560 --scheduler CALR --pos_enc

# python ../main_rl.py --arch abmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_dir /media/frank/FlashVol/all_data/clean/virchow2_feats/leica_1st_he_256_20x_feats/ --res_root ../results/virchow2/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 2560 --scheduler CALR

python ../main_rl.py --arch abmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_dir /media/frank/FlashVol/all_data/clean/virchow2_feats/leica_1st_he_256_20x_feats/ --res_root ../results/virchow2/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 2560 --scheduler CALR --pos_enc

# python ../main_rl.py --arch abmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_dir /media/frank/FlashVol/all_data/clean/virchow2_feats/leica_1st_he_256_20x_feats/ --res_root ../results/virchow2/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 2560 --scheduler CALR