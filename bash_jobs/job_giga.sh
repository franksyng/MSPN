#!/bin/sh

# ------------- #
# --- DSMIL --- #
# ------------- #
# PR
python ../main_rl.py --arch dsmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR  --pos_enc

python ../main_rl.py --arch dsmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_ms.py --arch dsmilms --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_bb gigapath --res_root ../results/gigapath/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_ms.py --arch dsmilmscat --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_bb gigapath --res_root ../results/gigapath/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

# ER
python ../main_rl.py --arch dsmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR  --pos_enc

python ../main_rl.py --arch dsmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_ms.py --arch dsmilms --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_bb gigapath --res_root ../results/gigapath/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_ms.py --arch dsmilmscat --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_bb gigapath --res_root ../results/gigapath/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

# HER2
python ../main_rl.py --arch dsmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR --pos_enc

python ../main_rl.py --arch dsmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_ms.py --arch dsmilms --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_bb gigapath --res_root ../results/gigapath/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_ms.py --arch dsmilmscat --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_bb gigapath --res_root ../results/gigapath/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

# Surv SurGen
python ../main_rl_surv.py --arch dsmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_surgen_surv.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_surgen_surv/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/surgen_256_20x_feats/ --res_root ../results/gigapath/surv  --task surgen_surv --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR --pos_enc

python ../main_rl_surv.py --arch dsmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_surgen_surv.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_surgen_surv/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/surgen_256_20x_feats/ --res_root ../results/gigapath/surv  --task surgen_surv --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_surv_ms.py --arch dsmilms --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_surgen_surv.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_surgen_surv/ --data_bb gigapath --res_root ../results/gigapath/surv  --task surgen_surv --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_surv_ms.py --arch dsmilmscat --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_surgen_surv.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_surgen_surv/ --data_bb gigapath --res_root ../results/gigapath/surv  --task surgen_surv --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR


# ------------- #
# --- CLAMSB -- #
# ------------- #
# PR
python ../main_rl.py --arch clamsb_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR  --pos_enc

python ../main_rl.py --arch clamsb --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_ms.py --arch clamsbms --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_bb gigapath --res_root ../results/gigapath/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_ms.py --arch clamsbmscat --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_bb gigapath --res_root ../results/gigapath/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

# ER
python ../main_rl.py --arch clamsb_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR  --pos_enc

python ../main_rl.py --arch clamsb --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_ms.py --arch clamsbms --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_bb gigapath --res_root ../results/gigapath/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_ms.py --arch clamsbmscat --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_bb gigapath --res_root ../results/gigapath/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

# HER2
python ../main_rl.py --arch clamsb_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR --pos_enc

python ../main_rl.py --arch clamsb --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_ms.py --arch clamsbms --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_bb gigapath --res_root ../results/gigapath/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_ms.py --arch clamsbmscat --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_bb gigapath --res_root ../results/gigapath/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

# Surv SurGen
python ../main_rl_surv.py --arch clamsb_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_surgen_surv.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_surgen_surv/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/surgen_256_20x_feats/ --res_root ../results/gigapath/surv  --task surgen_surv --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR --pos_enc

python ../main_rl_surv.py --arch clamsb --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_surgen_surv.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_surgen_surv/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/surgen_256_20x_feats/ --res_root ../results/gigapath/surv  --task surgen_surv --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_surv_ms.py --arch clamsbms --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_surgen_surv.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_surgen_surv/ --data_bb gigapath --res_root ../results/gigapath/surv  --task surgen_surv --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_surv_ms.py --arch clamsbmscat --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_surgen_surv.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_surgen_surv/ --data_bb gigapath --res_root ../results/gigapath/surv  --task surgen_surv --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR



# ------------- #
# --- CLAMMB -- #
# ------------- #
# PR
python ../main_rl.py --arch clammb_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR  --pos_enc

python ../main_rl.py --arch clammb --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_ms.py --arch clammbms --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_bb gigapath --res_root ../results/gigapath/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_ms.py --arch clammbmscat --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_bb gigapath --res_root ../results/gigapath/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

# ER
python ../main_rl.py --arch clammb_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR  --pos_enc

python ../main_rl.py --arch clammb --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_ms.py --arch clammbms --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_bb gigapath --res_root ../results/gigapath/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_ms.py --arch clammbmscat --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_bb gigapath --res_root ../results/gigapath/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

# HER2
python ../main_rl.py --arch clammb_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR --pos_enc

python ../main_rl.py --arch clammb --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_ms.py --arch clammbms --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_bb gigapath --res_root ../results/gigapath/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_ms.py --arch clammbmscat --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_bb gigapath --res_root ../results/gigapath/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

# Surv SurGen
python ../main_rl_surv.py --arch clammb_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_surgen_surv.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_surgen_surv/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/surgen_256_20x_feats/ --res_root ../results/gigapath/surv  --task surgen_surv --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR --pos_enc

python ../main_rl_surv.py --arch clammb --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_surgen_surv.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_surgen_surv/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/surgen_256_20x_feats/ --res_root ../results/gigapath/surv  --task surgen_surv --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_surv_ms.py --arch clammbms --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_surgen_surv.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_surgen_surv/ --data_bb gigapath --res_root ../results/gigapath/surv  --task surgen_surv --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_surv_ms.py --arch clammbmscat --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_surgen_surv.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_surgen_surv/ --data_bb gigapath --res_root ../results/gigapath/surv  --task surgen_surv --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR


# ------------- #
# --- ABMIL --- #
# ------------- #
# PR
python ../main_rl.py --arch abmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR  --pos_enc

python ../main_rl.py --arch abmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR 

python ../main_rl_ms.py --arch abmilms --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_bb gigapath --res_root ../results/gigapath/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_ms.py --arch abmilmscat --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_bb gigapath --res_root ../results/gigapath/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

# ER
python ../main_rl.py --arch abmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR  --pos_enc

python ../main_rl.py --arch abmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR 

python ../main_rl_ms.py --arch abmilms --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_bb gigapath --res_root ../results/gigapath/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_ms.py --arch abmilmscat --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_bb gigapath --res_root ../results/gigapath/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

# HER2
python ../main_rl.py --arch abmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR  --pos_enc

python ../main_rl.py --arch abmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/leica_1st_he_256_20x_feats/ --res_root ../results/gigapath/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_ms.py --arch abmilms --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_bb gigapath --res_root ../results/gigapath/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_ms.py --arch abmilmscat --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_bb gigapath --res_root ../results/gigapath/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

# Surv SurGen
python ../main_rl_surv.py --arch abmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_surgen_surv.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_surgen_surv/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/surgen_256_20x_feats/ --res_root ../results/gigapath/surv  --task surgen_surv --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR --pos_enc

python ../main_rl_surv.py --arch abmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_surgen_surv.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_surgen_surv/ --data_dir /media/frank/FlashVol/all_data/clean/gigapath_feats/surgen_256_20x_feats/ --res_root ../results/gigapath/surv  --task surgen_surv --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_surv_ms.py --arch abmilms --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_surgen_surv.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_surgen_surv/ --data_bb gigapath --res_root ../results/gigapath/surv  --task surgen_surv --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_surv_ms.py --arch abmilmscat --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_surgen_surv.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_surgen_surv/ --data_bb gigapath --res_root ../results/gigapath/surv  --task surgen_surv --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR


