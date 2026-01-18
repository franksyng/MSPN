#!/bin/sh

python ../main_rl.py --arch dsmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_dir /media/frank/FlashVol/all_data/clean/conch_feats/leica_1st_he_256_20x_feats/ --res_root ../results/conch/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CALR  --pos_enc

python ../main_rl.py --arch dsmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_dir /media/frank/FlashVol/all_data/clean/conch_feats/leica_1st_he_256_20x_feats/ --res_root ../results/conch/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CALR 

python ../main_rl.py --arch dsmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_dir /media/frank/FlashVol/all_data/clean/conch_feats/leica_1st_he_256_20x_feats/ --res_root ../results/conch/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CALR  --pos_enc

python ../main_rl.py --arch dsmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_dir /media/frank/FlashVol/all_data/clean/conch_feats/leica_1st_he_256_20x_feats/ --res_root ../results/conch/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CALR 

python ../main_rl.py --arch dsmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_dir /media/frank/FlashVol/all_data/clean/conch_feats/leica_1st_he_256_20x_feats/ --res_root ../results/conch/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CALR --pos_enc

python ../main_rl.py --arch dsmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_dir /media/frank/FlashVol/all_data/clean/conch_feats/leica_1st_he_256_20x_feats/ --res_root ../results/conch/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CALR

python ../main_rl_surv.py --arch dsmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_surgen_surv.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_surgen_surv/ --data_dir /media/frank/FlashVol/all_data/clean/conch_feats/surgen_256_20x_feats/ --res_root ../results/conch/surv  --task surgen_surv --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CALR --pos_enc

python ../main_rl_surv.py --arch dsmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_surgen_surv.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_surgen_surv/ --data_dir /media/frank/FlashVol/all_data/clean/conch_feats/surgen_256_20x_feats/ --res_root ../results/conch/surv  --task surgen_surv --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CALR

python ../main_rl_ms.py --arch dsmilms --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_bb conch --res_root ../results/conch/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CALR

python ../main_rl_ms.py --arch dsmilms --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_bb conch --res_root ../results/conch/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CALR

python ../main_rl_ms.py --arch dsmilms --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_bb conch --res_root ../results/conch/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CALR