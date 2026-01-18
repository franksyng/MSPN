#!/bin/sh

# CONCH
python ../main_rl.py --arch abmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_dir /media/frank/FlashVol/all_data/clean/conch_feats/leica_1st_he_256_20x_feats/ --res_root ../results/conch/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CAWR --debug --pos_enc

python ../main_rl.py --arch abmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_dir /media/frank/FlashVol/all_data/clean/conch_feats/leica_1st_he_256_20x_feats/ --res_root ../results/conch/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CAWR --debug

python ../main_rl.py --arch abmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_dir /media/frank/FlashVol/all_data/clean/conch_feats/leica_1st_he_256_20x_feats/ --res_root ../results/conch/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CAWR --debug --pos_enc

python ../main_rl.py --arch abmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_dir /media/frank/FlashVol/all_data/clean/conch_feats/leica_1st_he_256_20x_feats/ --res_root ../results/conch/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CAWR --debug

# python ../main_rl.py --arch abmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_dir /media/frank/FlashVol/all_data/clean/conch_feats/leica_1st_he_256_20x_feats/ --res_root ../results/conch/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CAWR --debug --pos_enc

# python ../main_rl.py --arch abmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_dir /media/frank/FlashVol/all_data/clean/conch_feats/leica_1st_he_256_20x_feats/ --res_root ../results/conch/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CAWR --debug

# Virchow2
python ../main_rl.py --arch abmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_dir /media/frank/FlashVol/all_data/clean/virchow2_feats/leica_1st_he_256_20x_feats/ --res_root ../results/virchow2/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 2560 --scheduler CAWR --debug --pos_enc

python ../main_rl.py --arch abmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_dir /media/frank/FlashVol/all_data/clean/virchow2_feats/leica_1st_he_256_20x_feats/ --res_root ../results/virchow2/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 2560 --scheduler CAWR --debug

python ../main_rl.py --arch abmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_dir /media/frank/FlashVol/all_data/clean/virchow2_feats/leica_1st_he_256_20x_feats/ --res_root ../results/virchow2/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 2560 --scheduler CAWR --debug --pos_enc

python ../main_rl.py --arch abmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_dir /media/frank/FlashVol/all_data/clean/virchow2_feats/leica_1st_he_256_20x_feats/ --res_root ../results/virchow2/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 2560 --scheduler CAWR --debug

# python ../main_rl.py --arch abmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_dir /media/frank/FlashVol/all_data/clean/virchow2_feats/leica_1st_he_256_20x_feats/ --res_root ../results/virchow2/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 2560 --scheduler CAWR --debug --pos_enc

# python ../main_rl.py --arch abmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_dir /media/frank/FlashVol/all_data/clean/virchow2_feats/leica_1st_he_256_20x_feats/ --res_root ../results/virchow2/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 2560 --scheduler CAWR --debug