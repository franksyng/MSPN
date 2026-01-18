#!/bin/sh

# CONCH
python ../main_rl_ms.py --arch hagmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_bb conch --res_root ../results/conch/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CALR

python ../main_rl_ms.py --arch hagmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_bb conch --res_root ../results/conch/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CALR

python ../main_rl_ms.py --arch hagmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_bb conch --res_root ../results/conch/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CALR --debug

# UNI2
python ../main_rl_ms.py --arch hagmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_bb uni --res_root ../results/uni/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_ms.py --arch hagmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_bb uni --res_root ../results/uni/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_ms.py --arch hagmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_bb uni --res_root ../results/uni/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR


# GigaPath
python ../main_rl_ms.py --arch hagmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_er/ --data_bb gigapath --res_root ../results/gigapath/er  --task er --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_ms.py --arch hagmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_pr/ --data_bb gigapath --res_root ../results/gigapath/pr  --task pr --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR

python ../main_rl_ms.py --arch hagmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_cohort_1_partial.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_her2/ --data_bb gigapath --res_root ../results/gigapath/her2  --task her2 --lr 2e-4 --gc 32 --epochs 150 --in_dim 1536 --scheduler CALR
