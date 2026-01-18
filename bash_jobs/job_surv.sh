#!/bin/sh

python ../main_rl_surv.py --arch abmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_luad_surv.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_surv/ --data_dir /media/frank/FlashVol/all_data/clean/conch_feats/nsclc_256_20x_feats/ --res_root ../results/conch/surv  --task luad_surv --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CALR   --pos_enc

python ../main_rl_surv.py --arch abmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_luad_surv.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_surv/ --data_dir /media/frank/FlashVol/all_data/clean/conch_feats/nsclc_256_20x_feats/ --res_root ../results/conch/surv  --task luad_surv --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CALR

python ../main_rl_surv.py --arch abmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_surgen_surv.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_surgen_surv/ --data_dir /media/frank/FlashVol/all_data/clean/conch_feats/surgen_256_20x_feats/ --res_root ../results/conch/surv  --task surgen_surv --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CALR   --pos_enc

python ../main_rl_surv.py --arch abmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_surgen_surv.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_surgen_surv/ --data_dir /media/frank/FlashVol/all_data/clean/conch_feats/surgen_256_20x_feats/ --res_root ../results/conch/surv  --task surgen_surv --lr 2e-4 --gc 32 --epochs 150 --in_dim 512 --scheduler CALR  


# python ../main_rl_surv.py --arch abmil_rpn --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_luad_surv.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_surv/ --data_dir /media/frank/FlashVol/all_data/clean/virchow2_feats/nsclc_256_20x_feats/ --res_root ../results/virchow2/surv  --task luad_surv --lr 2e-4 --gc 32 --epochs 150 --in_dim 2560 --scheduler CALR   --pos_enc

# python ../main_rl_surv.py --arch abmil --ann /home/frank/PycharmProjects/mrs_bc/annotations/annotations_luad_surv.csv --split_dir /home/frank/PycharmProjects/mrs_bc/annotations/5fold_splits_surv/ --data_dir /media/frank/FlashVol/all_data/clean/virchow2_feats/nsclc_256_20x_feats/ --res_root ../results/virchow2/surv  --task luad_surv --lr 2e-4 --gc 32 --epochs 150 --in_dim 2560 --scheduler CALR  
