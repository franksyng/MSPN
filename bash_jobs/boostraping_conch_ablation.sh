#!/bin/sh

# ========= CONCH ========= #
# ER
echo 'Processing CONCH ER'
# ABMILMSPN vs. others
# ABMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/1536-abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv

python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/1536-2048-abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv

python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/1536-2048-2560-3072-abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv

python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/1024-1536-2048-2560-3072-abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv


echo 'Processing CONCH PR'
# PR
# ABMILMSPN vs. others
# ABMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/1536-abmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv

python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/1536-2048-abmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv

python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/1536-2048-2560-3072-abmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv

python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/1024-1536-2048-2560-3072-abmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv


echo 'Processing CONCH HER2'
# HER2
# ABMILMSPN vs. others
# ABMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/1536-abmil_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv

python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/1536-2048-abmil_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv

python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/1536-2048-2560-3072-abmil_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv

python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/1024-1536-2048-2560-3072-abmil_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv


echo 'Processing CONCH Surv'
# urGen Surv
# ABMILMSPN vs. others
# ABMIL
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/1536-EVAL-abmil_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-abmil_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv

python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/1536-2048-EVAL-abmil_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-abmil_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv

python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/1536-2048-2560-3072-EVAL-abmil_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-abmil_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv

python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/1024-1536-2048-2560-3072-EVAL-abmil_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-abmil_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv
