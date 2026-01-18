#!/bin/sh

# ========= CONCH ========= #
# ER
echo 'Processing CONCH'
# ABMILMSPN vs. others
# ABMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmilprems_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmilpremscat_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv

python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmilprems_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmilpremscat_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv

python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmilprems_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmilpremscat_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv

python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-abmil_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-abmilprems_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-abmil_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-abmilpremscat_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv


echo 'Processing UNI'
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmilprems_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmilpremscat_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv

python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmilprems_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmilpremscat_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv

python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmilprems_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmilpremscat_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv

python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-abmil_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-abmilprems_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-abmil_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-abmilpremscat_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv


echo 'Processing GigaPath'
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmilprems_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmilpremscat_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv

python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmilprems_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmilpremscat_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv

python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmilprems_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmilpremscat_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv

python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-abmil_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-abmilprems_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-abmil_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-abmilpremscat_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv
