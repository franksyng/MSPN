#!/bin/sh

# ========= uni ========= #
# ER
echo 'Processing uni ER'
# ABMILMSPN vs. others
# ABMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmil_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# ABMILCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmilms_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# ABMILCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmilmscat_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# Maxpool
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/maxpool_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# Meanpool
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/meanpool_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# TransMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/transmil_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# (Pretrained) ABMILMSPN vs. ABMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmil_rpn_pre_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmilpre_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv

# DSMILMSPN vs. others
# DSMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/dsmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/dsmil_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# DSMILCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/dsmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/dsmilms_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# DSMILCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/dsmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/dsmilmscat_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv

# CLAM-SBMSPN vs. others
# CLAM-SB
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clamsb_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clamsb_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# CLAM-SBCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clamsb_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clamsbms_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# CLAM-SBCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clamsb_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clamsbmscat_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv

# CLAM-MBMSPN vs. others
# CLAM-MB
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clammb_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clammb_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# CLAM-MBCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clammb_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clammbms_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# CLAM-MBCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clammb_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clammbmscat_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv

echo 'Processing uni PR'
# PR
# ABMILMSPN vs. others
# ABMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmil_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# ABMILCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmilms_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# ABMILCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmilmscat_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# Maxpool
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/maxpool_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# Meanpool
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/meanpool_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# TransMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/transmil_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# (Pretrained) ABMILMSPN vs. ABMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmil_rpn_pre_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmilpre_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv

# DSMILMSPN vs. others
# DSMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/dsmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/dsmil_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# DSMILCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/dsmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/dsmilms_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# DSMILCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/dsmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/dsmilmscat_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv

# CLAM-SBMSPN vs. others
# CLAM-SB
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clamsb_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clamsb_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# CLAM-SBCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clamsb_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clamsbms_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# CLAM-SBCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clamsb_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clamsbmscat_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv

# CLAM-MBMSPN vs. others
# CLAM-MB
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clammb_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clammb_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# CLAM-MBCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clammb_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clammbms_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# CLAM-MBCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clammb_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clammbmscat_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv

echo 'Processing uni HER2'
# HER2
# ABMILMSPN vs. others
# ABMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmil_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmil_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# ABMILCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmil_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmilms_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# ABMILCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmil_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmilmscat_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# (Pretrained) ABMILMSPN vs. ABMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmil_rpn_pre_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/abmilpre_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv

# DSMILMSPN vs. others
# DSMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/dsmil_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/dsmil_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# DSMILCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/dsmil_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/dsmilms_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# DSMILCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/dsmil_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/dsmilmscat_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv

# CLAM-SBMSPN vs. others
# CLAM-SB
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clamsb_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clamsb_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# CLAM-SBCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clamsb_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clamsbms_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# CLAM-SBCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clamsb_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clamsbmscat_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# Maxpool
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clamsb_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/maxpool_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# Meanpool
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clamsb_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/meanpool_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# TransMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clamsb_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/transmil_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv

# CLAM-MBMSPN vs. others
# CLAM-MB
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clammb_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clammb_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# CLAM-MBCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clammb_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clammbms_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv
# CLAM-MBCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clammb_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/clammbmscat_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_uni.csv

echo 'Processing uni Surv'
# urGen Surv
# ABMILMSPN vs. others
# ABMIL
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-abmil_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-abmil_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv
# ABMILCSA
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-abmil_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-abmilms_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv
# ABMILCat
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-abmil_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-abmilmscat_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv
# (Pretrained) ABMILMSPN vs. ABMIL
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-abmil_rpn_pre_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-abmilpre_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv

# DSMILMSPN vs. others
# DSMIL
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-dsmil_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-dsmil_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv
# DSMILCSA
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-dsmil_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-dsmilms_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv
# DSMILCat
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-dsmil_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-dsmilmscat_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv

# CLAM-SBMSPN vs. others
# CLAM-SB
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-clamsb_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-clamsb_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv
# CLAM-SBCSA
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-clamsb_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-clamsbms_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv
# CLAM-SBCat
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-clamsb_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-clamsbmscat_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv
# Maxpool
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-clamsb_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-maxpool_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv
# Meanpool
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-clamsb_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-meanpool_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv
# TransMIL
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-clamsb_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-transmil_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv

# CLAM-MBMSPN vs. others
# CLAM-MB
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-clammb_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-clammb_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv
# CLAM-MBCSA
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-clammb_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-clammbms_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv
# CLAM-MBCat
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-clammb_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/uni/EVAL-clammbmscat_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_uni.csv