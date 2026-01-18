#!/bin/sh

# ========= CONCH ========= #
# ER
echo 'Processing CONCH ER'
# ABMILMSPN vs. others
# ABMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# ABMILCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmilms_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# ABMILCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmilmscat_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# Maxpool
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/maxpool_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# Meanpool
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/meanpool_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# TransMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/transmil_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# (Pretrained) ABMILMSPN vs. ABMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_rpn_pre_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmilpre_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv

# DSMILMSPN vs. others
# DSMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/dsmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/dsmil_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# DSMILCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/dsmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/dsmilms_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# DSMILCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/dsmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/dsmilmscat_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv

# CLAM-SBMSPN vs. others
# CLAM-SB
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clamsb_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clamsb_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# CLAM-SBCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clamsb_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clamsbms_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# CLAM-SBCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clamsb_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clamsbmscat_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv

# CLAM-MBMSPN vs. others
# CLAM-MB
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clammb_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clammb_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# CLAM-MBCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clammb_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clammbms_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# CLAM-MBCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clammb_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clammbmscat_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv

echo 'Processing CONCH PR'
# PR
# ABMILMSPN vs. others
# ABMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# ABMILCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmilms_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# ABMILCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmilmscat_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# Maxpool
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/maxpool_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# Meanpool
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/meanpool_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# TransMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/transmil_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# (Pretrained) ABMILMSPN vs. ABMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_rpn_pre_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmilpre_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv

# DSMILMSPN vs. others
# DSMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/dsmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/dsmil_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# DSMILCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/dsmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/dsmilms_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# DSMILCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/dsmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/dsmilmscat_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv

# CLAM-SBMSPN vs. others
# CLAM-SB
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clamsb_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clamsb_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# CLAM-SBCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clamsb_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clamsbms_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# CLAM-SBCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clamsb_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clamsbmscat_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv

# CLAM-MBMSPN vs. others
# CLAM-MB
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clammb_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clammb_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# CLAM-MBCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clammb_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clammbms_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# CLAM-MBCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clammb_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clammbmscat_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv

echo 'Processing CONCH HER2'
# HER2
# ABMILMSPN vs. others
# ABMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# ABMILCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmilms_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# ABMILCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmilmscat_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# (Pretrained) ABMILMSPN vs. ABMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmil_rpn_pre_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/abmilpre_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv

# DSMILMSPN vs. others
# DSMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/dsmil_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/dsmil_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# DSMILCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/dsmil_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/dsmilms_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# DSMILCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/dsmil_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/dsmilmscat_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv

# CLAM-SBMSPN vs. others
# CLAM-SB
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clamsb_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clamsb_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# CLAM-SBCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clamsb_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clamsbms_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# CLAM-SBCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clamsb_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clamsbmscat_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# Maxpool
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clamsb_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/maxpool_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# Meanpool
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clamsb_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/meanpool_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# TransMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clamsb_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/transmil_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv

# CLAM-MBMSPN vs. others
# CLAM-MB
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clammb_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clammb_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# CLAM-MBCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clammb_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clammbms_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv
# CLAM-MBCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clammb_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/clammbmscat_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_conch.csv

echo 'Processing CONCH Surv'
# urGen Surv
# ABMILMSPN vs. others
# ABMIL
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-abmil_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-abmil_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv
# ABMILCSA
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-abmil_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-abmilms_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv
# ABMILCat
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-abmil_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-abmilmscat_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv
# (Pretrained) ABMILMSPN vs. ABMIL
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-abmil_rpn_pre_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-abmilpre_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv

# DSMILMSPN vs. others
# DSMIL
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-dsmil_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-dsmil_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv
# DSMILCSA
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-dsmil_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-dsmilms_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv
# DSMILCat
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-dsmil_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-dsmilmscat_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv

# CLAM-SBMSPN vs. others
# CLAM-SB
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-clamsb_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-clamsb_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv
# CLAM-SBCSA
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-clamsb_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-clamsbms_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv
# CLAM-SBCat
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-clamsb_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-clamsbmscat_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv
# Maxpool
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-clamsb_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-maxpool_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv
# Meanpool
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-clamsb_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-meanpool_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv
# TransMIL
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-clamsb_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-transmil_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv

# CLAM-MBMSPN vs. others
# CLAM-MB
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-clammb_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-clammb_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv
# CLAM-MBCSA
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-clammb_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-clammbms_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv
# CLAM-MBCat
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-clammb_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/conch/EVAL-clammbmscat_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_conch.csv