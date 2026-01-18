#!/bin/sh

# ========= gigapath ========= #
# ER
echo 'Processing gigapath ER'
# ABMILMSPN vs. others
# ABMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmil_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# ABMILCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmilms_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# ABMILCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmilmscat_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# Maxpool
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/maxpool_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# Meanpool
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/meanpool_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# TransMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/transmil_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# (Pretrained) ABMILMSPN vs. ABMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmil_rpn_pre_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmilpre_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv

# DSMILMSPN vs. others
# DSMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/dsmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/dsmil_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# DSMILCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/dsmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/dsmilms_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# DSMILCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/dsmil_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/dsmilmscat_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv

# CLAM-SBMSPN vs. others
# CLAM-SB
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clamsb_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clamsb_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# CLAM-SBCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clamsb_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clamsbms_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# CLAM-SBCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clamsb_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clamsbmscat_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv

# CLAM-MBMSPN vs. others
# CLAM-MB
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clammb_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clammb_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# CLAM-MBCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clammb_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clammbms_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# CLAM-MBCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clammb_rpn_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clammbmscat_er_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv

echo 'Processing gigapath PR'
# PR
# ABMILMSPN vs. others
# ABMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmil_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# ABMILCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmilms_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# ABMILCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmilmscat_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# Maxpool
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/maxpool_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# Meanpool
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/meanpool_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# TransMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/transmil_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# (Pretrained) ABMILMSPN vs. ABMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmil_rpn_pre_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmilpre_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv

# DSMILMSPN vs. others
# DSMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/dsmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/dsmil_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# DSMILCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/dsmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/dsmilms_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# DSMILCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/dsmil_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/dsmilmscat_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv

# CLAM-SBMSPN vs. others
# CLAM-SB
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clamsb_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clamsb_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# CLAM-SBCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clamsb_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clamsbms_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# CLAM-SBCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clamsb_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clamsbmscat_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv

# CLAM-MBMSPN vs. others
# CLAM-MB
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clammb_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clammb_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# CLAM-MBCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clammb_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clammbms_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# CLAM-MBCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clammb_rpn_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clammbmscat_pr_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv

echo 'Processing gigapath HER2'
# HER2
# ABMILMSPN vs. others
# ABMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmil_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmil_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# ABMILCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmil_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmilms_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# ABMILCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmil_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmilmscat_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# (Pretrained) ABMILMSPN vs. ABMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmil_rpn_pre_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/abmilpre_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv

# DSMILMSPN vs. others
# DSMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/dsmil_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/dsmil_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# DSMILCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/dsmil_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/dsmilms_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# DSMILCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/dsmil_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/dsmilmscat_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv

# CLAM-SBMSPN vs. others
# CLAM-SB
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clamsb_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clamsb_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# CLAM-SBCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clamsb_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clamsbms_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# CLAM-SBCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clamsb_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clamsbmscat_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# Maxpool
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clamsb_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/maxpool_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# Meanpool
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clamsb_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/meanpool_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# TransMIL
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clamsb_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/transmil_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv

# CLAM-MBMSPN vs. others
# CLAM-MB
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clammb_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clammb_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# CLAM-MBCSA
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clammb_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clammbms_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv
# CLAM-MBCat
python ../boostraping.py --metric auc --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clammb_rpn_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/clammbmscat_her2_b1_gc32_adamw_e150_lr0.0002_ce_CALR_gigapath.csv

echo 'Processing gigapath Surv'
# urGen Surv
# ABMILMSPN vs. others
# ABMIL
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-abmil_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-abmil_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv
# ABMILCSA
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-abmil_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-abmilms_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv
# ABMILCat
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-abmil_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-abmilmscat_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv
# (Pretrained) ABMILMSPN vs. ABMIL
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-abmil_rpn_pre_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-abmilpre_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv

# DSMILMSPN vs. others
# DSMIL
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-dsmil_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-dsmil_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv
# DSMILCSA
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-dsmil_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-dsmilms_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv
# DSMILCat
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-dsmil_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-dsmilmscat_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv

# CLAM-SBMSPN vs. others
# CLAM-SB
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-clamsb_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-clamsb_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv
# CLAM-SBCSA
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-clamsb_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-clamsbms_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv
# CLAM-SBCat
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-clamsb_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-clamsbmscat_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv
# Maxpool
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-clamsb_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-maxpool_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv
# Meanpool
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-clamsb_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-meanpool_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv
# TransMIL
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-clamsb_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-transmil_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv

# CLAM-MBMSPN vs. others
# CLAM-MB
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-clammb_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-clammb_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv
# CLAM-MBCSA
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-clammb_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-clammbms_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv
# CLAM-MBCat
python ../boostraping.py --metric cindex --mA /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-clammb_rpn_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv --mB /home/frank/PycharmProjects/rpn_mil/oof_preds/gigapath/EVAL-clammbmscat_surgen_surv_b1_gc32_adamw_e150_lr0.0002_nll_surv_CALR_gigapath.csv