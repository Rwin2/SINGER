#!/bin/bash
# =============================================================================
# CENTROID V9 — Full Training Pipeline
# =============================================================================
# Launched: $(date)
# Branch: feature/centroid-v9
# BC cohort: ssv_BC_CENTROID_V9
# DAgger cohort: SSV_DAGGER_CENTROID_V9
#
# WHAT'S NEW: CLIPSeg centroid features (bearing, elevation, apparent_size)
# added to CommanderSV input via obj_com. Commander input: 147-dim (was 144).
#
# TO REVERT: git checkout dev
# Baseline model backup: cohorts/ssv_BC_6S/roster/InstinctJester/model_baseline_no_centroid.pth
# =============================================================================

set -e

cd /data/erwinpi/SINGER
export ACADOS_SOURCE_DIR=/data/erwinpi/FiGS-Standalone/acados
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/erwinpi/FiGS-Standalone/acados/lib

BC_CONFIG="configs/experiment/ssv_bc_centroid_v9.yml"
DAGGER_CONFIG="configs/experiment/ssv_dagger_centroid_v9.yml"
LOG_FILE="centroid_v9_training.log"

echo "=== CENTROID V9 PIPELINE ===" | tee $LOG_FILE
echo "Started: $(date)" | tee -a $LOG_FILE
echo "Branch: $(git branch --show-current)" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# ─── Step 3: Generate observations (with centroid features) ──────────────────
echo ">>> STEP 3: generate-observations — $(date)" | tee -a $LOG_FILE
conda run -n FiGS python ssv_muilti3dgs_campaign.py generate-observations \
    --config-file $BC_CONFIG 2>&1 | tee -a $LOG_FILE
echo ">>> STEP 3 DONE — $(date)" | tee -a $LOG_FILE

# ─── Step 4: Train history (HistoryEncoder — unchanged architecture) ─────────
echo ">>> STEP 4: train-history — $(date)" | tee -a $LOG_FILE
conda run -n FiGS python ssv_muilti3dgs_campaign.py train-history \
    --config-file $BC_CONFIG 2>&1 | tee -a $LOG_FILE
echo ">>> STEP 4 DONE — $(date)" | tee -a $LOG_FILE

# ─── Step 5: Train command (new 147-dim CommanderSV + VisionMLP) ─────────────
echo ">>> STEP 5: train-command — $(date)" | tee -a $LOG_FILE
conda run -n FiGS python ssv_muilti3dgs_campaign.py train-command \
    --config-file $BC_CONFIG 2>&1 | tee -a $LOG_FILE
echo ">>> STEP 5 DONE — $(date)" | tee -a $LOG_FILE

# ─── Step 6: DAgger ──────────────────────────────────────────────────────────
echo ">>> STEP 6: dagger — $(date)" | tee -a $LOG_FILE
conda run -n FiGS python ssv_muilti3dgs_campaign.py dagger \
    --config-file $DAGGER_CONFIG 2>&1 | tee -a $LOG_FILE
echo ">>> STEP 6 DONE — $(date)" | tee -a $LOG_FILE

# ─── Visualization ───────────────────────────────────────────────────────────
echo ">>> VISUALIZATION — $(date)" | tee -a $LOG_FILE
CUDA_VISIBLE_DEVICES=1 conda run -n FiGS python scripts/compare_trajectories_3d.py 2>&1 | tee -a $LOG_FILE
echo ">>> VISUALIZATION DONE — $(date)" | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "=== ALL DONE — $(date) ===" | tee -a $LOG_FILE
