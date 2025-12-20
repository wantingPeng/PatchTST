#!/bin/bash
# ============================================================================
# SHAP Analysis Runner Script for PatchTST
# ============================================================================
# This script runs SHAP analysis to generate beeswarm plots and other
# feature importance visualizations for the PatchTST model.
#
# Usage:
#   cd /home/wanting/PatchTST/PatchTST_supervised
#   source venv/bin/activate  # if using virtual environment
#   bash run_shap_analysis_patchtst.sh
# ============================================================================

# Exit on error
set -e

# Navigate to project root
cd "$(dirname "$0")"
PROJECT_ROOT=$(pwd)

echo "=============================================="
echo "SHAP Analysis for PatchTST"
echo "=============================================="
echo "Project root: ${PROJECT_ROOT}"

#CHECKPOINT_DIR="checkpoints_paramenter_analysis_patchtst/checkpoints_anormly_ratio_analysis/PatchTST_custom_pcb_anormly_ratio3.0_20251110_123510"
CHECKPOINT_DIR="checkpoints_paramenter_analysis_patchtst/checkpoints_anormly_ratio_analysis/PatchTST_custom_contact_anormly_ratio2.0_20251110_114123"
#CHECKPOINT_DIR='checkpoints_paramenter_analysis_patchtst/checkpoints_patch_len_analysis/PatchTST_custom_ring_patch_len28_20251108_133430'
MODEL_PATH="${CHECKPOINT_DIR}/model.pth"
CONFIG_PATH="${CHECKPOINT_DIR}/config.json"

# Default values (will be overridden by config.json if found)
SEQ_LEN=100
ENC_IN=27
PRED_LEN=100

# Auto-load configuration from config.json
if [[ -f "${CONFIG_PATH}" ]]; then
    echo "Loading configuration from: ${CONFIG_PATH}"
    
    # Use Python to parse JSON and export variables
    eval "$(python3 -c "
import json
import sys

try:
    with open('${CONFIG_PATH}', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Map config.json keys to shell variables
    if 'data_path' in config and config['data_path']:
        print(f'DATA_PATH=\"{config[\"data_path\"]}\"')
    if 'seq_len' in config:
        print(f'SEQ_LEN={config[\"seq_len\"]}')
    if 'enc_in' in config:
        print(f'ENC_IN={config[\"enc_in\"]}')
    if 'pred_len' in config:
        print(f'PRED_LEN={config[\"pred_len\"]}')
    
except Exception as e:
    print(f'# Warning: Failed to load config.json: {e}', file=sys.stderr)
    sys.exit(0)  # Continue with default values
")"
    
    echo "  Loaded configurations:"
    echo "    DATA_PATH: ${DATA_PATH}"
    echo "    SEQ_LEN: ${SEQ_LEN}"
    echo "    ENC_IN: ${ENC_IN}"
    echo "    PRED_LEN: ${PRED_LEN}"
else
    echo "Warning: config.json not found at ${CONFIG_PATH}"
    echo "Please set DATA_PATH manually or check CHECKPOINT_DIR"
    exit 1
fi

# Validate that DATA_PATH was loaded
if [[ -z "${DATA_PATH}" ]]; then
    echo "Error: DATA_PATH not found in config.json"
    echo "Please manually set DATA_PATH in the script"
    exit 1
fi

# SHAP configuration
N_BACKGROUND=200   # Number of background samples for SHAP
N_TEST=50          # Number of test samples to explain
STEP=1             # Step size for sliding window in training data

# Output configuration
OUTPUT_DIR="experiments/SHAP"
# Auto-generate experiment name from checkpoint directory
EXPERIMENT_NAME=$(basename "${CHECKPOINT_DIR}")

# Random seed for reproducibility
SEED=42

# ============================================================================
# Run SHAP Analysis
# ============================================================================

echo ""
echo "Configuration:"
echo "  Model path: ${MODEL_PATH}"
echo "  Data path: ${DATA_PATH}"
echo "  Sequence length: ${SEQ_LEN}"
echo "  Input channels: ${ENC_IN}"
echo "  Prediction length: ${PRED_LEN}"
echo "  Background samples: ${N_BACKGROUND}"
echo "  Test samples: ${N_TEST}"
echo "  Random seed: ${SEED}"
echo ""

python3 shap_analysis_patchtst.py \
    --model_path "${CHECKPOINT_DIR}" \
    --data_path "${DATA_PATH}" \
    --seq_len ${SEQ_LEN} \
    --enc_in ${ENC_IN} \
    --pred_len ${PRED_LEN} \
    --n_background ${N_BACKGROUND} \
    --n_test ${N_TEST} \
    --step ${STEP} \
    --output_dir "${OUTPUT_DIR}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --seed ${SEED}

echo ""
echo "=============================================="
echo "SHAP Analysis Complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
