#!/bin/bash

# SHAP Analysis Script for DLinear Model
# This script runs SHAP explanation on a trained DLinear model checkpoint

# Activate virtual environment
source venv/bin/activate

# Checkpoint directory containing model.pth and config.json
#CHECKPOINT_DIR="PatchTST_supervised/checkpoints_paramenter_analysis_dlinear/checkpoints_kernel_size_analysis/DLinear_custom_ring_kernel_size15_20251110_134746"
#CHECKPOINT_DIR="PatchTST_supervised/checkpoints_paramenter_analysis_dlinear/checkpoints_seq_len_analysis/DLinear_custom_contact_seq_len50_20251114_094454"
CHECKPOINT_DIR="PatchTST_supervised/checkpoints_paramenter_analysis_dlinear/checkpoints_kernel_size_analysis/DLinear_custom_pcb_kernel_size35_20251110_135123"
# Output directory for SHAP visualizations (optional, defaults to checkpoint_dir/shap_analysis)
OUTPUT_DIR="PatchTST_supervised/exp/shap_analysis"

# Number of background samples for SHAP from train set (default: 100)
NUM_BACKGROUND=100

# Number of test samples to explain with stratified sampling (default: 50)
# Will be split 50% anomaly windows, 50% normal windows
NUM_TEST=50

# Random seed for reproducibility
RANDOM_SEED=42

# Device (cuda or cpu)
DEVICE="cuda"

echo "========================================"
echo "Running SHAP Analysis..."
echo "========================================"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Background samples (from train): ${NUM_BACKGROUND}"
echo "Test samples (stratified): ${NUM_TEST}"
echo "Random seed: ${RANDOM_SEED}"
echo "Device: ${DEVICE}"
echo "========================================"

# Run SHAP analysis
python PatchTST_supervised/shap_explanation.py \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_background ${NUM_BACKGROUND} \
    --num_test ${NUM_TEST} \
    --random_seed ${RANDOM_SEED} \
    --device ${DEVICE}

echo ""
echo "========================================"
echo "SHAP Analysis Complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================"

