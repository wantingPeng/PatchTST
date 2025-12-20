#!/bin/bash

# Change to the project root directory (PatchTST_supervised)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
    #   --data_path 'dataset/dataset/downsampleData_scratch_1minut/contact/contact_cleaned_1minut_20250928_172122.parquet' \
    #   --data_path 'dataset/dataset/downsampleData_scratch_1minut/pcb/pcb_cleaned_1minut_20250928_161509.parquet' \
    #   --data_path 'dataset/dataset/downsampleData_scratch_1minut/ring/Ring_cleaned_1minut_20250928_170147.parquet' \


python -u exp/run_anomaly_detection.py \
      --is_training 0 \
      --test_only True \
      --data_path 'dataset/dataset/downsampleData_scratch_1minut/ring/Ring_cleaned_1minut_20250928_170147.parquet' \
      --model DLinear \
      --data custom \
      --features M \
      --seq_len 50 \
      --pred_len 50 \
      --enc_in 28 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 256 \
      --d_ff 512 \
      --patch_len 16 \
      --des 'Anomaly Detection' \
      --train_epochs 1 \
      --patience 3 \
      --batch_size 64 \
      --learning_rate 0.0001 \
      --kernel_size 25 \
      --anormly_ratio 3.0 \
      --checkpoint_dir checkpoints_paramenter_analysis_dlinear/checkpoints_seq_len_analysis/DLinear_custom_ring_seq_len50_20251114_094729\
