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


python -u exp/run_anomaly_detection.py \
      --is_training 1 \
      --test_only False \
      --data_path 'dataset/dataset/downsampleData_scratch_1minut/contact/contact_cleaned_1minut_20250928_172122.parquet' \
      --model Autoformer \
      --data custom \
      --features M \
      --seq_len 100 \
      --pred_len 100 \
      --enc_in 27 \
      --e_layers 3 \
      --n_heads 8 \
      --d_model 256 \
      --d_ff 512 \
      --des 'Anomaly Detection' \
      --train_epochs 1 \
      --patience 3 \
      --batch_size 64 \
      --learning_rate 0.0001 \
      --kernel_size 25 \
      --anormly_ratio 5.0 \
