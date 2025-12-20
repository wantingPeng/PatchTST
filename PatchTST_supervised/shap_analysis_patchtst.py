#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SHAP Analysis Script for PatchTST Model

This script performs SHAP (SHapley Additive exPlanations) analysis on the 
PatchTST model and generates beeswarm plots to visualize feature importance.

Usage:
    python shap_analysis_patchtst.py \
        --model_path <checkpoint_directory> \
        --data_path <parquet_file_path> \
        --seq_len <sequence_length> \
        --enc_in <input_channels> \
        --n_background <background_samples> \
        --n_test <test_samples> \
        --output_dir <output_directory> \
        --seed <random_seed>
"""

import os
import sys
import json
import logging
import argparse
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.PatchTST import Model as PatchTST
from data_provider.data_loader_anomaly import get_anomaly_loader


# ============================================================================
# Logging Configuration
# ============================================================================
def setup_logger(output_dir: str, script_name: str = "shap_analysis_patchtst") -> logging.Logger:
    """
    Setup logger with console handler only (no log files).
    
    Args:
        output_dir: Unused, kept for API compatibility
        script_name: Name of the logger
    
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Log format
    log_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler only (do not save logs to file)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    logger.info("Logger initialized (console only, no file logging).")
    
    return logger


# ============================================================================
# Random Seed Setting
# ============================================================================
def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ============================================================================
# Configuration Class
# ============================================================================
class Args:
    """Configuration class to mimic argparse.Namespace for model initialization."""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


# ============================================================================
# Model Wrapper for SHAP
# ============================================================================
class PatchTSTSHAPWrapper(nn.Module):
    """
    SHAP-compatible wrapper for PatchTST.
    
    Computes anomaly scores based on reconstruction error:
    - Uses MSE between input and reconstructed output
    - Averages over time steps to get a single score per sample
    """
    def __init__(self, model: PatchTST, device: torch.device, seq_len: int):
        """
        Initialize SHAP wrapper.
        
        Args:
            model: Trained PatchTST model
            device: Device to run inference on
            seq_len: Sequence length for time series
        """
        super().__init__()
        self.model = model
        self.device = device
        self.seq_len = seq_len
        self.criterion = nn.MSELoss(reduction='none')
        self.model.eval()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly scores based on reconstruction error.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, n_features]
        
        Returns:
            Aggregated anomaly scores tensor of shape [batch_size, 1]
            (averaged over time steps and features)
        """
        # Get model outputs (reconstruction)
        output = self.model(x)  # [batch, seq_len, n_features]
        
        # Compute reconstruction error per time step and feature: [batch, seq_len, n_features]
        reconstruction_error = self.criterion(x, output)
        
        # Aggregate to single score per sample: [batch]
        # Mean over time steps and features
        anomaly_score = torch.mean(reconstruction_error, dim=(1, 2))
        
        # Add extra dimension for SHAP compatibility: [batch, 1]
        return anomaly_score.unsqueeze(-1)


# ============================================================================
# Data Loading Functions
# ============================================================================
def load_data_for_shap(data_path: str, seq_len: int, step: int,
                        n_background: int, n_test: int,
                        batch_size: int = 32,
                        logger: logging.Logger = None) -> tuple:
    """
    Load and prepare data for SHAP analysis with stratified sampling.
    
    Uses stratified sampling to ensure test set contains a balanced mix of 
    normal and anomaly samples (targeting at least 30% anomaly samples).
    This ensures SHAP analysis captures both normal and anomalous patterns.
    
    Args:
        data_path: Path to the data file
        seq_len: Sequence length for time series
        step: Step size for sliding window
        n_background: Number of background samples for SHAP
        n_test: Number of test samples for SHAP (will be stratified)
        batch_size: Batch size for data loading
        logger: Logger instance
    
    Returns:
        Tuple of (background_data, test_data, test_labels, feature_names)
        - background_data: Random samples from training set [n_background, seq_len, n_features]
        - test_data: Stratified samples from test set [n_test, seq_len, n_features]
        - test_labels: Labels for test samples [n_test, seq_len]
        - feature_names: List of feature names
    """
    if logger:
        logger.info(f"Loading data from: {data_path}")
    
    # Load training data for background
    train_dataset, train_loader = get_anomaly_loader(
        data_path=data_path,
        batch_size=batch_size,
        seq_len=seq_len,
        step=step,
        mode='train',
        num_workers=0
    )
    
    # Load test data
    test_dataset, test_loader = get_anomaly_loader(
        data_path=data_path,
        batch_size=batch_size,
        seq_len=seq_len,
        step=seq_len,  # Non-overlapping windows for test
        mode='test',
        num_workers=0
    )
    
    # Get feature names from parquet file
    feature_names = None
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
        feature_cols = [c for c in df.columns if c not in ['TimeStamp', 'anomaly_label']]
        feature_names = feature_cols
        if logger:
            logger.info(f"Feature names extracted: {len(feature_names)} features")
    
    # Collect background samples
    background_samples = []
    for batch_idx, (data, labels) in enumerate(train_loader):
        background_samples.append(data.numpy())
        if len(background_samples) * data.shape[0] >= n_background:
            break
    
    background_data = np.concatenate(background_samples, axis=0)[:n_background]
    if logger:
        logger.info(f"Background data shape: {background_data.shape}")
    
    # Collect all test samples first for stratified sampling
    if logger:
        logger.info("Collecting all test samples for stratified sampling...")
    

    # 准备测试数据 - 使用分层采样确保包含异常样本
    print("\n准备测试数据（分层采样）...")
    all_test_samples = []
    all_test_labels = []
    
    # 首先收集所有测试数据
    for input_data, labels in test_loader:
        all_test_samples.append(input_data)
        all_test_labels.append(labels)
    
    all_test_data = torch.cat(all_test_samples, dim=0)
    all_test_labels = torch.cat(all_test_labels, dim=0)
    
    # 分离正常和异常样本
    # 检查每个窗口是否包含异常（至少有一个时间点是异常）
    anomaly_mask = (all_test_labels.sum(dim=1) > 50)  # [N] bool tensor
    normal_indices = torch.where(~anomaly_mask)[0]
    anomaly_indices = torch.where(anomaly_mask)[0]
    
    print(f"测试集总样本数: {len(all_test_data)}")
    print(f"  正常样本数: {len(normal_indices)} ({100*len(normal_indices)/len(all_test_data):.1f}%)")
    print(f"  异常样本数: {len(anomaly_indices)} ({100*len(anomaly_indices)/len(all_test_data):.1f}%)")
    
    # 分层采样：确保包含一定比例的异常样本
    n_anomaly_samples = min(int(n_test * 0.5), len(anomaly_indices))  # 目标：至少30%异常样本
    n_normal_samples = n_test - n_anomaly_samples
    
    # 如果异常样本不够，调整比例
    if n_anomaly_samples > len(anomaly_indices):
        n_anomaly_samples = len(anomaly_indices)
        n_normal_samples = n_test - n_anomaly_samples
    
    # 随机选择样本（使用固定种子）
    RANDOM_SEED = 42
    torch.manual_seed(RANDOM_SEED)
    selected_anomaly_idx = anomaly_indices[torch.randperm(len(anomaly_indices))[:n_anomaly_samples]]
    selected_normal_idx = normal_indices[torch.randperm(len(normal_indices))[:n_normal_samples]]
    
    # 合并并打乱
    selected_indices = torch.cat([selected_anomaly_idx, selected_normal_idx])
    selected_indices = selected_indices[torch.randperm(len(selected_indices))]
    
    # 提取选中的样本（仍为torch.Tensor）
    test_data = all_test_data[selected_indices]
    test_labels_array = all_test_labels[selected_indices]
    
    print(f"\n最终测试数据: {test_data.shape}")
    print(f"  包含异常样本: {n_anomaly_samples} ({100*n_anomaly_samples/n_test:.1f}%)")
    print(f"  包含正常样本: {n_normal_samples} ({100*n_normal_samples/n_test:.1f}%)")

    # 转为numpy，保持与后续SHAP流程一致
    test_data_np = test_data.cpu().numpy()
    test_labels_np = test_labels_array.cpu().numpy()
    
    return background_data, test_data_np, test_labels_np, feature_names


# ============================================================================
# SHAP Analysis Functions
# ============================================================================
def compute_shap_values(model: PatchTST, background_data: np.ndarray,
                        test_data: np.ndarray, device: torch.device,
                        seq_len: int, n_features: int,
                        logger: logging.Logger = None) -> tuple:
    """
    Compute SHAP values using DeepExplainer (gradient-based).
    
    Args:
        model: Trained model
        background_data: Background data for SHAP [n_background, seq_len, n_features]
        test_data: Test data to explain [n_test, seq_len, n_features]
        device: Device for computation
        seq_len: Sequence length
        n_features: Number of features
        logger: Logger instance
    
    Returns:
        Tuple of (shap_values, expected_value)
        - shap_values: array of shape [n_test, seq_len, n_features]
        - expected_value: baseline prediction value
    """
    if logger:
        logger.info("Computing SHAP values using DeepExplainer (gradient-based)...")
        logger.info(f"Background samples: {background_data.shape[0]}")
        logger.info(f"Test samples: {test_data.shape[0]}")
    
    # Create SHAP wrapper
    model_wrapper = PatchTSTSHAPWrapper(model, device, seq_len)
    
    # Convert to tensors
    background_tensor = torch.from_numpy(background_data).float().to(device)
    test_tensor = torch.from_numpy(test_data).float().to(device)
    
    if logger:
        logger.info(f"Background tensor shape: {background_tensor.shape}")
        logger.info(f"Test tensor shape: {test_tensor.shape}")
    
    # Create SHAP DeepExplainer
    if logger:
        logger.info("Creating SHAP DeepExplainer...")
        logger.info("This uses gradients and is much faster than KernelExplainer!")
    
    explainer = shap.DeepExplainer(model_wrapper, background_tensor)
    
    # Compute SHAP values
    if logger:
        logger.info("Computing SHAP values...")
    
    # Disable additivity check for complex models
    shap_values = explainer.shap_values(test_tensor, check_additivity=False)
    
    # Convert to numpy
    if isinstance(shap_values, torch.Tensor):
        shap_values = shap_values.cpu().numpy()
    
    # Handle extra output dimension if present
    if logger:
        logger.info(f"Raw SHAP values shape: {shap_values.shape}")
    
    # If there's an extra dimension at the end (output dimension), squeeze it
    if len(shap_values.shape) == 4 and shap_values.shape[-1] == 1:
        shap_values = shap_values.squeeze(-1)
        if logger:
            logger.info(f"Squeezed SHAP values shape: {shap_values.shape}")
    
    if logger:
        logger.info(f"DeepExplainer succeeded!")
        logger.info(f"Final SHAP values shape: {shap_values.shape}")
    
    # Get expected value (baseline)
    expected_value = explainer.expected_value
    if isinstance(expected_value, torch.Tensor):
        expected_value = expected_value.cpu().numpy()
    elif isinstance(expected_value, (list, np.ndarray)):
        expected_value = np.array(expected_value).flatten()[0] if len(np.array(expected_value).flatten()) > 0 else 0.0
    
    return shap_values, expected_value


# ============================================================================
# Visualization Functions
# ============================================================================
def plot_beeswarm(shap_values: np.ndarray, test_data: np.ndarray, 
                  feature_names: list, output_dir: str,
                  logger: logging.Logger = None):
    """
    Generate SHAP beeswarm plot.
    
    Aggregates SHAP values across time steps for feature importance visualization.
    
    Args:
        shap_values: SHAP values [n_test, seq_len, n_features]
        test_data: Test data [n_test, seq_len, n_features]
        feature_names: List of feature names
        output_dir: Directory to save plots
        logger: Logger instance
    """
    if logger:
        logger.info("Generating beeswarm plot...")
    
    n_test, seq_len, n_features = shap_values.shape
    if logger:
        logger.info(f"SHAP values shape for plotting: {shap_values.shape}")
    
    # Aggregate SHAP values across time steps
    # Use mean to aggregate temporal dimension
    shap_values_avg = np.mean(shap_values, axis=1)  # [n_test, n_features]

    # Also compute mean feature values for coloring
    feature_values = np.mean(test_data, axis=1)  # [n_test, n_features]
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    if logger:
        logger.info(f"Aggregated SHAP shape: {shap_values_avg.shape}")
        logger.info(f"Feature values shape: {feature_values.shape}")
    
    # Create SHAP Explanation object for beeswarm plot
    explanation = shap.Explanation(
        values=shap_values_avg,
        base_values=np.zeros(n_test),
        data=feature_values,
        feature_names=feature_names
    )
    
    # Generate beeswarm plot
    plt.figure(figsize=(12, max(8, n_features * 0.3)))
    shap.plots.beeswarm(explanation, show=False, max_display=min(20, n_features))
    plt.title("SHAP Summary (Beeswarm)")
    plt.tight_layout()
    
    beeswarm_path = os.path.join(output_dir, "shap_summary_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.info(f"Beeswarm plot saved to: {beeswarm_path}")



def plot_time_feature_heatmap(shap_values: np.ndarray, feature_names: list,
                               output_dir: str, logger: logging.Logger = None):
    """
    Generate heatmap showing SHAP values across time steps and features.
    
    Args:
        shap_values: SHAP values [n_test, seq_len, n_features]
        feature_names: List of feature names
        output_dir: Directory to save plots
        logger: Logger instance
    """
    if logger:
        logger.info("Generating time-feature heatmap...")
    
    n_test, seq_len, n_features = shap_values.shape
    
    # Average SHAP values across test samples
    mean_shap = np.mean(np.abs(shap_values), axis=0)  # [seq_len, n_features]
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    # Plot heatmap
    plt.figure(figsize=(14, max(8, n_features * 0.25)))
    plt.imshow(mean_shap.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    plt.colorbar(label='Mean |SHAP Value|')
    plt.xlabel('Time Step')
    plt.ylabel('Feature')
    plt.title('SHAP Values Heatmap (Time Steps × Features)')
    
    # Set y-tick labels
    if n_features <= 50:
        plt.yticks(range(n_features), feature_names)
    else:
        # Only show every nth label
        step = max(1, n_features // 30)
        plt.yticks(range(0, n_features, step), 
                   [feature_names[i] for i in range(0, n_features, step)])
    
    plt.tight_layout()
    
    heatmap_path = os.path.join(output_dir, "shap_feature_time_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.info(f"Time-feature heatmap saved to: {heatmap_path}")


def plot_time_feature_heatmap_single_sample(
    shap_values: np.ndarray,
    feature_names: list,
    output_dir: str,
    sample_idx: int = 0,
    logger: logging.Logger = None
):
    """
    Generate heatmap showing SHAP values across time steps and features
    for a single sample.

    Args:
        shap_values: SHAP values [n_test, seq_len, n_features]
        feature_names: List of feature names
        output_dir: Directory to save plots
        sample_idx: Index of the sample to visualize
        logger: Logger instance
    """
    if logger:
        logger.info(
            f"Generating time-feature heatmap for single sample (index={sample_idx})..."
        )

    n_test, seq_len, n_features = shap_values.shape

    if sample_idx < 0 or sample_idx >= n_test:
        msg = (
            f"sample_idx {sample_idx} is out of range for shap_values "
            f"with n_test={n_test}"
        )
        if logger:
            logger.error(msg)
        raise ValueError(msg)

    # SHAP values for a single sample: [seq_len, n_features]
    sample_shap = np.abs(shap_values[sample_idx])

    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]

    # Plot heatmap
    plt.figure(figsize=(14, max(8, n_features * 0.25)))
    plt.imshow(sample_shap.T, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(label="|SHAP Value|")
    plt.xlabel("Time Step")
    plt.ylabel("Feature")
    plt.title(f"SHAP Values Heatmap (Single Sample idx={sample_idx})")

    # Set y-tick labels
    if n_features <= 50:
        plt.yticks(range(n_features), feature_names)
    else:
        # Only show every nth label
        step = max(1, n_features // 30)
        plt.yticks(
            range(0, n_features, step),
            [feature_names[i] for i in range(0, n_features, step)],
        )

    plt.tight_layout()

    heatmap_path = os.path.join(
        output_dir, f"shap_feature_time_heatmap_sample{sample_idx}.png"
    )
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()

    if logger:
        logger.info(
            f"Time-feature heatmap for sample {sample_idx} saved to: {heatmap_path}"
        )


# ============================================================================
# Main Function
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='SHAP Analysis for PatchTST')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to checkpoint directory containing model.pth and config.json')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data file (parquet)')
    
    # Model configuration (can be loaded from config.json if not provided)
    parser.add_argument('--seq_len', type=int, default=None,
                        help='Sequence length for time series (loaded from config if not provided)')
    parser.add_argument('--enc_in', type=int, default=None,
                        help='Number of input channels/features (loaded from config if not provided)')
    parser.add_argument('--pred_len', type=int, default=None,
                        help='Prediction length (loaded from config if not provided)')
    
    # SHAP configuration
    parser.add_argument('--n_background', type=int, default=200,
                        help='Number of background samples for SHAP')
    parser.add_argument('--n_test', type=int, default=50,
                        help='Number of test samples to explain')
    parser.add_argument('--step', type=int, default=1,
                        help='Step size for sliding window in training data')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='experiments/SHAP',
                        help='Directory to save outputs')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for this experiment (auto-generated if not provided)')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Load config.json from checkpoint directory if it exists
    config_path = os.path.join(args.model_path, "config.json")
    model_config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        print(f"Loaded config from: {config_path}")
    
    # Use config values if not provided in command line
    seq_len = args.seq_len if args.seq_len is not None else model_config.get('seq_len', 100)
    enc_in = args.enc_in if args.enc_in is not None else model_config.get('enc_in', 27)
    pred_len = args.pred_len if args.pred_len is not None else model_config.get('pred_len', 100)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.experiment_name:
        experiment_dir = f"shap_analysis_{timestamp}_{args.experiment_name}"
    else:
        data_name = os.path.splitext(os.path.basename(args.data_path))[0]
        model_name = os.path.basename(args.model_path)
        experiment_dir = f"shap_analysis_{timestamp}_{model_name}"
    
    output_path = os.path.join(args.output_dir, experiment_dir)
    os.makedirs(output_path, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(output_path, "shap_analysis_patchtst")
    
    logger.info("=" * 60)
    logger.info("SHAP Analysis for PatchTST")
    logger.info("=" * 60)
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Output directory: {output_path}")
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  model_path: {args.model_path}")
    logger.info(f"  data_path: {args.data_path}")
    logger.info(f"  seq_len: {seq_len}")
    logger.info(f"  enc_in: {enc_in}")
    logger.info(f"  pred_len: {pred_len}")
    logger.info(f"  n_background: {args.n_background}")
    logger.info(f"  n_test: {args.n_test} (with stratified sampling)")
    logger.info(f"  step: {args.step}")
    logger.info("")
    logger.info("Note: Using stratified sampling to ensure ~30% anomaly samples in test set")
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading model...")
    
    # Create model configuration
    model_config_obj = Args(model_config)
    
    # Build model
    model = PatchTST(model_config_obj)
    
    # Load model weights
    model_file = os.path.join(args.model_path, "model.pth")
    if not os.path.exists(model_file):
        logger.error(f"Model file not found: {model_file}")
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    model.load_state_dict(torch.load(model_file, map_location=device))
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded from: {model_file}")
    
    # Load data
    background_data, test_data, test_labels, feature_names = load_data_for_shap(
        args.data_path, seq_len, args.step, args.n_background, args.n_test,
        batch_size=32, logger=logger
    )
    
    # Compute SHAP values
    shap_values, expected_value = compute_shap_values(
        model, background_data, test_data, device,
        seq_len, enc_in, logger
    )
    
    # Save SHAP values
    shap_path = os.path.join(output_path, "shap_values.npy")
    np.save(shap_path, shap_values)
    logger.info(f"SHAP values saved to: {shap_path}")
    
    # Generate visualizations
    plot_beeswarm(shap_values, test_data, feature_names, output_path, logger)
    plot_time_feature_heatmap(shap_values, feature_names, output_path, logger)
    plot_time_feature_heatmap_single_sample(shap_values, feature_names, output_path, 0, logger)
    
    # Save analysis info
    analysis_info = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "n_background_samples": args.n_background,
        "n_test_samples": args.n_test,
        "seq_len": seq_len,
        "enc_in": enc_in,
        "pred_len": pred_len,
        "step": args.step,
        "seed": args.seed,
        "feature_names": feature_names,
        "timestamp": timestamp,
        "expected_value": float(expected_value) if isinstance(expected_value, (int, float, np.number)) else str(expected_value)
    }
    
    info_path = os.path.join(output_path, "analysis_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_info, f, ensure_ascii=False, indent=2)
    logger.info(f"Analysis info saved to: {info_path}")
    
    logger.info("=" * 60)
    logger.info("SHAP Analysis completed successfully!")
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

