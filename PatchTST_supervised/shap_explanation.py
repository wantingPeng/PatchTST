"""
SHAP Explanation Script for DLinear Anomaly Detection Model

This script generates SHAP visualizations (beeswarm, waterfall, heatmap) 
for the trained DLinear model to interpret feature importance in anomaly detection.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import os
import json
import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.DLinear import Model as DLinearModel
from data_provider.data_loader_anomaly import get_anomaly_loader


class Args:
    """Configuration class to hold model arguments"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


class ModelWrapper(nn.Module):
    """
    Wrapper for DLinear model - Returns ONE aggregated anomaly score per sample
    """
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.anomaly_criterion = nn.MSELoss(reduction='none')
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, features)
        Returns:
            score: (batch_size, 1) - ONE aggregated anomaly score per sample
        """
        outputs = self.model(x)
        reconstruction_error = self.anomaly_criterion(x, outputs)  # (batch, seq_len, features)
        # Aggregate over BOTH time and feature dimensions
        score = torch.mean(reconstruction_error, dim=-1)  # (batch, seq_len)
        score = torch.mean(score, dim=1, keepdim=True)   # (batch, 1) ← KEY!
        return score


def get_feature_names_from_data(data_path):
    """
    Load parquet schema and return feature column names (excluding TimeStamp/anomaly_label)
    """
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_parquet(data_path)
    columns = list(df.columns)
    
    cleaned_columns = []
    for col in columns:
        if col in ('TimeStamp', 'anomaly_label'):
            continue
        cleaned_columns.append(col)
    
    return cleaned_columns


def load_model_and_config(checkpoint_dir, device='cuda'):
    """
    Load trained model and configuration from checkpoint directory
    
    Args:
        checkpoint_dir: Path to checkpoint directory containing model.pth and config.json
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        model: Loaded PyTorch model
        config: Configuration dictionary
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Load configuration
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Loaded configuration from: {config_path}")
    
    # Create Args object from config
    args = Args(config)
    
    # Fix data path if relative
    if hasattr(args, 'data_path') and not os.path.isfile(args.data_path):
        # Try prepending PatchTST_supervised/ for relative paths
        alt_path = os.path.join('PatchTST_supervised', args.data_path)
        if os.path.isfile(alt_path):
            args.data_path = alt_path
            print(f"Adjusted data path to: {args.data_path}")
    
    # Initialize model
    model = DLinearModel(args).float()
    
    # Load model weights
    model_path = checkpoint_dir / "model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Loaded model weights from: {model_path}")
    print(f"Model architecture: {args.model}")
    print(f"Sequence length: {args.seq_len}, Features: {args.enc_in}")
    
    return model, config, args


def prepare_shap_data(args, num_background=100, num_test=50, random_seed=42):
    """
    Prepare background and test data for SHAP analysis
    
    Args:
        args: Configuration arguments
        num_background: Number of background samples for SHAP (from train set)
        num_test: Number of test samples to explain (50% anomaly, 50% normal)
        random_seed: Random seed for reproducibility
    
    Returns:
        background_data: Background data for SHAP explainer (from train set - all normal)
        test_data: Test data to explain (stratified sampling)
        feature_names: List of feature names
        test_labels: Labels for test samples
    """
    print("\n" + "="*80)
    print("Preparing data for SHAP analysis...")
    print("="*80)
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Get feature names from data file to ensure plots show real feature labels
    feature_names = get_feature_names_from_data(args.data_path)
    if len(feature_names) != args.enc_in:
        print(f"[Warning] Number of feature names ({len(feature_names)}) does not match enc_in ({args.enc_in})")
    
    # ========== 1. Load background data from TRAIN set (all normal samples) ==========
    print("\n[1/2] Loading background data from TRAIN set (normal samples only)...")
    train_dataset, train_loader = get_anomaly_loader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        step=args.step,
        mode='train',
        num_workers=0
    )
    
    # Collect train samples (should be all normal based on data_loader_anomaly.py)
    train_samples = []
    for batch_x, batch_y in train_loader:
        train_samples.append(batch_x)
        if len(train_samples) * batch_x.shape[0] >= num_background:
            break
    
    train_samples = torch.cat(train_samples, dim=0)
    
    # Randomly select background samples from train set
    if len(train_samples) > num_background:
        indices = torch.randperm(len(train_samples))[:num_background]
        background_data = train_samples[indices].numpy()
    else:
        background_data = train_samples.numpy()
    
    print(f"Background samples (from train): {background_data.shape}")
    
    # ========== 2. Load test data with STRATIFIED SAMPLING ==========
    print("\n[2/2] Loading test data with stratified sampling (50% anomaly, 50% normal)...")
    test_dataset, test_loader = get_anomaly_loader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        step=args.step,
        mode='test',
        num_workers=0
    )
    
    # Collect all test samples and labels
    all_test_samples = []
    all_test_labels = []
    
    for batch_x, batch_y in test_loader:
        all_test_samples.append(batch_x)
        all_test_labels.append(batch_y)
    
    all_test_data = torch.cat(all_test_samples, dim=0)
    all_test_labels = torch.cat(all_test_labels, dim=0)
    
    print(f"Total test samples collected: {all_test_data.shape[0]}")
    
    # Identify anomaly windows: windows where >50% of time steps are anomalies
    # For each window, count anomaly points (label=1)
    anomaly_counts = all_test_labels.sum(dim=1)  # Sum over time dimension
    anomaly_ratio = anomaly_counts / args.seq_len  # Ratio of anomalies in each window
    
    # Windows with >50% anomaly time steps are considered "anomaly windows"
    anomaly_mask = (anomaly_ratio > 0.5)  # Boolean tensor [N]
    normal_indices = torch.where(~anomaly_mask)[0]
    anomaly_indices = torch.where(anomaly_mask)[0]
    
    print(f"\nWindow classification:")
    print(f"  Normal windows: {len(normal_indices)} ({100*len(normal_indices)/len(all_test_data):.1f}%)")
    print(f"  Anomaly windows (>50% anomaly): {len(anomaly_indices)} ({100*len(anomaly_indices)/len(all_test_data):.1f}%)")
    
    # Stratified sampling: 50% anomaly windows, 50% normal windows
    n_anomaly_samples = min(num_test // 2, len(anomaly_indices))
    n_normal_samples = num_test - n_anomaly_samples
    
    # Adjust if not enough anomaly samples
    if n_anomaly_samples > len(anomaly_indices):
        n_anomaly_samples = len(anomaly_indices)
        n_normal_samples = num_test - n_anomaly_samples
        print(f"\n[Warning] Not enough anomaly windows. Adjusting to {n_anomaly_samples} anomaly + {n_normal_samples} normal")
    
    # Adjust if not enough normal samples
    if n_normal_samples > len(normal_indices):
        n_normal_samples = len(normal_indices)
        n_anomaly_samples = num_test - n_normal_samples
        print(f"\n[Warning] Not enough normal windows. Adjusting to {n_anomaly_samples} anomaly + {n_normal_samples} normal")
    
    # Random selection with fixed seed
    torch.manual_seed(random_seed)
    selected_anomaly_idx = anomaly_indices[torch.randperm(len(anomaly_indices))[:n_anomaly_samples]]
    selected_normal_idx = normal_indices[torch.randperm(len(normal_indices))[:n_normal_samples]]
    
    # Combine and shuffle
    selected_indices = torch.cat([selected_anomaly_idx, selected_normal_idx])
    selected_indices = selected_indices[torch.randperm(len(selected_indices))]
    
    # Extract selected samples
    test_data = all_test_data[selected_indices].numpy()
    test_labels = all_test_labels[selected_indices].numpy()
    
    print(f"\nFinal stratified test samples: {test_data.shape}")
    print(f"  Anomaly windows: {n_anomaly_samples} ({100*n_anomaly_samples/(n_anomaly_samples+n_normal_samples):.1f}%)")
    print(f"  Normal windows: {n_normal_samples} ({100*n_normal_samples/(n_anomaly_samples+n_normal_samples):.1f}%)")
    
    print("="*80)
    
    return background_data, test_data, feature_names, test_labels


def compute_shap_values(model, background_data, test_data, device='cuda'):
    """
    Compute SHAP values using DeepExplainer
    
    Args:
        model: PyTorch model
        background_data: Background samples for SHAP
        test_data: Test samples to explain
        device: Device for computation
    
    Returns:
        shap_values: SHAP values for test samples
        explainer: SHAP explainer object
    """
    print("\n" + "="*80)
    print("Computing SHAP values...")
    print("="*80)
    
    # Wrap model for SHAP (computes anomaly scores)
    wrapped_model = ModelWrapper(model)
    wrapped_model.to(device)
    wrapped_model.eval()
    
    # Convert to tensors
    background_tensor = torch.FloatTensor(background_data).to(device)
    test_tensor = torch.FloatTensor(test_data).to(device)
    
    print(f"Background data shape: {background_tensor.shape}")
    print(f"Test data shape: {test_tensor.shape}")
    print(f"Initializing SHAP DeepExplainer with {background_data.shape[0]} background samples...")
    
    # Create SHAP explainer
    explainer = shap.DeepExplainer(wrapped_model, background_tensor)
    
    print("Computing SHAP values for test samples...")
    print("(This may take a few minutes...)")
    
    # Compute SHAP values
    shap_values = explainer.shap_values(test_tensor, check_additivity=False)
    
    print(f"SHAP values computed. Shape: {shap_values.shape}")
    print(f"  Expected shape: (n_samples={test_data.shape[0]}, seq_len={test_data.shape[1]}, features={test_data.shape[2]})")
    
    return shap_values, explainer


def create_visualizations(shap_values, test_data, feature_names, output_dir, args):
    """
    Create SHAP visualizations: beeswarm, waterfall, and heatmap
    
    Args:
        shap_values: Computed SHAP values (n_samples, seq_len, features)
        test_data: Test samples (n_samples, seq_len, features)
        feature_names: List of feature names
        output_dir: Directory to save visualizations
        args: Configuration arguments
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("Creating SHAP visualizations...")
    print("="*80)
    
    seq_len = args.seq_len
    n_features = args.enc_in
    n_samples = shap_values.shape[0]
    
    # SHAP DeepExplainer 对多输出模型返回 4 维张量：
    # (n_outputs, n_samples, n_input_features, n_input_time_steps)
    # 这里模型输出长度 = seq_len，输入时间长度也 = seq_len
    print(f"SHAP values shape: {shap_values.shape}")
    
    if len(shap_values.shape) == 4:
        # Correct dimension order: (n_samples, seq_len, features, n_outputs=1)
        n_samples_4d, n_time_4d, n_feat_4d, n_outputs_4d = shap_values.shape
        print(f"4D SHAP: samples={n_samples_4d}, time={n_time_4d}, features={n_feat_4d}, outputs={n_outputs_4d}")
        
        # Remove singleton output dimension
        if n_outputs_4d == 1:
            shap_values_reshaped = shap_values[:, :, :, 0]  # (samples, time, features)
            print(f"Removed singleton output dim → {shap_values_reshaped.shape}")
        else:
            shap_values_reshaped = np.mean(shap_values, axis=-1)
            print(f"[Warning] Expected n_outputs=1, got {n_outputs_4d}, averaged")
    elif len(shap_values.shape) == 3:
        # 已经是 (n_samples, seq_len, n_features)
        shap_values_reshaped = shap_values
    else:
        raise ValueError(f"Unsupported SHAP values shape: {shap_values.shape}")
    
    print(f"Reshaped SHAP values for visualization: {shap_values_reshaped.shape}")
    
    # Use MAX (not MEAN) to preserve variance
    print("[Fix] Using MAX  SHAP value to avoid over-averaging")
    idx_max = shap_values_reshaped.argmax(axis=1)
    shap_values_avg = np.take_along_axis(shap_values_reshaped, idx_max[:, None, :], axis=1).squeeze(1)  # Shape: (n_samples, n_features)
    
    print(f"Peak SHAP values (max  from each time window): {shap_values_avg.shape}")
    
    # Average test data across time for feature values
    test_data_avg = np.mean(test_data, axis=1)  # Shape: (n_samples, n_features)
    
    # ========== 1. Beeswarm Plot ==========
    print("\n[1/3] Creating beeswarm plot...")
    plt.figure(figsize=(12, 8))
    
    # Create SHAP Explanation object for beeswarm plot
    shap_explanation = shap.Explanation(
        values=shap_values_avg,
        base_values=np.zeros(n_samples),
        data=test_data_avg,
        feature_names=feature_names
    )
    
    shap.plots.beeswarm(shap_explanation, show=False, max_display=20)
    plt.title("SHAP Beeswarm Plot", fontsize=14, pad=20)
    plt.tight_layout()
    plt.xticks(rotation=30)

    beeswarm_path = output_dir / "shap_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=300, bbox_inches='tight')
    print(f"Saved beeswarm plot to: {beeswarm_path}")
    plt.close()
    

    
    # ========== 3. Heatmap Plot ==========
    print("\n[3/3] Creating heatmap plot...")
    
    # Use first 20 samples and top 15 features for better visualization
    n_display_features = min(15, n_features)
    
    # Get top features by mean **absolute** SHAP magnitude for ranking,
    # but actual heatmap values remain signed.
    mean_abs_shap = np.mean(np.abs(shap_values_avg), axis=0)
    top_feature_indices = np.argsort(mean_abs_shap)[-n_display_features:][::-1]
    top_feature_names = [feature_names[i] for i in top_feature_indices]
    
    # ========== 4. Time-series Heatmap (optional: SHAP across time for one sample) ==========
    print("\n[4/4] Creating time-series heatmap for sample 0...")
    
    sample_idx = 0
    shap_values_ts = shap_values_reshaped[sample_idx]  # Shape: (seq_len, n_features)
    
    # Use top features
    shap_values_ts_top = shap_values_ts[:, top_feature_indices]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    im = ax.imshow(shap_values_ts_top.T, cmap='RdBu_r', aspect='auto',
                   vmin=-np.abs(shap_values_ts_top).max(),
                   vmax=np.abs(shap_values_ts_top).max())
    
    # Set ticks and labels
    ax.set_xticks(np.arange(0, seq_len, max(1, seq_len // 10)))
    ax.set_yticks(np.arange(n_display_features))
    ax.set_xticklabels(np.arange(0, seq_len, max(1, seq_len // 10)))
    ax.set_yticklabels(top_feature_names)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('SHAP Value', rotation=270, labelpad=20, fontsize=12)
    
    # Add title and labels
    ax.set_title(f"SHAP Values Over Time - Sample {sample_idx} (Top Features)", fontsize=14, pad=20)
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Features", fontsize=12)
    
    plt.tight_layout()
    
    ts_heatmap_path = output_dir / "shap_timeseries_heatmap_sample0.png"
    plt.savefig(ts_heatmap_path, dpi=300, bbox_inches='tight')
    print(f"Saved time-series heatmap to: {ts_heatmap_path}")
    plt.close()
    
    print("\n" + "="*80)
    print("All visualizations created successfully!")
    print("="*80)


def save_shap_summary(shap_values, feature_names, output_dir, args):
    """
    Save SHAP summary statistics to CSV
    
    Args:
        shap_values: Computed SHAP values
        feature_names: List of feature names
        output_dir: Directory to save summary
        args: Configuration arguments
    """
    output_dir = Path(output_dir)
    
    # Reshape SHAP values if needed
    seq_len = args.seq_len
    n_features = args.enc_in
    n_samples = shap_values.shape[0]
    
    if len(shap_values.shape) == 4:
        # 与 create_visualizations 中相同的维度处理逻辑
        n_outputs, n_samples_shap, n_feat_shap, n_time_shap = shap_values.shape
        shap_values_mean_out = np.mean(shap_values, axis=0)           # (samples, feat, time)
        shap_values_reshaped = np.transpose(shap_values_mean_out, (0, 2, 1))  # (samples, time, feat)
    elif len(shap_values.shape) == 3:
        shap_values_reshaped = shap_values
    else:
        raise ValueError(f"Unsupported SHAP values shape in save_shap_summary: {shap_values.shape}")
    
    # Calculate feature importance (mean absolute SHAP value across samples and time)
    # 这里依然使用绝对值来做“重要性”度量，与图像中的有符号 SHAP 互补
    feature_importance = np.mean(np.abs(shap_values_reshaped), axis=(0, 1))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': feature_importance
    })
    df = df.sort_values('Mean_Abs_SHAP', ascending=False)
    
    # Save to CSV
    summary_path = output_dir / "shap_feature_importance.csv"
    df.to_csv(summary_path, index=False)
    
    print(f"\nSaved feature importance summary to: {summary_path}")
    print("\nTop 10 Most Important Features:")
    print(df.head(10).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='SHAP Explanation for DLinear Anomaly Detection')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Path to checkpoint directory containing model.pth and config.json')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save SHAP visualizations (default: checkpoint_dir/shap_analysis)')
    parser.add_argument('--num_background', type=int, default=100,
                        help='Number of background samples for SHAP from train set (default: 100)')
    parser.add_argument('--num_test', type=int, default=50,
                        help='Number of test samples to explain with stratified sampling (default: 50)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for computation (default: cuda)')
    
    args_cmd = parser.parse_args()
    
    # Set output directory
    if args_cmd.output_dir is None:
        args_cmd.output_dir = os.path.join(args_cmd.checkpoint_dir, 'shap_analysis')
    
    print("\n" + "="*80)
    print("SHAP Explanation Script for DLinear Anomaly Detection")
    print("="*80)
    print(f"Checkpoint directory: {args_cmd.checkpoint_dir}")
    print(f"Output directory: {args_cmd.output_dir}")
    print(f"Device: {args_cmd.device}")
    print(f"Background samples (from train): {args_cmd.num_background}")
    print(f"Test samples (stratified): {args_cmd.num_test}")
    print(f"Random seed: {args_cmd.random_seed}")
    print("="*80)
    
    # Check device availability
    if args_cmd.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args_cmd.device = 'cpu'
    
    # Step 1: Load model and configuration
    model, config, args = load_model_and_config(args_cmd.checkpoint_dir, args_cmd.device)
    
    # Step 2: Prepare data (background from train, test with stratified sampling)
    background_data, test_data, feature_names, test_labels = prepare_shap_data(
        args, args_cmd.num_background, args_cmd.num_test, args_cmd.random_seed
    )
    
    # Step 3: Compute SHAP values
    shap_values, explainer = compute_shap_values(
        model, background_data, test_data, args_cmd.device
    )
    
    # Step 4: Create visualizations
    create_visualizations(
        shap_values, test_data, feature_names, args_cmd.output_dir, args
    )
    
    # Step 5: Save summary statistics
    save_shap_summary(shap_values, feature_names, args_cmd.output_dir, args)
    
    print("\n" + "="*80)
    print("SHAP analysis completed successfully!")
    print(f"All results saved to: {args_cmd.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

