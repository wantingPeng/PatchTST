"""
Parameter Analysis for PatchTST Anomaly Detection
This script trains models with different parameter values and visualizes F1 score performance.
Supports analyzing: win_size, n_heads, e_layers, anormly_ratio, patch_len, d_model
Adapted for PatchTST/PatchTST_supervised project
"""

import os
import sys
import json
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# EXPERIMENT CONFIGURATION - MODIFY THESE PARAMETERS
# ============================================================================

# Select which parameter to analyze (choose one: 'win_size', 'k', 'anormly_ratio')
PARAM_TO_ANALYZE = 'seq_len'  # Change this to 'k' or 'anormly_ratio' for other analyses

# Datasets to test
DATASETS = ['contact', 'ring', 'pcb']

# Parameter values to test (modify based on PARAM_TO_ANALYZE)
# Note: n_heads must be divisors of d_model for Transformer models
# For d_model=[128, 256, 512, 1024], valid n_heads are powers of 2: [1, 2, 4, 8, 16, 32]
PARAM_VALUES = {
    'n_heads': [1, 2, 4, 8],  # Changed from [1,3,5,7] to ensure divisibility with d_model
    'e_layers': [1,3,5,7,9,11],
    'anormly_ratio': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0,7.0,8.0,9.0,10.0],
    'patch_len': [10,16,22,28,34],
    # seq_len and pred_len should change together for anomaly detection (reconstruction task)
    'seq_len': [20,40,60,80,100,120,140],  # When analyzing seq_len, pred_len will be set to the same value
    'kernel_size': [5,15,25,35,45,55,65,75,85],
}

# Fixed parameters (used when not being analyzed)
FIXED_PARAMS = {
    'train_epochs': 1,
    'batch_size': 64,
    'learning_rate': 1e-4,
    'seq_len': 100,
    'pred_len': 100,
    'n_heads': 2,
    'e_layers': 3,
    'd_model': 256,
    'd_ff': 512,
    'kernel_size': 25,
    'patch_len': 10,
    'stride': 5,
    'anormly_ratio': 3.0,
    'model': 'PatchTST',
    'data': 'custom',
    'features': 'M',
    'des': 'Anomaly Detection'
}

# Output configuration
OUTPUT_DIR = 'experiments/results/PatchTST2_pca'
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# ============================================================================
# DATASET CONFIGURATIONS
# ============================================================================

DATASET_CONFIGS = {
     'contact': {
         'data_path': 'dataset/dataset/downsampleData_scratch_1minut/contact/contact_cleaned_1minut_20250928_172122.parquet',
         'enc_in': 10,
         'dec_in': 10,
         'c_out': 10,
     },
    'ring': {
        'data_path': 'dataset/dataset/downsampleData_scratch_1minut/ring/Ring_cleaned_1minut_20250928_170147.parquet',
        'enc_in': 10,
        'dec_in': 10,
        'c_out': 10,
     },
     'pcb': {
         'data_path': 'dataset/dataset/downsampleData_scratch_1minut/pcb/pcb_cleaned_1minut_20250928_161509.parquet',
        'enc_in': 10,
        'dec_in': 10,
        'c_out': 10,
     }
}

# Plot styling
PLOT_CONFIG = {
    'colors': {'contact': 'red', 'ring': 'green', 'pcb': 'blue'},
    'markers': {'contact': 'o', 'ring': 's', 'pcb': '^'},
    'labels': {'contact': 'Contact', 'ring': 'Ring', 'pcb': 'PCB'}
}

PARAM_DISPLAY_NAMES = {
    'n_heads': 'Number of Heads',
    'e_layers': 'Encoder Layers',
    'anormly_ratio': 'Anomaly Ratio',
    'patch_len': 'Patch Length',
    'd_model': 'D Model',
    'seq_len': 'Sequence Length (seq_len=pred_len)',
    'pred_len': 'Prediction Length (seq_len=pred_len)',
}

# ============================================================================
# EXPERIMENT FUNCTIONS
# ============================================================================

def get_dataset_config(dataset_name):
    """Get configuration for each dataset."""
    return DATASET_CONFIGS.get(dataset_name)


def run_single_experiment(dataset_name, param_name, param_value, fixed_params):
    """
    Run a single experiment with specified parameter value.
    
    Args:
        dataset_name: Name of the dataset ('contact', 'ring', 'pcb')
        param_name: Name of the parameter being tested
        param_value: Value of the parameter to test
        fixed_params: Dictionary of fixed parameters
        
    Returns:
        dict: Results containing precision, recall, f_score, accuracy
    """
    config = get_dataset_config(dataset_name)
    if config is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create unique model save path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f'{dataset_name}_{param_name}{param_value}_{timestamp}'
    
    # Build parameters dictionary
    params = fixed_params.copy()
    params[param_name] = param_value
    
    # For anomaly detection (reconstruction task), seq_len and pred_len should be the same
    # When analyzing seq_len, automatically set pred_len to the same value
    if param_name == 'seq_len':
        params['pred_len'] = param_value

    
    # exp_anomaly.py creates path as: checkpoints/setting/ where setting = model_data_des
    # So if we want: checkpoints_n_heads_analysis/PatchTST_custom_experiment_id/checkpoint_xxx/
    # We pass: checkpoints='checkpoints_n_heads_analysis', des=experiment_id
    # Then setting = PatchTST_custom_experiment_id
    # And path = checkpoints_n_heads_analysis/PatchTST_custom_experiment_id/
    
    checkpoints_parent = f'checkpoints2_{param_name}_analysis_PatchTST_pca'
    setting = f"{params['model']}_{params['data']}_{experiment_id}"
    model_save_path = os.path.join(checkpoints_parent, setting)
    
    # Build command for run_anomaly_detection.py
    cmd = [
        'python', '-u', 'exp/run_anomaly_detection.py',
        '--is_training', '1',
        '--test_only', 'False',
        '--data_path', config['data_path'],
        '--model', params['model'],
        '--data', params['data'],
        '--features', params['features'],
        '--seq_len', str(params['seq_len']),
        '--pred_len', str(params['pred_len']),
        '--enc_in', str(config['enc_in']),
        '--dec_in', str(config['dec_in']),
        '--c_out', str(config['c_out']),
        '--batch_size', str(params['batch_size']),
        '--learning_rate', str(params['learning_rate']),
        '--anormly_ratio', str(params['anormly_ratio']),
        '--train_epochs', str(params['train_epochs']),
        '--patience', '10',
        '--n_heads', str(params['n_heads']),
        '--d_model', str(params['d_model']),
        '--e_layers', str(params['e_layers']),
        '--d_ff', str(params['d_ff']),
        '--patch_len', str(params['patch_len']),
        '--stride', str(params['stride']),
        '--des', experiment_id,
        '--checkpoints', checkpoints_parent
    ]
    
    print(f"\n{'='*80}")
    print(f"Running: {dataset_name} | {param_name}={param_value}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    # Run the command
    try:
        # Ensure we run from project root so exp/* imports work
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        subprocess.run(cmd, check=True, capture_output=False, text=True, cwd=project_root)
        
        # Find result.json directly in model_save_path
        # Structure: {checkpoints_parent}/{setting}/result.json
        if not os.path.exists(model_save_path):
            print(f"Warning: Model save path not found: {model_save_path}")
            return None
            
        result_json_path = os.path.join(model_save_path, 'result.json')
        
        # Read results
        if os.path.exists(result_json_path):
            with open(result_json_path, 'r') as f:
                results = json.load(f)
                return results['summary']
        else:
            print(f"Warning: Result file not found at {result_json_path}")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def run_all_experiments(datasets, param_name, param_values, fixed_params):
    """
    Run experiments for all datasets and parameter values.
    
    Args:
        datasets: List of dataset names
        param_name: Name of the parameter being tested
        param_values: List of parameter values to test
        fixed_params: Dictionary of fixed parameters
        
    Returns:
        dict: Results organized by dataset and parameter value
    """
    results = {dataset: {} for dataset in datasets}
    
    total_experiments = len(datasets) * len(param_values)
    current_experiment = 0
    
    for dataset in datasets:
        for param_value in param_values:
            current_experiment += 1
            print(f"\n{'#'*80}")
            print(f"# Experiment {current_experiment}/{total_experiments}")
            print(f"# Dataset: {dataset}, {param_name}: {param_value}")
            print(f"{'#'*80}\n")
            
            result = run_single_experiment(
                dataset_name=dataset,
                param_name=param_name,
                param_value=param_value,
                fixed_params=fixed_params
            )
            
            if result:
                results[dataset][param_value] = result
                print(f"\n✓ Results: Precision={result['precision']:.4f}, "
                      f"Recall={result['recall']:.4f}, F-score={result['f_score']:.4f}")
            else:
                print(f"\n✗ Failed to get results for {dataset} with {param_name}={param_value}")
    
    return results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_results(results, param_name, output_dir=OUTPUT_DIR):
    """
    Plot F1 scores for different parameter values and datasets.
    
    Args:
        results: Dictionary containing results for each dataset and parameter value
        param_name: Name of the parameter being analyzed
        output_dir: Directory to save output files
    """
    # Set up the plot
    plt.figure(figsize=(10, 6))
    plt.style.use(PLOT_STYLE)
    
    colors = PLOT_CONFIG['colors']
    markers = PLOT_CONFIG['markers']
    labels = PLOT_CONFIG['labels']
    
    # Plot each dataset
    for dataset, dataset_results in results.items():
        if not dataset_results:
            continue
            
        # Extract parameter values and f1 scores
        param_vals = sorted(dataset_results.keys())
        f1_scores = [dataset_results[pv]['f_score'] * 100 for pv in param_vals]
        
        # Plot line with markers
        plt.plot(param_vals, f1_scores, 
                marker=markers[dataset], 
                color=colors[dataset], 
                linewidth=2, 
                markersize=8,
                label=labels[dataset])
        
        # Add value annotations next to each point
        for param_val, f1_score in zip(param_vals, f1_scores):
            plt.annotate(f'{f1_score:.1f}', 
                        xy=(param_val, f1_score),
                        xytext=(5, 5), 
                        textcoords='offset points',
                        fontsize=9,
                        color=colors[dataset],
                        weight='bold')
    
    # Customize plot
    param_display_name = PARAM_DISPLAY_NAMES.get(param_name, param_name)
    plt.xlabel(param_display_name, fontsize=12, weight='bold')
    plt.ylabel('F1 Score (%)', fontsize=12, weight='bold')
    plt.title(f'F1 Score vs {param_display_name} for Different Datasets', 
             fontsize=14, weight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Set axis limits
    all_param_vals = []
    all_f1_scores = []
    for dataset_results in results.values():
        for pv, metrics in dataset_results.items():
            all_param_vals.append(pv)
            all_f1_scores.append(metrics['f_score'] * 100)
    
    # Ensure x-axis always shows configured parameter values
    configured_param_vals = PARAM_VALUES.get(param_name, [])
    if configured_param_vals:
        # Convert configured values to same type as experiment outputs when possible
        normalized_config_vals = []
        for val in configured_param_vals:
            if isinstance(val, (int, float)):
                normalized_config_vals.append(val)
            else:
                try:
                    # Attempt numeric conversion (covers cases like "10")
                    numeric_val = float(val)
                    # Cast back to int if it represents an integer value
                    if numeric_val.is_integer():
                        numeric_val = int(numeric_val)
                    normalized_config_vals.append(numeric_val)
                except (TypeError, ValueError):
                    normalized_config_vals.append(val)
        all_param_vals.extend(normalized_config_vals)

    if all_param_vals:
        x_min, x_max = min(all_param_vals), max(all_param_vals)
        x_range = x_max - x_min
        plt.xlim(x_min - x_range * 0.05, x_max + x_range * 0.05)
        unique_ticks = sorted(set(all_param_vals))
        plt.xticks(unique_ticks)
    
    if all_f1_scores:
        y_min = max(0, min(all_f1_scores) - 5)
        y_max = min(100, max(all_f1_scores) + 5)
        plt.ylim(y_min, y_max)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    
    png_path = os.path.join(output_dir, f'{param_name}_f1_scores.png')
    pdf_path = os.path.join(output_dir, f'{param_name}_f1_scores.pdf')
    
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    
    print(f"\n{'='*80}")
    print(f"Plot saved:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")
    print(f"{'='*80}\n")
    
    plt.show()




def save_results_to_json(results, param_name, output_dir=OUTPUT_DIR):
    """
    Save results to JSON file.
    
    Args:
        results: Dictionary containing results
        param_name: Name of the parameter being analyzed
        output_dir: Directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f'{param_name}_results.json')
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to JSON: {json_path}")


def load_results_from_json(param_name, output_dir=OUTPUT_DIR):
    """
    Load results from JSON file.
    
    Args:
        param_name: Name of the parameter being analyzed
        output_dir: Directory containing output files
        
    Returns:
        dict: Loaded results
    """
    json_path = os.path.join(output_dir, f'{param_name}_results.json')
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Results file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    # Convert string keys back to appropriate types
    converted_results = {}
    for dataset, dataset_results in results.items():
        converted_results[dataset] = {}
        for param_val_str, metrics in dataset_results.items():
            # Try to convert to int first, then float, otherwise keep as string
            try:
                param_val = int(param_val_str)
            except ValueError:
                try:
                    param_val = float(param_val_str)
                except ValueError:
                    param_val = param_val_str
            converted_results[dataset][param_val] = metrics
    
    return converted_results


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to run the parameter analysis."""
    
    print("\n" + "="*80)
    print("PatchTST - PARAMETER ANALYSIS FOR ANOMALY DETECTION")
    print("="*80)
    print(f"Parameter to analyze: {PARAM_TO_ANALYZE}")
    print(f"Parameter values: {PARAM_VALUES[PARAM_TO_ANALYZE]}")
    print(f"Datasets: {DATASETS}")
    print(f"Fixed parameters: {FIXED_PARAMS}")
    print("="*80 + "\n")
    
    # Change to project root directory (PatchTST_supervised)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    print(f"Working directory: {os.getcwd()}\n")
    
    # Get parameter values to test
    param_values = PARAM_VALUES[PARAM_TO_ANALYZE]
    
    # Run experiments
    print(f"\nStarting experiments...")
    results = run_all_experiments(
        datasets=DATASETS,
        param_name=PARAM_TO_ANALYZE,
        param_values=param_values,
        fixed_params=FIXED_PARAMS
    )
    
    # Save results
    print(f"\nSaving results...")
    save_results_to_json(results, PARAM_TO_ANALYZE, OUTPUT_DIR)
    
    # Plot results
    print(f"\nGenerating plots...")
    plot_results(results, PARAM_TO_ANALYZE, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"All results saved in: {OUTPUT_DIR}")
    print("="*80 + "\n")


# ============================================================================
# UTILITY: PLOT ONLY MODE
# ============================================================================

def plot_only_mode(param_name):
    """
    Load existing results and regenerate plots only.
    
    Args:
        param_name: Name of the parameter to plot
    """
    print(f"\n{'='*80}")
    print("PLOT ONLY MODE")
    print(f"Loading results for: {param_name}")
    print(f"{'='*80}\n")
    
    # Change to project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    print(f"Working directory: {os.getcwd()}\n")
    
    try:
        results = load_results_from_json(param_name, OUTPUT_DIR)
        print("✓ Results loaded successfully\n")
        
        plot_results(results, param_name, OUTPUT_DIR)
        
        print(f"\n{'='*80}")
        print("PLOT GENERATION COMPLETE!")
        print(f"{'='*80}\n")
        
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("Please run the full analysis first.")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    # Set to True to only regenerate plots from existing results
    PLOT_ONLY = False
    
    if PLOT_ONLY:
        plot_only_mode(PARAM_TO_ANALYZE)
    else:
        main()

