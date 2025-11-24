"""
Find Best Results from Checkpoint Directories

This script searches for all result.json files in a given directory,
finds the best F-score for each dataset (contact, ring, pcb),
and saves the results to a summary file.

Usage:
    python find_best_results.py --checkpoint_dir <path_to_checkpoints>
    
Example:
    python find_best_results.py --checkpoint_dir /home/wanting/PatchTST/PatchTST_supervised/checkpoints_paramenter_analysis_dlinear/pca
"""

import os
import json
from pathlib import Path
from datetime import datetime


def find_all_result_json_files(root_dir):
    """
    Recursively find all result.json files in the given directory.
    
    Args:
        root_dir: Root directory to search
        
    Returns:
        list: List of paths to result.json files
    """
    result_files = []
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"Error: Directory does not exist: {root_dir}")
        return result_files
    
    # Recursively search for result.json files
    for result_file in root_path.rglob('result.json'):
        result_files.append(result_file)
    
    print(f"Found {len(result_files)} result.json files")
    return result_files


def extract_dataset_name(file_path):
    """
    Extract dataset name from file path.
    
    Args:
        file_path: Path to result.json file
        
    Returns:
        str: Dataset name ('contact', 'ring', 'pcb') or None
    """
    path_str = str(file_path).lower()
    
    # Check for dataset names in the path
    if 'contact' in path_str:
        return 'contact'
    elif 'ring' in path_str:
        return 'ring'
    elif 'pcb' in path_str:
        return 'pcb'
    else:
        return None


def parse_result_json(file_path):
    """
    Parse result.json file and extract metrics.
    
    Args:
        file_path: Path to result.json file
        
    Returns:
        dict: Dictionary containing precision, recall, f_score, or None if failed
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract summary metrics
        if 'summary' in data:
            summary = data['summary']
            return {
                'precision': summary.get('precision', 0.0),
                'recall': summary.get('recall', 0.0),
                'f_score': summary.get('f_score', 0.0),
                'accuracy': summary.get('accuracy', 0.0)
            }
        else:
            print(f"Warning: 'summary' key not found in {file_path}")
            return None
            
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None


def find_best_results(checkpoint_dir, output_file):
    """
    Find best F-score results for each dataset.
    
    Args:
        checkpoint_dir: Directory containing checkpoint subdirectories with result.json
        output_file: Full path to output JSON file
        
    Returns:
        dict: Best results for each dataset
    """
    # Find all result.json files
    result_files = find_all_result_json_files(checkpoint_dir)
    
    if not result_files:
        print("No result.json files found!")
        return None
    
    # Group results by dataset
    dataset_results = {
        'contact': [],
        'ring': [],
        'pcb': []
    }
    
    # Parse all result files
    for result_file in result_files:
        dataset = extract_dataset_name(result_file)
        
        if dataset is None:
            print(f"Warning: Could not determine dataset for {result_file}")
            continue
        
        metrics = parse_result_json(result_file)
        
        if metrics is not None:
            dataset_results[dataset].append({
                'file_path': str(result_file),
                'metrics': metrics
            })
    
    # Find best F-score for each dataset
    best_results = {}
    
    for dataset, results in dataset_results.items():
        if not results:
            print(f"Warning: No results found for dataset '{dataset}'")
            continue
        
        # Sort by F-score (descending)
        sorted_results = sorted(results, 
                               key=lambda x: x['metrics']['f_score'], 
                               reverse=True)
        
        best_result = sorted_results[0]
        best_results[dataset] = best_result
        
        print(f"  File: {best_result['file_path']}")
        print(f"\n{dataset.upper()} - Best F-score: {best_result['metrics']['f_score']:.4f}")
        print(f"  Precision: {best_result['metrics']['precision']:.4f}")
        print(f"  Recall: {best_result['metrics']['recall']:.4f}")
        print(f"  Accuracy: {best_result['metrics']['accuracy']:.4f}")
    
    # Save results to file
    save_results(best_results, checkpoint_dir, output_file)
    
    return best_results


def save_results(best_results, checkpoint_dir, output_file):
    """
    Save best results to JSON file.
    
    Args:
        best_results: Dictionary of best results for each dataset
        checkpoint_dir: Original checkpoint directory
        output_file: Full path to output JSON file
    """
    # Convert output_file to absolute path
    output_file = Path(output_file).resolve()
    
    # Create parent directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare data for saving
    save_data = {
        'checkpoint_dir': str(checkpoint_dir),
        'timestamp': timestamp,
        'best_results': {}
    }
    
    # Fill in best results data
    for dataset, result in best_results.items():
        save_data['best_results'][dataset] = {
            'file_path': result['file_path'],
            'precision': result['metrics']['precision'],
            'recall': result['metrics']['recall'],
            'f_score': result['metrics']['f_score'],
            'accuracy': result['metrics']['accuracy']
        }
    
    # Save as JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Results saved to JSON: {output_file}")
    print(f"{'='*80}\n")
    
    # Print summary table
    print("\nSummary Table:")
    for dataset, result in best_results.items():
        print(f"\n{dataset.upper()}:")
        print(f"  Precision: {result['metrics']['precision']:.4f}")
        print(f"  Recall:    {result['metrics']['recall']:.4f}")
        print(f"  F-score:   {result['metrics']['f_score']:.4f}")
        print(f"  Accuracy:  {result['metrics']['accuracy']:.4f}")
        print(f"  File:      {result['file_path']}")
    
    return output_file
    
def main():
    """Main function - configure parameters here."""
    
    # ============================================================================
    # CONFIGURATION - MODIFY THESE PARAMETERS
    # ============================================================================
    
    # Checkpoint directory to search for result.json files
    checkpoint_dir = 'PatchTST_supervised/checkpoints_paramenter_analysis_patchtst'
    
    # Output file path (fixed to PatchTST_supervised/exp/find_best_results.json)
    script_dir = Path(__file__).parent
    output_file = script_dir / 'find_best_results_patchtst.json'
    
    # ============================================================================
    
    print("\n" + "="*80)
    print("FINDING BEST RESULTS FROM CHECKPOINTS")
    print("="*80)
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Output file: {output_file}")
    print("="*80 + "\n")
    
    # Find best results
    best_results = find_best_results(checkpoint_dir, output_file)
    
    if best_results:
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80 + "\n")
    else:
        print("\n" + "="*80)
        print("NO RESULTS FOUND!")
        print("="*80 + "\n")


if __name__ == '__main__':
    main()
