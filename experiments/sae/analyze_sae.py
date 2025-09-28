import argparse
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from scipy import stats
from tqdm import tqdm

def analyze_sae_with_ttest(
    exp_id: str,
    target_files: list,
    label_map: dict,
    output_dir: Path,
    num_features: int,
):
    """
    Perform significance analysis using t-tests on SAE unit activations.
    """
    tumor_activations = []
    normal_activations = []

    for file_path in tqdm(target_files, desc=f"[{exp_id}] Processing slides for t-test"):
        slide_id = file_path.stem
        label = label_map.get(slide_id)
        if label not in ('tumor', 'normal'):
            continue
        is_tumor = (label == 'tumor')

        features = torch.load(file_path, map_location=torch.device('cpu'))
        pooled = torch.max(features, dim=0).values
        if is_tumor:
            tumor_activations.append(pooled)
        else:
            normal_activations.append(pooled)

    print(f"Processed {len(tumor_activations)} tumor and {len(normal_activations)} normal slides.")
    if not tumor_activations or not normal_activations:
        print(f"Warning: Missing tumor or normal slides for exp {exp_id}. Skipping t-test.")
        return

    tumor_stack = torch.stack(tumor_activations)
    normal_stack = torch.stack(normal_activations)

    results = []
    for unit in tqdm(range(num_features), desc=f"[{exp_id}] Performing t-tests"):
        t_vals = tumor_stack[:, unit].numpy()
        n_vals = normal_stack[:, unit].numpy()
        t_stat, p_val = stats.ttest_ind(t_vals, n_vals, equal_var=False, nan_policy='omit')
        results.append({
            'unit_id': unit,
            't_statistic': t_stat,
            'p_value': p_val,
            'tumor_mean_activation': t_vals.mean(),
            'normal_mean_activation': n_vals.mean(),
        })

    results_df = pd.DataFrame(results)
    sorted_df = results_df.sort_values(
        by=['p_value', 't_statistic'], 
        ascending=[True, False]
    )

    save_dir = output_dir / exp_id / "analysis"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_file = save_dir / "sae_analysis_t_test.csv"
    sorted_df.to_csv(out_file, index=False)

    print(f"\n--- Top 5 significant units (t-test) for experiment {exp_id} ---")
    print(sorted_df.head(5))
    print(f"T-test results saved to {out_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Run SAE activation analysis for a given experiment using t-tests."
    )
    parser.add_argument(
        'exp_id', type=str,
        help='Experiment ID to analyze'
    )
    parser.add_argument(
        '--split_csv_path', type=Path, required=True,
        help='Path to the split CSV file for this experiment'
    )
    parser.add_argument(
        '--feature_dir', type=Path, required=True,
        help='Directory containing .pt feature files from SAE'
    )
    parser.add_argument(
        '--dataset_csv', type=Path, required=True,
        help='CSV file mapping slide_id to label'
    )
    parser.add_argument(
        '--output_dir', type=Path, required=True,
        help='Directory to save analysis outputs'
    )
    parser.add_argument(
        '--num_features', type=int, required=True,
        help='Number of features/units in SAE activation vectors'
    )
    args = parser.parse_args()

    print(f"\n{'='*20} Starting analysis for experiment: {args.exp_id} {'='*20}")

    # --- 1. Load and prepare common data ---
    print("Step 1: Loading data and identifying target files...")
    try:
        labels_df = pd.read_csv(args.dataset_csv)
        labels_df['slide_id'] = labels_df['slide_id'].astype(str)
        # Convert label to string and lowercase for robust matching
        labels_df['label'] = labels_df['label'].astype(str).str.lower()
        label_map = dict(zip(labels_df['slide_id'], labels_df['label']))

        split_df = pd.read_csv(args.split_csv_path)
        raw_train = split_df['train'].dropna().astype(str).tolist()
        train_slides = set(Path(val).stem for val in raw_train)

        all_files = list(args.feature_dir.glob('*.pt'))
        target_files = [f for f in all_files if f.stem in train_slides]
        
        print(f"Found {len(train_slides)} slide IDs in the training set.")
        print(f"Found {len(target_files)} feature files matching the training set.")
        if not target_files:
            print("Error: No matching feature files found for the training split. Exiting.")
            return

    except FileNotFoundError as e:
        print(f"Error: A required file was not found: {e}. Exiting.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}. Exiting.")
        return

    # --- 2. Run t-test analysis ---
    print(f"\nStep 2: Running analysis with t-tests...")
    analyze_sae_with_ttest(
        args.exp_id,
        target_files,
        label_map,
        args.output_dir,
        args.num_features,
    )
    
    print(f"\n{'='*20} Analysis for experiment {args.exp_id} complete. {'='*20}")

if __name__ == '__main__':
    main()