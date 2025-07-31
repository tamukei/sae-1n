from __future__ import print_function

import os
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from typing import Optional
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig
import wandb

from utils.file_utils import save_pkl
from utils.seed import seed_torch
from utils.core_utils import train
from dataset.dataset_generic import Generic_MIL_Dataset


def get_selected_sae_units(cfg_features: DictConfig) -> Optional[np.ndarray]:
    """
    Retrieves top SAE unit IDs based on a specified metric from a summary CSV.
    """
    if not cfg_features.get("use_sae_selected_units", False):
        return None

    summary_path = Path(cfg_features.sae_unit_summary_path)
    num_top_units = cfg_features.num_top_sae_units
    metric = cfg_features.sae_unit_selection_metric

    if not summary_path.exists():
        print(f"Warning: SAE unit summary file not found: {summary_path}. Skipping unit selection.")
        return None

    try:
        df_summary = pd.read_csv(summary_path)
    except Exception as e:
        print(f"Error reading SAE summary file {summary_path}: {e}. Skipping unit selection.")
        return None

    if 'unit_id' not in df_summary.columns:
        print(f"Error: 'unit_id' column not found in {summary_path}. Skipping unit selection.")
        return None
    if metric not in df_summary.columns:
        print(f"Error: Metric '{metric}' not found in columns of {summary_path}. Columns: {df_summary.columns.tolist()}. Skipping unit selection.")
        return None

    # Sort by metric (descending for AUPRC, ascending for P Value)
    df_sorted = df_summary.sort_values(by=metric, ascending=True)
    
    selected_units_series = df_sorted['unit_id'].head(num_top_units)
    
    if selected_units_series.empty:
        print(f"Warning: No units selected from {summary_path} with metric {metric} and top {num_top_units}. Skipping unit selection.")
        return None
        
    selected_unit_indices = selected_units_series.values.astype(int)
    
    print(f"Selected top {len(selected_unit_indices)} SAE units based on '{metric}'. Min index: {selected_unit_indices.min()}, Max index: {selected_unit_indices.max()}")
    return selected_unit_indices

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:

    # === Wandb Initialization ===
    experiment_code = cfg.exp.get("experiment_code", Path(__file__).resolve().parent.name)
    config_name = HydraConfig.get().runtime.choices.exp
    wandb_run_name = f"{experiment_code}/{config_name}"

    wandb.init(
        project=cfg.exp.task,
        name=wandb_run_name,
        config=OmegaConf.to_container(cfg.exp, resolve=True), 
    )

    # === Results Directory Setup ===
    results_dir_base = cfg.exp.get("results_dir_base", "output") 
    results_dir = Path(results_dir_base) / experiment_code / config_name
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    # === SAE Unit Selection ===
    selected_unit_indices = None
    if cfg.exp.features.get("use_sae_selected_units", False):
        selected_unit_indices = get_selected_sae_units(cfg.exp.features)
        if selected_unit_indices is not None:
            original_in_dim = cfg.exp.model.in_dim
            # Modify Hydra config for model input dimension
            with open_dict(cfg):
                cfg.exp.model.in_dim = len(selected_unit_indices)
            print(f"Updated model input dimension from {original_in_dim} to {cfg.exp.model.in_dim} due to SAE unit selection.")
        else:
            print("SAE unit selection was enabled, but no units were selected or an error occurred. Using original in_dim.")
            with open_dict(cfg): # Ensure consistency
                cfg.exp.features.use_sae_selected_units = False

    # === K-Fold Cross Validation Setup ===
    if cfg.exp.k_start == -1:
        start = 0
    else:
        start = cfg.exp.k_start
    if cfg.exp.k_end == -1:
        end = cfg.exp.k
    else:
        end = cfg.exp.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)

    # === Dataset Initialization ===
    print('\nLoad Dataset')
    if cfg.exp.task == 'camelyon16':
        dataset_full = Generic_MIL_Dataset(
            csv_path=cfg.exp.dataset.csv_path, 
            data_dir=cfg.exp.dataset.data_dir, 
            shuffle=False,
            seed=cfg.exp.seed,
            print_info=True,
            label_dict={'normal': 0, 'tumor': 1},
            patient_strat=False,
            ignore=[],
            selected_unit_indices=selected_unit_indices
        )
    else:
        raise NotImplementedError(f"Task {cfg.exp.task} not implemented.")
    
    # === Split Directory ===
    split_dir_path = Path(cfg.exp.split_dir)
    print('split_dir: ', split_dir_path)
    assert os.path.isdir(split_dir_path), f"Split directory {split_dir_path} not found."

    # === Log settings ===
    print("################# Hydra Config ###################")
    print(OmegaConf.to_yaml(cfg))

    # === Main Loop for Folds ===
    for i in folds:
        print(f"\n===== Processing Fold: {i} =====")
        seed_torch(cfg.exp.seed)

        train_dataset, val_dataset, test_dataset = dataset_full.return_splits(
            backbone=cfg.exp.features.backbone,
            patch_size=cfg.exp.features.patch_size,
            from_id=False,
            csv_path=str(split_dir_path / f'splits_{i}.csv'),
            selected_unit_indices=selected_unit_indices
        )
        
        datasets = (train_dataset, val_dataset, test_dataset)
        if cfg.exp.preloading == 'yes':
            for d_idx, d in enumerate(datasets):
                if d is not None:
                    print(f"Preloading for dataset split {d_idx}...")
                    d.pre_loading()
                else:
                    print(f"Dataset split {d_idx} is None, skipping preloading.")
            
        results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, cfg.exp, results_dir)

        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)

        #write results to pkl
        filename = results_dir / f'split_{i}_results.pkl'
        save_pkl(filename, results)
        print(f"Fold {i} results saved to {filename}")

    # === Final Results Aggregation ===
    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(folds) != cfg.exp.k:
        save_name = f'summary_partial_{start}_{end}.csv'
    else:
        save_name = 'summary.csv'
    final_df.to_csv(results_dir / save_name)
    print(f"Summary saved to {results_dir / save_name}")

    mean_auc_test = final_df['test_auc'].mean()
    std_auc_test = final_df['test_auc'].std()
    mean_auc_val = final_df['val_auc'].mean()
    std_auc_val = final_df['val_auc'].std()
    mean_acc_test = final_df['test_acc'].mean()
    std_acc_test = final_df['test_acc'].std()
    mean_acc_val = final_df['val_acc'].mean()
    std_acc_val = final_df['val_acc'].std()

    wandb.log({
        "mean_auc_test": mean_auc_test, "std_auc_test": std_auc_test,
        "mean_auc_val": mean_auc_val, "std_auc_val": std_auc_val,
        "mean_acc_test": mean_acc_test, "std_acc_test": std_acc_test,
        "mean_acc_val": mean_acc_val, "std_acc_val": std_acc_val,
    })

    df_append = pd.DataFrame({
        'folds': ['mean', 'std'],
        'test_auc': [mean_auc_test, std_auc_test],
        'val_auc': [mean_auc_val, std_auc_val],
        'test_acc': [mean_acc_test, std_acc_test],
        'val_acc': [mean_acc_val, std_acc_val],
    })
    final_df = pd.concat([final_df, df_append])

    if len(folds) != cfg.exp.k:
        save_name = f'summary_partial_{start}_{end}.csv'
    else:
        save_name = 'summary.csv'
    final_df.to_csv(results_dir / save_name)
    print(f"Final summary saved to {results_dir / save_name}")

    final_df['folds'] = final_df['folds'].astype(str)
    table = wandb.Table(dataframe=final_df)
    wandb.log({"summary_table": table}) 

    wandb.finish()

if __name__ == "__main__":
    main()