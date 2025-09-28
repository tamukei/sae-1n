import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import types
import sys

from utils.splits import save_splits
from utils.descriptor import generate_descriptor

def create_splits_camelyon16(metadata_path, output_base_dir, k=5, seed=42, task="camelyon16"):
    """
    Camelyon16: patient-level k-fold splits via StratifiedGroupKFold.
    Outputs per fold: split CSVs (column & boolean) and a train/val descriptor.

    Expected columns:
        slide_id (str), case_id (str), label (...), split in {'train','test'}
    """
    # Validate input
    if k < 2:
        print(f"Error: Number of folds (k) must be >= 2 for StratifiedGroupKFold. Received k={k}.")
        sys.exit(1)

    # Load metadata
    try:
        df = pd.read_csv(metadata_path, dtype={'slide_id': str, 'case_id': str})
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading metadata file {metadata_path}: {e}")
        sys.exit(1)

    # Prepare output
    split_dir = os.path.join(output_base_dir, f"{task}_{seed}")
    os.makedirs(split_dir, exist_ok=True)

    # Fixed test / split trainval
    test_df = df[df['split'] == 'test'].copy()
    trainval_df = df[df['split'] == 'train'].copy().reset_index(drop=True)

    print("-" * 30)
    print(f"Total slides: {len(df)}")
    print(f"Test slides (fixed): {len(test_df)}")
    print(f"Train/Val slides (to be split): {len(trainval_df)}")
    print("-" * 30)

    if trainval_df.empty:
        print("Warning: No slides available for training/validation split.")
        return

    # Patient-level stratification
    patient_data = trainval_df.drop_duplicates(subset=['case_id']).reset_index(drop=True)
    patient_ids_for_split = patient_data['case_id'].values
    patient_labels_for_split = patient_data['label'].values
    X_for_split = np.arange(len(patient_ids_for_split))

    # Label dict for descriptor
    unique_labels = sorted(trainval_df['label'].unique())
    if not unique_labels:
        print("Error: No labels found in the train/val data.")
        sys.exit(1)
    label_dict = {label: i for i, label in enumerate(unique_labels)}
    print(f"Using Label Dictionary: {label_dict}")

    print(f"Using StratifiedGroupKFold with k={k}")
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)

    if len(X_for_split) < k:
        print(f"Error: Not enough unique patients ({len(X_for_split)}) to create {k} folds.")
        sys.exit(1)
    if len(np.unique(patient_labels_for_split)) < 2:
        print("Warning: Only one class present in train/val data. Stratification will have no effect.")

    # Generate folds
    try:
        fold_indices_gen = sgkf.split(X_for_split, patient_labels_for_split, groups=patient_ids_for_split)
    except ValueError as e:
        print(f"Error during StratifiedGroupKFold split: {e}")
        print("This might happen if a group spans multiple classes, a class has < k members, or other constraints are violated.")
        print("Consider checking data integrity.")
        sys.exit(1)

    # Save per-fold outputs
    for i, (train_patient_idx, val_patient_idx) in enumerate(fold_indices_gen):
        train_patients = patient_ids_for_split[train_patient_idx]
        val_patients = patient_ids_for_split[val_patient_idx]

        train_df_fold = trainval_df[trainval_df['case_id'].isin(train_patients)].copy()
        val_df_fold = trainval_df[trainval_df['case_id'].isin(val_patients)].copy()
        test_df_fold = test_df.copy()

        train_slide_ids = train_df_fold['slide_id'].tolist()
        val_slide_ids = val_df_fold['slide_id'].tolist()
        test_slide_ids = test_df_fold['slide_id'].tolist()

        print(f"\n--- Fold {i} ---")
        print(f"  Train patients: {len(train_patients)}, Train slides: {len(train_slide_ids)}")
        print(f"  Val patients: {len(val_patients)}, Val slides: {len(val_slide_ids)}")
        print(f"  Test slides: {len(test_slide_ids)}")

        # Descriptor
        descriptor_df = generate_descriptor(train_df_fold, val_df_fold, label_dict)
        descriptor_filename = os.path.join(split_dir, f'splits_{i}_descriptor.csv')
        descriptor_df.to_csv(descriptor_filename, index=True)
        print(f"Saved descriptor to: {descriptor_filename}")

        # Split CSVs
        train_split_obj = types.SimpleNamespace(slide_data=pd.DataFrame({'slide_id': train_slide_ids}))
        val_split_obj   = types.SimpleNamespace(slide_data=pd.DataFrame({'slide_id': val_slide_ids}))
        test_split_obj  = types.SimpleNamespace(slide_data=pd.DataFrame({'slide_id': test_slide_ids}))
        split_datasets = [train_split_obj, val_split_obj, test_split_obj]
        column_keys = ['train', 'val', 'test']

        filename_col = os.path.join(split_dir, f'splits_{i}.csv')
        filename_bool = os.path.join(split_dir, f'splits_{i}_bool.csv')
        save_splits(split_datasets, column_keys, filename_col,  boolean_style=False)
        save_splits(split_datasets, column_keys, filename_bool, boolean_style=True)

    print("-" * 30)
    print(f"Split files and descriptors saved in: {split_dir}")
    print("-" * 30)

def create_splits_panda(metadata_path, output_base_dir, repeats=5, seed=42, task="panda"):
    """
    PANDA: repeated 80/10/10 splits with SGKF over (label, data_provider) at patient level.
    Outputs per repeat: split CSVs (column & boolean) and a train/val descriptor.

    Expected columns:
        slide_id (str), case_id (str), label in {...}, data_provider (str)
    """
    # Load metadata
    try:
        df = pd.read_csv(metadata_path, dtype={'slide_id': str, 'case_id': str})
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading metadata file {metadata_path}: {e}")
        sys.exit(1)

    # Check schema
    required_cols = {"slide_id", "case_id", "label", "data_provider"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Error: PANDA metadata must contain columns {required_cols}, but missing {missing}")
        sys.exit(1)

    # Output dir
    split_dir = os.path.join(output_base_dir, f"{task}_{seed}")
    os.makedirs(split_dir, exist_ok=True)

    # One row per case; detect conflicts
    agg = (df.groupby("case_id")
             .agg(label=("label", "nunique"),
                  provider=("data_provider", "nunique"))
             .reset_index())
    conflicts = agg[(agg["label"] > 1) | (agg["provider"] > 1)]
    if not conflicts.empty:
        print("Error: Found case_ids with multiple labels and/or data_providers. "
              "Please ensure each case_id maps to a single (label, data_provider).")
        print(conflicts.head())
        sys.exit(1)

    patient_data = (df.groupby("case_id")
                      .agg(label=("label", "first"),
                           data_provider=("data_provider", "first"))
                      .reset_index())

    # Strata per patient
    patient_data["strata"] = patient_data["label"].astype(str) + "||" + patient_data["data_provider"].astype(str)

    # Arrays for SGKF
    groups = patient_data["case_id"].values
    y_strata = patient_data["strata"].values
    X_dummy = np.arange(len(groups))

    # Label dict for descriptor
    unique_labels = sorted(df["label"].unique().tolist())
    label_dict = {label: i for i, label in enumerate(unique_labels)}
    print(f"Using Label Dictionary: {label_dict}")

    # Start with 10 folds (approx 8/1/1)
    n_splits = 10
    stratum_counts = pd.Series(y_strata).value_counts()
    min_stratum = int(stratum_counts.min())
    if min_stratum < n_splits:
        print(f"Warning: Some (label, data_provider) strata have only {min_stratum} cases "
              f"(< {n_splits}). Reducing n_splits to {min_stratum} to avoid SGKF errors.")
        n_splits = max(2, min_stratum)

    for r in range(repeats):
        rs = seed + r
        print("-" * 30)
        print(f"PANDA repeat {r+1}/{repeats} with random_state={rs} and n_splits={n_splits}")
        print("-" * 30)

        # Assign a fold id per patient
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rs)
        fold_id = np.empty(len(groups), dtype=int)
        fold_id.fill(-1)
        for fold_idx, (_, test_index) in enumerate(sgkf.split(X_dummy, y_strata, groups=groups)):
            fold_id[test_index] = fold_idx
        if (fold_id < 0).any():
            print("Error: Fold assignment failed.")
            sys.exit(1)

        # Map folds to ~80/10/10
        n_train_folds = int(np.ceil(0.8 * n_splits))
        remaining = n_splits - n_train_folds
        if remaining < 2:
            n_val_folds = 1 if n_splits >= 2 else 0
            n_test_folds = 1 if n_splits >= 3 else 0
        else:
            n_val_folds = 1
            n_test_folds = 1

        all_folds = np.arange(n_splits)
        rng = np.random.RandomState(rs)
        rng.shuffle(all_folds)

        train_folds = all_folds[:n_train_folds]
        leftover = all_folds[n_train_folds:]
        if len(leftover) >= 2:
            val_fold = leftover[0:1]
            test_fold = leftover[1:2]
        elif len(leftover) == 1:
            val_fold = leftover[0:1]
            test_fold = np.array([], dtype=int)
            print("Warning: Not enough folds to allocate both val and test; test will be empty in this repeat.")
        else:
            val_fold = np.array([], dtype=int)
            test_fold = np.array([], dtype=int)
            print("Warning: Not enough folds to allocate val/test; all folds go to train in this repeat.")

        # Slide-level splits
        pid2fold = dict(zip(groups, fold_id))
        df["fold_id"] = df["case_id"].map(pid2fold)

        train_df_fold = df[df["fold_id"].isin(train_folds)].copy()
        val_df_fold   = df[df["fold_id"].isin(val_fold)].copy()
        test_df_fold  = df[df["fold_id"].isin(test_fold)].copy()

        print(f"Train slides: {len(train_df_fold)} | Val slides: {len(val_df_fold)} | Test slides: {len(test_df_fold)}")

        # Descriptor (train/val)
        descriptor_df = generate_descriptor(train_df_fold, val_df_fold, label_dict)
        descriptor_filename = os.path.join(split_dir, f"splits_{r}_descriptor.csv")
        descriptor_df.to_csv(descriptor_filename, index=True)
        print(f"Saved descriptor to: {descriptor_filename}")

        # Split CSVs
        train_split_obj = types.SimpleNamespace(slide_data=pd.DataFrame({"slide_id": train_df_fold["slide_id"].tolist()}))
        val_split_obj   = types.SimpleNamespace(slide_data=pd.DataFrame({"slide_id": val_df_fold["slide_id"].tolist()}))
        test_split_obj  = types.SimpleNamespace(slide_data=pd.DataFrame({"slide_id": test_df_fold["slide_id"].tolist()}))
        split_datasets  = [train_split_obj, val_split_obj, test_split_obj]
        column_keys     = ["train", "val", "test"]

        filename_col  = os.path.join(split_dir, f"splits_{r}.csv")
        filename_bool = os.path.join(split_dir, f'splits_{r}_bool.csv')
        save_splits(split_datasets, column_keys, filename_col,  boolean_style=False)
        save_splits(split_datasets, column_keys, filename_bool, boolean_style=True)

    print("-" * 30)
    print(f"PANDA split files and descriptors saved in: {split_dir}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create patient-level stratified train/val/test splits'
    )
    parser.add_argument('--metadata_path', type=str,
                        help='Path to the metadata CSV file.')
    parser.add_argument('--output_dir', type=str,
                        help="Base directory to save the split CSV files.")
    parser.add_argument('--k', type=int, default=5,
                        help='Number of folds for cross-validation (>= 2). Used only for camelyon16.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (also used in output subdirectory naming).')
    parser.add_argument('--task', type=str, default='camelyon16', choices=['camelyon16', 'panda'],
                        help='Task name.')
    parser.add_argument('--repeats', type=int, default=5,
                        help='Number of random repeats for PANDA 80/10/10 splits.')

    args = parser.parse_args()

    if args.task.lower() == 'panda':
        create_splits_panda(args.metadata_path, args.output_dir, repeats=args.repeats, seed=args.seed, task='panda')
    else:
        if args.k < 2:
            print(f"Error: Number of folds (k) must be >= 2. Received k={args.k}.")
            sys.exit(1)
        create_splits_camelyon16(args.metadata_path, args.output_dir, args.k, args.seed, args.task)