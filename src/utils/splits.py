import os
import pandas as pd
import numpy as np

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
    """
    Save split CSVs (column or boolean style).

    Args:
        split_datasets (list): List of objects, each with `slide_data['slide_id']` (pd.Series).
        column_keys (list): Names for columns (e.g., ['train', 'val', 'test']).
        filename (str): Output CSV path.
        boolean_style (bool): If True, save in boolean style.
    """
    # Ensure output dir exists
    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Extract slide_id Series
    try:
        splits = [d.slide_data['slide_id'] for d in split_datasets]
    except (AttributeError, KeyError, TypeError) as e:
        print("Error: split_datasets must contain objects with slide_data['slide_id'] as pd.Series.")
        raise ValueError("Invalid input structure for save_splits") from e

    if not boolean_style:
        # Column style
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
        df.to_csv(filename, index=False)
        print(f"Saved column-style splits to: {filename}")
    else:
        # Boolean style
        df = pd.concat(splits, ignore_index=True, axis=0)
        index = df.values.tolist()
        unique_index = sorted(df.unique().tolist())

        one_hot = np.eye(len(split_datasets)).astype(bool)
        counts = [len(dset.slide_data['slide_id']) for dset in split_datasets]
        bool_array = np.repeat(one_hot, counts, axis=0)

        df_bool = pd.DataFrame(bool_array, index=pd.Index(index).unique(), columns=column_keys)
        df_bool = df_bool.reindex(unique_index)

        df_bool.index.name = 'slide_id'
        df_bool.to_csv(filename, index=True)
        print(f"Saved boolean-style splits to: {filename}")
