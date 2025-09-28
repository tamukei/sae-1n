import pandas as pd
import numpy as np

def generate_descriptor(train_df, val_df, label_dict):
    """
    Create a descriptor DataFrame showing class counts for train and val sets.

    Args:
        train_df (pd.DataFrame): Training slides ('slide_id', 'label').
        val_df (pd.DataFrame): Validation slides ('slide_id', 'label').
        label_dict (dict): Map from string labels to int indices (e.g., {'normal': 0, 'tumor': 1}).

    Returns:
        pd.DataFrame: Descriptor with class labels as index, columns ['train', 'val'].
    """
    # Invert dict: index -> label
    index_to_label = {v: k for k, v in label_dict.items()}
    class_labels = [index_to_label[i] for i in sorted(index_to_label.keys())]

    # Initialize descriptor
    descriptor = pd.DataFrame(0, index=class_labels, columns=['train', 'val'], dtype=np.int32)

    # Train counts
    if not train_df.empty:
        train_counts = train_df['label'].value_counts()
        for label_str, count in train_counts.items():
            if label_str in descriptor.index:
                descriptor.loc[label_str, 'train'] = count
            else:
                print(f"Warning: Label '{label_str}' in train_df not in label_dict.")

    # Val counts
    if not val_df.empty:
        val_counts = val_df['label'].value_counts()
        for label_str, count in val_counts.items():
            if label_str in descriptor.index:
                descriptor.loc[label_str, 'val'] = count
            else:
                print(f"Warning: Label '{label_str}' in val_df not in label_dict.")

    print("\nSplit Descriptor:")
    print(descriptor)
    return descriptor
