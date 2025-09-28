import os
import pandas as pd
import re
import argparse
from tqdm import tqdm

def create_metadata_camelyon16(feature_dir, annotation_dir, output_dir, output_filename):
    os.makedirs(output_dir, exist_ok=True)

    slide_ids = []
    case_ids = []
    labels = []
    splits = []

    print(f"Scanning directory: {feature_dir}")
    try:
        h5_files = sorted([f for f in os.listdir(feature_dir) if f.endswith('.h5')])
    except FileNotFoundError:
        print(f"Error: Directory not found - {feature_dir}")
        return

    if not h5_files:
        print(f"Error: No .h5 files found in {feature_dir}")
        return

    print(f"Found {len(h5_files)} .h5 files. Processing...")
    for filename in tqdm(h5_files, desc="Creating Metadata (camelyon16)"):
        slide_id = filename[:-3]
        label = "unknown"
        case_id = "unknown"
        split = "unknown"

        if slide_id.startswith("test_"):
            split = "test"
            xml_path = os.path.join(annotation_dir, f"{slide_id}.xml")
            label = "tumor" if os.path.exists(xml_path) else "normal"
            match = re.search(r"test_(\d+)", slide_id)
            case_id = f"test_{match.group(1)}" if match else slide_id

        elif slide_id.startswith("normal_"):
            label = "normal"
            split = "train"
            match = re.search(r"normal_(\d+)", slide_id)
            case_id = f"normal_{match.group(1)}" if match else slide_id

        elif slide_id.startswith("tumor_"):
            label = "tumor"
            split = "train"
            match = re.search(r"tumor_(\d+)", slide_id)
            case_id = f"tumor_{match.group(1)}" if match else slide_id

        else:
            print(f"Warning: Unknown file prefix for {slide_id}. Skipping.")
            continue

        slide_ids.append(slide_id)
        case_ids.append(case_id)
        labels.append(label)
        splits.append(split)

    if not slide_ids:
        print("Error: No valid slides found to create the CSV.")
        return

    df = pd.DataFrame({
        'case_id': case_ids,
        'slide_id': slide_ids,
        'label': labels,
        'split': splits
    })

    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path, index=False)

    print("-" * 30)
    print(f"Successfully created metadata CSV: {output_path}")
    print(f"Total slides: {len(df)}")
    print("Split distribution:")
    print(df['split'].value_counts())
    print("-" * 30)
    print("Label distribution (excluding test set):")
    print(df[df['split'] == 'train']['label'].value_counts())
    print("-" * 30)

def create_metadata_panda(feature_dir, panda_csv, output_dir, output_filename):
    os.makedirs(output_dir, exist_ok=True)

    try:
        h5_files = sorted([f for f in os.listdir(feature_dir) if f.endswith('.h5')])
    except FileNotFoundError:
        print(f"Error: Directory not found - {feature_dir}")
        return

    if not h5_files:
        print(f"Error: No .h5 files found in {feature_dir}")
        return

    if not os.path.exists(panda_csv):
        print(f"Error: PANDA CSV not found - {panda_csv}")
        return

    try:
        meta = pd.read_csv(panda_csv, usecols=['image_id', 'isup_grade', 'data_provider'])
    except ValueError as e:
        try:
            meta = pd.read_csv(panda_csv, usecols=['image_id', 'isup_grade'])
            meta['data_provider'] = 'unknown'
            print("Warning: 'data_provider' column not found in PANDA CSV. Filled with 'unknown'.")
        except Exception as ee:
            print(f"Error: Failed to read PANDA CSV ({panda_csv}): {ee}")
            return
    except Exception as e:
        print(f"Error: Failed to read PANDA CSV ({panda_csv}): {e}")
        return

    # isup_grade==0 -> normal, else tumor
    meta['label'] = meta['isup_grade'].apply(lambda x: 'normal' if int(x) == 0 else 'tumor')

    meta['image_id'] = meta['image_id'].astype(str)
    label_map = dict(zip(meta['image_id'], meta['label']))
    provider_map = dict(zip(meta['image_id'], meta['data_provider']))

    case_ids, slide_ids, labels, providers = [], [], [], []
    missing_in_csv = 0

    print(f"Found {len(h5_files)} .h5 files. Processing...")
    for filename in tqdm(h5_files, desc="Creating Metadata (PANDA)"):
        slide_id = filename[:-3]
        case_id = slide_id

        label = label_map.get(slide_id)
        if label is None:
            missing_in_csv += 1
            continue

        provider = provider_map.get(slide_id, 'unknown')

        case_ids.append(case_id)
        slide_ids.append(slide_id)
        labels.append(label)
        providers.append(provider)

    if not slide_ids:
        print("Error: No valid slides found after matching with PANDA CSV.")
        return

    df = pd.DataFrame({
        'case_id': case_ids,
        'slide_id': slide_ids,
        'label': labels,
        'data_provider': providers
    })

    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path, index=False)

    print("-" * 30)
    print(f"Successfully created PANDA metadata CSV: {output_path}")
    print(f"Total .h5 files: {len(h5_files)}")
    print(f"Used slides (matched in CSV): {len(df)}")
    if missing_in_csv:
        print(f"Skipped (not found in train.csv): {missing_in_csv}")
    print("Label distribution:")
    print(df['label'].value_counts())
    print("Data provider distribution:")
    print(df['data_provider'].value_counts())
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create metadata CSV for datasets.')
    parser.add_argument('--task', type=str, default='camelyon16', choices=['camelyon16', 'panda'],
                        help='Task to run: camelyon16 or panda')
    parser.add_argument('--feature_dir', type=str,
                        help='Directory containing the .h5 feature files.')
    parser.add_argument('--annotation_dir', type=str,
                        help='Directory containing the annotation .xml files.')
    parser.add_argument('--output_dir', type=str,
                        help='Directory to save the output metadata CSV file.')
    parser.add_argument('--output_filename', type=str, default='camelyon16_metadata.csv',
                        help='Name for the output metadata CSV file.')
    parser.add_argument('--panda_csv', type=str,
                        help='PANDA train.csv path (must contain image_id and isup_grade).')

    args = parser.parse_args()

    if args.task == 'camelyon16':
        create_metadata_camelyon16(args.feature_dir, args.annotation_dir, args.output_dir, args.output_filename)
    elif args.task == 'panda':
        create_metadata_panda(args.feature_dir, args.panda_csv, args.output_dir, args.output_filename)