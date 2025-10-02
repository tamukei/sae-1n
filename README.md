# Patch-level phenotype identification via weakly supervised neuron selection in sparse autoencoders for CLIP-derived pathology embeddings

![Figure](figures/fig1_pipeline.png)

Computer-aided analysis of whole slide images (WSIs) has advanced rapidly with the emergence of multi-modal pathology foundation models. In this study, we propose a weakly supervised neuron selection approach to extract disentangled representations from CLIP-derived pathology foundation models, leveraging the interpretability of sparse autoencoders. Specifically, neurons are ordered and selected using whole-slide level labels within a multiple instance learning (MIL) framework. We investigate the impact of different pre-trained image embeddings derived from general and pathology images and demonstrate that a selected single neuron can effectively enable patch-level phenotype identification. Experiments on the Camelyon16 and PANDA datasets demonstrate both the effectiveness and explainability of the proposed method, as well as its generalization ability for tumor patch identification.

---

## Project Layout

Below is an example of the **final state** after running preprocessing, feature extraction, and training. `<PROJECT_ROOT>` is a placeholder for your workspace root directory (you may name it `WSI`, etc.).

```text
<PROJECT_ROOT>/
├── sae-1n/                          # this repo (code)
│   ├── experiments/
│   ├── requirements.txt
│   ├── dataset_csv/                 # (created by metadata scripts)
│   ├── splits/                      # (created by split scripts)
│   ├── output/                      # (created during training/analysis)
│   │   └── sae/
│   │       └── camelyon16_conch_0/
│   │           ├── analysis/
│   │           ├── checkpoints/
│   │           └── encoded_features/
│   │               └── camelyon16/
│   │                   └── features_conch_v1/
│   │                       ├── h5_files/
│   │                       └── pt_files/
│   └── ...
├── data/
│   ├── CAMELYON16
│   └── PANDA
├── features/
│   ├── camelyon16/
│   │   └── clam_processed/
│   │       └── features_conch_v1/
│   │           ├── h5_files/
│   │           └── pt_files/
│   └── panda/
│       └── clam_processed/
│           └── features_conch_v1/
│               ├── h5_files/
│               └── pt_files/
````

---

## Installation

Run the following from your `<PROJECT_ROOT>`:

```bash
# clone the repo under your project root
mkdir <PROJECT_ROOT> && cd <PROJECT_ROOT>
git clone https://github.com/tamukei/sae-1n.git
cd sae-1n

# create and activate environment
conda create -n sae-1n python=3.12 -y
conda activate sae-1n

# install dependencies
pip install -r requirements.txt
```

---

## Datasets

This project uses the following public pathology datasets. Please refer to the official websites for access, license terms, and download instructions:

* **[Camelyon16 Challenge Dataset](https://camelyon16.grand-challenge.org/)**
* **[PANDA Challenge Dataset (Prostate cANcer graDe Assessment)](https://panda.grand-challenge.org/)**

---

## Preprocessing with CLAM

We use [**CLAM**](https://github.com/mahmoodlab/CLAM) for WSI preprocessing, including tiling/segmentation and patch-level feature extraction. Install and configure CLAM following the official repository instructions.

### 1) Create patches (example: Camelyon16)

```bash
python create_patches_fp.py \
  --source <PROJECT_ROOT>/data/CAMELYON16/images \
  --save_dir <PROJECT_ROOT>/features/camelyon16/clam_processed \
  --patch_size 256 \
  --patch_level 1 \
  --preset <PROJECT_ROOT>/CLAM/presets/bwh_biopsy.csv \
  --seg \
  --patch \
  --stitch
```

### 2) Extract features (example: Camelyon16)

```bash
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py \
  --data_h5_dir <PROJECT_ROOT>/features/camelyon16/clam_processed \
  --data_slide_dir <PROJECT_ROOT>/data/CAMELYON16/images \
  --csv_path <PROJECT_ROOT>/features/camelyon16/clam_processed/process_list_autogen.csv \
  --feat_dir <PROJECT_ROOT>/features/camelyon16/clam_processed/features_conch_v1 \
  --batch_size 512 \
  --slide_ext .tif \
  --model_name "conch_v1" \
  --no_auto_skip
```

> Change `--model_name` to match the encoder you intend to use. For supported options and checkpoints, see the CLAM repository.

---

## Metadata

Run the following from `<PROJECT_ROOT>/sae-1n`.

**Camelyon16**

```bash
python create_metadata.py \
  --task camelyon16 \
  --feature_dir <PROJECT_ROOT>/features/camelyon16/clam_processed/features_conch_v1/h5_files \
  --annotation_dir <PROJECT_ROOT>/data/CAMELYON16/annotations \
  --output_dir <PROJECT_ROOT>/dataset_csv \
  --output_filename camelyon16_metadata.csv
```

**PANDA**

```bash
python create_metadata.py \
  --task panda \
  --feature_dir <PROJECT_ROOT>/features/panda/clam_processed/features_conch_v1/h5_files \
  --output_dir <PROJECT_ROOT>/dataset_csv \
  --output_filename panda_metadata.csv \
  --panda_csv <PROJECT_ROOT>/data/PANDA/train.csv
```

---

## Splits

**Camelyon16 (K-fold CV)**

```bash
python create_splits.py \
  --metadata_path <PROJECT_ROOT>/dataset_csv/camelyon16_metadata.csv \
  --output_dir <PROJECT_ROOT>/splits \
  --k 5 \
  --seed 42 \
  --task camelyon16
```

**PANDA**

```bash
python create_splits.py \
  --metadata_path <PROJECT_ROOT>/dataset_csv/panda_metadata.csv \
  --output_dir <PROJECT_ROOT>/splits \
  --seed 42 \
  --repeats 1 \
  --task panda
```

---

## Training & Analysis

Run all commands from: `<PROJECT_ROOT>/sae-1n`
Outputs are written under: `<PROJECT_ROOT>/sae-1n/output/`

### 1) Train the Sparse Autoencoder (SAE)

```bash
python experiments/sae/main.py exp=camelyon16_conch_0
```

### 2) Save SAE Encoded Features

```bash
python experiments/sae/save_sae_features.py <PROJECT_ROOT>/sae-1n/output/sae/camelyon16_conch_0
```

### 3) Analyze SAE Neurons

```bash
python experiments/sae/analyze_sae.py camelyon16_conch_0 \
  --split_csv_path <PROJECT_ROOT>/splits/camelyon16_42/splits_0.csv \
  --feature_dir <PROJECT_ROOT>/sae-1n/output/sae/camelyon16_conch_0/encoded_features/camelyon16/features_conch_v1/pt_files \
  --dataset_csv <PROJECT_ROOT>/dataset_csv/camelyon16_metadata.csv \
  --output_dir <PROJECT_ROOT>/sae-1n/output/sae \
  --num_features 2048
```

### 4) Train the MIL Classifier

```bash
python experiments/mil/main.py exp=camelyon16_conch_sae_maxmil
```

**Notes on `exp=` configs**

* The value passed to `exp=` is the **YAML filename (without `.yaml`)** of the experiment config.
* SAE configs live under: `experiments/sae/exp/`
* MIL configs live under: `experiments/mil/exp/`
* The names shown above (e.g., `camelyon16_conch_0`, `camelyon16_conch_sae_maxmil`) are **examples**. Use the YAMLs you intend to run.

---

## Citation

If you find this repository helpful, please cite our work.

---

## Reference Repositories

This repository is primarily inspired by the following projects:

* [BatchTopK](https://github.com/bartbussmann/BatchTopK)
* [CLAM](https://github.com/mahmoodlab/CLAM)
* [MambaMIL](https://github.com/isyangshu/MambaMIL)
* [CONCH](https://github.com/mahmoodlab/CONCH)
