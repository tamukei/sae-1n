import argparse
import torch
import h5py
import os
import glob
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import json
from tqdm import tqdm
import numpy as np

from utils import SAE_MODEL_MAP


def load_sae_model_from_checkpoint(exp_cfg: DictConfig, checkpoint_path: Path, device: str) -> torch.nn.Module:
    """Loads an SAE model from a checkpoint file and its configuration."""
    sae_config_dict = OmegaConf.to_container(exp_cfg.sae, resolve=True)
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    try:
        torch_dtype = dtype_map[exp_cfg.sae.dtype]
    except KeyError:
        raise ValueError(f"Unsupported dtype in loaded config: {exp_cfg.sae.dtype}")
    sae_config_dict["dtype"] = torch_dtype

    sae_class = SAE_MODEL_MAP.get(exp_cfg.sae.model_type)
    if sae_class is None:
        raise ValueError(f"Unknown SAE model_type in loaded config: {exp_cfg.sae.model_type}")
    if "act_size" not in sae_config_dict or sae_config_dict["act_size"] is None:
        raise ValueError("act_size not found or is None in the loaded SAE configuration.")

    model = sae_class(sae_config_dict)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.to(device)
    model.eval()
    print(f"Loaded SAE model '{exp_cfg.sae.model_type}' from {checkpoint_path} to {device}")
    return model

def main(args):
    exp_output_dir = Path(args.exp_output_dir).resolve()
    if not exp_output_dir.is_dir():
        print(f"Error: Experiment output directory not found: {exp_output_dir}")
        return

    checkpoint_dir = exp_output_dir / "checkpoints"
    sae_model_path = checkpoint_dir / "sae_best.pt"
    sae_config_path = checkpoint_dir / "config_best.json"

    if not sae_model_path.exists():
        print(f"Error: Best SAE model checkpoint not found: {sae_model_path}")
        return
    if not sae_config_path.exists():
        print(f"Error: Best SAE config not found: {sae_config_path}")
        return

    with open(sae_config_path, 'r') as f:
        loaded_exp_cfg_dict = json.load(f)
    cfg = OmegaConf.create({"exp": loaded_exp_cfg_dict})
    
    device = cfg.exp.sae.device
    sae_model = load_sae_model_from_checkpoint(cfg.exp, sae_model_path, device)
    
    original_feature_dir = Path(cfg.exp.data.feature_dir)
    if not original_feature_dir.is_dir():
        print(f"Error: Original feature directory from config not found: {original_feature_dir}")
        return
    
    feature_set_name = original_feature_dir.name
    try:
        project_name_dir = original_feature_dir.parent.parent 
        project_name = project_name_dir.name
        
        path_parts = original_feature_dir.parts
        if 'features' in path_parts:
            features_idx = path_parts.index('features')
            if features_idx + 1 < len(path_parts):
                project_name = path_parts[features_idx + 1]
                print(f"Derived project_name: {project_name}")
                print(f"Derived feature_set_name: {feature_set_name}")
            else:
                print(f"Warning: Could not derive project_name after 'features' in path. Using fallback.")
                project_name = "unknown_project" # Fallback
        else:
            print(f"Warning: 'features' directory not found in path. Using fallback for project_name.")
            project_name = "unknown_project" # Fallback

    except IndexError:
        print(f"Warning: Could not derive project_name from path structure {original_feature_dir}. Using fallback.")
        project_name = "unknown_project" # Fallback
        # feature_set_name is already original_feature_dir.name

    h5_dataset_name_original_features = cfg.exp.data.get("h5_dataset_name", "features")
    h5_dataset_name_original_coords = "coords"

    encoded_output_base = exp_output_dir / "encoded_features" / project_name / feature_set_name
    encoded_pt_dir = encoded_output_base / "pt_files"
    encoded_h5_dir = encoded_output_base / "h5_files"

    os.makedirs(encoded_pt_dir, exist_ok=True)
    os.makedirs(encoded_h5_dir, exist_ok=True)
    print(f"Saving encoded features to: {encoded_output_base.resolve()}")

    source_dir_h5 = original_feature_dir / "h5_files"
    source_files_h5 = list(source_dir_h5.glob("*.h5"))
    
    if not source_files_h5:
        print(f"No .h5 files found in {source_dir_h5}")
        return

    print(f"Found {len(source_files_h5)} H5 files to process.") 

    sae_input_dtype = next(sae_model.parameters()).dtype

    for f_path in tqdm(source_files_h5, desc="Encoding features"):
        slide_id = f_path.stem
        original_coords = None
        try:
            if f_path.suffix == ".h5":
                with h5py.File(f_path, 'r') as hf:
                    if h5_dataset_name_original_features not in hf:
                        print(f"Warning: Dataset '{h5_dataset_name_original_features}' not found in {f_path}. Skipping.")
                        continue
                    features = torch.from_numpy(hf[h5_dataset_name_original_features][:])
                    
                    if h5_dataset_name_original_coords in hf:
                        original_coords = hf[h5_dataset_name_original_coords][:]
                        if not isinstance(original_coords, np.ndarray):
                            print(f"Warning: Coords in {f_path} are not a numpy array. Skipping coords.")
                            original_coords = None
                        elif features.shape[0] != original_coords.shape[0]:
                            print(f"Warning: Mismatch between number of features ({features.shape[0]}) and coords ({original_coords.shape[0]}) in {f_path}. Skipping coords.")
                            original_coords = None
                    else:
                        print(f"Info: Dataset '{h5_dataset_name_original_coords}' not found in {f_path}. No coords will be saved for this H5.")
            else:
                print(f"Warning: Skipping unsupported file type: {f_path}")
                continue
            
            if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[0] == 0:
                print(f"Warning: Invalid data in {f_path} (expected 2D tensor with >0 rows). Skipping.")
                continue

            features = features.to(device=device, dtype=sae_input_dtype)

            with torch.no_grad():
                encoded_features = sae_model.encode(features)
            
            encoded_features_cpu = encoded_features.cpu()

            pt_save_path = encoded_pt_dir / f"{slide_id}.pt"
            torch.save(encoded_features_cpu, pt_save_path)

            h5_save_path = encoded_h5_dir / f"{slide_id}.h5"
            with h5py.File(h5_save_path, 'w') as hf:
                hf.create_dataset("features", data=encoded_features_cpu.numpy())
                if original_coords is not None:
                    hf.create_dataset("coords", data=original_coords)
        
        except Exception as e:
            print(f"Error processing {f_path}: {e}")
            continue
            
    print("Feature encoding and saving complete.")
    print(f"Encoded .pt files saved to: {encoded_pt_dir.resolve()}")
    print(f"Encoded .h5 files saved to: {encoded_h5_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encode features using a trained SAE model and save them in .pt and .h5 formats (with coords for H5 if available)."
    )
    parser.add_argument(
        "exp_output_dir",
        type=str,
        help="Path to the experiment output directory"
             " This directory should contain 'checkpoints/sae_best.pt' and 'checkpoints/config_best.json'."
    )
    args = parser.parse_args()
    main(args)