import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig
from pathlib import Path

from activation_store import FeatureActivationStore
from training import train_sae
from utils import seed_everything, SAE_MODEL_MAP


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    try:
        hydra_cfg = HydraConfig.get()
        exp_run_name = hydra_cfg.runtime.choices.exp or cfg.exp.get("name", "unknown_run")
        script_path = Path(__file__).resolve()
        exp_group_dir = script_path.parent
        exp_group_name = exp_group_dir.name
        project_root = exp_group_dir.parent.parent
        output_base_dir = project_root / "output"
        output_dir = output_base_dir / exp_group_name / exp_run_name
    except Exception:
        exp_run_name = cfg.exp.get("name", "fallback_run")
        output_dir = Path("./outputs") / exp_run_name

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")
    with open_dict(cfg):
        cfg.output_dir = str(output_dir)
        cfg.exp.exp_group_name = exp_group_name
        cfg.exp.exp_run_name = exp_run_name

    print(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.exp.seed)

    # Initialize train activation store
    try:
        train_store = FeatureActivationStore(cfg, split_name="train")
        total_train = train_store.get_total_patches()
        feature_dim = train_store.get_feature_dim()
        print(f"Train patches: {total_train}, feature dim: {feature_dim}")
    except Exception as e:
        print(f"Error initializing train store: {e}")
        return

    # Initialize validation activation store if available
    val_store = None
    if cfg.exp.data.get("split_csv_path"):
        try:
            tmp_store = FeatureActivationStore(cfg, split_name="val")
            total_val = tmp_store.get_total_patches()
            if total_val > 0:
                val_store = tmp_store
                print(f"Val patches: {total_val}")
            else:
                print("No validation patches, skipping.")
        except Exception as e:
            print(f"Val store skipped: {e}")

    # Compute num_tokens
    if "num_epochs" in cfg.exp.training:
        num_epochs = cfg.exp.training.num_epochs
        num_tokens = total_train * num_epochs
        with open_dict(cfg):
            cfg.exp.training.num_tokens = num_tokens
        print(f"num_tokens: {num_tokens}")
    elif "num_tokens" not in cfg.exp.training:
        raise ValueError("Specify training.num_epochs or training.num_tokens")
    else:
        print(f"Using num_tokens: {cfg.exp.training.num_tokens}")

    # Ensure act_size matches detected feature dimension
    if train_store:
        detected_dim = train_store.get_feature_dim()
        if cfg.exp.sae.get("act_size") is None:
            with open_dict(cfg):
                cfg.exp.sae.act_size = detected_dim
        elif cfg.exp.sae.act_size != detected_dim:
            raise ValueError("Configured act_size does not match detected dimension")
        print(f"act_size: {cfg.exp.sae.act_size}")

    # Determine dict_size
    if "dict_size" not in cfg.exp.sae and "dict_size_ratio" in cfg.exp.sae:
        with open_dict(cfg):
            cfg.exp.sae.dict_size = int(cfg.exp.sae.act_size * cfg.exp.sae.dict_size_ratio)
        print(f"dict_size: {cfg.exp.sae.dict_size}")
    elif "dict_size" not in cfg.exp.sae:
        raise ValueError("Specify sae.dict_size or sae.dict_size_ratio")
    else:
        print(f"dict_size: {cfg.exp.sae.dict_size}")

    # Map dtype string to torch dtype
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    try:
        torch_dtype = dtype_map[cfg.exp.sae.dtype]
    except KeyError:
        raise ValueError(f"Unsupported dtype: {cfg.exp.sae.dtype}")
    print(f"dtype: {torch_dtype}")

    # Initialize SAE model
    sae_config = OmegaConf.to_container(cfg.exp.sae, resolve=True)
    sae_config["dtype"] = torch_dtype
    sae_class = SAE_MODEL_MAP.get(cfg.exp.sae.model_type)
    if sae_class is None:
        raise ValueError(f"Unknown SAE model_type: {cfg.exp.sae.model_type}")
    sae = sae_class(sae_config).to(cfg.exp.sae.device)
    num_params = sum(p.numel() for p in sae.parameters() if p.requires_grad)
    print(f"Initialized {cfg.exp.sae.model_type} with {num_params:,} trainable parameters")

    # Start training
    train_sae(sae=sae, train_activation_store=train_store, val_activation_store=val_store, cfg=cfg)
    print("Training complete")

if __name__ == "__main__":
    main()