import wandb
import torch
import os
import json
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Optional, Union


def init_wandb(cfg: DictConfig):
    if "exp" not in cfg:
        raise ValueError("Missing 'exp' in cfg.")
    exp_cfg = cfg.exp
    config_dict = OmegaConf.to_container(exp_cfg, resolve=True)

    group = exp_cfg.get("exp_group_name")
    run_name = exp_cfg.get("exp_run_name")
    if group and run_name:
        name = f"{group}_{run_name}"
    else:
        name = exp_cfg.get("name", "sae_run")

    print(f"W&B init: project='{exp_cfg.training.wandb_project}', entity='{exp_cfg.training.wandb_entity}', name='{name}'")
    try:
        run = wandb.init(
            project=exp_cfg.training.wandb_project,
            entity=exp_cfg.training.wandb_entity,
            name=name,
            config=config_dict,
            reinit=True,
        )
        print(f"W&B run URL: {run.url if run else 'None'}")
        return run
    except Exception as e:
        print(f"W&B init error: {e}")
        return None


def log_wandb(metrics: Dict, step: int, run, cfg_exp: DictConfig):
    if run is None:
        return
    payload = {}
    for k, v in metrics.items():
        if v is None:
            continue
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                payload[k] = v.item()
            else:
                print(f"Skipping non-scalar tensor metric '{k}'.")
        elif isinstance(v, (int, float)):
            payload[k] = v
        else:
            try:
                payload[k] = float(v)
            except:
                print(f"Skipping metric '{k}' of type {type(v)}.")

    if payload:
        try:
            run.log(payload, step=step)
        except Exception as e:
            print(f"W&B log error at step {step}: {e}")


def save_checkpoint(
    run,
    sae,
    cfg_exp: DictConfig,
    step: Union[int, str],
    index: Optional[int] = None,
    output_dir: Optional[str] = None,
):
    if output_dir is None:
        output_dir = "checkpoints"
        print("No output_dir provided, using './checkpoints'.")

    name_base = cfg_exp.get("name", "sae_model").replace("/", "_")
    suffix = step if isinstance(step, str) else f"{step}"
    if index is not None:
        suffix = f"idx{index}_{suffix}"

    sae_file = os.path.join(output_dir, "checkpoints", f"sae_{suffix}.pt")
    cfg_file = os.path.join(output_dir, "checkpoints", f"config_{suffix}.json")
    os.makedirs(os.path.dirname(sae_file), exist_ok=True)

    try:
        torch.save(sae.state_dict(), sae_file)
        print(f"SAE saved: {sae_file}")
    except Exception as e:
        print(f"Error saving SAE: {e}")
        return

    config_data = OmegaConf.to_container(cfg_exp, resolve=True)
    try:
        with open(cfg_file, "w") as f:
            json.dump(config_data, f, indent=4)
        print(f"Config saved: {cfg_file}")
    except Exception as e:
        print(f"Error saving config: {e}")

    if run:
        try:
            artifact_name = f"{name_base}_{suffix}"
            metadata = {
                "step": step if isinstance(step, int) else run.step,
                "type": "best_model" if step == "best" else "checkpoint",
                "index": index,
            }
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                description=(
                    f"{'Best model' if step == 'best' else f'Checkpoint {step}'}"
                    + (f" (Index {index})" if index is not None else "")
                ),
                metadata=metadata,
            )
            artifact.add_file(sae_file, name=os.path.basename(sae_file))
            artifact.add_file(cfg_file, name=os.path.basename(cfg_file))
            aliases = ["best"] if step == "best" else None
            run.log_artifact(artifact, aliases=aliases)
            print(f"W&B artifact '{artifact_name}' logged.")
        except Exception as e:
            print(f"Error logging W&B artifact: {e}")
    else:
        print(f"Checkpoint '{suffix}' saved locally.")
