import torch
import tqdm
from logs import init_wandb, log_wandb, save_checkpoint
from omegaconf import DictConfig
from typing import Optional, Dict


def run_validation(
    sae,
    val_activation_store,
    current_step: int,
    wandb_run,
    cfg_exp: DictConfig
) -> Optional[Dict[str, float]]:
    sae.eval()
    if not val_activation_store:
        print(f"Step {current_step}: No validation. Skipping.")
        sae.train()
        return None

    print(f"Running validation at step {current_step} over validation store...")

    total_val_patches = val_activation_store.get_total_patches()
    batch_size = cfg_exp.training.batch_size

    num_val_steps = (total_val_patches + batch_size - 1) // batch_size

    val_pbar = tqdm.tqdm(total=num_val_steps, desc=f"Validating SAE (Step {current_step})", leave=False)

    metrics = {k: [] for k in ["loss", "l2_loss", "l1_loss", "l0_norm", "l1_norm", "aux_loss", "num_dead_features"]}

    batches_processed = 0
    with torch.no_grad():
        while batches_processed < num_val_steps:
            try:
                batch = val_activation_store.next_batch()
                if batch is None or batch.numel() == 0 or batch.size(0) == 0:
                    print("Warning: val_activation_store.next_batch() returned empty or None batch during validation.")
                    if batch is None:
                        break
                    continue 
            except RuntimeError as e:
                print(f"Error getting validation batch: {e}. Ending validation early.")
                break

            val_pbar.update(1)
            out = sae(batch)
            for k in metrics:
                v = out.get(k)
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    metrics[k].append(v.item())
                elif isinstance(v, (int, float)):
                    metrics[k].append(v)
            batches_processed += 1
            if batches_processed >= num_val_steps:
                break
    val_pbar.close()

    avg_metrics = {k: sum(v_list) / len(v_list) for k, v_list in metrics.items() if v_list}
    if avg_metrics:
        payload = {f"val/{k}": v for k, v in avg_metrics.items()}
        log_wandb(payload, current_step, wandb_run, cfg_exp)
        print(f"Step {current_step} Val Loss={avg_metrics.get('loss', 0):.4f}")
    else:
        print(f"Step {current_step}: No valid val metrics.")

    sae.train()
    return avg_metrics


def train_sae(
    sae,
    train_activation_store,
    val_activation_store: Optional[object],
    cfg: DictConfig
):
    num_tokens = cfg.exp.training.num_tokens
    batch_size = cfg.exp.training.batch_size
    num_batches = num_tokens // batch_size
    print(f"Training {num_tokens} tokens in {num_batches} batches (bs={batch_size})")

    optimizer = torch.optim.Adam(
        sae.parameters(),
        lr=cfg.exp.training.lr,
        betas=(cfg.exp.training.beta1, cfg.exp.training.beta2)
    )
    pbar = tqdm.trange(num_batches, desc="Training", leave=True)

    wandb_run = init_wandb(cfg)
    ckpt_freq = cfg.exp.training.get("checkpoint_freq", 10000)
    val_freq = cfg.exp.training.get("validation_freq", 500)
    max_grad_norm = cfg.exp.training.max_grad_norm
    output_dir = cfg.output_dir
    device = sae.config["device"]

    best_val_loss = float('inf')
    best_step = -1
    total_tokens = 0

    for i in pbar:
        sae.train()
        try:
            batch = train_activation_store.next_batch()
            if batch is None or batch.numel() == 0 or batch.size(0) == 0:
                continue
            batch = batch.to(device)
        except Exception:
            break

        out = sae(batch)
        train_payload = {f"train/{k}": v for k, v in out.items() if k in ["loss", "l2_loss", "l1_loss", "l0_norm", "l1_norm", "aux_loss", "num_dead_features"]}
        if "feature_acts" in out and isinstance(out["feature_acts"], torch.Tensor):
            dead = (out["feature_acts"].sum(0) == 0).sum().item()
            train_payload["train/n_dead_in_batch"] = dead
        log_wandb(train_payload, i, wandb_run, cfg.exp)

        loss = out["loss"]
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "L0": f"{out.get('l0_norm', torch.tensor(0.0)).item():.4f}"})

        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), max_grad_norm)
        if hasattr(sae, 'make_decoder_weights_and_grad_unit_norm'):
            sae.make_decoder_weights_and_grad_unit_norm()
        optimizer.step()
        optimizer.zero_grad()

        total_tokens += batch.size(0)

        if val_activation_store and i > 0 and (i % val_freq == 0 or i == num_batches - 1):
            val_metrics = run_validation(sae, val_activation_store, i, wandb_run, cfg.exp)
            if val_metrics and val_metrics.get('loss', float('inf')) < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_step = i
                print(f"New best Val Loss={best_val_loss:.4f} at step {best_step}. Saving best model.")
                save_checkpoint(wandb_run, sae, cfg.exp, step="best", output_dir=output_dir)
                if wandb_run:
                    wandb_run.summary["best_val_loss"] = best_val_loss
                    wandb_run.summary["best_val_loss_step"] = best_step

        if i > 0 and i % ckpt_freq == 0:
            save_checkpoint(wandb_run, sae, cfg.exp, i, output_dir=output_dir)

    print(f"Finished training {i+1} steps, processed {total_tokens} tokens.")

    if val_activation_store and (i % val_freq != 0 and i < num_batches - 1):
        print("Final validation...")
        val_metrics = run_validation(sae, val_activation_store, i, wandb_run, cfg.exp)
        if val_metrics and val_metrics.get('loss', float('inf')) < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_step = i
            print(f"Best Val Loss {best_val_loss:.4f} at final step {best_step}. Saving.")
            save_checkpoint(wandb_run, sae, cfg.exp, step="best", output_dir=output_dir)
            if wandb_run:
                wandb_run.summary["best_val_loss"] = best_val_loss
                wandb_run.summary["best_val_loss_step"] = best_step

    print("Saving final checkpoint.")
    save_checkpoint(wandb_run, sae, cfg.exp, i, output_dir=output_dir)

    if best_step != -1:
        print(f"Best Val Loss was {best_val_loss:.4f} at step {best_step}.")
    else:
        print("No validation was performed.")

    if wandb_run:
        wandb_run.finish()
    print("Training complete.")
