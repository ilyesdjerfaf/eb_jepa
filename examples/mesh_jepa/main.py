"""
Mesh JEPA Training Script

Train a self-supervised temporal mesh prediction model on DFAUST using
Joint Embedding Predictive Architecture (JEPA) with VC regularization.

DiffusionNet encoder + GRU predictor. Supports HKS or XYZ input features.
"""

import json
import time
from pathlib import Path

import fire
import torch
from omegaconf import OmegaConf
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from eb_jepa.architectures import Projector
from eb_jepa.datasets.mesh import DFAUSTDataset
from eb_jepa.jepa import JEPA
from eb_jepa.logging import get_logger
from eb_jepa.losses import SquareLossSeq, VCLoss
from eb_jepa.training_utils import (
    get_default_dev_name,
    get_unified_experiment_dir,
    load_config,
    log_config,
    log_epoch,
    log_model_info,
    save_checkpoint,
    setup_device,
    setup_seed,
    setup_wandb,
)
from examples.mesh_jepa.encoder import DiffusionNetEncoder
from examples.mesh_jepa.predictor import MeshPredictor

logger = get_logger(__name__)


def run(
    fname: str = None,
    folder: str = None,
    **overrides,
):
    cfg = load_config(fname, overrides if overrides else None)

    # Setup
    device = setup_device(cfg.meta.device)
    setup_seed(cfg.meta.seed)

    # Experiment directory
    if folder:
        exp_dir = Path(folder)
        exp_dir.mkdir(parents=True, exist_ok=True)
        exp_name = exp_dir.name
    else:
        sweep_name = get_default_dev_name()
        exp_name = f"mesh_{cfg.data.feature_type}_w{cfg.model.width}_d{cfg.model.depth}"
        exp_dir = get_unified_experiment_dir(
            example_name="mesh_jepa",
            sweep_name=sweep_name,
            exp_name=exp_name,
            seed=cfg.meta.seed,
        )

    wandb_run = setup_wandb(
        project="eb_jepa_mesh",
        config={"example": "mesh_jepa", **OmegaConf.to_container(cfg, resolve=True)},
        run_dir=exp_dir,
        run_name=exp_name,
        tags=["mesh_jepa", cfg.data.feature_type, f"seed_{cfg.meta.seed}"],
        enabled=cfg.logging.log_wandb,
    )

    # Dataset
    train_subjects = cfg.data.get(
        "train_subjects", [50002, 50004, 50007, 50009, 50020, 50021, 50022, 50025]
    )
    test_subjects = cfg.data.get("test_subjects", [50026, 50027])

    train_set = DFAUSTDataset(
        data_dir=cfg.data.data_dir,
        seq_len=cfg.data.seq_len,
        feature_type=cfg.data.feature_type,
        subjects=train_subjects,
        max_clips=cfg.data.get("max_clips", None),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    logger.info(
        f"Dataset: {len(train_set)} clips, {len(train_loader)} batches/epoch, "
        f"feature_type={cfg.data.feature_type}"
    )

    # Model
    in_channels = 16 if cfg.data.feature_type == "hks" else 3
    encoder = DiffusionNetEncoder(
        in_channels=in_channels,
        out_dim=cfg.model.henc,
        width=cfg.model.width,
        depth=cfg.model.depth,
        n_eigen=cfg.model.n_eigen,
        dropout=cfg.model.get("dropout", True),
    )

    # Register operators
    ops = train_set.get_operators()
    encoder.register_operators(
        ops["eigenvalues"].to(device),
        ops["eigenvectors"].to(device),
        ops["mass"].to(device),
    )

    predictor = MeshPredictor(
        state_dim=cfg.model.henc,
        hidden_dim=cfg.model.hpre,
        num_layers=cfg.model.get("predictor_layers", 1),
    )

    projector = Projector(cfg.loss.proj_spec)
    regularizer = VCLoss(cfg.loss.std_coeff, cfg.loss.cov_coeff, proj=projector)
    predcost = SquareLossSeq(projector)

    jepa = JEPA(encoder, encoder, predictor, regularizer, predcost).to(device)

    # Log model info
    encoder_params = sum(p.numel() for p in encoder.parameters())
    predictor_params = sum(p.numel() for p in predictor.parameters())
    log_model_info(jepa, {"encoder": encoder_params, "predictor": predictor_params})

    # Optimizer + scheduler
    optimizer = Adam(
        jepa.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.optim.epochs)

    log_config(cfg)

    # Training loop
    logger.info(f"Starting training for {cfg.optim.epochs} epochs...")
    global_step = 0
    train_start_time = time.time()
    epoch_times = []
    history = []

    for epoch in range(cfg.optim.epochs):
        epoch_start = time.time()
        jepa.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            features = batch["features"].to(device)  # (B, T, V, C)

            optimizer.zero_grad()

            # Encode sequence: (B, T, V, C) → (B, D, T, 1, 1)
            state = encoder(features)

            # JEPA unroll (predictor + loss) on pre-encoded states
            # We bypass jepa.unroll's encoder call by passing states directly
            rloss, rloss_unweight, rloss_dict = jepa.regularizer(state, None)
            ploss = 0.0

            # Autoregressive multi-step prediction
            predicted_states = state[:, :, :1]  # First frame as context
            for i in range(cfg.model.steps):
                context = predicted_states[:, :, -1:]
                pred_step = predictor(context, None)[:, :, -1:]
                predicted_states = torch.cat([predicted_states, pred_step], dim=2)
                ploss += (
                    predcost(pred_step, state[:, :, i + 1 : i + 2]) / cfg.model.steps
                )

            loss = rloss + ploss
            loss.backward()

            # Gradient clipping (stabilizes RNN training)
            grad_clip = cfg.model.get("grad_clip", None)
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(jepa.parameters(), grad_clip)

            optimizer.step()

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "pred": f"{ploss.item():.4f}",
                    "vc": f"{rloss.item():.4f}",
                }
            )
            global_step += 1

        scheduler.step()
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # Logging
        if epoch % cfg.logging.get("log_every", 1) == 0:
            metrics = {
                "epoch": epoch,
                "train/loss": loss.item(),
                "train/pred_loss": ploss.item(),
                "train/vc_loss": rloss.item(),
                "train/lr": scheduler.get_last_lr()[0],
                "train/epoch_time_s": epoch_time,
            }
            for k, v in rloss_dict.items():
                metrics[f"train/{k}"] = float(v)

            history.append(metrics)

            if wandb_run:
                import wandb

                wandb.log(metrics, step=global_step)

            log_epoch(
                epoch,
                {"loss": loss.item(), "pred": ploss.item(), "vc": rloss.item()},
                total_epochs=cfg.optim.epochs,
            )

        # Save checkpoint
        if epoch % cfg.logging.get("save_every", 10) == 0 and epoch > 0:
            save_checkpoint(
                exp_dir / f"epoch_{epoch}.pth.tar",
                model=jepa,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
            )

    # Training time summary
    total_train_time = time.time() - train_start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    # Measure inference time (single clip)
    jepa.eval()
    with torch.no_grad():
        sample_features = next(iter(train_loader))["features"][:1].to(device)
        t0 = time.time()
        for _ in range(10):
            z = encoder(sample_features)
            pred = predictor(z[:, :, :1], None)
        inference_time = (time.time() - t0) / 10

    timing = {
        "total_training_time_s": total_train_time,
        "avg_epoch_time_s": avg_epoch_time,
        "inference_time_per_clip_s": inference_time,
        "num_epochs": cfg.optim.epochs,
        "num_batches_per_epoch": len(train_loader),
        "device": str(device),
    }

    logger.info(f"Training time: {total_train_time:.1f}s ({avg_epoch_time:.1f}s/epoch)")
    logger.info(f"Inference time: {inference_time*1000:.1f}ms/clip")

    # Final checkpoint (include timing)
    save_checkpoint(
        exp_dir / "final.pth.tar",
        model=jepa,
        optimizer=optimizer,
        epoch=cfg.optim.epochs,
        step=global_step,
        timing=timing,
    )

    # Save training history
    with open(exp_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save config to exp dir for eval to find
    import yaml

    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(OmegaConf.to_container(cfg, resolve=True), f)

    if wandb_run:
        import wandb

        wandb.log(timing)
        wandb.finish()

    logger.info(f"Training complete! Model saved to {exp_dir}")


if __name__ == "__main__":
    fire.Fire(run)
