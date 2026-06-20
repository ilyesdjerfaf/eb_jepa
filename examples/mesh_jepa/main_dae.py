"""
Mesh DAE (Denoising Autoencoder) Baseline Training Script

Same temporal structure as JEPA: encode context → GRU predicts future.
Key difference: loss is MSE in FEATURE SPACE (reconstruct HKS/XYZ),
not in latent space. No VICReg needed (reconstruction prevents collapse).

Purpose: demonstrate that JEPA's abstract prediction objective learns
better representations than predicting in input space.
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

from eb_jepa.datasets.mesh import DFAUSTDataset
from eb_jepa.jepa import JEPA
from eb_jepa.logging import get_logger
from eb_jepa.training_utils import (
    get_default_dev_name,
    get_unified_experiment_dir,
    load_config,
    log_config,
    log_epoch,
    setup_device,
    setup_seed,
    setup_wandb,
)
from examples.mesh_jepa.decoder import MeshDecoder
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
        exp_name = f"dae_{cfg.data.feature_type}_w{cfg.model.width}_d{cfg.model.depth}"
        exp_dir = get_unified_experiment_dir(
            example_name="mesh_jepa",
            sweep_name=sweep_name,
            exp_name=exp_name,
            seed=cfg.meta.seed,
        )

    wandb_run = setup_wandb(
        project="eb_jepa_mesh",
        config={"example": "mesh_dae", **OmegaConf.to_container(cfg, resolve=True)},
        run_dir=exp_dir,
        run_name=exp_name,
        tags=["mesh_dae", cfg.data.feature_type, f"seed_{cfg.meta.seed}"],
        enabled=cfg.logging.log_wandb,
    )

    # Dataset
    train_subjects = cfg.data.get(
        "train_subjects", [50002, 50004, 50007, 50009, 50020, 50021, 50022, 50025]
    )

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

    # Model — Encoder (same as JEPA)
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

    # Predictor (same GRU as JEPA)
    predictor = MeshPredictor(
        state_dim=cfg.model.henc,
        hidden_dim=cfg.model.hpre,
        num_layers=cfg.model.get("predictor_layers", 1),
    )

    # Decoder (new — maps latent back to feature space)
    decoder = MeshDecoder(
        latent_dim=cfg.model.henc,
        out_channels=in_channels,
        width=cfg.model.get("decoder_width", 512),
        depth=cfg.model.get("decoder_depth", 3),
    )
    n_vertices = ops["eigenvectors"].shape[0]
    decoder.set_n_vertices(n_vertices)

    # Move to device
    encoder = encoder.to(device)
    predictor = predictor.to(device)
    decoder = decoder.to(device)

    # Log model info
    encoder_params = sum(p.numel() for p in encoder.parameters())
    predictor_params = sum(p.numel() for p in predictor.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    total_params = encoder_params + predictor_params + decoder_params
    logger.info(
        f"Parameters — encoder: {encoder_params:,}, predictor: {predictor_params:,}, "
        f"decoder: {decoder_params:,}, total: {total_params:,}"
    )

    # Optimizer + scheduler (all components)
    all_params = (
        list(encoder.parameters())
        + list(predictor.parameters())
        + list(decoder.parameters())
    )
    optimizer = Adam(all_params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.optim.epochs)

    # Mass for weighted reconstruction loss
    mass = ops["mass"].to(device)
    mass_sum = mass.sum()

    log_config(cfg)

    # Training loop
    logger.info(f"Starting DAE training for {cfg.optim.epochs} epochs...")
    global_step = 0
    train_start_time = time.time()
    epoch_times = []
    history = []

    for epoch in range(cfg.optim.epochs):
        epoch_start = time.time()
        encoder.train()
        predictor.train()
        decoder.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            features = batch["features"].to(device)  # (B, T, V, C)
            B, T, V, C = features.shape

            optimizer.zero_grad()

            # Encode full sequence: (B, T, V, C) → (B, D, T, 1, 1)
            state = encoder(features)

            # Autoregressive multi-step prediction + reconstruction loss
            steps = cfg.model.steps
            recon_loss = 0.0

            predicted_states = state[:, :, :1]  # First frame as context
            for i in range(steps):
                context = predicted_states[:, :, -1:]
                pred_step = predictor(context, None)[:, :, -1:]  # (B, D, 1, 1, 1)
                predicted_states = torch.cat([predicted_states, pred_step], dim=2)

                # Decode predicted latent → feature space
                z_pred = pred_step[:, :, 0, 0, 0]  # (B, D)
                recon = decoder(z_pred)  # (B, V, C)

                # Ground truth features for frame i+1
                target = features[:, i + 1]  # (B, V, C)

                # Mass-weighted MSE (discretization-invariant)
                diff_sq = ((recon - target) ** 2).sum(dim=-1)  # (B, V)
                frame_loss = torch.einsum("bv,v->b", diff_sq, mass) / mass_sum
                recon_loss += frame_loss.mean() / steps

            loss = recon_loss
            loss.backward()

            # Gradient clipping
            grad_clip = cfg.model.get("grad_clip", None)
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(all_params, grad_clip)

            optimizer.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            global_step += 1

            # Per-step logging
            step_metrics = {
                "step": global_step,
                "epoch": epoch,
                "train/loss": loss.item(),
                "train/recon_loss": recon_loss.item(),
                "train/lr": scheduler.get_last_lr()[0],
            }
            history.append(step_metrics)

            if wandb_run:
                import wandb

                wandb.log(step_metrics, step=global_step)

        scheduler.step()
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        if epoch % cfg.logging.get("log_every", 1) == 0:
            log_epoch(
                epoch,
                {"loss": loss.item(), "recon": recon_loss.item()},
                total_epochs=cfg.optim.epochs,
            )

        # Save checkpoint
        if epoch % cfg.logging.get("save_every", 10) == 0 and epoch > 0:
            ckpt = {
                "encoder_state_dict": encoder.state_dict(),
                "predictor_state_dict": predictor.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "step": global_step,
            }
            torch.save(ckpt, exp_dir / f"epoch_{epoch}.pth.tar")

    # Training time summary
    total_train_time = time.time() - train_start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    # Measure inference time
    encoder.eval()
    predictor.eval()
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

    # Final checkpoint — save encoder in JEPA-compatible format for eval.py
    # Wrap encoder+predictor into a JEPA shell so eval.py can load it
    from eb_jepa.architectures import Projector
    from eb_jepa.losses import SquareLossSeq, VCLoss

    projector = Projector(cfg.loss.proj_spec)
    regularizer = VCLoss(cfg.loss.std_coeff, cfg.loss.cov_coeff, proj=projector)
    predcost = SquareLossSeq(None)
    jepa = JEPA(encoder, encoder, predictor, regularizer, predcost).to(device)

    final_ckpt = {
        "model_state_dict": jepa.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": cfg.optim.epochs,
        "step": global_step,
        "timing": timing,
    }
    torch.save(final_ckpt, exp_dir / "final.pth.tar")

    # Save training history
    with open(exp_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save config
    import yaml

    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(OmegaConf.to_container(cfg, resolve=True), f)

    if wandb_run:
        import wandb

        wandb.log(timing)
        wandb.finish()

    logger.info(f"DAE training complete! Model saved to {exp_dir}")


if __name__ == "__main__":
    fire.Fire(run)
