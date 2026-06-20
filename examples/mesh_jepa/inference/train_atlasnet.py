"""Train AtlasNet decoder on extracted embeddings.

Takes pre-extracted (embedding, point_cloud) pairs and trains AtlasNet
to reconstruct the point cloud from the embedding. Uses Chamfer distance.

Data augmentations (AtlasNet paper + 3D-CODED):
  - Random point subsampling (each batch sees different 2500 GT points)
  - Random scaling (uniform [0.8, 1.2])
  - Random rotation (SO(3) — full random rotation matrix)
  - Point jitter (Gaussian noise σ=0.005)

Normalization:
  - Point clouds are already centered + unit-sphere normalized from preprocessing
  - We store per-dataset stats (mean, std of embeddings) for reproducibility

Usage:
    uv run python -m examples.mesh_jepa.inference.train_atlasnet \
        --data_dir examples/mesh_jepa/inference/data/jepa_large_hks \
        --output_dir examples/mesh_jepa/inference/models/jepa_large_hks \
        --epochs 200
"""

import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from eb_jepa.logging import get_logger
from examples.mesh_jepa.inference.atlasnet import AtlasNet

logger = get_logger(__name__)


def chamfer_distance(pred, target):
    """Chamfer distance between two point clouds.

    pred: (B, N, 3)
    target: (B, M, 3)
    Returns: scalar (mean over batch)
    """
    diff_p2t = pred.unsqueeze(2) - target.unsqueeze(1)  # (B, N, M, 3)
    dist_p2t = (diff_p2t**2).sum(dim=-1)  # (B, N, M)
    min_p2t = dist_p2t.min(dim=2).values.mean(dim=1)  # (B,)
    min_t2p = dist_p2t.min(dim=1).values.mean(dim=1)  # (B,)
    return (min_p2t + min_t2p).mean()


def chamfer_distance_chunked(pred, target, chunk_size=1000):
    """Memory-efficient Chamfer distance for large point clouds."""
    B, N, _ = pred.shape
    M = target.shape[1]

    min_p2t_all = []
    for i in range(0, N, chunk_size):
        pred_chunk = pred[:, i : i + chunk_size]
        diff = pred_chunk.unsqueeze(2) - target.unsqueeze(1)
        dist = (diff**2).sum(dim=-1)
        min_p2t_all.append(dist.min(dim=2).values)
    min_p2t = torch.cat(min_p2t_all, dim=1).mean(dim=1)

    min_t2p_all = []
    for i in range(0, M, chunk_size):
        target_chunk = target[:, i : i + chunk_size]
        diff = target_chunk.unsqueeze(2) - pred.unsqueeze(1)
        dist = (diff**2).sum(dim=-1)
        min_t2p_all.append(dist.min(dim=2).values)
    min_t2p = torch.cat(min_t2p_all, dim=1).mean(dim=1)

    return (min_p2t + min_t2p).mean()


def random_rotation_matrix(batch_size, device):
    """Sample uniform random SO(3) rotation matrices (Gram-Schmidt)."""
    z = torch.randn(batch_size, 3, 3, device=device)
    q, r = torch.linalg.qr(z)
    sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    q = q * sign.unsqueeze(-2)
    det = torch.det(q)
    q[det < 0, :, 0] *= -1
    return q


class EmbeddingPointCloudDataset(Dataset):
    """Dataset with online augmentation for AtlasNet training."""

    def __init__(
        self,
        embeddings,
        point_clouds,
        n_sample_points=2500,
        augment=True,
        jitter_std=0.005,
        scale_range=(0.8, 1.2),
    ):
        self.embeddings = torch.from_numpy(embeddings).float()
        self.point_clouds = torch.from_numpy(point_clouds).float()
        self.n_sample_points = n_sample_points
        self.augment = augment
        self.jitter_std = jitter_std
        self.scale_range = scale_range

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        emb = self.embeddings[idx]  # (D,)
        pc = self.point_clouds[idx]  # (V, 3)

        # Random subsampling — different points each time (paper standard)
        n_pts = pc.shape[0]
        if self.n_sample_points < n_pts:
            perm = torch.randperm(n_pts)[: self.n_sample_points]
            pc = pc[perm]

        if self.augment:
            # Random scaling (uniform [0.8, 1.2])
            scale = (
                torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]).item()
            )
            pc = pc * scale

            # Random SO(3) rotation
            R = random_rotation_matrix(1, pc.device)[0]  # (3, 3)
            pc = pc @ R.T

            # Point jitter (Gaussian noise)
            pc = pc + torch.randn_like(pc) * self.jitter_std

        return emb, pc


def visualize_reconstruction(
    model, embeddings, point_clouds, epoch, output_dir, device, n_vis=4, n_dense=2000
):
    """Save reconstruction screenshots: GT vs predicted point clouds (3 views)."""
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    model.eval()
    indices = np.linspace(0, len(embeddings) - 1, n_vis, dtype=int)

    fig, axes = plt.subplots(
        n_vis, 3, figsize=(15, 4 * n_vis), subplot_kw={"projection": "3d"}
    )
    if n_vis == 1:
        axes = axes[np.newaxis, :]

    views = [
        (20, 45, "Front-Left"),
        (5, 90, "Side"),
        (80, 0, "Top"),
    ]

    with torch.no_grad():
        for row, idx in enumerate(indices):
            emb = torch.from_numpy(embeddings[idx : idx + 1]).float().to(device)
            pred_pc = model(emb, n_points_per_patch=n_dense)[0].cpu().numpy()
            gt_pc = point_clouds[idx]

            # Subsample for cleaner visualization
            gt_sub = gt_pc[
                np.random.choice(len(gt_pc), min(2000, len(gt_pc)), replace=False)
            ]
            pred_sub = pred_pc[
                np.random.choice(len(pred_pc), min(2000, len(pred_pc)), replace=False)
            ]

            for col, (elev, azim, title) in enumerate(views):
                ax = axes[row, col]
                ax.scatter(
                    gt_sub[:, 0],
                    gt_sub[:, 1],
                    gt_sub[:, 2],
                    c="steelblue",
                    s=0.3,
                    alpha=0.4,
                    label="GT",
                )
                ax.scatter(
                    pred_sub[:, 0],
                    pred_sub[:, 1],
                    pred_sub[:, 2],
                    c="coral",
                    s=0.3,
                    alpha=0.4,
                    label="Pred",
                )
                ax.view_init(elev=elev, azim=azim)
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_zlim(-1, 1)
                ax.set_title(f"Sample {idx} — {title}", fontsize=9)
                ax.set_axis_off()
                if row == 0 and col == 0:
                    ax.legend(markerscale=10, fontsize=8)

    fig.suptitle(f"Epoch {epoch}", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(vis_dir / f"epoch_{epoch:04d}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    model.train()


def run(
    data_dir: str,
    output_dir: str,
    n_patches: int = 25,
    points_per_patch: int = 100,
    hidden_dim: int = 256,
    depth: int = 4,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "auto",
    n_sample_points: int = 2500,
    augment: bool = True,
    jitter_std: float = 0.005,
    scale_range_low: float = 0.8,
    scale_range_high: float = 1.2,
    vis_every: int = 20,
    n_vis: int = 4,
):
    """Train AtlasNet on extracted embeddings.

    Defaults follow AtlasNet paper (Groueix et al. 2018) adapted for 256-dim latent:
    - 25 patches (paper default for complex shapes)
    - 100 points per patch = 2500 total (paper trains with 2500 points)
    - 4-layer MLPs per patch, hidden=256
    - ~6.6M decoder params
    - Augmentation: random rotation + scaling + jitter (paper + 3D-CODED)
    - LR schedule: decay by 10x at epochs 100 and 150 (paper)

    Args:
        data_dir: Directory with embeddings.npy + point_clouds.npy
        output_dir: Where to save trained model
        n_patches: Number of AtlasNet patches (25 = paper default)
        points_per_patch: Points generated per patch (100, total=2500 for training)
        hidden_dim: MLP hidden width per patch (256 for 256-dim latent)
        depth: MLP depth per patch (4 = paper default)
        n_sample_points: GT points subsampled per shape (2500 = paper)
        augment: Enable data augmentation (rotation + scale + jitter)
        jitter_std: Gaussian noise std for point jitter
    """
    from eb_jepa.training_utils import setup_device

    device = setup_device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data_dir = Path(data_dir)
    embeddings = np.load(data_dir / "embeddings.npy")
    point_clouds = np.load(data_dir / "point_clouds.npy")

    logger.info(
        f"Loaded {embeddings.shape[0]} samples: "
        f"embeddings {embeddings.shape}, point_clouds {point_clouds.shape}"
    )

    if point_clouds.shape[-1] != 3:
        logger.error(
            f"Point clouds have {point_clouds.shape[-1]} channels, expected 3 (XYZ). "
            f"Make sure to extract with vertices=True or use xyz feature_type."
        )
        return

    latent_dim = embeddings.shape[1]

    # Compute and save normalization stats
    emb_mean = embeddings.mean(axis=0)
    emb_std = embeddings.std(axis=0)
    pc_mean = point_clouds.mean(axis=(0, 1))
    pc_std = point_clouds.std()

    norm_stats = {
        "emb_mean": emb_mean.tolist(),
        "emb_std": emb_std.tolist(),
        "pc_mean": pc_mean.tolist(),
        "pc_std": float(pc_std),
        "n_samples": int(embeddings.shape[0]),
        "n_vertices": int(point_clouds.shape[1]),
    }
    with open(output_dir / "norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)

    # Normalize embeddings (zero-mean, unit-std per dimension)
    emb_std_safe = np.where(emb_std > 1e-8, emb_std, 1.0)
    embeddings_norm = (embeddings - emb_mean) / emb_std_safe

    logger.info(
        f"Normalization: emb mean={emb_mean.mean():.4f}, std={emb_std.mean():.4f}, "
        f"pc mean={pc_mean}, pc_std={pc_std:.4f}"
    )

    # Dataset with augmentation
    dataset = EmbeddingPointCloudDataset(
        embeddings_norm,
        point_clouds,
        n_sample_points=n_sample_points,
        augment=augment,
        jitter_std=jitter_std,
        scale_range=(scale_range_low, scale_range_high),
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0
    )

    # Model
    model = AtlasNet(
        latent_dim=latent_dim,
        n_patches=n_patches,
        points_per_patch=points_per_patch,
        hidden_dim=hidden_dim,
        depth=depth,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"AtlasNet: {n_patches} patches, {points_per_patch} pts/patch, "
        f"hidden={hidden_dim}, depth={depth}, {total_params:,} params"
    )
    logger.info(
        f"Augmentation: {'ON' if augment else 'OFF'} "
        f"(rotation=SO(3), scale=[{scale_range_low},{scale_range_high}], "
        f"jitter_std={jitter_std})"
    )

    # Optimizer + LR schedule (paper: decay by 10x at milestones)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    milestones = [int(epochs * 0.5), int(epochs * 0.75)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    logger.info(
        f"Training AtlasNet for {epochs} epochs "
        f"(LR decay at epochs {milestones})..."
    )

    # Training
    history = []
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        for emb_batch, pc_batch in pbar:
            emb_batch = emb_batch.to(device)
            pc_batch = pc_batch.to(device)

            optimizer.zero_grad()

            # Generate point cloud from embedding
            pred_pc = model(emb_batch)  # (B, total_points, 3)

            # Chamfer distance (pred has n_patches*points_per_patch points,
            # target has n_sample_points — both are ~2500)
            loss = chamfer_distance(pred_pc, pc_batch)
            loss.backward()

            # Gradient clipping (stabilizes training with augmentation)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"chamfer": f"{loss.item():.6f}"})

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        history.append(
            {"epoch": epoch, "chamfer_loss": avg_loss, "lr": scheduler.get_last_lr()[0]}
        )

        if epoch % 10 == 0:
            logger.info(
                f"Epoch {epoch}: chamfer={avg_loss:.6f}, lr={scheduler.get_last_lr()[0]:.1e}"
            )

        # Periodic reconstruction visualization
        if epoch % vis_every == 0 or epoch == epochs - 1:
            visualize_reconstruction(
                model,
                embeddings_norm,
                point_clouds,
                epoch,
                output_dir,
                device,
                n_vis=n_vis,
            )
            logger.info(
                f"  → Saved visualization: visualizations/epoch_{epoch:04d}.png"
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_dir / "best.pth")

    # Save final
    torch.save(model.state_dict(), output_dir / "final.pth")
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save config (includes all hyperparams for reproducibility)
    config = {
        "latent_dim": latent_dim,
        "n_patches": n_patches,
        "points_per_patch": points_per_patch,
        "hidden_dim": hidden_dim,
        "depth": depth,
        "epochs": epochs,
        "lr": lr,
        "milestones": milestones,
        "batch_size": batch_size,
        "n_sample_points": n_sample_points,
        "augment": augment,
        "jitter_std": jitter_std,
        "scale_range": [scale_range_low, scale_range_high],
        "best_chamfer": best_loss,
        "total_params": total_params,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save normalization stats alongside model for inference
    np.save(output_dir / "emb_mean.npy", emb_mean)
    np.save(output_dir / "emb_std.npy", emb_std_safe)

    logger.info(f"Training complete. Best Chamfer: {best_loss:.6f}")
    logger.info(f"Model saved to {output_dir}")


if __name__ == "__main__":
    fire.Fire(run)
