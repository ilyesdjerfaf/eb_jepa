"""Generate meshes from embeddings using trained AtlasNet + surface reconstruction.

Pipeline:
    1. Load trained AtlasNet
    2. Generate dense point cloud from embedding
    3. Surface reconstruction (Delaunay + alpha shape filtering)
    4. Save as .ply files

No open3d or scikit-image required — uses scipy + trimesh only.

Usage:
    python -m examples.mesh_jepa.inference.reconstruct \
        --model_dir /lustre/work/.../inference/models/jepa_large_hks \
        --data_dir /lustre/work/.../inference/data/jepa_large_hks \
        --output_dir /lustre/work/.../inference/meshes/jepa_large_hks \
        --n_samples 20
"""

import json
from collections import Counter
from pathlib import Path

import fire
import numpy as np
import torch
import trimesh
from scipy.spatial import Delaunay
from tqdm import tqdm

from eb_jepa.logging import get_logger
from examples.mesh_jepa.inference.atlasnet import AtlasNet

logger = get_logger(__name__)


def alpha_shape_mesh(points, alpha=None):
    """Reconstruct surface mesh from point cloud using alpha shapes.

    Uses 3D Delaunay triangulation, extracts boundary faces,
    then filters by maximum edge length (alpha parameter).

    Args:
        points: (N, 3) numpy array
        alpha: max edge length threshold. If None, auto-computed from point density.
    Returns:
        vertices (N, 3), faces (F, 3) or (None, None) on failure
    """
    if len(points) < 4:
        return None, None

    # Auto-compute alpha from point density
    if alpha is None:
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        dists, _ = tree.query(points, k=6)
        avg_dist = dists[:, 1:].mean()
        alpha = avg_dist * 3.0

    # Delaunay triangulation
    try:
        tri = Delaunay(points)
    except Exception as e:
        logger.warning(f"Delaunay failed: {e}")
        return None, None

    # Extract boundary faces (faces belonging to exactly one tetrahedron)
    face_count = Counter()
    for simplex in tri.simplices:
        for face in [
            (simplex[0], simplex[1], simplex[2]),
            (simplex[0], simplex[1], simplex[3]),
            (simplex[0], simplex[2], simplex[3]),
            (simplex[1], simplex[2], simplex[3]),
        ]:
            face_count[tuple(sorted(face))] += 1

    boundary_faces = [list(f) for f, c in face_count.items() if c == 1]

    if not boundary_faces:
        return None, None

    # Alpha filter: remove faces with edges longer than alpha
    filtered = []
    for f in boundary_faces:
        edges = [
            np.linalg.norm(points[f[0]] - points[f[1]]),
            np.linalg.norm(points[f[1]] - points[f[2]]),
            np.linalg.norm(points[f[0]] - points[f[2]]),
        ]
        if max(edges) < alpha:
            filtered.append(f)

    if not filtered:
        return None, None

    return points, np.array(filtered)


def run(
    model_dir: str,
    data_dir: str,
    output_dir: str,
    n_samples: int = 20,
    points_per_patch: int = 2000,
    alpha: float = None,
    device: str = "auto",
    save_pointclouds: bool = True,
):
    """Generate meshes from embeddings.

    Args:
        model_dir: Directory with trained AtlasNet (best.pth + config.json)
        data_dir: Directory with embeddings.npy (+ point_clouds.npy for GT)
        output_dir: Where to save generated .ply meshes
        n_samples: Number of samples to reconstruct
        points_per_patch: Dense sampling for reconstruction (more = smoother)
        alpha: Alpha shape threshold (None = auto from point density)
    """
    from eb_jepa.training_utils import setup_device

    device = setup_device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load AtlasNet config + weights
    model_dir = Path(model_dir)
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    model = AtlasNet(
        latent_dim=config["latent_dim"],
        n_patches=config["n_patches"],
        points_per_patch=config["points_per_patch"],
        hidden_dim=config["hidden_dim"],
        depth=config["depth"],
    ).to(device)
    model.load_state_dict(torch.load(model_dir / "best.pth", map_location=device))
    model.eval()

    # Load embeddings and normalize (same as training)
    data_dir = Path(data_dir)
    embeddings = np.load(data_dir / "embeddings.npy")

    # Apply embedding normalization if stats are available
    emb_mean_path = model_dir / "emb_mean.npy"
    if emb_mean_path.exists():
        emb_mean = np.load(emb_mean_path)
        emb_std = np.load(model_dir / "emb_std.npy")
        embeddings = (embeddings - emb_mean) / emb_std
        logger.info("Applied embedding normalization from training stats")

    # Load ground truth if available
    gt_path = data_dir / "point_clouds.npy"
    gt_available = gt_path.exists()
    if gt_available:
        gt_clouds = np.load(gt_path)

    n_samples = min(n_samples, len(embeddings))
    indices = np.linspace(0, len(embeddings) - 1, n_samples, dtype=int)

    total_pts = points_per_patch * config["n_patches"]
    logger.info(
        f"Reconstructing {n_samples} meshes "
        f"(dense: {points_per_patch} pts/patch × {config['n_patches']} patches = "
        f"{total_pts} points)"
    )

    for i, idx in enumerate(tqdm(indices, desc="Reconstructing")):
        emb = torch.from_numpy(embeddings[idx : idx + 1]).float().to(device)

        # Generate dense point cloud
        with torch.no_grad():
            points, patch_ids = model.forward_per_patch(
                emb, n_points_per_patch=points_per_patch
            )
        points_np = points[0].cpu().numpy()

        # Save point cloud
        if save_pointclouds:
            cloud = trimesh.PointCloud(points_np)
            cloud.export(str(output_dir / f"sample_{i:03d}_pc.ply"))

        # Surface reconstruction (alpha shape)
        verts, faces = alpha_shape_mesh(points_np, alpha=alpha)

        if verts is not None:
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            mesh.export(str(output_dir / f"sample_{i:03d}_mesh.ply"))
            logger.info(
                f"  Sample {i}: {len(points_np)} pts → "
                f"{len(verts)} verts, {len(faces)} faces"
            )
        else:
            logger.warning(
                f"  Sample {i}: reconstruction failed, saved point cloud only"
            )

        # Save ground truth for comparison
        if gt_available and gt_clouds.shape[-1] == 3:
            gt_cloud = trimesh.PointCloud(gt_clouds[idx])
            gt_cloud.export(str(output_dir / f"sample_{i:03d}_gt.ply"))

    # Summary
    summary = {
        "n_samples": n_samples,
        "points_per_patch": points_per_patch,
        "total_generated_points": total_pts,
        "alpha": alpha,
        "model_dir": str(model_dir),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Done! Meshes saved to {output_dir}")


if __name__ == "__main__":
    fire.Fire(run)
