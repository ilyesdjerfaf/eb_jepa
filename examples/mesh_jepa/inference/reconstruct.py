"""Generate meshes from embeddings using trained AtlasNet + Poisson reconstruction.

Pipeline:
    1. Load trained AtlasNet
    2. Generate dense point cloud from embedding
    3. Estimate normals (PCA-based)
    4. Poisson surface reconstruction → watertight mesh
    5. Save as .ply files

Usage:
    uv run python -m examples.mesh_jepa.inference.reconstruct \
        --model_dir examples/mesh_jepa/inference/models/jepa_large_hks \
        --data_dir examples/mesh_jepa/inference/data/jepa_large_hks \
        --output_dir examples/mesh_jepa/inference/meshes/jepa_large_hks \
        --n_samples 20
"""

import json
from pathlib import Path

import fire
import numpy as np
import torch
from tqdm import tqdm

from eb_jepa.logging import get_logger
from examples.mesh_jepa.inference.atlasnet import AtlasNet

logger = get_logger(__name__)


def estimate_normals(points, k=30):
    """Estimate normals via PCA on k-nearest neighbors.

    points: (N, 3) numpy array
    Returns: normals (N, 3)
    """
    from scipy.spatial import KDTree

    tree = KDTree(points)
    _, idx = tree.query(points, k=k)
    normals = np.zeros_like(points)

    for i in range(len(points)):
        neighbors = points[idx[i]]
        centered = neighbors - neighbors.mean(axis=0)
        cov = centered.T @ centered
        _, vecs = np.linalg.eigh(cov)
        normals[i] = vecs[:, 0]  # smallest eigenvector = normal

    # Orient normals consistently (point outward from centroid)
    centroid = points.mean(axis=0)
    for i in range(len(points)):
        if np.dot(normals[i], points[i] - centroid) < 0:
            normals[i] *= -1

    return normals


def poisson_reconstruct(points, normals, depth=8):
    """Poisson surface reconstruction using Open3D.

    Returns: open3d TriangleMesh
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    # Trim low-density vertices (removes reconstruction artifacts at boundaries)
    densities = np.asarray(densities)
    threshold = np.quantile(densities, 0.05)
    vertices_to_remove = densities < threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    return mesh


def run(
    model_dir: str,
    data_dir: str,
    output_dir: str,
    n_samples: int = 20,
    points_per_patch: int = 2000,
    poisson_depth: int = 8,
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
        poisson_depth: Octree depth for Poisson (higher = more detail)
    """
    import open3d as o3d

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

    logger.info(
        f"Reconstructing {n_samples} meshes (dense sampling: {points_per_patch} pts/patch)..."
    )

    chamfer_scores = []

    for i, idx in enumerate(tqdm(indices, desc="Reconstructing")):
        emb = torch.from_numpy(embeddings[idx : idx + 1]).float().to(device)

        # Generate dense point cloud
        with torch.no_grad():
            points, patch_ids = model.forward_per_patch(
                emb, n_points_per_patch=points_per_patch
            )
        points_np = points[0].cpu().numpy()  # (total_points, 3)

        # Save point cloud
        if save_pointclouds:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_np)
            o3d.io.write_point_cloud(str(output_dir / f"sample_{i:03d}_pc.ply"), pcd)

        # Estimate normals
        normals = estimate_normals(points_np)

        # Poisson surface reconstruction
        mesh = poisson_reconstruct(points_np, normals, depth=poisson_depth)

        # Save mesh
        o3d.io.write_triangle_mesh(str(output_dir / f"sample_{i:03d}_mesh.ply"), mesh)

        # Save ground truth for comparison
        if gt_available and gt_clouds.shape[-1] == 3:
            gt_pcd = o3d.geometry.PointCloud()
            gt_pcd.points = o3d.utility.Vector3dVector(gt_clouds[idx])
            o3d.io.write_point_cloud(str(output_dir / f"sample_{i:03d}_gt.ply"), gt_pcd)

        logger.info(
            f"  Sample {i}: {len(points_np)} generated points → "
            f"{len(mesh.vertices)} mesh vertices, {len(mesh.triangles)} faces"
        )

    # Summary
    summary = {
        "n_samples": n_samples,
        "points_per_patch": points_per_patch,
        "total_generated_points": config["n_patches"] * points_per_patch,
        "poisson_depth": poisson_depth,
        "model_dir": str(model_dir),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Done! Meshes saved to {output_dir}")


if __name__ == "__main__":
    fire.Fire(run)
