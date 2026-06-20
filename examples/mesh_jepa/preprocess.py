"""
Preprocess DFAUST dataset for Mesh JEPA training.

Steps:
1. Load raw HDF5 registrations
2. Temporal subsample to 15fps (keep every 4th frame)
3. Per-frame: center, normalize to unit sphere
4. Per-frame: compute cotangent Laplacian + eigendecomposition + HKS
5. Save operators (from mean pose, for DiffusionNet), per-sequence data, manifest

Usage:
    uv run python -m examples.mesh_jepa.preprocess \
        --data_dir datasets/dfaust/raw \
        --out_dir datasets/dfaust/processed \
        --n_eigen 128 --n_hks 16 --temporal_stride 4 \
        --actions jumping_jacks punching running_on_spot
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import scipy.sparse.linalg


def load_hdf5_registrations(data_dir: Path):
    """Load DFAUST registrations from HDF5 files."""
    import h5py

    sequences = {}
    faces = None
    for fname in sorted(data_dir.glob("registrations_*.hdf5")):
        print(f"Loading {fname.name}...")
        with h5py.File(fname, "r") as f:
            for key in f.keys():
                if key == "faces":
                    faces = np.array(f[key])
                    continue
                data = f[key]
                verts = np.array(data).transpose(2, 0, 1)  # (V,3,T) → (T,V,3)
                parts = key.split("_", 1)
                sequences[key] = {
                    "vertices": verts,
                    "subject": parts[0],
                    "action": parts[1] if len(parts) > 1 else key,
                    "n_frames": verts.shape[0],
                }
    return sequences, faces


def compute_cotangent_laplacian(vertices, faces):
    """Compute cotangent Laplacian and mass matrix for a triangle mesh."""
    import robust_laplacian

    L, M = robust_laplacian.mesh_laplacian(
        np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)
    )
    return L, M


def eigendecompose(L, M, n_eigen=128):
    """Compute first n_eigen eigenvectors/eigenvalues of the Laplacian."""
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
        L, k=n_eigen, M=M, sigma=-1e-8, which="LM"
    )
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues.astype(np.float32), eigenvectors.astype(np.float32)


def compute_hks(eigenvectors, eigenvalues, n_scales=16, t_min=0.1, t_max=1000.0):
    """Compute Heat Kernel Signature for all vertices.

    HKS(v, t) = sum_i exp(-lambda_i * t) * phi_i(v)^2

    Returns: (V, n_scales)
    """
    t_values = np.logspace(np.log10(t_min), np.log10(t_max), n_scales)
    phi_sq = eigenvectors**2  # (V, K)
    exp_terms = np.exp(
        -eigenvalues[np.newaxis, :] * t_values[:, np.newaxis]
    )  # (n_scales, K)
    hks = phi_sq @ exp_terms.T  # (V, n_scales)
    return hks.astype(np.float32)


def compute_frame_hks(vertices, faces, n_eigen, n_hks):
    """Compute per-frame HKS: Laplacian + eigendecomp + HKS for one frame.

    Each frame has different vertex positions → different cotangent weights
    → different eigenvectors → different HKS. This makes HKS a pose-dependent,
    rotation-invariant descriptor.

    Returns: (V, n_hks)
    """
    L, M = compute_cotangent_laplacian(vertices, faces)
    eigenvalues, eigenvectors = eigendecompose(L, M, n_eigen=n_eigen)
    hks = compute_hks(eigenvectors, eigenvalues, n_scales=n_hks)
    return hks


def normalize_vertices(vertices):
    """Center and normalize to unit sphere."""
    center = vertices.mean(axis=0)
    vertices = vertices - center
    scale = np.abs(vertices).max()
    if scale > 0:
        vertices = vertices / scale
    return vertices


def main():
    parser = argparse.ArgumentParser(description="Preprocess DFAUST for Mesh JEPA")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets/dfaust/raw",
        help="Path to raw DFAUST HDF5 files",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="datasets/dfaust/processed",
        help="Output directory for preprocessed data",
    )
    parser.add_argument(
        "--n_eigen", type=int, default=128, help="Number of Laplacian eigenvectors"
    )
    parser.add_argument(
        "--n_hks", type=int, default=16, help="Number of HKS time scales"
    )
    parser.add_argument(
        "--temporal_stride",
        type=int,
        default=4,
        help="Keep every Nth frame (4 = 15fps)",
    )
    parser.add_argument(
        "--actions",
        nargs="*",
        default=None,
        help="Actions to process (default: all). E.g.: jumping_jacks punching running_on_spot",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Suffix appended to output dir (e.g., '3actions' → processed_3actions/)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    if args.suffix:
        out_dir = out_dir.parent / f"{out_dir.name}_{args.suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DFAUST Preprocessing for Mesh JEPA")
    print("=" * 60)

    # Step 1: Load raw data
    print("\n[1/5] Loading raw HDF5 data...")
    sequences, faces = load_hdf5_registrations(data_dir)
    print(f"  Loaded {len(sequences)} sequences, faces shape: {faces.shape}")

    if args.actions:
        sequences = {k: v for k, v in sequences.items() if v["action"] in args.actions}
        print(f"  Filtered to {len(sequences)} sequences for actions: {args.actions}")

    # Step 2: Compute mean-pose Laplacian (for DiffusionNet operators)
    print("\n[2/5] Computing mean-pose Laplacian (for DiffusionNet operators)...")
    all_first_frames = [seq["vertices"][0] for seq in sequences.values()]
    mean_pose = np.mean(all_first_frames, axis=0)  # (V, 3)
    L, M = compute_cotangent_laplacian(mean_pose, faces)
    mean_eigenvalues, mean_eigenvectors = eigendecompose(L, M, n_eigen=args.n_eigen)
    mass = np.array(M.diagonal(), dtype=np.float32)
    print(f"  Vertices: {mean_pose.shape[0]}")
    print(
        f"  Eigenvalues range: [{mean_eigenvalues[0]:.6f}, {mean_eigenvalues[-1]:.2f}]"
    )

    # Save operators (used by DiffusionNet for diffusion)
    np.savez(
        out_dir / "operators.npz",
        eigenvalues=mean_eigenvalues,
        eigenvectors=mean_eigenvectors,
        mass=mass,
        faces=faces,
    )
    print(f"  Saved operators to {out_dir / 'operators.npz'}")

    # Step 3: Process each sequence (normalize + per-frame HKS)
    print(
        f"\n[3/5] Processing sequences (stride={args.temporal_stride}, per-frame HKS)..."
    )
    print("  (Per-frame eigendecomposition — this may take a while)")
    manifest_rows = []
    total_frames_processed = 0

    for i, (key, seq_data) in enumerate(sequences.items()):
        verts_full = seq_data["vertices"]  # (T, V, 3)

        # Temporal subsample
        verts_subsampled = verts_full[:: args.temporal_stride]  # (T', V, 3)
        T = verts_subsampled.shape[0]

        # Per-frame: normalize + compute HKS
        verts_normalized = np.empty((T, verts_subsampled.shape[1], 3), dtype=np.float32)
        hks_per_frame = np.empty(
            (T, verts_subsampled.shape[1], args.n_hks), dtype=np.float32
        )

        for t in range(T):
            verts_normalized[t] = normalize_vertices(verts_subsampled[t])
            hks_per_frame[t] = compute_frame_hks(
                verts_subsampled[t], faces, args.n_eigen, args.n_hks
            )

        # Save
        out_path = out_dir / f"{key}.npz"
        np.savez_compressed(
            out_path,
            vertices=verts_normalized,
            hks=hks_per_frame,
            subject=seq_data["subject"],
            action=seq_data["action"],
            n_frames=T,
        )

        total_frames_processed += T
        manifest_rows.append(
            {
                "filename": f"{key}.npz",
                "subject": seq_data["subject"],
                "action": seq_data["action"],
                "n_frames": T,
                "n_frames_original": seq_data["n_frames"],
            }
        )

        print(
            f"  [{i+1}/{len(sequences)}] {key}: {T} frames ({total_frames_processed} total)"
        )

    # Step 4: Write manifest
    print("\n[4/5] Writing manifest...")
    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "subject",
                "action",
                "n_frames",
                "n_frames_original",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Step 5: Summary
    print("\n[5/5] Summary")
    print("=" * 60)
    total_frames = sum(r["n_frames"] for r in manifest_rows)
    n_actions = len(set(r["action"] for r in manifest_rows))
    n_subjects = len(set(r["subject"] for r in manifest_rows))
    print(f"  Sequences: {len(manifest_rows)}")
    print(f"  Actions: {n_actions}")
    print(f"  Subjects: {n_subjects}")
    print(f"  Total frames (15fps): {total_frames}")
    print(f"  Vertices per frame: {mean_pose.shape[0]}")
    print(f"  Faces: {len(faces)}")
    print(f"  Eigenvectors: {args.n_eigen}")
    print(f"  HKS scales: {args.n_hks}")
    print(f"  Output dir: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
