"""
Explore and visualize the DFAUST dataset.

Usage:
    uv run python -m examples.mesh_jepa.explore_dfaust --data_dir datasets/dfaust/raw

Expected data_dir contents (from dfaust.is.tue.mpg.de registrations download):
    registrations_m.hdf5  (male subjects)
    registrations_f.hdf5  (female subjects)

    OR individual .npz/.npy files per sequence.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_hdf5_registrations(data_dir: Path):
    """Load DFAUST registrations from HDF5 files.

    Format: each dataset is keyed as '{subject}_{action}' with shape (6890, 3, T).
    We transpose to (T, 6890, 3) for consistency.
    """
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
                # Shape is (V, 3, T) — transpose to (T, V, 3)
                verts = np.array(data).transpose(2, 0, 1)
                parts = key.split("_", 1)
                sequences[key] = {
                    "vertices": verts,
                    "subject": parts[0],
                    "action": parts[1] if len(parts) > 1 else key,
                    "n_frames": verts.shape[0],
                }
    return sequences, faces


def load_npz_registrations(data_dir: Path):
    """Load DFAUST registrations from individual .npz files."""
    sequences = {}
    for fname in sorted(data_dir.glob("*.npz")):
        data = np.load(fname)
        key = fname.stem
        parts = key.split("_", 1)
        sequences[key] = {
            "vertices": data["vertices"] if "vertices" in data else data[data.files[0]],
            "subject": parts[0] if len(parts) > 1 else "unknown",
            "action": parts[1] if len(parts) > 1 else key,
            "n_frames": (
                data["vertices"].shape[0]
                if "vertices" in data
                else data[data.files[0]].shape[0]
            ),
        }
    return sequences


def load_registrations(data_dir: Path):
    """Auto-detect and load registrations. Returns (sequences, faces)."""
    hdf5_files = list(data_dir.glob("*.hdf5")) + list(data_dir.glob("*.h5"))
    npz_files = list(data_dir.glob("*.npz"))

    if hdf5_files:
        print(f"Found {len(hdf5_files)} HDF5 file(s)")
        return load_hdf5_registrations(data_dir)
    elif npz_files:
        print(f"Found {len(npz_files)} NPZ file(s)")
        return load_npz_registrations(data_dir), None
    else:
        print(f"Contents of {data_dir}:")
        for f in sorted(data_dir.iterdir()):
            print(f"  {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
        raise FileNotFoundError(
            f"No .hdf5 or .npz registration files found in {data_dir}. "
            "Download registrations from dfaust.is.tue.mpg.de"
        )


def print_dataset_stats(sequences: dict):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("DFAUST Dataset Summary")
    print("=" * 60)

    subjects = sorted(set(s["subject"] for s in sequences.values()))
    actions = sorted(set(s["action"] for s in sequences.values()))
    total_frames = sum(s["n_frames"] for s in sequences.values())

    print(f"Total sequences: {len(sequences)}")
    print(f"Total frames:    {total_frames}")
    print(f"Subjects ({len(subjects)}): {subjects}")
    print(f"Actions ({len(actions)}):")
    for a in actions:
        print(f"  - {a}")

    sample = next(iter(sequences.values()))
    v = sample["vertices"]
    print(f"\nMesh topology:")
    print(f"  Vertices per frame: {v.shape[1]}")
    print(f"  Coordinates:        {v.shape[2]}D")

    print(f"\nFrames per sequence:")
    frame_counts = [s["n_frames"] for s in sequences.values()]
    print(
        f"  Min: {min(frame_counts)}, Max: {max(frame_counts)}, "
        f"Mean: {np.mean(frame_counts):.0f}, Total: {sum(frame_counts)}"
    )

    print("\nSubject × Action matrix:")
    print(f"{'':>12}", end="")
    for a in actions[:7]:
        print(f"{a[:8]:>9}", end="")
    if len(actions) > 7:
        print(" ...")
    print()
    for subj in subjects:
        print(f"{subj:>12}", end="")
        for a in actions[:7]:
            key = f"{subj}_{a}"
            if key in sequences:
                print(f"{sequences[key]['n_frames']:>9}", end="")
            else:
                print(f"{'—':>9}", end="")
        print()


def visualize_mesh_frame(vertices, faces=None, title="Mesh Frame", ax=None):
    """Plot a single mesh frame as 3D scatter (or wireframe if faces provided)."""
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        s=0.3,
        alpha=0.5,
        c=vertices[:, 1],
        cmap="viridis",
    )
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    max_range = (vertices.max(axis=0) - vertices.min(axis=0)).max() / 2
    mid = vertices.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    return ax


def visualize_sequence_strip(vertices_seq, n_frames=8, title=""):
    """Show N evenly-spaced frames from a sequence as subplots."""
    T = vertices_seq.shape[0]
    indices = np.linspace(0, T - 1, n_frames, dtype=int)

    fig = plt.figure(figsize=(4 * n_frames, 4))
    fig.suptitle(title, fontsize=14)
    for i, idx in enumerate(indices):
        ax = fig.add_subplot(1, n_frames, i + 1, projection="3d")
        visualize_mesh_frame(vertices_seq[idx], title=f"t={idx}", ax=ax)
        ax.set_axis_off()
    plt.tight_layout()
    return fig


def visualize_motion_heatmap(vertices_seq, title="Per-vertex displacement"):
    """Show per-vertex motion magnitude over time."""
    diffs = np.linalg.norm(np.diff(vertices_seq, axis=0), axis=-1)
    mean_motion = diffs.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im = axes[0].imshow(diffs[:100].T, aspect="auto", cmap="hot")
    axes[0].set_xlabel("Time (frame)")
    axes[0].set_ylabel("Vertex index")
    axes[0].set_title("Displacement magnitude (first 100 frames)")
    plt.colorbar(im, ax=axes[0])

    axes[1].hist(mean_motion, bins=50, alpha=0.7)
    axes[1].set_xlabel("Mean displacement per frame")
    axes[1].set_ylabel("Count (vertices)")
    axes[1].set_title("Distribution of vertex mobility")
    axes[1].axvline(
        mean_motion.mean(),
        color="r",
        linestyle="--",
        label=f"mean={mean_motion.mean():.4f}",
    )
    axes[1].legend()

    fig.suptitle(title)
    plt.tight_layout()
    return fig


def visualize_action_comparison(sequences: dict, n_frames=5):
    """Show one frame from different actions for the same subject."""
    subjects = sorted(set(s["subject"] for s in sequences.values()))
    subject = subjects[0]

    subject_seqs = {k: v for k, v in sequences.items() if v["subject"] == subject}
    actions = sorted(subject_seqs.keys())[:6]

    fig = plt.figure(figsize=(4 * len(actions), 4))
    fig.suptitle(f"Subject {subject} — different actions (middle frame)", fontsize=14)
    for i, key in enumerate(actions):
        seq = subject_seqs[key]
        mid = seq["n_frames"] // 2
        ax = fig.add_subplot(1, len(actions), i + 1, projection="3d")
        visualize_mesh_frame(seq["vertices"][mid], title=seq["action"][:12], ax=ax)
        ax.set_axis_off()
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Explore DFAUST dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets/dfaust/raw",
        help="Path to DFAUST registrations",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="datasets/dfaust/figures",
        help="Where to save visualization figures",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Don't call plt.show() (for headless environments)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print(f"Please download DFAUST registrations to: {data_dir}")
        print("  1. Go to https://dfaust.is.tue.mpg.de")
        print("  2. Register / Log in")
        print("  3. Download 'Registrations' (HDF5 format)")
        print(f"  4. Place files in {data_dir}/")
        return

    sequences, faces = load_registrations(data_dir)
    print_dataset_stats(sequences)

    if faces is not None:
        print(f"\nMesh faces: {faces.shape} (connectivity shared across all frames)")
        np.save(save_dir / "faces.npy", faces)
        print(f"  Saved faces to {save_dir}/faces.npy")

    # Pick a sample sequence for visualization
    sample_key = list(sequences.keys())[0]
    sample = sequences[sample_key]
    print(f"\nVisualizing: {sample_key} ({sample['n_frames']} frames)")

    # 1. Frame strip
    fig = visualize_sequence_strip(
        sample["vertices"], n_frames=8, title=f"Sequence: {sample_key}"
    )
    fig.savefig(save_dir / "frame_strip.png", dpi=100, bbox_inches="tight")
    print(f"  Saved: {save_dir}/frame_strip.png")

    # 2. Motion heatmap
    fig = visualize_motion_heatmap(sample["vertices"], title=f"Motion: {sample_key}")
    fig.savefig(save_dir / "motion_heatmap.png", dpi=100, bbox_inches="tight")
    print(f"  Saved: {save_dir}/motion_heatmap.png")

    # 3. Action comparison
    fig = visualize_action_comparison(sequences)
    fig.savefig(save_dir / "action_comparison.png", dpi=100, bbox_inches="tight")
    print(f"  Saved: {save_dir}/action_comparison.png")

    # 4. Basic sanity checks
    print("\n" + "=" * 60)
    print("Sanity Checks")
    print("=" * 60)
    v = sample["vertices"]
    print(f"  Vertex range X: [{v[:, :, 0].min():.3f}, {v[:, :, 0].max():.3f}]")
    print(f"  Vertex range Y: [{v[:, :, 1].min():.3f}, {v[:, :, 1].max():.3f}]")
    print(f"  Vertex range Z: [{v[:, :, 2].min():.3f}, {v[:, :, 2].max():.3f}]")
    print(
        f"  Mean inter-frame displacement: {np.linalg.norm(np.diff(v, axis=0), axis=-1).mean():.5f}"
    )
    print(
        f"  Max inter-frame displacement:  {np.linalg.norm(np.diff(v, axis=0), axis=-1).max():.5f}"
    )

    if not args.no_show:
        plt.show()

    print("\nDone! Next steps:")
    print("  1. Run preprocessing: python -m examples.mesh_jepa.preprocess")
    print("  2. Verify HKS features look reasonable")
    print("  3. Start JEPA training")


if __name__ == "__main__":
    main()
