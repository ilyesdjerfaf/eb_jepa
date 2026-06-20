"""Extract per-frame embeddings + corresponding XYZ point clouds.

Explodes the temporal dimension: each row = one frame's embedding + its mesh.
For a clip of T frames, produces T (embedding, point_cloud) pairs.

This builds the dataset needed by AtlasNet: we decode embeddings back into
3D meshes to evaluate how much geometric information the encoder retains.

Usage:
    uv run python -m examples.mesh_jepa.inference.extract_embeddings \
        --model_path checkpoints/mesh_jepa/large_model/hks/final.pth.tar \
        --data_dir /lustre/work/vivatech-discretizers/shared/dfaust/processed \
        --output_dir examples/mesh_jepa/inference/data/jepa_large_hks \
        --feature_type hks
"""

import csv
from pathlib import Path

import fire
import numpy as np
import torch
from tqdm import tqdm

from eb_jepa.logging import get_logger

logger = get_logger(__name__)


def extract(
    model_path: str,
    data_dir: str,
    output_dir: str,
    feature_type: str = "hks",
    batch_size: int = 8,
    seq_len: int = 16,
    device: str = "auto",
    max_sequences: int = None,
    subjects: list = None,
):
    """Extract per-frame embeddings and XYZ meshes from a trained encoder.

    The output is one .npz file with:
      - embeddings: (N_frames, D) — encoder output per frame
      - point_clouds: (N_frames, V, 3) — XYZ vertex positions per frame
      - labels: (N_frames,) — action label per frame

    Each frame is an independent row (temporal dimension exploded).
    """
    import yaml

    from eb_jepa.training_utils import setup_device

    device = setup_device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)

    # Load config from checkpoint dir
    ckpt_dir = Path(model_path).parent
    config_path = ckpt_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = None

    # Load checkpoint
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # Determine encoder type
    encoder_type = "diffusionnet"
    if cfg:
        encoder_type = cfg.get("model", {}).get("encoder_type", "diffusionnet")

    # Build encoder
    in_channels = 16 if feature_type == "hks" else 3

    if encoder_type == "mlp":
        from examples.mesh_jepa.encoder_mlp import MLPEncoder

        width = cfg["model"]["width"] if cfg else 768
        depth = cfg["model"]["depth"] if cfg else 9
        henc = cfg["model"]["henc"] if cfg else 256
        encoder = MLPEncoder(
            in_channels=in_channels, out_dim=henc, width=width, depth=depth
        )
    else:
        from examples.mesh_jepa.encoder import DiffusionNetEncoder

        width = cfg["model"]["width"] if cfg else 384
        depth = cfg["model"]["depth"] if cfg else 8
        henc = cfg["model"]["henc"] if cfg else 256
        n_eigen = cfg["model"]["n_eigen"] if cfg else 128
        encoder = DiffusionNetEncoder(
            in_channels=in_channels,
            out_dim=henc,
            width=width,
            depth=depth,
            n_eigen=n_eigen,
        )

    # Register operators
    ops = np.load(data_dir / "operators.npz")
    eigenvalues = torch.from_numpy(ops["eigenvalues"]).to(device)
    eigenvectors = torch.from_numpy(ops["eigenvectors"]).to(device)
    mass = torch.from_numpy(ops["mass"]).to(device)
    encoder.register_operators(eigenvalues, eigenvectors, mass)

    # Load weights
    state_dict = ckpt.get("model_state_dict", ckpt)
    encoder_keys = {
        k.replace("encoder.", ""): v
        for k, v in state_dict.items()
        if k.startswith("encoder.")
    }
    encoder.load_state_dict(encoder_keys, strict=False)
    encoder = encoder.to(device)
    encoder.eval()

    # Load sequences directly (bypass dataset class to get raw xyz + features)
    manifest_path = data_dir / "manifest.csv"
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        entries = list(reader)

    # Filter by subjects (to match JEPA training split)
    if subjects:
        subjects_str = [str(s) for s in subjects]
        entries = [e for e in entries if e["subject"] in subjects_str]

    if max_sequences:
        entries = entries[:max_sequences]

    from eb_jepa.datasets.mesh.dataset import ACTION_LABELS

    all_embeddings = []
    all_point_clouds = []
    all_labels = []

    logger.info(
        f"Extracting from {len(entries)} sequences, feature_type={feature_type}"
    )

    with torch.no_grad():
        for entry in tqdm(entries, desc="Sequences"):
            seq_data = np.load(data_dir / entry["filename"], allow_pickle=True)
            vertices = seq_data["vertices"]  # (T, V, 3) — always available
            n_frames = int(entry["n_frames"])
            action = str(seq_data["action"])
            label = ACTION_LABELS[action]

            # Select input features for encoder
            if feature_type == "hks":
                features = seq_data["hks"]  # (T, V, 16)
            else:
                features = vertices  # (T, V, 3)

            # Process in chunks of seq_len to avoid OOM
            for start in range(0, n_frames - seq_len + 1, seq_len):
                end = start + seq_len
                feat_chunk = torch.from_numpy(features[start:end].copy()).float()
                feat_chunk = feat_chunk.unsqueeze(0).to(device)  # (1, T, V, C)

                # Encode: (1, T, V, C) → (1, D, T, 1, 1)
                z = encoder(feat_chunk)
                z = z[0, :, :, 0, 0]  # (D, T)
                z = z.permute(1, 0)  # (T, D)

                all_embeddings.append(z.cpu().numpy())
                all_point_clouds.append(vertices[start:end])  # (T, V, 3)
                all_labels.append(np.full(seq_len, label))

    # Concatenate all frames
    embeddings = np.concatenate(all_embeddings, axis=0)  # (N, D)
    point_clouds = np.concatenate(all_point_clouds, axis=0)  # (N, V, 3)
    labels = np.concatenate(all_labels, axis=0)  # (N,)

    # Save
    np.save(output_dir / "embeddings.npy", embeddings)
    np.save(output_dir / "point_clouds.npy", point_clouds)
    np.save(output_dir / "labels.npy", labels)

    logger.info(
        f"Saved {embeddings.shape[0]} frames: "
        f"embeddings {embeddings.shape}, point_clouds {point_clouds.shape}"
    )
    logger.info(f"Output: {output_dir}")


if __name__ == "__main__":
    fire.Fire(extract)
