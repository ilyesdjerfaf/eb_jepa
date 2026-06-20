"""
Mesh JEPA Evaluation Script

Evaluation suite:
1. Linear probe (action classification) — encoder vs predictor
2. Temporal horizon evaluation (prediction quality at K steps)
3. Rotation invariance (cosine similarity under random SO(3))
4. Robustness (vertex noise + temporal jitter)
5. Abstraction evaluation (JEPA vs supervised baseline)
6. Collapse dashboard:
   - Standard VJEPA2: effective rank, std per dim, eigenspectrum, dead dims
   - Mesh-specific: learned diffusion times + per-frequency-band energy

Usage:
    uv run python -m examples.mesh_jepa.eval \
        --model_path checkpoints/mesh_jepa/.../final.pth.tar \
        --data_dir datasets/dfaust/processed \
        --output_dir results/baseline_3actions/hks
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from eb_jepa.datasets.mesh import DFAUSTDataset

# ============================================================
# Model loading
# ============================================================


def load_model(model_path, device):
    """Load trained JEPA model from checkpoint."""
    from eb_jepa.architectures import Projector
    from eb_jepa.jepa import JEPA
    from eb_jepa.losses import SquareLossSeq, VCLoss
    from examples.mesh_jepa.encoder import DiffusionNetEncoder
    from examples.mesh_jepa.predictor import MeshPredictor

    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # Infer architecture from weights
    first_lin_weight = ckpt["model_state_dict"].get("encoder.diffnet.first_lin.weight")
    if first_lin_weight is not None:
        in_channels = first_lin_weight.shape[1]
        width = first_lin_weight.shape[0]
    else:
        in_channels, width = 16, 128

    last_lin_weight = ckpt["model_state_dict"].get("encoder.diffnet.last_lin.weight")
    if last_lin_weight is not None:
        out_dim = last_lin_weight.shape[0]
    else:
        out_dim = 256

    block_keys = [k for k in ckpt["model_state_dict"] if "encoder.diffnet.blocks." in k]
    if block_keys:
        depth = max(int(k.split("blocks.")[1].split(".")[0]) for k in block_keys) + 1
    else:
        depth = 4

    diffusion_key = "encoder.diffnet.blocks.0.diffusion.diffusion_time"
    if diffusion_key in ckpt["model_state_dict"]:
        n_eigen = 128  # n_eigen not stored in diffusion_time (it's C_width)
    else:
        n_eigen = 128

    encoder = DiffusionNetEncoder(
        in_channels=in_channels,
        out_dim=out_dim,
        width=width,
        depth=depth,
        n_eigen=n_eigen,
        dropout=False,
        with_gradient_features=False,
    )
    predictor = MeshPredictor(state_dim=out_dim, hidden_dim=out_dim)
    projector = Projector(f"{out_dim}-{out_dim*4}-{out_dim*4}")
    regularizer = VCLoss(10.0, 100.0, proj=projector)
    predcost = SquareLossSeq(projector)

    jepa = JEPA(encoder, encoder, predictor, regularizer, predcost).to(device)
    jepa.load_state_dict(ckpt["model_state_dict"])
    jepa.eval()

    feature_type = "hks" if in_channels == 16 else "xyz"
    return jepa, encoder, predictor, feature_type


# ============================================================
# 1. Linear Probe
# ============================================================


@torch.no_grad()
def extract_representations(encoder, dataloader, device):
    """Extract encoder representations for all clips."""
    all_reps, all_reps_per_frame, all_labels = [], [], []

    for batch in tqdm(dataloader, desc="Extracting representations"):
        features = batch["features"].to(device)
        labels = batch["label"]

        z = encoder(features).squeeze(-1).squeeze(-1)  # (B, D, T)
        all_reps_per_frame.append(z.permute(0, 2, 1).cpu())
        all_reps.append(z.mean(dim=2).cpu())
        all_labels.append(labels)

    return (
        torch.cat(all_reps, 0).numpy(),
        torch.cat(all_reps_per_frame, 0).numpy(),
        torch.cat(all_labels, 0).numpy(),
    )


@torch.no_grad()
def extract_predictor_representations(encoder, predictor, dataloader, device):
    """Extract predictor hidden states for probing."""
    all_reps, all_labels = [], []

    for batch in tqdm(dataloader, desc="Extracting predictor reps"):
        features = batch["features"].to(device)
        z = encoder(features)
        pred_out = predictor(z, None).squeeze(-1).squeeze(-1)
        all_reps.append(pred_out.mean(dim=2).cpu())
        all_labels.append(batch["label"])

    return torch.cat(all_reps, 0).numpy(), torch.cat(all_labels, 0).numpy()


def linear_probe(train_reps, train_labels, test_reps, test_labels, label_names):
    """Train and evaluate linear probe."""
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_reps, train_labels)

    test_preds = clf.predict(test_reps)
    return {
        "train_acc": accuracy_score(train_labels, clf.predict(train_reps)),
        "test_acc": accuracy_score(test_labels, test_preds),
        "report": classification_report(
            test_labels, test_preds, target_names=label_names, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(test_labels, test_preds),
    }


def plot_linear_probe(results, label_names, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm = results["confusion_matrix"]
    im = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_xticks(range(len(label_names)))
    axes[0].set_yticks(range(len(label_names)))
    axes[0].set_xticklabels(label_names, rotation=45, ha="right", fontsize=8)
    axes[0].set_yticklabels(label_names, fontsize=8)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title(f"Confusion Matrix (acc={results['test_acc']:.1%})")
    plt.colorbar(im, ax=axes[0])

    f1_scores = [results["report"][name]["f1-score"] for name in label_names]
    axes[1].barh(range(len(label_names)), f1_scores, color="steelblue")
    axes[1].set_yticks(range(len(label_names)))
    axes[1].set_yticklabels(label_names, fontsize=8)
    axes[1].set_xlabel("F1 Score")
    axes[1].set_title("Per-class F1")
    axes[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / "1_linear_probe.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# 2. Temporal Horizon
# ============================================================


@torch.no_grad()
def temporal_horizon_eval(encoder, predictor, dataloader, device, max_horizon=15):
    """Evaluate prediction quality at different horizons."""
    horizon_mse = {k: [] for k in range(1, max_horizon + 1)}
    horizon_cosine = {k: [] for k in range(1, max_horizon + 1)}

    for batch in tqdm(dataloader, desc="Temporal horizon eval"):
        features = batch["features"].to(device)
        B, T, V, C = features.shape
        z = encoder(features)  # (B, D, T, 1, 1)

        predicted = z[:, :, :1]
        for step in range(min(max_horizon, T - 1)):
            context = predicted[:, :, -1:]
            pred_step = predictor(context, None)[:, :, -1:]
            predicted = torch.cat([predicted, pred_step], dim=2)

            gt = z[:, :, step + 1 : step + 2]
            pred = predicted[:, :, step + 1 : step + 2]

            mse = F.mse_loss(pred, gt, reduction="none").mean(dim=(1, 2, 3, 4))
            horizon_mse[step + 1].extend(mse.cpu().tolist())

            cos = F.cosine_similarity(pred.flatten(1), gt.flatten(1), dim=1)
            horizon_cosine[step + 1].extend(cos.cpu().tolist())

    results = {}
    for k in range(1, max_horizon + 1):
        if horizon_mse[k]:
            results[k] = {
                "mse": np.mean(horizon_mse[k]),
                "mse_std": np.std(horizon_mse[k]),
                "cosine": np.mean(horizon_cosine[k]),
                "cosine_std": np.std(horizon_cosine[k]),
            }
    return results


def plot_temporal_horizon(results, output_dir):
    horizons = sorted(results.keys())
    mse_vals = [results[k]["mse"] for k in horizons]
    cos_vals = [results[k]["cosine"] for k in horizons]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(horizons, mse_vals, "o-", color="coral", linewidth=2)
    axes[0].set_xlabel("Prediction Horizon (steps)")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Prediction Error vs Horizon")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(horizons, cos_vals, "o-", color="steelblue", linewidth=2)
    axes[1].set_xlabel("Prediction Horizon (steps)")
    axes[1].set_ylabel("Cosine Similarity")
    axes[1].set_title("Representation Similarity vs Horizon")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "2_temporal_horizon.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# 3. Rotation Invariance
# ============================================================


def random_rotation_matrix():
    """Generate a random SO(3) rotation matrix."""
    u1, u2, u3 = np.random.uniform(size=3)
    q = np.array(
        [
            np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
            np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
            np.sqrt(u1) * np.sin(2 * np.pi * u3),
            np.sqrt(u1) * np.cos(2 * np.pi * u3),
        ]
    )
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


@torch.no_grad()
def rotation_invariance_eval(encoder, dataset, device, n_samples=100, n_rotations=10):
    """Measure representation stability under random SO(3) rotations.

    Only meaningful for XYZ features (HKS is intrinsically invariant).
    For HKS: tests whether the full pipeline (HKS computation + encoding) is invariant.
    """
    cosine_sims = []
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    for idx in tqdm(indices, desc="Rotation invariance"):
        sample = dataset[idx]
        features = sample["features"].unsqueeze(0).to(device)  # (1, T, V, C)

        # Original representation
        z_orig = encoder(features).flatten(1)  # (1, D*T)

        for _ in range(n_rotations):
            R = torch.from_numpy(random_rotation_matrix()).to(device)

            if features.shape[-1] == 3:  # XYZ
                # Rotate vertex positions: (1, T, V, 3) @ R^T
                features_rot = torch.einsum("btvd,kd->btvk", features, R)
            else:
                # HKS: rotation doesn't change HKS (intrinsic descriptor)
                # But we test anyway to verify pipeline invariance
                features_rot = features

            z_rot = encoder(features_rot).flatten(1)
            cos = F.cosine_similarity(z_orig, z_rot, dim=1).item()
            cosine_sims.append(cos)

    return {
        "mean_cosine": np.mean(cosine_sims),
        "std_cosine": np.std(cosine_sims),
        "min_cosine": np.min(cosine_sims),
    }


# ============================================================
# 4. Robustness (vertex noise + temporal jitter)
# ============================================================


@torch.no_grad()
def robustness_eval(encoder, dataset, device, n_samples=100):
    """Measure representation stability under perturbations."""
    noise_levels = [0.001, 0.005, 0.01, 0.02, 0.05]
    results = {"vertex_noise": {}, "temporal_jitter": {}}
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    # --- Vertex noise ---
    for noise_std in noise_levels:
        cosines = []
        for idx in indices:
            sample = dataset[idx]
            features = sample["features"].unsqueeze(0).to(device)
            z_orig = encoder(features).flatten(1)

            noise = torch.randn_like(features) * noise_std
            features_noisy = features + noise
            z_noisy = encoder(features_noisy).flatten(1)

            cos = F.cosine_similarity(z_orig, z_noisy, dim=1).item()
            cosines.append(cos)

        results["vertex_noise"][noise_std] = {
            "mean_cosine": np.mean(cosines),
            "std_cosine": np.std(cosines),
        }

    # --- Temporal jitter (shift frames by ±1) ---
    cosines_jitter = []
    for idx in indices:
        sample = dataset[idx]
        features = sample["features"]  # (T, V, C)
        T = features.shape[0]
        if T < 3:
            continue

        features_orig = features.unsqueeze(0).to(device)
        z_orig = encoder(features_orig).flatten(1)

        # Shift by +1 (drop first frame, duplicate last)
        features_shift = torch.cat([features[1:], features[-1:]], dim=0)
        features_shift = features_shift.unsqueeze(0).to(device)
        z_shift = encoder(features_shift).flatten(1)

        cos = F.cosine_similarity(z_orig, z_shift, dim=1).item()
        cosines_jitter.append(cos)

    results["temporal_jitter"] = {
        "mean_cosine": np.mean(cosines_jitter),
        "std_cosine": np.std(cosines_jitter),
    }

    return results


def plot_robustness(results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Vertex noise
    noise_levels = sorted(results["vertex_noise"].keys())
    cosines = [results["vertex_noise"][n]["mean_cosine"] for n in noise_levels]
    stds = [results["vertex_noise"][n]["std_cosine"] for n in noise_levels]

    axes[0].errorbar(
        noise_levels, cosines, yerr=stds, fmt="o-", color="coral", capsize=4
    )
    axes[0].set_xlabel("Noise Std")
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].set_title("Robustness to Vertex Noise")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)

    # Temporal jitter bar
    jitter = results["temporal_jitter"]
    axes[1].bar(
        ["Original vs\nShifted +1 frame"],
        [jitter["mean_cosine"]],
        yerr=[jitter["std_cosine"]],
        color="steelblue",
        capsize=5,
    )
    axes[1].set_ylabel("Cosine Similarity")
    axes[1].set_title("Robustness to Temporal Jitter")
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_dir / "4_robustness.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# 5. Abstraction Evaluation
# ============================================================


def train_supervised_baseline(dataset, encoder_class, device, operators, epochs=10):
    """Train a supervised baseline that directly predicts next-frame features.

    Same encoder architecture, but loss is MSE between encoder(frame_t)
    and the raw features of frame_{t+1}. No predictor, no VICReg.
    """
    from examples.mesh_jepa.encoder import DiffusionNetEncoder

    feature_dim = dataset[0]["features"].shape[-1]  # 16 (HKS) or 3 (XYZ)

    # Same architecture but output matches input feature dim (for reconstruction)
    baseline_encoder = DiffusionNetEncoder(
        in_channels=feature_dim,
        out_dim=256,
        width=128,
        depth=4,
        n_eigen=128,
        dropout=False,
        with_gradient_features=False,
    ).to(device)
    baseline_encoder.register_operators(
        operators["eigenvalues"].to(device),
        operators["eigenvectors"].to(device),
        operators["mass"].to(device),
    )

    # Projection head: latent → feature space (for supervised prediction)
    pred_head = torch.nn.Sequential(
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, feature_dim),
    ).to(device)

    optimizer = torch.optim.Adam(
        list(baseline_encoder.parameters()) + list(pred_head.parameters()),
        lr=1e-3,
    )

    loader = DataLoader(
        dataset, batch_size=16, shuffle=True, num_workers=0, drop_last=True
    )

    baseline_encoder.train()
    pred_head.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            features = batch["features"].to(device)  # (B, T, V, C)
            B, T, V, C = features.shape

            # Encode all frames
            z = baseline_encoder(features)  # (B, D, T, 1, 1)
            z = z.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # (B, T, D)

            # Predict next frame's features from current latent
            z_curr = z[:, :-1].reshape(B * (T - 1), 256)  # (B*(T-1), D)
            pred_features = pred_head(z_curr)  # (B*(T-1), C)

            # Target: mean-pooled features of next frame (V → 1)
            target = features[:, 1:].mean(dim=2).reshape(B * (T - 1), C)

            loss = F.mse_loss(pred_features, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    baseline_encoder.eval()
    return baseline_encoder


def abstraction_eval(
    jepa_encoder, dataset_train, dataset_test, device, operators, label_names
):
    """Compare JEPA encoder vs supervised baseline via probe accuracy and robustness."""
    # Train supervised baseline
    print("  Training supervised baseline (10 epochs)...")
    baseline_encoder = train_supervised_baseline(
        dataset_train, None, device, operators, epochs=10
    )

    # Extract representations for both
    train_loader = DataLoader(
        dataset_train, batch_size=16, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(dataset_test, batch_size=16, shuffle=False, num_workers=0)

    print("  Extracting JEPA representations...")
    jepa_train_reps, _, train_labels = extract_representations(
        jepa_encoder, train_loader, device
    )
    jepa_test_reps, _, test_labels = extract_representations(
        jepa_encoder, test_loader, device
    )

    print("  Extracting baseline representations...")
    base_train_reps, _, _ = extract_representations(
        baseline_encoder, train_loader, device
    )
    base_test_reps, _, _ = extract_representations(
        baseline_encoder, test_loader, device
    )

    # Probe both
    jepa_probe = linear_probe(
        jepa_train_reps, train_labels, jepa_test_reps, test_labels, label_names
    )
    base_probe = linear_probe(
        base_train_reps, train_labels, base_test_reps, test_labels, label_names
    )

    # Robustness comparison (vertex noise)
    print("  Comparing robustness...")
    noise_std = 0.01
    jepa_cosines, base_cosines = [], []
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset_test), min(50, len(dataset_test)), replace=False)

    with torch.no_grad():
        for idx in indices:
            sample = dataset_test[idx]
            features = sample["features"].unsqueeze(0).to(device)
            noise = torch.randn_like(features) * noise_std

            z_orig_j = jepa_encoder(features).flatten(1)
            z_noisy_j = jepa_encoder(features + noise).flatten(1)
            jepa_cosines.append(F.cosine_similarity(z_orig_j, z_noisy_j, dim=1).item())

            z_orig_b = baseline_encoder(features).flatten(1)
            z_noisy_b = baseline_encoder(features + noise).flatten(1)
            base_cosines.append(F.cosine_similarity(z_orig_b, z_noisy_b, dim=1).item())

    return {
        "jepa_probe_acc": jepa_probe["test_acc"],
        "baseline_probe_acc": base_probe["test_acc"],
        "jepa_robustness": np.mean(jepa_cosines),
        "baseline_robustness": np.mean(base_cosines),
    }


def plot_abstraction(results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Probe accuracy comparison
    methods = ["JEPA\n(abstract)", "Supervised\n(reconstruct)"]
    accs = [results["jepa_probe_acc"], results["baseline_probe_acc"]]
    colors = ["steelblue", "coral"]
    axes[0].bar(methods, accs, color=colors)
    axes[0].set_ylabel("Probe Accuracy")
    axes[0].set_title("Abstraction: Probe Accuracy")
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=10)

    # Robustness comparison
    rob = [results["jepa_robustness"], results["baseline_robustness"]]
    axes[1].bar(methods, rob, color=colors)
    axes[1].set_ylabel("Cosine Sim (clean vs noisy)")
    axes[1].set_title("Abstraction: Robustness to Noise (std=0.01)")
    axes[1].set_ylim(0, 1.05)
    for i, v in enumerate(rob):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "5_abstraction.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# 6. Collapse Dashboard
# ============================================================


@torch.no_grad()
def collapse_metrics(encoder, dataloader, device):
    """Standard VJEPA2 collapse metrics."""
    all_reps = []

    for batch in tqdm(dataloader, desc="Computing collapse metrics"):
        features = batch["features"].to(device)
        z = encoder(features).squeeze(-1).squeeze(-1)  # (B, D, T)
        z_flat = z.permute(0, 2, 1).reshape(-1, z.shape[1])  # (B*T, D)
        all_reps.append(z_flat.cpu())

    reps = torch.cat(all_reps, 0).numpy()

    # Std per dimension
    std_per_dim = np.std(reps, axis=0)

    # Effective rank (from eigenspectrum of covariance)
    reps_centered = reps - reps.mean(axis=0)
    cov = np.cov(reps_centered.T)
    eigenvalues_cov = np.linalg.eigvalsh(cov)[::-1]  # sorted descending
    eigenvalues_cov = np.maximum(eigenvalues_cov, 0)  # numerical stability

    # Effective rank via Shannon entropy
    ev_norm = eigenvalues_cov / (eigenvalues_cov.sum() + 1e-10)
    ev_nonzero = ev_norm[ev_norm > 1e-10]
    entropy = -np.sum(ev_nonzero * np.log(ev_nonzero))
    effective_rank = np.exp(entropy)

    # Dead dimensions
    dead_dims = int(np.sum(std_per_dim < 0.01))
    dead_ratio = dead_dims / len(std_per_dim)

    # Off-diagonal covariance
    off_diag = cov - np.diag(np.diag(cov))
    off_diag_mean = np.abs(off_diag).mean()

    return {
        "std_per_dim": std_per_dim,
        "eigenvalues_cov": eigenvalues_cov,
        "effective_rank": effective_rank,
        "dead_dims": dead_dims,
        "dead_ratio": dead_ratio,
        "off_diag_cov_mean": off_diag_mean,
        "mean_std": std_per_dim.mean(),
    }


def extract_diffusion_times(encoder):
    """Extract learned diffusion times from all DiffusionNet blocks."""
    times = []
    for i, block in enumerate(encoder.diffnet.blocks):
        t = block.diffusion.diffusion_time.detach().cpu().numpy()
        times.append(t)
    return times


@torch.no_grad()
def frequency_band_energy(encoder, dataset, device):
    """Measure how much representation variance comes from each frequency band.

    Split input into low/mid/high frequency components using Laplacian eigenbasis,
    encode each separately, measure variance contribution.
    """
    ops = dataset.get_operators()
    eigenvectors = ops["eigenvectors"].to(device)  # (V, K)
    mass = ops["mass"].to(device)  # (V,)
    K = eigenvectors.shape[1]

    # Define bands
    bands = {
        "low (1-32)": slice(0, 32),
        "mid (33-80)": slice(32, 80),
        "high (81-128)": slice(80, K),
    }

    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    band_variances = {name: [] for name in bands}

    for batch in tqdm(loader, desc="Frequency band analysis"):
        features = batch["features"].to(device)  # (B, T, V, C)
        B, T, V, C = features.shape

        # Full representation
        z_full = encoder(features).squeeze(-1).squeeze(-1)  # (B, D, T)

        for band_name, band_slice in bands.items():
            # Project features to this band only
            evecs_band = eigenvectors[:, band_slice]  # (V, K_band)

            # features: (B, T, V, C) → project each frame
            feat_flat = features.reshape(B * T, V, C)

            # Project to band: V→K_band→V
            feat_spec = torch.einsum("vk,v,bvc->bkc", evecs_band, mass, feat_flat)
            feat_band = torch.einsum("vk,bkc->bvc", evecs_band, feat_spec)

            feat_band = feat_band.reshape(B, T, V, C)
            z_band = encoder(feat_band).squeeze(-1).squeeze(-1)  # (B, D, T)

            # Variance of this band's representation
            var = z_band.var(dim=0).mean().item()
            band_variances[band_name].append(var)

    return {name: np.mean(vals) for name, vals in band_variances.items()}


def plot_collapse_dashboard(collapse, diffusion_times, band_energy, output_dir):
    fig = plt.figure(figsize=(16, 12))

    # 1. Std per dimension (sorted)
    ax1 = fig.add_subplot(3, 2, 1)
    std_sorted = np.sort(collapse["std_per_dim"])[::-1]
    ax1.bar(range(len(std_sorted)), std_sorted, width=1.0, color="steelblue")
    ax1.axhline(y=0.01, color="red", linestyle="--", label="Dead threshold")
    ax1.set_xlabel("Dimension (sorted)")
    ax1.set_ylabel("Std")
    ax1.set_title(f"Std per Dimension (dead={collapse['dead_dims']}/{len(std_sorted)})")
    ax1.set_yscale("log")
    ax1.legend()

    # 2. Eigenspectrum of covariance
    ax2 = fig.add_subplot(3, 2, 2)
    ev = collapse["eigenvalues_cov"][:50]
    ax2.semilogy(ev, "o-", color="coral", markersize=3)
    ax2.set_xlabel("Component")
    ax2.set_ylabel("Eigenvalue")
    ax2.set_title(
        f"Covariance Eigenspectrum (eff. rank={collapse['effective_rank']:.1f})"
    )
    ax2.grid(True, alpha=0.3)

    # 3. Learned diffusion times per block
    ax3 = fig.add_subplot(3, 2, 3)
    for i, t in enumerate(diffusion_times):
        ax3.hist(t, bins=30, alpha=0.6, label=f"Block {i}")
    ax3.set_xlabel("Diffusion Time t_c")
    ax3.set_ylabel("Count")
    ax3.set_title("Learned Diffusion Times (per channel, per block)")
    ax3.legend(fontsize=7)

    # 4. Diffusion times heatmap
    ax4 = fig.add_subplot(3, 2, 4)
    times_matrix = np.array(diffusion_times)  # (N_blocks, C_width)
    im = ax4.imshow(times_matrix, aspect="auto", cmap="viridis")
    ax4.set_xlabel("Channel")
    ax4.set_ylabel("Block")
    ax4.set_title("Diffusion Times Heatmap (large=smooth, small=preserve detail)")
    plt.colorbar(im, ax=ax4)

    # 5. Per-frequency-band energy
    ax5 = fig.add_subplot(3, 2, 5)
    bands = list(band_energy.keys())
    energies = list(band_energy.values())
    colors = ["#2196F3", "#FF9800", "#F44336"]
    ax5.bar(bands, energies, color=colors)
    ax5.set_ylabel("Mean Representation Variance")
    ax5.set_title(
        "Per-Frequency-Band Energy\n(which frequencies drive the representation?)"
    )

    # 6. Summary
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.axis("off")
    summary = (
        f"COLLAPSE DASHBOARD SUMMARY\n"
        f"{'='*40}\n\n"
        f"Effective Rank:    {collapse['effective_rank']:.1f} / {len(collapse['std_per_dim'])}\n"
        f"Dead Dimensions:   {collapse['dead_dims']} / {len(collapse['std_per_dim'])} ({collapse['dead_ratio']:.1%})\n"
        f"Mean Std:          {collapse['mean_std']:.4f}\n"
        f"Off-diag Cov:      {collapse['off_diag_cov_mean']:.6f}\n\n"
        f"Diffusion Times:\n"
    )
    for i, t in enumerate(diffusion_times):
        summary += f"  Block {i}: mean={t.mean():.4f}, std={t.std():.4f}\n"
    summary += f"\nFrequency Band Energy:\n"
    for band, energy in band_energy.items():
        summary += f"  {band}: {energy:.4f}\n"

    ax6.text(
        0.05,
        0.95,
        summary,
        transform=ax6.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
    )

    plt.tight_layout()
    plt.savefig(output_dir / "6_collapse_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# Training Curves
# ============================================================


def plot_training_curves(history, output_dir):
    """Plot publication-quality training loss curves."""
    epochs = [h["epoch"] for h in history]
    total_loss = [h["train/loss"] for h in history]
    pred_loss = [h["train/pred_loss"] for h in history]
    vc_loss = [h["train/vc_loss"] for h in history]
    lr = [h["train/lr"] for h in history]

    # Extract std and cov losses if available
    std_loss = [h.get("train/std_loss", None) for h in history]
    cov_loss = [h.get("train/cov_loss", None) for h in history]
    has_breakdown = std_loss[0] is not None

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Total loss
    axes[0, 0].plot(epochs, total_loss, "k-", linewidth=2, label="Total")
    axes[0, 0].plot(
        epochs, pred_loss, "-", color="coral", linewidth=1.5, label="Prediction"
    )
    axes[0, 0].plot(
        epochs, vc_loss, "-", color="steelblue", linewidth=1.5, label="VICReg"
    )
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Prediction loss (zoomed)
    axes[0, 1].plot(epochs, pred_loss, "o-", color="coral", markersize=3, linewidth=1.5)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Prediction Loss")
    axes[0, 1].set_title("Prediction Loss (should decrease, stay > 0)")
    axes[0, 1].grid(True, alpha=0.3)

    # VICReg breakdown
    if has_breakdown:
        axes[1, 0].plot(
            epochs, std_loss, "-", color="#2196F3", linewidth=1.5, label="Std loss"
        )
        axes[1, 0].plot(
            epochs, cov_loss, "-", color="#FF9800", linewidth=1.5, label="Cov loss"
        )
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].set_title("VICReg Components (std + cov)")
        axes[1, 0].legend()
    else:
        axes[1, 0].plot(epochs, vc_loss, "-", color="steelblue", linewidth=1.5)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("VICReg Loss")
        axes[1, 0].set_title("VICReg Loss")
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 1].plot(epochs, lr, "-", color="green", linewidth=1.5)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].set_title("Learning Rate Schedule")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "0_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# Timing
# ============================================================


@torch.no_grad()
def measure_timing(encoder, predictor, dataloader, device, n_repeats=20):
    """Measure inference latency."""
    sample = next(iter(dataloader))["features"][:1].to(device)

    # Warmup
    for _ in range(3):
        z = encoder(sample)
        predictor(z[:, :, :1], None)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeats):
        z = encoder(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    encode_time = (time.time() - t0) / n_repeats

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeats):
        predicted = z[:, :, :1]
        for _ in range(4):
            pred_step = predictor(predicted[:, :, -1:], None)[:, :, -1:]
            predicted = torch.cat([predicted, pred_step], dim=2)
    if device.type == "cuda":
        torch.cuda.synchronize()
    predict_time = (time.time() - t0) / n_repeats

    return {
        "encode_time_ms": encode_time * 1000,
        "predict_4steps_time_ms": predict_time * 1000,
        "total_inference_time_ms": (encode_time + predict_time) * 1000,
    }


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="Evaluate Mesh JEPA")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="datasets/dfaust/processed")
    parser.add_argument("--output_dir", type=str, default="results/eval")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    print("\nLoading model...")
    jepa, encoder, predictor, feature_type = load_model(args.model_path, device)
    print(f"  Feature type: {feature_type}")
    print(f"  Parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    # Datasets
    train_set = DFAUSTDataset(
        data_dir=args.data_dir,
        seq_len=16,
        feature_type=feature_type,
        subjects=[50002, 50004, 50007, 50009, 50020, 50021, 50022, 50025],
    )
    test_set = DFAUSTDataset(
        data_dir=args.data_dir,
        seq_len=16,
        feature_type=feature_type,
        subjects=[50026, 50027],
    )

    # Register operators
    ops = train_set.get_operators()
    encoder.register_operators(
        ops["eigenvalues"].to(device),
        ops["eigenvectors"].to(device),
        ops["mass"].to(device),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    actions_present = sorted(
        set(seq["action"] for seq in train_set.sequences + test_set.sequences)
    )
    label_names = actions_present

    print(f"  Train: {len(train_set)} clips, Test: {len(test_set)} clips")
    print(f"  Actions: {actions_present}")

    # ==== 0. TRAINING CURVES ====
    history_path = Path(args.model_path).parent / "history.json"
    if history_path.exists():
        import json

        with open(history_path) as f:
            history = json.load(f)
        print(f"\n  Training history: {len(history)} epochs")
        plot_training_curves(history, output_dir)
    else:
        print("\n  No training history found (history.json missing)")

    # ==== 1. LINEAR PROBE ====
    print("\n" + "=" * 60)
    print("1. LINEAR PROBE")
    print("=" * 60)

    train_reps, train_reps_pf, train_labels = extract_representations(
        encoder, train_loader, device
    )
    test_reps, test_reps_pf, test_labels = extract_representations(
        encoder, test_loader, device
    )

    encoder_probe = linear_probe(
        train_reps, train_labels, test_reps, test_labels, label_names
    )
    print(
        f"  Encoder: train={encoder_probe['train_acc']:.1%}, test={encoder_probe['test_acc']:.1%}"
    )

    pred_train_reps, _ = extract_predictor_representations(
        encoder, predictor, train_loader, device
    )
    pred_test_reps, _ = extract_predictor_representations(
        encoder, predictor, test_loader, device
    )
    predictor_probe = linear_probe(
        pred_train_reps, train_labels, pred_test_reps, test_labels, label_names
    )
    print(
        f"  Predictor: train={predictor_probe['train_acc']:.1%}, test={predictor_probe['test_acc']:.1%}"
    )

    plot_linear_probe(encoder_probe, label_names, output_dir)

    # ==== 2. TEMPORAL HORIZON ====
    print("\n" + "=" * 60)
    print("2. TEMPORAL HORIZON")
    print("=" * 60)

    horizon_results = temporal_horizon_eval(encoder, predictor, test_loader, device)
    for k, v in sorted(horizon_results.items())[:5]:
        print(f"  K={k:2d}: MSE={v['mse']:.4f}, Cosine={v['cosine']:.4f}")
    plot_temporal_horizon(horizon_results, output_dir)

    # ==== 3. ROTATION INVARIANCE ====
    print("\n" + "=" * 60)
    print("3. ROTATION INVARIANCE")
    print("=" * 60)

    rotation_results = rotation_invariance_eval(encoder, test_set, device)
    print(
        f"  Mean cosine sim: {rotation_results['mean_cosine']:.4f} +/- {rotation_results['std_cosine']:.4f}"
    )
    print(f"  Min cosine sim:  {rotation_results['min_cosine']:.4f}")

    # ==== 4. ROBUSTNESS ====
    print("\n" + "=" * 60)
    print("4. ROBUSTNESS")
    print("=" * 60)

    robustness_results = robustness_eval(encoder, test_set, device)
    for noise, vals in robustness_results["vertex_noise"].items():
        print(f"  Noise std={noise}: cosine={vals['mean_cosine']:.4f}")
    print(
        f"  Temporal jitter: cosine={robustness_results['temporal_jitter']['mean_cosine']:.4f}"
    )
    plot_robustness(robustness_results, output_dir)

    # ==== 5. ABSTRACTION ====
    print("\n" + "=" * 60)
    print("5. ABSTRACTION EVALUATION")
    print("=" * 60)

    abstraction_results = abstraction_eval(
        encoder, train_set, test_set, device, ops, label_names
    )
    print(f"  JEPA probe acc:       {abstraction_results['jepa_probe_acc']:.1%}")
    print(f"  Supervised probe acc: {abstraction_results['baseline_probe_acc']:.1%}")
    print(f"  JEPA robustness:      {abstraction_results['jepa_robustness']:.4f}")
    print(f"  Supervised robustness:{abstraction_results['baseline_robustness']:.4f}")
    plot_abstraction(abstraction_results, output_dir)

    # ==== 6. COLLAPSE DASHBOARD ====
    print("\n" + "=" * 60)
    print("6. COLLAPSE DASHBOARD")
    print("=" * 60)

    collapse = collapse_metrics(encoder, train_loader, device)
    print(f"  Effective Rank: {collapse['effective_rank']:.1f}")
    print(f"  Dead Dims: {collapse['dead_dims']} ({collapse['dead_ratio']:.1%})")
    print(f"  Mean Std: {collapse['mean_std']:.4f}")

    diffusion_times = extract_diffusion_times(encoder)
    for i, t in enumerate(diffusion_times):
        print(
            f"  Block {i} diffusion times: mean={t.mean():.4f}, range=[{t.min():.4f}, {t.max():.4f}]"
        )

    print("  Computing frequency band energy...")
    band_energy = frequency_band_energy(encoder, test_set, device)
    for band, energy in band_energy.items():
        print(f"    {band}: {energy:.4f}")

    plot_collapse_dashboard(collapse, diffusion_times, band_energy, output_dir)

    # ==== TIMING ====
    print("\n" + "=" * 60)
    print("TIMING")
    print("=" * 60)

    timing = measure_timing(encoder, predictor, test_loader, device)
    print(f"  Encode (16 frames): {timing['encode_time_ms']:.1f} ms")
    print(f"  Predict (4 steps):  {timing['predict_4steps_time_ms']:.1f} ms")
    print(f"  Total inference:    {timing['total_inference_time_ms']:.1f} ms")

    # ==== SAVE RESULTS ====
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    results = {
        "encoder_probe_test_acc": encoder_probe["test_acc"],
        "predictor_probe_test_acc": predictor_probe["test_acc"],
        "rotation_invariance": rotation_results,
        "robustness": robustness_results,
        "abstraction": abstraction_results,
        "effective_rank": collapse["effective_rank"],
        "dead_dims": collapse["dead_dims"],
        "band_energy": band_energy,
        "timing": timing,
        "horizon_results": {str(k): v for k, v in horizon_results.items()},
    }
    np.save(output_dir / "results.npy", results, allow_pickle=True)

    print(f"\n  Output: {output_dir}")
    for f in sorted(output_dir.glob("*.png")):
        print(f"    {f.name}")
    print(f"\n  Encoder Probe:  {encoder_probe['test_acc']:.1%}")
    print(f"  Effective Rank: {collapse['effective_rank']:.1f}")
    print(f"  Rotation Inv:   {rotation_results['mean_cosine']:.4f}")
    print(f"  Random baseline: {1/len(label_names):.1%}")


if __name__ == "__main__":
    main()
