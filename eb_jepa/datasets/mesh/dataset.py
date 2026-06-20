"""DFAUST temporal mesh dataset for Mesh JEPA training.

Loads preprocessed sequences (vertices + per-frame HKS) and serves
fixed-length temporal clips for self-supervised JEPA training.

Each clip is a sequence of frames with either HKS or XYZ features.
The dataset also provides action labels for downstream linear probing.
"""

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


ACTION_LABELS = {
    "chicken_wings": 0,
    "hips": 1,
    "jiggle_on_toes": 2,
    "jumping_jacks": 3,
    "knees": 4,
    "light_hopping_loose": 5,
    "light_hopping_stiff": 6,
    "one_leg_jump": 7,
    "one_leg_loose": 8,
    "punching": 9,
    "running_on_spot": 10,
    "shake_arms": 11,
    "shake_hips": 12,
    "shake_shoulders": 13,
}


class DFAUSTDataset(Dataset):
    """Temporal mesh clips from preprocessed DFAUST data.

    Returns clips of seq_len consecutive frames with chosen features.
    Features are either per-frame HKS (intrinsic, rotation-invariant)
    or XYZ vertex positions (extrinsic).
    """

    def __init__(
        self,
        data_dir,
        seq_len=16,
        feature_type="hks",
        subjects=None,
        actions=None,
    ):
        """
        Args:
            data_dir: Path to preprocessed data (datasets/dfaust/processed/)
            seq_len: Number of frames per clip
            feature_type: "hks" (16 channels) or "xyz" (3 channels)
            subjects: List of subject IDs to include (None = all)
            actions: List of action names to include (None = all)
        """
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.feature_type = feature_type

        # Load manifest
        manifest_path = self.data_dir / "manifest.csv"
        with open(manifest_path) as f:
            reader = csv.DictReader(f)
            entries = list(reader)

        # Filter by subjects/actions
        if subjects is not None:
            subjects = [str(s) for s in subjects]
            entries = [e for e in entries if e["subject"] in subjects]
        if actions is not None:
            entries = [e for e in entries if e["action"] in actions]

        # Build clip index: (sequence_idx, start_frame)
        self.sequences = []
        self.clips = []

        for seq_idx, entry in enumerate(entries):
            seq_path = self.data_dir / entry["filename"]
            seq_data = np.load(seq_path, allow_pickle=True)
            n_frames = int(entry["n_frames"])

            self.sequences.append(
                {
                    "vertices": seq_data["vertices"],  # (T, V, 3)
                    "hks": seq_data["hks"],  # (T, V, 16)
                    "action": str(seq_data["action"]),
                    "subject": str(seq_data["subject"]),
                    "n_frames": n_frames,
                }
            )

            # All valid clip start positions
            n_clips = max(0, n_frames - seq_len + 1)
            for start in range(n_clips):
                self.clips.append((seq_idx, start))

        # Load operators (for DiffusionNet)
        ops = np.load(self.data_dir / "operators.npz")
        self.eigenvalues = ops["eigenvalues"]  # (K,)
        self.eigenvectors = ops["eigenvectors"]  # (V, K)
        self.mass = ops["mass"]  # (V,)
        self.faces = ops["faces"]  # (F, 3)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        seq_idx, start = self.clips[idx]
        seq = self.sequences[seq_idx]

        end = start + self.seq_len

        # Select features based on type
        if self.feature_type == "hks":
            features = seq["hks"][start:end]  # (T, V, 16)
        elif self.feature_type == "xyz":
            features = seq["vertices"][start:end]  # (T, V, 3)
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

        # Action label for probing
        label = ACTION_LABELS[seq["action"]]

        return {
            "features": torch.from_numpy(features.copy()),  # (T, V, C)
            "label": label,
        }

    def get_operators(self):
        """Return Laplacian operators as tensors (for DiffusionNet)."""
        return {
            "eigenvalues": torch.from_numpy(self.eigenvalues),
            "eigenvectors": torch.from_numpy(self.eigenvectors),
            "mass": torch.from_numpy(self.mass),
        }


def make_loader(
    data_dir,
    seq_len=16,
    feature_type="hks",
    subjects=None,
    actions=None,
    batch_size=32,
    num_workers=4,
    shuffle=True,
):
    ds = DFAUSTDataset(
        data_dir=data_dir,
        seq_len=seq_len,
        feature_type=feature_type,
        subjects=subjects,
        actions=actions,
    )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
