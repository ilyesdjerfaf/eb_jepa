"""Gray-Scott reaction-diffusion (The Well) as a 2D physical video for JEPA.

Each HDF5 file (one F,k regime) holds ``t0_fields/A`` and ``t0_fields/B`` of shape
``(n_traj, 1001, 128, 128)`` float32. A training item is a clip of ``n_frames``
frames with the two chemical fields stacked as channels -> ``[2, T, 128, 128]``,
z-scored with the dataset stats. ``time_stride`` spaces frames so the slow
dynamics are visible between them.

Data loading is PROVIDED (plumbing). The modelling choices on top of these clips
(encoder, temporal JEPA objective, eval decoder/metric) live in
``examples/gray_scott/`` and are where the ``# TODO``s are.
"""

import glob
import os
from dataclasses import dataclass

import numpy as np
import torch

try:
    import h5py
except ImportError:
    h5py = None

ROOT = "/lustre/work/pdl17890/udl806719/datasets/the_well/gray_scott_reaction_diffusion"
# per-channel stats from data/stats.yaml (A, B)
MEAN = np.array([0.729227819893941, 0.09658732411527585], dtype=np.float32)
STD = np.array([0.23988766176449572, 0.12366442840472558], dtype=np.float32)
NT = 1001  # timesteps per trajectory


@dataclass
class GrayScottConfig:
    data_root: str = ROOT
    split: str = "train"  # train | valid | test
    channels: int = 2  # two chemical fields A, B
    img_size: int = 128
    n_frames: int = 16  # frames per clip
    time_stride: int = 4  # spacing between frames (Gray-Scott evolves slowly)
    epoch_size: int = 8000
    batch_size: int = 8
    num_workers: int = 8


class GrayScottDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: GrayScottConfig):
        if h5py is None:
            raise ImportError("h5py required (uv pip install h5py)")
        self.cfg = cfg
        self.files = sorted(
            glob.glob(os.path.join(cfg.data_root, "data", cfg.split, "*.hdf5"))
        )
        if not self.files:
            raise FileNotFoundError(
                f"No .hdf5 in {os.path.join(cfg.data_root, 'data', cfg.split)} — "
                "download the dataset first (see examples/gray_scott/README.md)."
            )
        # (file -> n_traj) index — read n_traj cheaply from the header
        self.ntraj = []
        for p in self.files:
            with h5py.File(p, "r") as f:
                self.ntraj.append(f["t0_fields/A"].shape[0])
        self.span = (cfg.n_frames - 1) * cfg.time_stride + 1
        self._rng = np.random.default_rng()
        self._handles = {}

    def __len__(self):
        return self.cfg.epoch_size

    def _h(self, path):
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
        return self._handles[path]

    def __getitem__(self, idx):
        # per-worker RNG so each item draws an independent random clip
        self._rng = np.random.default_rng(torch.randint(0, 2**31 - 1, (1,)).item())
        fi = int(self._rng.integers(len(self.files)))
        f = self._h(self.files[fi])
        tr = int(self._rng.integers(self.ntraj[fi]))
        t0 = int(self._rng.integers(0, NT - self.span + 1))
        sl = slice(t0, t0 + self.span, self.cfg.time_stride)
        A = f["t0_fields/A"][tr, sl]  # [T, 128, 128]
        B = f["t0_fields/B"][tr, sl]
        x = np.stack([A, B], axis=0).astype(np.float32)  # [2, T, 128, 128]
        x = (x - MEAN[:, None, None, None]) / STD[:, None, None, None]
        return {"video": torch.from_numpy(x)}


def make_loader(cfg: GrayScottConfig, shuffle=True):
    return torch.utils.data.DataLoader(
        GrayScottDataset(cfg),
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
