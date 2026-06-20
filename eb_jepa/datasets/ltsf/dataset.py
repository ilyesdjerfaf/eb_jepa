"""LTSF dataset — ETT long-term forecasting (Track 5).

Loads an ETT csv (7 channels: HUFL HULL MUFL MULL LUFL LULL OT), applies the
canonical Time-Series-Library 12/4/4-month train/val/test split, and fits a
StandardScaler on the TRAIN portion only (no leakage). Yields sliding windows:

  mode="ssl"        -> one input window x [C, L]          (JEPA pretraining)
  mode="forecast"   -> (x [C, L], y [C, H])               (probe / supervised)

Borders follow thuml/Time-Series-Library exactly so the MSE/MAE are comparable
to the published DLinear / PatchTST / iTransformer numbers.

Data loading is PROVIDED (plumbing). The modelling choices you make on top of
these windows (encoder, SSL objective, forecast probe) live in
``examples/ltsf/`` and are where the ``# TODO``s are.
"""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

ETT_CHANNELS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def _borders(name, seq_len):
    """TSLib train/val/test borders. ETTh* hourly (unit=1), ETTm* 15-min (unit=4)."""
    unit = 4 if name.lower().startswith("ettm") else 1
    d = 30 * 24 * unit
    b1 = [0, 12 * d - seq_len, 12 * d + 4 * d - seq_len]
    b2 = [12 * d, 12 * d + 4 * d, 12 * d + 8 * d]
    return b1, b2


@dataclass
class LTSFConfig:
    csv: str = "/lustre/work/pdl17890/udl806719/datasets/LTSF/ETT-small/ETTh1.csv"
    name: str = "ETTh1"
    flag: str = "train"  # train | val | test
    mode: str = "forecast"  # ssl (unlabeled windows) | forecast ((x, y))
    seq_len: int = 336  # input window length L
    pred_len: int = 96  # forecast horizon H
    in_channels: int = 7
    batch_size: int = 64
    num_workers: int = 8


def _load_norm(cfg: LTSFConfig):
    """Return the normalized [T, C] array and the (border1, border2) for cfg.flag."""
    if not os.path.exists(cfg.csv):
        raise FileNotFoundError(
            f"{cfg.csv} not found — download ailuntz/ETT-small (see examples/ltsf/README.md)"
        )
    df = pd.read_csv(cfg.csv)
    data = df[ETT_CHANNELS].values.astype(np.float32)  # [T, 7]
    b1s, b2s = _borders(cfg.name, cfg.seq_len)
    tr0, tr1 = b1s[0], b2s[0]  # StandardScaler on TRAIN only
    mu = data[tr0:tr1].mean(0, keepdims=True)
    sd = data[tr0:tr1].std(0, keepdims=True) + 1e-8
    data = (data - mu) / sd
    fi = {"train": 0, "val": 1, "test": 2}[cfg.flag]
    return data, b1s[fi], b2s[fi]


class LTSFDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: LTSFConfig):
        self.cfg = cfg
        self.data, self.b1, self.b2 = _load_norm(cfg)
        self.seq, self.pred = cfg.seq_len, cfg.pred_len
        # last valid window start so that [start : start+seq+pred] fits in [b1, b2]
        span = self.seq + (0 if cfg.mode == "ssl" else self.pred)
        self.n = max(0, (self.b2 - self.b1) - span + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = self.b1 + i
        x = self.data[s : s + self.seq].T.copy()  # [C, L]
        if self.cfg.mode == "ssl":
            # SSL: one input window (predictive JEPA) — for a two-view objective
            # return two augmented copies (see the # TODO in examples/ltsf/main.py).
            return torch.from_numpy(x)
        y = self.data[s + self.seq : s + self.seq + self.pred].T.copy()  # [C, H]
        return torch.from_numpy(x), torch.from_numpy(y)


def make_loader(cfg: LTSFConfig, shuffle=None, drop_last=None):
    ds = LTSFDataset(cfg)
    is_train = cfg.flag == "train"
    return torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=is_train if shuffle is None else shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=is_train if drop_last is None else drop_last,
        persistent_workers=cfg.num_workers > 0,
    )


def make_ssl_loaders(cfg: LTSFConfig):
    """Train/val SSL loaders over input windows only (no horizon, no labels)."""
    tr = LTSFConfig(**{**cfg.__dict__, "flag": "train", "mode": "ssl"})
    va = LTSFConfig(**{**cfg.__dict__, "flag": "val", "mode": "ssl"})
    return make_loader(tr), make_loader(va, shuffle=False, drop_last=False)
