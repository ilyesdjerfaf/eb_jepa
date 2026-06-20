"""LTSF — downstream forecasting eval (answers the track's transferability question).

The feature/forecast-extraction harness is provided. What you implement (`# TODO`)
is the forecast probe + metric, and the comparison that makes the result
meaningful: the frozen SSL encoder vs a random-encoder floor vs a supervised
end-to-end baseline vs a strong DLinear/NLinear linear baseline.

All MSE/MAE are on normalized data (the dataset z-scores on TRAIN only), so the
numbers are comparable to the published DLinear / PatchTST / iTransformer results.

Run:  python -m examples.ltsf.eval --ckpt <.../latest.pth.tar> --pred_len 96
"""

import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.ltsf.dataset import LTSFConfig, LTSFDataset
from examples.ltsf.main import build_encoder


def _collect(name, csv, seq_len, pred_len, flag):
    """Provided: stack the whole split into arrays X[N,C,L], Y[N,C,H] (ETT is small)."""
    ds = LTSFDataset(
        LTSFConfig(
            csv=csv,
            name=name,
            flag=flag,
            mode="forecast",
            seq_len=seq_len,
            pred_len=pred_len,
            num_workers=8,
        )
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=256, shuffle=False, num_workers=8
    )
    X, Y = [], []
    for x, y in loader:
        X.append(x.numpy())
        Y.append(y.numpy())
    return np.concatenate(X), np.concatenate(Y)


@torch.no_grad()
def extract_features(encoder, X, device):
    """Provided: frozen encoder -> [N, D] window representations."""
    encoder.eval()
    out = []
    for i in range(0, len(X), 512):
        xb = torch.from_numpy(X[i : i + 512]).to(device)
        out.append(encoder.represent(xb).cpu().numpy())
    return np.concatenate(out)


def mse_mae(y, p):
    return float(np.mean((y - p) ** 2)), float(np.mean(np.abs(y - p)))


# --------------------------------------------------------------------------- #
# FORECAST PROBE + METRIC  — # TODO
# --------------------------------------------------------------------------- #
def probe(Ztr, Ytr, Zte, Yte):
    """TODO: fit a linear forecast head on the FROZEN train features and score on
    test, returning (mse, mae) on normalized data.
      * Z* are [N, D] window representations from a frozen encoder; Y* are the
        targets [N, C, H]. Fit e.g. a ridge regression Z -> flatten(Y) (sklearn
        Ridge on Ytr.reshape(N, C*H)), predict, reshape back, then `mse_mae`.
    To make the number meaningful, also run this probe on (a) a RANDOM untrained
    encoder (floor) and (b) a supervised end-to-end encoder+linear head (= direct
    forecasting), and compare against the dlinear_baseline below."""
    raise NotImplementedError(
        "TODO: implement the forecast probe + metric (see docstring)"
    )


# --------------------------------------------------------------------------- #
# STRONG LINEAR BASELINE  — # TODO (optional but this is the bar to beat)
# --------------------------------------------------------------------------- #
def dlinear_baseline(X, Y, Xte, Yte, seq_len, pred_len):
    """TODO (optional): the DLinear/NLinear baseline the JEPA must beat.
    NLinear: subtract the last value, apply one channel-shared linear map L->H on
    the raw input, add the last value back. Treat each (sample, channel) as a row
    [N*C, L] -> [N*C, H] (sklearn Ridge), reshape, add last value, then `mse_mae`.
    On ETT this is famously hard to beat — see Zeng et al. (DLinear)."""
    raise NotImplementedError(
        "TODO: implement the DLinear/NLinear baseline (see docstring)"
    )


def main():
    ckpt = sys.argv[sys.argv.index("--ckpt") + 1]
    pred_len = (
        int(sys.argv[sys.argv.index("--pred_len") + 1])
        if "--pred_len" in sys.argv
        else 96
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    encoder = build_encoder(cfg.model).to(device)
    encoder.load_state_dict(state["encoder"])
    encoder.eval()

    name, csv, seq_len = cfg.data.name, cfg.data.csv, cfg.data.seq_len
    Xtr, Ytr = _collect(name, csv, seq_len, pred_len, "train")
    Xte, Yte = _collect(name, csv, seq_len, pred_len, "test")
    Ztr, Zte = extract_features(encoder, Xtr, device), extract_features(
        encoder, Xte, device
    )

    mse, mae = probe(Ztr, Ytr, Zte, Yte)
    print(f"[ltsf-eval] {name} H={pred_len} | frozen-JEPA MSE={mse:.4f} MAE={mae:.4f}")


if __name__ == "__main__":
    main()
