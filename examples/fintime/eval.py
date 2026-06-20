"""FinTime — downstream evaluation (answers the track's transferability question).

The feature-extraction harness is provided. What you implement (`# TODO`) is the
probe + metric, and the comparison that makes the result meaningful: the frozen
SSL encoder vs a random-encoder floor vs a supervised end-to-end baseline.

Run:  python -m examples.fintime.eval --ckpt <.../latest.pth.tar>
"""

import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.fintime.dataset import FinTimeConfig, FinTimeDataset
from examples.fintime.main import build_encoder


@torch.no_grad()
def extract_features(encoder, split, target, device):
    """Provided: frozen encoder -> [N, D] features + labels for `split`."""
    ds = FinTimeDataset(FinTimeConfig(split=split, mode="supervised", target=target))
    loader = torch.utils.data.DataLoader(
        ds, batch_size=512, shuffle=False, num_workers=8
    )
    X, y = [], []
    for xb, yb in loader:
        X.append(encoder.represent(xb.to(device)).cpu().numpy())
        y.append(yb.numpy())
    return np.concatenate(X), np.concatenate(y)


# --------------------------------------------------------------------------- #
# PROBE + METRIC  — # TODO
# --------------------------------------------------------------------------- #
def probe(Xtr, ytr, Xte, yte, target):
    """TODO: fit a linear probe on the FROZEN train features (no leakage:
    standardize on train only) and score on test. Return a metrics dict.
      * classification (target='direction'): accuracy / balanced-acc / F1 / AUROC
        (sklearn LogisticRegression on the [N, D] features)
      * regression    (target='return')    : MSE / MAE / directional-accuracy / corr
    To make the number meaningful, also run this probe on (a) a RANDOM untrained
    encoder and (b) a supervised end-to-end baseline, and compare."""
    raise NotImplementedError("TODO: implement the probe + metric (see docstring)")


def main():
    ckpt = sys.argv[sys.argv.index("--ckpt") + 1]
    target = (
        sys.argv[sys.argv.index("--target") + 1]
        if "--target" in sys.argv
        else "direction"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    encoder = build_encoder(cfg.model).to(device)
    encoder.load_state_dict(state["encoder"])
    encoder.eval()

    Xtr, ytr = extract_features(encoder, "train", target, device)
    Xte, yte = extract_features(encoder, "test", target, device)
    print("[fintime-eval]", probe(Xtr, ytr, Xte, yte, target))


if __name__ == "__main__":
    main()
