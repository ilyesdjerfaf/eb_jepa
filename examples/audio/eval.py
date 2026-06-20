"""Audio — downstream evaluation (35-keyword spotting via frozen SSL features).

The feature-extraction harness is provided. What you implement (`# TODO`) is the
linear probe + metric, and the comparison that makes the number meaningful: the
frozen SSL encoder vs a random-encoder floor vs a supervised end-to-end baseline.

Metric: 35-way accuracy on the OFFICIAL Speech Commands v2 test split
(chance = 100/35 = 2.86 %).

Run:  python -m examples.audio.eval --ckpt <.../latest.pth.tar>
"""

import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.audio.dataset import AudioConfig, make_loader
from examples.audio.main import build_encoder


@torch.no_grad()
def extract_features(encoder, split, dcfg, device):
    """Provided: frozen encoder -> [N, D] features + integer labels for `split`."""
    cfg = AudioConfig(**{**dcfg, "split": split, "mode_ssl": "supervised"})
    loader = make_loader(cfg, shuffle=False)
    X, y = [], []
    for xb, yb in loader:
        X.append(encoder.represent(xb.to(device)).cpu().numpy())
        y.append(np.asarray(yb))
    return np.concatenate(X), np.concatenate(y)


# --------------------------------------------------------------------------- #
# PROBE + METRIC  — # TODO
# --------------------------------------------------------------------------- #
def probe(Xtr, ytr, Xte, yte, n_classes):
    """TODO: fit a LINEAR probe on the FROZEN train features and score on the
    official test split as 35-way accuracy (chance = 100/n_classes %).
      * standardize on TRAIN only (no leakage), then sklearn LogisticRegression
        (multinomial) — or a torch nn.Linear trained with cross-entropy.
      * return a metrics dict, e.g. {"acc": ..., "balanced_acc": ..., "chance": ...}
    To make the number meaningful, also run this probe on (a) a RANDOM untrained
    encoder and (b) a supervised end-to-end baseline, and compare."""
    raise NotImplementedError(
        "TODO: implement the linear probe + 35-way accuracy (see docstring)"
    )


def main():
    ckpt = sys.argv[sys.argv.index("--ckpt") + 1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    dcfg = OmegaConf.to_container(cfg.data, resolve=True)
    encoder = build_encoder(cfg.model).to(device)
    encoder.load_state_dict(state["encoder"])
    encoder.eval()

    Xtr, ytr = extract_features(encoder, "train", dcfg, device)
    Xte, yte = extract_features(encoder, "test", dcfg, device)
    print("[audio-eval]", probe(Xtr, ytr, Xte, yte, dcfg["n_classes"]))


if __name__ == "__main__":
    main()
