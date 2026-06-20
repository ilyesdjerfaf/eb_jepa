"""Gray-Scott — downstream evaluation (The Well's open question, in field space).

The Well asks: does latent prediction give more *stable* long-horizon rollouts
than the field-space neural-operator surrogates (FNO / U-Net)? To answer it we
roll the frozen JEPA predictor forward in LATENT space, DECODE each latent back
to a 2-channel field, and score multi-step VRMSE against ground truth and a
PERSISTENCE baseline (optionally vs FNO / U-Net surrogates).

The rollout-extraction harness is provided. What you implement (``# TODO``) is the
latent->field DECODER and the VRMSE metric that makes the comparison meaningful.

Run:  python -m examples.gray_scott.eval --ckpt <.../latest.pth.tar> --H 10
"""

import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.gray_scott.dataset import GrayScottConfig, make_loader
from examples.gray_scott.main import build_encoder, build_jepa

C = 2  # context_length (StateOnlyPredictor predicts from the previous 2 frames)


def load_jepa(ckpt, device):
    """Provided: rebuild encoder + JEPA from a training checkpoint and freeze."""
    cfg = OmegaConf.create(ckpt["cfg"])
    encoder = build_encoder(cfg.model).to(device)
    jepa = build_jepa(encoder, cfg.model).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    jepa.load_state_dict(ckpt["jepa"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)
    return jepa, encoder


@torch.no_grad()
def rollout_latents(jepa, x, H, device):
    """Provided: autoregressive latent rollout from C context frames.

    Feeds the first C frames of the clip and rolls the predictor forward H steps
    in latent space (``ctxt_window_time=C`` — the StateOnlyPredictor needs 2
    context frames, else the autoregressive loop yields an empty time axis).
    Returns the predicted latent sequence ``[B, D, C+H, h, w]``."""
    pred, _ = jepa.unroll(
        x[:, :, :C],
        actions=None,
        nsteps=H,
        unroll_mode="autoregressive",
        ctxt_window_time=C,
        compute_loss=False,
        return_all_steps=False,
    )
    return pred


# --------------------------------------------------------------------------- #
# LATENT -> FIELD DECODER  — # TODO
# --------------------------------------------------------------------------- #
def build_decoder(dstc, device):
    """TODO: return a trained latent->field decoder mapping ``[B, D, T, H, W]`` ->
    ``[B, 2, T, H, W]`` (the JEPA has no decoder — predicting latents is the point,
    so to score VRMSE in field space you must add one).

    A small conv stack suffices when the encoder is stride-1 / no-pool (latent is
    full 128x128): ``Conv2d(D, hid, 3) -> GELU -> Conv2d(hid, hid, 3) -> GELU ->
    Conv2d(hid, 2, 1)`` applied per frame. Train it with the JEPA FROZEN to
    minimise ``MSE(decode(encode(field)), field)`` on the train split, then load
    its weights here. Its reconstruction error is the JEPA's irreducible field
    floor (``decode(encode(truth))``), so report that floor alongside the rollout."""
    raise NotImplementedError(
        "TODO: build + load the latent->field decoder (see docstring)"
    )


# --------------------------------------------------------------------------- #
# METRIC  — # TODO
# --------------------------------------------------------------------------- #
def vrmse_per_horizon(jepa, encoder, decoder, loader, device, H):
    """TODO: per-horizon field-space VRMSE for JEPA vs a persistence baseline
    (and, optionally, FNO / U-Net surrogates trained iso-protocol).

    VRMSE (The Well) = sqrt( <(pred-true)^2>_space / <(true-<true>)^2>_space ).
    AGGREGATE numerator and denominator across the batch and take the ratio ONCE
    at the end — per-sample ratios blow up on near-uniform frames (Gray-Scott
    channel B has tiny spatial variance). Protocol, iso for all models:
      * ground truth   : true[h] = x[:, :, C-1+h]            for h = 1..H
      * JEPA           : decode(rollout_latents(...))        (latent -> field)
      * persistence    : repeat the last context field x[:, :, C-1]
      * decoder_floor  : decode(encode(true field))          (irreducible floor)
    Return a dict ``{name: np.ndarray[H] of VRMSE}``. Lower-than-persistence and
    the gap to the surrogates is the answer to The Well's question."""
    raise NotImplementedError("TODO: implement the VRMSE metric (see docstring)")


def main():
    ckpt_path = sys.argv[sys.argv.index("--ckpt") + 1]
    H = int(sys.argv[sys.argv.index("--H") + 1]) if "--H" in sys.argv else 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    jepa, encoder = load_jepa(ckpt, device)
    dstc = int(OmegaConf.create(ckpt["cfg"]).model.dstc)
    decoder = build_decoder(dstc, device)
    print(f"[gs-eval] loaded (epoch {ckpt.get('epoch')}), H={H}", flush=True)

    dcfg = GrayScottConfig(
        split="valid",
        n_frames=C + H,
        time_stride=4,
        epoch_size=400,
        batch_size=8,
        num_workers=8,
    )
    loader = make_loader(dcfg, shuffle=False)
    scores = vrmse_per_horizon(jepa, encoder, decoder, loader, device, H)
    for name, arr in scores.items():
        print(
            f"   {name:14s} h1={arr[0]:.3f} h{H}={arr[-1]:.3f} | {np.round(arr, 3).tolist()}",
            flush=True,
        )


if __name__ == "__main__":
    main()
