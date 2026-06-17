#!/usr/bin/env bash
# Source this file to set up the EB-JEPA environment variables.
# Usage: source env.sh
# In SLURM scripts: source "$(dirname "$0")/../env.sh"  (adjust path as needed)

# Your personal work directory — defaults to the project work partition under your
# username. Override by setting EBJEPA_WORK before sourcing. WORK IS NOT YOUR HOME:
# clone the repo and run everything from /lustre/work (the home quota blocks git/venvs).
WORK=${EBJEPA_WORK:-/lustre/work/pdl17890/$USER}
export EBJEPA_WORK="$WORK"                  # exported so python (launch_sbatch) sees it
ARCH=$(uname -m)                           # x86_64 on login node, aarch64 on compute nodes
export EBJEPA_COMPUTE_ARCH=${EBJEPA_COMPUTE_ARCH:-aarch64}  # target arch for SLURM jobs

# Repo root (this file lives at the repo root) — SLURM scripts read $EBJEPA_REPO.
export EBJEPA_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Cluster utility scripts
export PATH="$EBJEPA_REPO/cluster:$PATH"

# uv binary (arch-specific, avoids Exec format error across node types)
export UV_INSTALL_DIR=$WORK/uv_bin/$ARCH
export PATH="$UV_INSTALL_DIR:$HOME/.local/bin:$PATH"

# uv cache and venv (arch-specific so x86_64 and aarch64 don't collide)
export UV_CACHE_DIR=$WORK/uv_cache/$ARCH
export UV_PROJECT_ENVIRONMENT=$WORK/venvs/eb_jepa_$ARCH

# Keep ALL caches on /work — the /lustre/home quota is small and fills up fast
# (model/dataset downloads, torch.compile kernels, pip wheels, matplotlib, ...).
export XDG_CACHE_HOME=$WORK/.cache              # catch-all (pip, matplotlib, fontconfig, ...)
export HF_HOME=$WORK/.cache/huggingface         # HuggingFace hub + datasets + transformers
export TORCH_HOME=$WORK/.cache/torch            # torch hub weights
export TRITON_CACHE_DIR=$WORK/.cache/triton     # torch.compile / triton kernels
export PIP_CACHE_DIR=$WORK/.cache/pip
export WANDB_DIR=$WORK/wandb                     # W&B run files + cache
export WANDB_CACHE_DIR=$WORK/.cache/wandb

# EB-JEPA paths
export EBJEPA_CKPTS=${EBJEPA_CKPTS:-$WORK/checkpoints}
# Dataset folder. Defaults to your own $WORK/datasets; set EBJEPA_DSETS to point at
# a shared/provided dataset folder if one is available on your cluster.
export EBJEPA_DSETS=${EBJEPA_DSETS:-$WORK/datasets}

# W&B: export WANDB_DISABLED=true before sourcing to turn off logging cluster-wide
export WANDB_DISABLED=${WANDB_DISABLED:-false}
