#!/bin/bash
#SBATCH --job-name=eb_jepa_test
#SBATCH --partition=defq
#SBATCH --account=pdl17890
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/work/pdl17890/%u/logs/slurm_test_%j.out
#SBATCH --error=/lustre/work/pdl17890/%u/logs/slurm_test_%j.err

set -e

REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"

echo "=== Host: $(hostname) | Arch: $ARCH | Date: $(date) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a') ==="
echo "=== venv: $UV_PROJECT_ENVIRONMENT ==="

module load python312
echo "=== Python: $(python3 --version) ==="

# Install uv for this arch if needed, then sync deps
if ! uv --version &>/dev/null; then
    echo ">>> Installing uv for $ARCH..."
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
fi
echo "=== uv: $(uv --version) ==="

echo ">>> uv sync..."
uv sync --dev --project "$REPO"

echo ">>> Running tests..."
uv run --project "$REPO" pytest "$REPO/tests/" -v

echo "=== Done ==="
