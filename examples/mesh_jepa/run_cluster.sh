#!/bin/bash
# Run Mesh JEPA experiment on cluster (parallel training on 2 GPUs, then eval)
# Usage: nohup bash examples/mesh_jepa/run_cluster.sh > logs/run_full.log 2>&1 &

source env.sh
cd "$(dirname "$0")/../.."

echo "=== Starting Mesh JEPA experiment ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

CONFIG="examples/mesh_jepa/experiments/default.yaml"

# Train HKS on GPU 0 and XYZ on GPU 1 in parallel
echo ""
echo "=== Training HKS (GPU 0) and XYZ (GPU 1) in parallel ==="
CUDA_VISIBLE_DEVICES=0 uv run python -m examples.mesh_jepa.run_experiment --config "$CONFIG" train --feature_type hks &
PID_HKS=$!
CUDA_VISIBLE_DEVICES=1 uv run python -m examples.mesh_jepa.run_experiment --config "$CONFIG" train --feature_type xyz &
PID_XYZ=$!

# Wait for both trainings to finish
wait $PID_HKS
echo "HKS training done (exit code: $?)"
wait $PID_XYZ
echo "XYZ training done (exit code: $?)"

# Evaluate both models
echo ""
echo "=== Evaluation ==="
uv run python -m examples.mesh_jepa.run_experiment --config "$CONFIG" eval

echo ""
echo "=== Done ==="
echo "Date: $(date)"
