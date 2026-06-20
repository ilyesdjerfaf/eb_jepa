# Experiment Configurations

Each YAML file defines a complete experiment: preprocessing, training, and evaluation parameters. The `run_experiment.py` orchestrator reads these configs to drive the full pipeline.

## Configs

| Config | Purpose | Encoder | Params | Epochs |
|--------|---------|---------|--------|--------|
| `large.yaml` | **Main experiment** — doubled expressivity | DiffusionNet 384×8 | 5.7M | 25 |
| `dae_baseline.yaml` | DAE baseline — reconstruct features | DiffusionNet 384×8 + decoder | 5.5M | 25 |
| `mlp_baseline.yaml` | MLP baseline — no geometric bias | Per-vertex MLP 768×9 | 4.9M | 60 |
| `default.yaml` | Standard DiffusionNet | DiffusionNet 256×6 | 2.5M | 60 |
| `overfit.yaml` | Capacity sanity check (2 clips) | DiffusionNet 256×6 | 2.5M | 200 |
| `sanity_10ep.yaml` | Quick verification | DiffusionNet 256×6 | 2.5M | 10 |
| `quick_test.yaml` | Jury demo (CPU, 2 min) | DiffusionNet 128×4 | 0.6M | 2 |

## Config Structure

```yaml
experiment:
  name: "experiment_name"        # Controls all output folder names
  seed: 42
  device: auto                   # "auto", "cuda", or "cpu"
  feature_types: [hks, xyz]      # Which features to train on

preprocessing:
  raw_data_dir: path/to/raw      # DFAUST HDF5 files
  processed_dir: path/to/out     # Override output dir (optional)
  actions: [...]                 # Subset of 14 DFAUST actions
  n_eigen: 128                   # Laplacian eigenvectors
  n_hks: 16                      # HKS time scales
  temporal_stride: 4             # 60fps → 15fps

training:
  # Data
  seq_len: 16                    # Frames per clip (~1s at 15fps)
  batch_size: 16
  train_subjects: [...]          # 8 training subjects
  test_subjects: [...]           # 2 held-out subjects

  # Encoder
  encoder_type: diffusionnet     # "diffusionnet", "mlp", or "dae"
  width: 384                     # Hidden width
  depth: 8                       # Number of blocks/layers
  henc: 256                      # Output latent dimension

  # Predictor
  hpre: 256                      # GRU hidden dim
  predictor_layers: 2
  steps: 4                       # Multi-step rollout
  grad_clip: 2.0

  # Loss (VICReg)
  std_coeff: 10.0                # Variance coefficient
  cov_coeff: 100.0               # Covariance coefficient
  proj_spec: "256-1024-1024"     # Projector architecture

  # Optimizer
  epochs: 25
  lr: 1.0e-3
  weight_decay: 1.0e-5

eval:
  batch_size: 16
  max_horizon: 15                # Max autoregressive steps for horizon eval
```

## Usage

```bash
# Run full pipeline
uv run python -m examples.mesh_jepa.run_experiment --config experiments/large.yaml all

# Train only HKS
uv run python -m examples.mesh_jepa.run_experiment --config experiments/large.yaml train --feature_type hks

# Force re-run
uv run python -m examples.mesh_jepa.run_experiment --config experiments/large.yaml --force eval
```

## Output Paths

All derived from `experiment.name`:
```
checkpoints/mesh_jepa/{name}/{hks,xyz}/   # Models + history.json + config.yaml
results/{name}/{hks,xyz}/                  # Evaluation plots + results.npy
```

## Creating a New Experiment

1. Copy `large.yaml` as a starting point
2. Change `experiment.name` to something unique
3. Adjust parameters as needed
4. Run: `python -m examples.mesh_jepa.run_experiment --config your_config.yaml all`

The orchestrator will warn if the name collides with existing outputs.
