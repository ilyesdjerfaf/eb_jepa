# Mesh JEPA — Temporal 3D Mesh Self-Supervised Learning

A JEPA (Joint-Embedding Predictive Architecture) for temporal 3D meshes. Learns motion-aware representations from the DFAUST dataset by predicting future mesh representations in latent space, without reconstructing raw geometry.

## Architecture

```
Frame t: mesh(V,F) + features → DiffusionNet → global pool → z_t ∈ R^256
                                                                    ↓
                                                   GRU Predictor(z_t) → ẑ_{t+1}
                                                                    ↓
Frame t+1: mesh(V,F) + features → DiffusionNet → global pool → z_{t+1} (target)
                                                                    ↓
                                          Loss = ||ẑ_{t+1} - z_{t+1}||² + λ · VCLoss(z)
```

**Encoder**: DiffusionNet (spectral diffusion on surfaces) + mass-weighted global mean pooling  
**Predictor**: GRU (autoregressive, state-only)  
**Regularizer**: VICReg (variance + covariance anti-collapse)  
**Input features**: HKS (per-frame, intrinsic, rotation-invariant) or XYZ (extrinsic vertex positions)

---

## Project Structure

```
examples/mesh_jepa/
├── README.md                  # This file
├── __init__.py
│
│   # --- Orchestration (like a Makefile, but in Python) ---
├── run_experiment.py          # Experiment orchestrator — calls the scripts below
│                              #   with the right args derived from the experiment YAML.
│                              #   Targets: preprocess, train, eval, all, summary
│                              #   Skips steps if output exists (use --force to override)
│
│   # --- Worker scripts (do the actual computation) ---
├── preprocess.py              # Preprocessing: Laplacian, per-frame HKS, normalization
├── main.py                    # Training: builds model, runs training loop, saves checkpoint
├── eval.py                    # Evaluation: probes, metrics, visualizations, timing
│
│   # --- Model components ---
├── encoder.py                 # DiffusionNetEncoder (DiffusionNet + global pool → JEPA 5D)
├── predictor.py               # MeshPredictor (state-only GRU for temporal prediction)
├── diffusion_net/             # DiffusionNet implementation (spectral mesh network)
│   ├── __init__.py
│   └── layers.py              # DiffusionNet, DiffusionNetBlock, LearnedTimeDiffusion
│
│   # --- Utilities ---
├── explore_dfaust.py          # Dataset exploration / statistics
├── visualize_dfaust.py        # Mesh rendering utilities (strips, GIFs, interactive)
│
│   # --- Experiment configs ---
└── experiments/               # YAML files that define full experiments
    ├── default.yaml           # Full experiment (30 epochs, for GPU)
    └── quick_test.yaml        # Jury/demo config (2 epochs, CPU, proves code works end-to-end)
```

### How run_experiment.py works

`run_experiment.py` is like a **Makefile in Python**. It doesn't train or evaluate anything itself — it reads the experiment YAML and calls the worker scripts (`preprocess.py`, `main.py`, `eval.py`) as subprocesses with the correct arguments and output paths.

- **Targets**: `preprocess`, `train`, `eval`, `all`, `summary` — like `make preprocess`
- **Dependency checks**: skips steps if output already exists — like Make checking file timestamps
- **`--force`**: re-runs even if output exists — like `make -B`
- **Why Python, not Make**: needs to parse YAML, resolve paths, generate intermediate configs

---

## Demo (for jury / verification)

Run the full pipeline on CPU in a few minutes — proves everything works end-to-end:

```bash
# If preprocessed data already exists (skips preprocessing):
uv run python -m examples.mesh_jepa.run_experiment \
    --config examples/mesh_jepa/experiments/quick_test.yaml all

# If starting from scratch (includes preprocessing — takes longer):
uv run python -m examples.mesh_jepa.run_experiment \
    --config examples/mesh_jepa/experiments/quick_test.yaml all --preprocess
```

This trains both HKS and XYZ models (2 epochs) → evaluates both (probe accuracy, collapse metrics, inference visualization) → prints summary. No GPU needed.

---

## Quick Start

### Full pipeline (one command)

```bash
uv run python -m examples.mesh_jepa.run_experiment \
    --config examples/mesh_jepa/experiments/default.yaml all
```

This runs: preprocessing → training (HKS + XYZ) → evaluation (both) → summary.

### Step by step

```bash
# 1. Preprocess DFAUST (compute per-frame HKS, normalize vertices)
uv run python -m examples.mesh_jepa.run_experiment \
    --config examples/mesh_jepa/experiments/default.yaml preprocess

# 2. Train both models
uv run python -m examples.mesh_jepa.run_experiment \
    --config examples/mesh_jepa/experiments/default.yaml train

# 3. Evaluate both models
uv run python -m examples.mesh_jepa.run_experiment \
    --config examples/mesh_jepa/experiments/default.yaml eval

# 4. Print results summary
uv run python -m examples.mesh_jepa.run_experiment \
    --config examples/mesh_jepa/experiments/default.yaml summary
```

---

## Commands Reference

### Experiment runner (recommended)

```bash
# Full pipeline (train + eval, assumes preprocessed data exists)
uv run python -m examples.mesh_jepa.run_experiment --config <yaml> all

# Full pipeline including preprocessing
uv run python -m examples.mesh_jepa.run_experiment --config <yaml> all --preprocess

# Individual steps
uv run python -m examples.mesh_jepa.run_experiment --config <yaml> preprocess
uv run python -m examples.mesh_jepa.run_experiment --config <yaml> train
uv run python -m examples.mesh_jepa.run_experiment --config <yaml> eval
uv run python -m examples.mesh_jepa.run_experiment --config <yaml> summary

# Train/eval only one feature type
uv run python -m examples.mesh_jepa.run_experiment --config <yaml> train --feature_type hks
uv run python -m examples.mesh_jepa.run_experiment --config <yaml> eval --feature_type xyz

# Force re-run (overwrite existing outputs)
uv run python -m examples.mesh_jepa.run_experiment --config <yaml> --force train
```

### Individual scripts (advanced)

```bash
# Preprocessing (standalone)
uv run python -m examples.mesh_jepa.preprocess \
    --data_dir datasets/dfaust/raw \
    --out_dir datasets/dfaust/processed \
    --n_eigen 128 --n_hks 16 --temporal_stride 4 \
    --actions jumping_jacks punching running_on_spot

# Training (standalone)
uv run python -m examples.mesh_jepa.main \
    --fname path/to/config.yaml \
    --folder path/to/output_dir

# Evaluation (standalone)
uv run python -m examples.mesh_jepa.eval \
    --model_path checkpoints/mesh_jepa/.../final.pth.tar \
    --data_dir datasets/dfaust/processed \
    --output_dir logs/eval_output
```

---

## Experiment Configuration

A single YAML file defines the entire experiment. Example (`experiments/default.yaml`):

```yaml
experiment:
  name: "baseline_3actions"       # Controls all output folder names
  seed: 42
  device: auto
  feature_types: [hks, xyz]       # Both trained from same preprocessed data

preprocessing:
  raw_data_dir: datasets/dfaust/raw
  actions: [jumping_jacks, punching, running_on_spot]
  n_eigen: 128                    # Laplacian eigenvectors
  n_hks: 16                       # HKS time scales
  temporal_stride: 4              # 60fps → 15fps

training:
  seq_len: 16                     # Frames per clip
  batch_size: 16
  epochs: 30
  lr: 1.0e-3
  width: 128                      # DiffusionNet hidden width
  depth: 4                        # DiffusionNet blocks
  henc: 256                       # Encoder output dim
  hpre: 256                       # Predictor hidden dim
  steps: 4                        # Multi-step rollout
  std_coeff: 10.0                 # VICReg variance weight
  cov_coeff: 100.0                # VICReg covariance weight
  ...

eval:
  batch_size: 16
  n_inference_clips: 3
  max_horizon: 15
```

### Output folder structure

All paths are derived from `experiment.name`:

```
datasets/dfaust/processed_{name}/         # Preprocessed data (vertices, HKS, operators)
checkpoints/mesh_jepa/{name}/hks/         # Trained HKS model + config
checkpoints/mesh_jepa/{name}/xyz/         # Trained XYZ model + config
results/{name}/hks/                       # Evaluation outputs (PNGs, GIF, results.npy)
results/{name}/xyz/                       # Evaluation outputs (PNGs, GIF, results.npy)
```

This keeps everything from one experiment grouped by name:
- `datasets/` — raw and processed data
- `checkpoints/` — trained model weights
- `results/` — evaluation figures and metrics (pitch-ready)

---

## Evaluation Outputs

After running eval, each model produces:

| File | Content |
|------|---------|
| `0_training_curves.png` | Loss curves: total, prediction, VICReg (std+cov), learning rate |
| `1_linear_probe.png` | Confusion matrix + per-class F1 (encoder & predictor probes) |
| `2_temporal_horizon.png` | Prediction MSE + cosine similarity vs rollout depth K=1..15 |
| `4_robustness.png` | Cosine similarity under vertex noise (5 levels) + temporal jitter |
| `5_abstraction.png` | JEPA vs supervised baseline: probe accuracy + noise robustness |
| `6_collapse_dashboard.png` | Full collapse dashboard (see below) |
| `results.npy` | All numeric results (for programmatic comparison) |

### Evaluation suite details

0. **Training Curves** — Publication-quality loss plots from `history.json` (saved during training). Shows total loss, prediction loss, VICReg components (std + cov), and learning rate schedule.
1. **Linear Probe** — Freeze encoder, train logistic regression on action labels. Compare encoder vs predictor representations. Confusion matrix + per-class F1.
2. **Temporal Horizon** — Autoregressive rollout K=1..15, MSE + cosine similarity at each step. Shows how gracefully prediction degrades.
3. **Rotation Invariance** — Apply random SO(3) rotations, measure cosine similarity between original and rotated representations. HKS should be invariant, XYZ should not.
4. **Robustness** — Vertex noise at 5 magnitudes (0.001→0.05) + temporal jitter (shift ±1 frame). Measures representation stability under perturbations.
5. **Abstraction** — Trains a supervised baseline (same architecture, MSE loss predicting next-frame features directly). Compares probe accuracy and noise robustness. Proves JEPA's advantage of predicting in abstract latent space vs reconstructing raw features.
6. **Collapse Dashboard**:
   - Standard VJEPA2: std per dim, covariance eigenspectrum, effective rank, dead dims
   - Mesh-specific: learned diffusion times per block/channel (reveals what spatial frequencies the model attends to)
   - Mesh-specific: per-frequency-band energy (splits input into low/mid/high Laplacian bands, encodes each, measures representation variance — detects spectral collapse)

---

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| **Full resolution** | 6890 vertices, no decimation | Preserves exact inter-frame displacements that the predictor learns |
| **Per-frame HKS** | Laplacian + eigendecomp per frame | Static HKS (fixed Laplacian) is identical every frame — useless for temporal prediction |
| **Mean-pose operators** | For DiffusionNet diffusion | DiffusionNet needs fixed operators; mean pose is more neutral than arbitrary first frame |
| **15fps** | Temporal stride 4 | Adjacent frames at 60fps are near-identical (~0.9mm); at 15fps motion is visible (~3.5mm) |
| **State-only predictor** | GRU without action input | No actions in DFAUST — predictor learns motion dynamics from state alone |
| **Autoregressive unroll** | Single-step GRU prediction | Matches RNN nature; each step predicts from its own output |

---

## Dependencies

Core (in `pyproject.toml`):
- `robust-laplacian` — Cotangent Laplacian computation
- `scipy` — Sparse eigendecomposition
- `h5py` — HDF5 loading (raw DFAUST)
- `torch` — Model, training
- `scikit-learn` — Linear probe, PCA, t-SNE
- `matplotlib` — All visualizations
