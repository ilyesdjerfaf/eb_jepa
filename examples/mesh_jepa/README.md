# Temporal 3D Mesh JEPA

Self-supervised representation learning on temporal 3D meshes using Joint-Embedding Predictive Architecture. Learns motion-aware representations from the DFAUST dataset by predicting future mesh states in abstract latent space — without ever reconstructing raw geometry.

**Paper basis**: EB-JEPA (arXiv 2602.03604) extended to 3D mesh sequences.

## Method

```
                          DiffusionNet Encoder
                          (spectral diffusion on surface)
                                    │
Frame t:  mesh(6890 verts) + HKS ──►│──► global pool ──► z_t ∈ ℝ²⁵⁶
                                    │                        │
                                    │                   GRU Predictor
                                    │                        │
                                    │                        ▼
Frame t+1: mesh + HKS ─────────────►│──► global pool ──► z_{t+1} (target)

Loss = ‖ẑ_{t+1} - z_{t+1}‖² + VICReg(z)
```

| Component | Architecture | Role |
|-----------|-------------|------|
| Encoder | DiffusionNet (Sharp et al. 2022) | Surface-aware feature extraction via learned spectral diffusion |
| Predictor | GRU (state-only, autoregressive) | Temporal dynamics in latent space |
| Regularizer | VICReg (std=10, cov=100) | Prevents representation collapse |
| Input | HKS (intrinsic, rotation-invariant) or XYZ (extrinsic) | Per-frame geometric descriptors |

## Results

We compare 4 models with matched parameter budgets (~5.5M):

| Model | Description | Probe Acc | Rotation Invariant |
|-------|-------------|-----------|-------------------|
| **JEPA-HKS** | DiffusionNet + HKS features | Best | Yes (intrinsic) |
| **JEPA-XYZ** | DiffusionNet + XYZ features | Good | No |
| **DAE-HKS** | Same encoder, reconstruction loss | Lower | Yes |
| **MLP-HKS** | Per-vertex MLP, no geometry | Worst | Yes |

Key findings:
1. **JEPA > DAE**: Predicting in latent space learns more abstract, transferable representations than reconstructing input features
2. **DiffusionNet > MLP**: Geometric inductive bias (spectral diffusion) enables spatial communication between vertices
3. **HKS = rotation invariance**: Intrinsic features give invariance for free, without data augmentation

## Project Structure

```
examples/mesh_jepa/
├── README.md
├── run_experiment.py          # Orchestrator (like Makefile) — drives full pipeline
├── preprocess.py              # DFAUST → per-frame HKS + operators
├── main.py                    # JEPA training loop
├── main_dae.py                # DAE baseline training (reconstruction loss)
├── eval.py                    # Full evaluation suite (7 metrics)
│
├── encoder.py                 # DiffusionNet encoder → JEPA 5D format
├── encoder_mlp.py             # MLP baseline encoder (no geometry)
├── decoder.py                 # Per-vertex decoder (for DAE)
├── predictor.py               # GRU temporal predictor
│
├── diffusion_net/             # DiffusionNet reimplementation
│   └── layers.py              # See diffusion_net/README.md
│
├── experiments/               # YAML configs defining full experiments
│   └── *.yaml                 # See experiments/README.md
│
├── inference/                 # Mesh generation from embeddings
│   ├── extract_embeddings.py  # Encoder → (embedding, point_cloud) pairs
│   ├── atlasnet.py            # AtlasNet decoder (patch-based)
│   ├── train_atlasnet.py      # Train decoder (Chamfer distance)
│   ├── reconstruct.py         # Dense sampling + surface reconstruction
│   └── compare_models.py      # Cross-model comparison plots
│
├── explore_dfaust.py          # Dataset statistics
└── visualize_dfaust.py        # Mesh rendering utilities
```

## Quick Start

### Full experiment (one command)

```bash
uv run python -m examples.mesh_jepa.run_experiment \
    --config examples/mesh_jepa/experiments/large.yaml all
```

### Step by step

```bash
# 1. Preprocess (compute per-frame HKS, Laplacian operators)
uv run python -m examples.mesh_jepa.run_experiment \
    --config examples/mesh_jepa/experiments/large.yaml preprocess

# 2. Train (HKS + XYZ in parallel)
uv run python -m examples.mesh_jepa.run_experiment \
    --config examples/mesh_jepa/experiments/large.yaml train

# 3. Evaluate
uv run python -m examples.mesh_jepa.run_experiment \
    --config examples/mesh_jepa/experiments/large.yaml eval

# 4. Cross-model comparison
python -m examples.mesh_jepa.inference.compare_models \
    --data_dir /path/to/processed \
    --output_dir /path/to/comparison \
    --models '{"JEPA-HKS":"checkpoints/.../hks/final.pth.tar", ...}'
```

### Demo (CPU, 2 minutes)

```bash
uv run python -m examples.mesh_jepa.run_experiment \
    --config examples/mesh_jepa/experiments/quick_test.yaml all --preprocess
```

## Cluster Deployment (HTW DALIA)

```bash
# Get a compute node (aarch64, GB200 GPU)
srun --pty --gres=gpu:4 --reservation=Vivatech bash

# Setup environment
cd eb_jepa-1 && source env.sh

# Train (use python directly, not uv — uv is x86 only)
CUDA_VISIBLE_DEVICES=0 nohup python -m examples.mesh_jepa.run_experiment \
    --config examples/mesh_jepa/experiments/large.yaml all \
    > logs/large.log 2>&1 &
```

## Evaluation Suite

| # | Evaluation | What it measures |
|---|-----------|-----------------|
| 0 | Training Curves | Loss convergence (pred + VICReg components) |
| 1 | Linear Probe | Action classification from frozen representations |
| 2 | Temporal Horizon | Prediction quality degradation at K=1..15 steps |
| 3 | Rotation Invariance | Embedding stability under random SO(3) |
| 4 | Robustness | Vertex noise (5 levels) + temporal jitter |
| 5 | Abstraction | JEPA vs supervised baseline |
| 6 | Collapse Dashboard | Effective rank, spectral analysis, diffusion times |

## Inference Pipeline

Generate 3D meshes from learned embeddings to visualize what the encoder captures:

```
Encoder → Embeddings → AtlasNet (25 patches) → Point Cloud → Surface Mesh
```

See `inference/README.md` for details.

## Dataset

**DFAUST** (Dynamic FAUST): 10 subjects, 14 actions, 6890 vertices (SMPL topology), 60fps.

We use: 3 actions (jumping_jacks, punching, running_on_spot), 15fps, full resolution.
- Train: 8 subjects (1191 clips)
- Test: 2 subjects (362 clips)

## References

- Assran et al. "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (I-JEPA, 2023)
- Srivastava et al. "EB-JEPA: Energy-Based Joint-Embedding Predictive Architectures" (arXiv 2602.03604)
- Sharp et al. "DiffusionNet: Discretization Agnostic Learning on Surfaces" (ACM TOG, 2022)
- Bardes et al. "VICReg: Variance-Invariance-Covariance Regularization" (ICLR, 2022)
- Groueix et al. "A Papier-Mache Approach to Learning 3D Surface Generation" (AtlasNet, CVPR 2018)
