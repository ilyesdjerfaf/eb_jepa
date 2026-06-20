# Inference Pipeline — Mesh Generation from Embeddings

Generates 3D meshes from learned encoder embeddings using **AtlasNet** (Groueix et al., CVPR 2018) followed by surface reconstruction. This evaluates how much geometric information is retained in the latent representations.

## Pipeline Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  Trained Encoder │────►│  extract_embed.  │────►│  embeddings.npy    │
│  (JEPA/DAE/MLP) │     │  + point clouds  │     │  point_clouds.npy  │
└─────────────────┘     └──────────────────┘     └────────┬───────────┘
                                                           │
                        ┌──────────────────┐               │
                        │  train_atlasnet  │◄──────────────┘
                        │  (Chamfer loss)  │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐     ┌────────────────────┐
                        │   reconstruct    │────►│  .ply meshes       │
                        │  (alpha shape)   │     │  (point clouds +   │
                        └──────────────────┘     │   surface meshes)  │
                                                 └────────────────────┘
```

## Files

| File | Role |
|------|------|
| `extract_embeddings.py` | Run encoder over dataset, save per-frame (embedding, xyz) pairs |
| `atlasnet.py` | AtlasNet decoder architecture (25 learned 2D→3D patch mappings) |
| `train_atlasnet.py` | Train AtlasNet with Chamfer distance, augmentation, periodic visualization |
| `reconstruct.py` | Dense point cloud generation + Delaunay alpha-shape surface reconstruction |
| `compare_models.py` | Cross-model comparison: resampling, rotation, reconstruction, probing |

## Step 1: Extract Embeddings

Runs the trained encoder over all frames and saves paired (embedding, point_cloud) data.

```bash
python -m examples.mesh_jepa.inference.extract_embeddings \
    --model_path checkpoints/mesh_jepa/large_model/hks/final.pth.tar \
    --data_dir /path/to/processed \
    --output_dir /path/to/inference/data/jepa_large_hks \
    --feature_type hks \
    --subjects '[50002,50004,50007,50009,50020,50021,50022,50025]'
```

Output:
```
data/jepa_large_hks/
├── embeddings.npy      # (N_frames, 256)
├── point_clouds.npy    # (N_frames, 6890, 3)
└── labels.npy          # (N_frames,) action labels
```

## Step 2: Train AtlasNet

Trains the decoder to reconstruct point clouds from embeddings using Chamfer distance.

```bash
CUDA_VISIBLE_DEVICES=0 python -m examples.mesh_jepa.inference.train_atlasnet \
    --data_dir /path/to/inference/data/jepa_large_hks \
    --output_dir /path/to/inference/models/jepa_large_hks \
    --n_patches 25 --hidden_dim 512 --depth 5 \
    --n_sample_points 6890 --augment False \
    --epochs 200
```

### AtlasNet Architecture

Based on Groueix et al. 2018, adapted for 256-dim latent:

| Parameter | Value |
|-----------|-------|
| Patches | 25 (each is a learned 2D → 3D mapping) |
| MLP per patch | Linear(258, h) → ReLU → ... → Linear(h, 3) |
| Hidden dim | 512 |
| Depth | 5 layers |
| Training points | 6890 (full mesh, 276/patch) |
| Inference points | 50,000 (2000/patch, for smooth reconstruction) |
| Loss | Chamfer distance |
| Augmentation | Random SO(3) rotation + scaling [0.8, 1.2] + jitter σ=0.005 |
| LR schedule | MultiStepLR, decay ×0.1 at 50% and 75% of training |

### Visualization

Every 20 epochs, saves `visualizations/epoch_XXXX.png` showing GT (blue) vs predicted (coral) point clouds from 3 viewpoints. Lets you monitor reconstruction quality during training.

## Step 3: Reconstruct Meshes

Generates dense point clouds from AtlasNet, then applies Delaunay + alpha shape filtering to produce watertight triangle meshes.

```bash
python -m examples.mesh_jepa.inference.reconstruct \
    --model_dir /path/to/inference/models/jepa_large_hks \
    --data_dir /path/to/inference/data/jepa_large_hks \
    --output_dir /path/to/inference/meshes/jepa_large_hks \
    --n_samples 20 --device cpu
```

Output per sample:
- `sample_XXX_pc.ply` — generated point cloud (50k points)
- `sample_XXX_mesh.ply` — reconstructed surface mesh
- `sample_XXX_gt.ply` — ground truth point cloud

## Cross-Model Comparison

Evaluates all models on 4 axes and generates presentation-ready plots:

```bash
python -m examples.mesh_jepa.inference.compare_models \
    --data_dir /path/to/processed \
    --output_dir /path/to/comparison \
    --models '{"JEPA-HKS":"path/to/ckpt", "DAE-HKS":"...", "MLP-HKS":"..."}'
```

### Metrics

| # | Metric | What it measures |
|---|--------|-----------------|
| 1 | Resampling robustness | Cosine sim between embeddings of original vs vertex-subsampled meshes |
| 2 | Rotation + translation | Cosine sim under random SE(3) transforms |
| 3 | Feature reconstruction | R² of linear decode from embedding → mean feature vector |
| 4 | Linear probing | Action classification accuracy (frozen encoder + logistic regression) |

### Expected Results

- **JEPA-HKS**: Best probe accuracy, rotation-invariant, moderate feature R² (abstract representations discard unpredictable details)
- **DAE-HKS**: Highest feature R² (trained to reconstruct), but lower probe accuracy
- **MLP-HKS**: Worst resampling robustness (no spatial communication between vertices)
- **JEPA-XYZ**: Good probe, but sensitive to rotation (extrinsic features)

## Dependencies

All dependencies are already in `pyproject.toml`:
- `torch` — model inference
- `trimesh` — PLY I/O, mesh operations
- `scipy` — Delaunay triangulation, KDTree
- `matplotlib` — visualization during training
- `fire` — CLI interface
