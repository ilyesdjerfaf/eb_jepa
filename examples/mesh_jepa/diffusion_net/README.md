# DiffusionNet Implementation

Faithful reimplementation of **DiffusionNet** (Sharp et al., ACM TOG 2022) — a discretization-agnostic neural network for learning on 3D surfaces using spectral diffusion.

**Reference**: "DiffusionNet: Discretization Agnostic Learning on Surfaces"  
**Original repo**: [github.com/nmwsharp/diffusion-net](https://github.com/nmwsharp/diffusion-net)

## Architecture

```
Input features (V, C_in)
        │
   first_lin: Linear(C_in → C_width)
        │
   ┌────┴────┐
   │  Block 1 │ ──► DiffusionNetBlock (diffusion + MLP + residual)
   └────┬────┘
   ┌────┴────┐
   │  Block 2 │ ──► ...
   └────┬────┘
        │ × N_blocks
        │
   last_lin: Linear(C_width → C_out)
        │
   global_mean_pool (mass-weighted)
        │
   Output (D,)
```

## Components

### `DiffusionNet` (main model)
- `first_lin`: projects input features to working width
- `blocks`: N DiffusionNetBlocks (learned diffusion + spatial MLP)
- `last_lin`: projects to output dimension
- `global_mean_pool`: mass-weighted average over vertices (discretization-invariant)

### `DiffusionNetBlock`
Each block applies:
1. **Learned spectral diffusion** — diffuses features along the surface with learned per-channel timescales
2. **Concatenation** — `[x_input, x_diffused]` (2 × C_width)
3. **MLP** — pointwise transformation with residual connection

No LayerNorm (matches official implementation).

### `LearnedTimeDiffusion`
The core innovation. Each channel `c` has a learned diffusion time `t_c > 0`. Diffusion in the spectral domain:

```
x_diffused = Φ · diag(exp(-λ_k · t_c)) · Φᵀ · x
```

Where `Φ` = Laplacian eigenvectors, `λ_k` = eigenvalues. This is equivalent to heat diffusion on the surface for time `t_c`, but learned end-to-end.

### `SpatialGradientFeatures`
Complex-linear transform on gradient features (currently using diffusion-only mode, gradients disabled).

### `MiniMLP`
Per-vertex pointwise MLP with dropout (p=0.5). No batch normalization.

## Inputs Required

DiffusionNet requires precomputed Laplacian operators:

| Operator | Shape | Description |
|----------|-------|-------------|
| `eigenvalues` | (K,) | First K eigenvalues of cotangent Laplacian |
| `eigenvectors` | (V, K) | Corresponding eigenvectors |
| `mass` | (V,) | Lumped mass matrix (vertex areas) |

These are computed once during preprocessing from the mean-pose mesh.

## Key Properties

- **Discretization agnostic**: works on any triangulation of the same surface
- **Spatially adaptive**: learned diffusion times let the network attend to different spatial frequencies per channel
- **Rotation equivariant** (with gradient features) / **invariant** (with global pooling)
- **Efficient**: operates in spectral domain (K eigenvectors, not V×V matrices)

## Parameters

For our Mesh JEPA large model (width=384, depth=8, K=128):
- Encoder params: ~4.84M
- Dominant cost: MLP layers in each block (2 × C_width → C_width)
