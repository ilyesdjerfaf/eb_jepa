"""MLP-based mesh encoder for JEPA (baseline comparison).

Replaces DiffusionNet with a simple per-vertex MLP + global mean pool.
No spectral diffusion, no surface awareness — just pointwise processing.
Used to demonstrate the value of DiffusionNet's geometric inductive bias.
"""

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """Per-vertex MLP encoder + mass-weighted global mean pool.

    Processes each vertex independently (no spatial communication),
    then pools over the mesh. Same parameter count as DiffusionNet
    for fair comparison.
    """

    def __init__(self, in_channels, out_dim=256, width=768, depth=9, dropout=True):
        super().__init__()
        layers = []
        in_d = in_channels
        for i in range(depth):
            layers.append(nn.Linear(in_d, width))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(p=0.5))
            in_d = width
        layers.append(nn.Linear(width, out_dim))
        self.mlp = nn.Sequential(*layers)
        self.out_dim = out_dim

    def register_operators(self, eigenvalues, eigenvectors, mass):
        """Register mass for pooling (eigenvalues/eigenvectors unused by MLP)."""
        self.register_buffer("mass", mass)

    def forward_single(self, x):
        """Encode a single frame.

        x: (B, V, C_in)
        Returns: (B, D)
        """
        per_vertex = self.mlp(x)  # (B, V, D)
        pooled = torch.einsum("bvd,v->bd", per_vertex, self.mass)
        pooled = pooled / self.mass.sum()
        return pooled

    def forward(self, x):
        """Encode a temporal sequence of frames.

        x: (B, T, V, C_in)
        Returns: (B, D, T, 1, 1) — JEPA 5D convention
        """
        B, T, V, C = x.shape
        x_flat = x.reshape(B * T, V, C)
        z_flat = self.forward_single(x_flat)  # (B*T, D)
        z = z_flat.reshape(B, T, self.out_dim)
        z = z.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        return z
