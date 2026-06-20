"""Per-vertex MLP decoder for DAE baseline.

Maps global latent (256-dim) back to per-vertex features (HKS or XYZ)
by broadcasting the latent to all vertices and applying a shared MLP.
"""

import torch.nn as nn


class MeshDecoder(nn.Module):
    """Broadcast + per-vertex MLP decoder.

    Takes a global latent vector and reconstructs per-vertex features.
    Uses mass-weighted reconstruction loss externally.
    """

    def __init__(self, latent_dim=256, out_channels=16, width=512, depth=3):
        super().__init__()
        layers = []
        in_d = latent_dim
        for i in range(depth):
            layers.append(nn.Linear(in_d, width))
            layers.append(nn.ReLU())
            in_d = width
        layers.append(nn.Linear(width, out_channels))
        self.mlp = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, z):
        """Decode latent to per-vertex features.

        z: (B, D) global latent
        Returns: (B, V, C_out) per-vertex reconstructed features
        """
        B, D = z.shape
        V = self.n_vertices
        z_broadcast = z.unsqueeze(1).expand(B, V, D)  # (B, V, D)
        return self.mlp(z_broadcast)  # (B, V, C_out)

    def set_n_vertices(self, n_vertices):
        self.n_vertices = n_vertices
