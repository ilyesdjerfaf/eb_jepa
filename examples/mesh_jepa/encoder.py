"""DiffusionNet-based mesh encoder for JEPA.

Wraps DiffusionNet with the JEPA 5D output convention.
Registers Laplacian operators as persistent buffers (fixed topology).
"""

import torch
import torch.nn as nn

from examples.mesh_jepa.diffusion_net import DiffusionNet


class DiffusionNetEncoder(nn.Module):
    """Mesh encoder: DiffusionNet (with global_mean output) → latent vector.

    DiffusionNet handles the mass-weighted global pooling internally.
    Outputs representations in JEPA's 5D convention: [B, D, T, 1, 1].
    """

    def __init__(
        self,
        in_channels,
        out_dim=256,
        width=128,
        depth=4,
        n_eigen=128,
        dropout=True,
        with_gradient_features=False,
        with_gradient_rotations=True,
    ):
        """
        Args:
            in_channels: Input features per vertex (16 for HKS, 3 for XYZ)
            out_dim: Output latent dimension
            width: DiffusionNet hidden width
            depth: Number of DiffusionNet blocks
            n_eigen: Number of Laplacian eigenvectors
            dropout: Use dropout in DiffusionNet MLPs
            with_gradient_features: Use spatial gradient features (requires gradX/gradY)
            with_gradient_rotations: Use complex rotations in gradient features
        """
        super().__init__()
        self.diffnet = DiffusionNet(
            C_in=in_channels,
            C_out=out_dim,
            C_width=width,
            N_block=depth,
            dropout=dropout,
            with_gradient_features=with_gradient_features,
            with_gradient_rotations=with_gradient_rotations,
            outputs_at="global_mean",
        )
        self.out_dim = out_dim

    def register_operators(self, eigenvalues, eigenvectors, mass):
        """Register precomputed Laplacian operators as persistent buffers."""
        self.register_buffer("eigenvalues", eigenvalues)
        self.register_buffer("eigenvectors", eigenvectors)
        self.register_buffer("mass", mass)

    def forward_single(self, x):
        """Encode a single frame (or batch of single frames).

        x: (B, V, C_in)
        Returns: (B, D)
        """
        return self.diffnet(x, self.mass, self.eigenvalues, self.eigenvectors)

    def forward(self, x):
        """Encode a temporal sequence of frames.

        x: (B, T, V, C_in)
        Returns: (B, D, T, 1, 1) — JEPA 5D convention
        """
        B, T, V, C = x.shape

        # Flatten time into batch for parallel processing
        x_flat = x.reshape(B * T, V, C)
        z_flat = self.forward_single(x_flat)  # (B*T, D)

        # Reshape to JEPA 5D: (B, D, T, 1, 1)
        z = z_flat.reshape(B, T, self.out_dim)
        z = z.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        return z
