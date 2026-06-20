"""AtlasNet decoder — patch-based surface generation from embeddings.

Implements the sampling version of AtlasNet (Groueix et al. 2018):
multiple learned 2D→3D mappings (patches), each conditioned on a
global latent vector. Together the patches approximate the surface.

Reference: "A Papier-Mâché Approach to Learning 3D Surface Generation"
"""

import torch
import torch.nn as nn


class AtlasNetPatch(nn.Module):
    """Single patch: maps (latent + 2D UV sample) → 3D point."""

    def __init__(self, latent_dim=256, hidden_dim=256, depth=3):
        super().__init__()
        layers = []
        in_d = latent_dim + 2  # latent + UV coordinates
        for i in range(depth):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.ReLU())
            in_d = hidden_dim
        layers.append(nn.Linear(hidden_dim, 3))
        self.mlp = nn.Sequential(*layers)

    def forward(self, latent, uv):
        """
        latent: (B, D)
        uv: (B, N, 2) — sampled 2D coordinates on unit square
        Returns: (B, N, 3) — 3D points
        """
        B, N, _ = uv.shape
        latent_expand = latent.unsqueeze(1).expand(B, N, -1)  # (B, N, D)
        x = torch.cat([latent_expand, uv], dim=-1)  # (B, N, D+2)
        return self.mlp(x)


class AtlasNet(nn.Module):
    """AtlasNet decoder: multiple patches collectively reconstruct a surface.

    Each patch learns a different 2D→3D mapping conditioned on the latent.
    At inference, sample UV points on each patch → union = point cloud.
    """

    def __init__(
        self,
        latent_dim=256,
        n_patches=10,
        points_per_patch=689,
        hidden_dim=256,
        depth=3,
    ):
        super().__init__()
        self.n_patches = n_patches
        self.points_per_patch = points_per_patch

        self.patches = nn.ModuleList(
            [
                AtlasNetPatch(latent_dim=latent_dim, hidden_dim=hidden_dim, depth=depth)
                for _ in range(n_patches)
            ]
        )

    def forward(self, latent, n_points_per_patch=None):
        """Generate point cloud from latent.

        latent: (B, D)
        n_points_per_patch: override points per patch (default: self.points_per_patch)
        Returns: (B, n_patches * n_points_per_patch, 3)
        """
        n_pts = n_points_per_patch or self.points_per_patch
        B = latent.shape[0]
        device = latent.device

        all_points = []
        for patch in self.patches:
            uv = torch.rand(B, n_pts, 2, device=device)
            points = patch(latent, uv)  # (B, n_pts, 3)
            all_points.append(points)

        return torch.cat(all_points, dim=1)  # (B, total_points, 3)

    def forward_per_patch(self, latent, n_points_per_patch=None):
        """Generate point cloud with patch labels (for visualization).

        Returns: points (B, total, 3), patch_ids (B, total)
        """
        n_pts = n_points_per_patch or self.points_per_patch
        B = latent.shape[0]
        device = latent.device

        all_points = []
        all_ids = []
        for i, patch in enumerate(self.patches):
            uv = torch.rand(B, n_pts, 2, device=device)
            points = patch(latent, uv)
            all_points.append(points)
            all_ids.append(torch.full((B, n_pts), i, device=device))

        points = torch.cat(all_points, dim=1)
        patch_ids = torch.cat(all_ids, dim=1)
        return points, patch_ids
