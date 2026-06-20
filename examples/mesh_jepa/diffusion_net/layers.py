"""DiffusionNet — Discretization Agnostic Learning on Surfaces.

Faithful reimplementation of Sharp et al. (2022).
Reference: github.com/nmwsharp/diffusion-net

Operates in the spectral domain using precomputed Laplacian eigenbasis.
Core per block: learned diffusion + spatial gradients (optional) + MLP + residual.
"""

import torch
import torch.nn as nn


def to_basis(values, eigenvectors, mass):
    """Project to spectral domain (mass-weighted).

    values: (B, V, C)
    eigenvectors: (V, K)
    mass: (V,)
    Returns: (B, K, C)
    """
    return torch.einsum("vk,v,bvc->bkc", eigenvectors, mass, values)


def from_basis(values, eigenvectors):
    """Project from spectral domain back to spatial.

    values: (B, K, C)
    eigenvectors: (V, K)
    Returns: (B, V, C)
    """
    return torch.einsum("vk,bkc->bvc", eigenvectors, values)


class LearnedTimeDiffusion(nn.Module):
    """Learned diffusion layer (spectral method).

    Learns one diffusion time per channel. Applies heat diffusion in eigenspace:
        x_out_c = sum_k exp(-lambda_k * t_c) * <phi_k, x_in_c>_M * phi_k

    Parameters: C_width scalar diffusion times (one per feature channel).
    """

    def __init__(self, C_width):
        super().__init__()
        self.diffusion_time = nn.Parameter(torch.zeros(C_width))

    def forward(self, x, eigenvectors, eigenvalues, mass):
        """
        x: (B, V, C)
        eigenvectors: (V, K)
        eigenvalues: (K,)
        mass: (V,)
        Returns: (B, V, C)
        """
        # Clamp diffusion times to positive
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)

        # Project to spectral domain: (B, K, C)
        x_spec = to_basis(x, eigenvectors, mass)

        # Apply diffusion: exp(-lambda_k * t_c) for each eigenvalue k, channel c
        # eigenvalues: (K,) → (K, 1), diffusion_time: (C,) → (1, C)
        diffusion_coefs = torch.exp(
            -eigenvalues.unsqueeze(-1) * self.diffusion_time.unsqueeze(0)
        )  # (K, C)
        x_diffuse_spec = diffusion_coefs.unsqueeze(0) * x_spec  # (B, K, C)

        # Project back to spatial domain: (B, V, C)
        return from_basis(x_diffuse_spec, eigenvectors)


class SpatialGradientFeatures(nn.Module):
    """Compute gradient features with learned complex-linear transform.

    Takes gradient vectors (computed via gradX/gradY operators) and applies
    a learned rotation+scaling in the tangent plane via complex multiplication.
    """

    def __init__(self, C_width, with_gradient_rotations=True):
        super().__init__()
        self.with_gradient_rotations = with_gradient_rotations

        if with_gradient_rotations:
            self.A_re = nn.Linear(C_width, C_width, bias=False)
            self.A_im = nn.Linear(C_width, C_width, bias=False)
        else:
            self.A = nn.Linear(C_width, C_width, bias=False)

    def forward(self, x_grads):
        """
        x_grads: (B, V, C, 2) — gradient vectors (real, imag components)
        Returns: (B, V, C) — scalar gradient features
        """
        if self.with_gradient_rotations:
            x_real = x_grads[..., 0]  # (B, V, C)
            x_imag = x_grads[..., 1]  # (B, V, C)
            # Complex multiplication: (A_re + i*A_im) * (real + i*imag)
            Breal = self.A_re(x_real) - self.A_im(x_imag)
            Bimag = self.A_re(x_imag) + self.A_im(x_real)
            # Dot product (inner product with original)
            dots = x_real * Breal + x_imag * Bimag
        else:
            x_real = x_grads[..., 0]
            x_imag = x_grads[..., 1]
            Breal = self.A(x_real)
            Bimag = self.A(x_imag)
            dots = x_real * Breal + x_imag * Bimag

        return torch.tanh(dots)  # (B, V, C)


class MiniMLP(nn.Sequential):
    """Small MLP with optional dropout (p=0.5 before each layer except first).

    No batch norm or layer norm — matches the official implementation exactly.
    """

    def __init__(self, layer_sizes, dropout=True):
        super().__init__()
        for i in range(len(layer_sizes) - 1):
            is_last = i + 2 == len(layer_sizes)
            if dropout and i > 0:
                self.add_module(f"dropout_{i}", nn.Dropout(p=0.5))
            self.add_module(
                f"linear_{i}", nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            )
            if not is_last:
                self.add_module(f"relu_{i}", nn.ReLU())


class DiffusionNetBlock(nn.Module):
    """One DiffusionNet block: diffusion + (optional gradients) + MLP + skip connection.

    The MLP input is the concatenation of:
    - x_in (original features): C channels
    - x_diffuse (diffused features): C channels
    - x_grad_features (gradient features, optional): C channels

    Total MLP input: 2C (without gradients) or 3C (with gradients).
    """

    def __init__(
        self,
        C_width,
        mlp_hidden_dims=None,
        dropout=True,
        with_gradient_features=True,
        with_gradient_rotations=True,
    ):
        super().__init__()

        if mlp_hidden_dims is None:
            mlp_hidden_dims = [C_width, C_width]

        self.with_gradient_features = with_gradient_features
        self.diffusion = LearnedTimeDiffusion(C_width)

        if with_gradient_features:
            self.gradient_features = SpatialGradientFeatures(
                C_width, with_gradient_rotations
            )
            mlp_in = 3 * C_width  # [x_in, x_diffuse, x_grad]
        else:
            mlp_in = 2 * C_width  # [x_in, x_diffuse]

        self.mlp = MiniMLP([mlp_in] + mlp_hidden_dims + [C_width], dropout=dropout)

    def forward(self, x_in, eigenvectors, eigenvalues, mass, gradX=None, gradY=None):
        """
        x_in: (B, V, C)
        eigenvectors: (V, K)
        eigenvalues: (K,)
        mass: (V,)
        gradX, gradY: (V, V) sparse — optional gradient operators
        Returns: (B, V, C)
        """
        B, V, C = x_in.shape

        # Diffusion
        x_diffuse = self.diffusion(x_in, eigenvectors, eigenvalues, mass)

        # Gradient features (optional)
        if self.with_gradient_features and gradX is not None and gradY is not None:
            x_grads = []
            for b in range(B):
                x_gradX = torch.mm(gradX, x_diffuse[b])  # (V, C)
                x_gradY = torch.mm(gradY, x_diffuse[b])  # (V, C)
                x_grads.append(torch.stack((x_gradX, x_gradY), dim=-1))  # (V, C, 2)
            x_grad = torch.stack(x_grads, dim=0)  # (B, V, C, 2)
            x_grad_features = self.gradient_features(x_grad)  # (B, V, C)
            features = torch.cat((x_in, x_diffuse, x_grad_features), dim=-1)
        else:
            features = torch.cat((x_in, x_diffuse), dim=-1)  # (B, V, 2C)

        # MLP + residual
        x_out = self.mlp(features) + x_in

        return x_out


class DiffusionNet(nn.Module):
    """DiffusionNet — Discretization Agnostic Learning on Surfaces.

    Stacks N_block DiffusionNetBlocks, each applying:
    1. Learned heat diffusion (spatial mixing along surface)
    2. Spatial gradient features (optional, requires gradX/gradY)
    3. Pointwise MLP on concatenated [original, diffused, gradients]
    4. Residual/skip connection

    Input: per-vertex features (B, V, C_in) or (V, C_in)
    Output: per-vertex features (B, V, C_out) or global mean (B, C_out)
    """

    def __init__(
        self,
        C_in,
        C_out,
        C_width=128,
        N_block=4,
        mlp_hidden_dims=None,
        dropout=True,
        with_gradient_features=True,
        with_gradient_rotations=True,
        last_activation=None,
        outputs_at="global_mean",
    ):
        """
        Args:
            C_in: Input feature dimension per vertex
            C_out: Output feature dimension
            C_width: Hidden width of all blocks
            N_block: Number of DiffusionNet blocks
            mlp_hidden_dims: Hidden layer sizes for per-block MLP (default: [C_width, C_width])
            dropout: Use dropout (p=0.5) in MLPs
            with_gradient_features: Use spatial gradient features (requires gradX/gradY)
            with_gradient_rotations: Use complex rotations in gradient features
            last_activation: Activation after final linear (None = no activation)
            outputs_at: "vertices" (per-vertex) or "global_mean" (mass-weighted mean pool)
        """
        super().__init__()

        if mlp_hidden_dims is None:
            mlp_hidden_dims = [C_width, C_width]

        self.C_in = C_in
        self.C_out = C_out
        self.C_width = C_width
        self.outputs_at = outputs_at
        self.last_activation = last_activation

        self.first_lin = nn.Linear(C_in, C_width)

        self.blocks = nn.ModuleList(
            [
                DiffusionNetBlock(
                    C_width=C_width,
                    mlp_hidden_dims=mlp_hidden_dims,
                    dropout=dropout,
                    with_gradient_features=with_gradient_features,
                    with_gradient_rotations=with_gradient_rotations,
                )
                for _ in range(N_block)
            ]
        )

        self.last_lin = nn.Linear(C_width, C_out)

    def forward(self, x_in, mass, eigenvalues, eigenvectors, gradX=None, gradY=None):
        """
        x_in: (B, V, C_in) or (V, C_in) — per-vertex input features
        mass: (V,) — lumped mass matrix diagonal
        eigenvalues: (K,) — Laplacian eigenvalues
        eigenvectors: (V, K) — Laplacian eigenvectors
        gradX, gradY: (V, V) sparse — optional gradient operators

        Returns:
            If outputs_at == "vertices": (B, V, C_out)
            If outputs_at == "global_mean": (B, C_out)
        """
        # Handle unbatched input
        appended_batch = False
        if x_in.dim() == 2:
            x_in = x_in.unsqueeze(0)
            appended_batch = True

        # Input projection (no activation)
        x = self.first_lin(x_in)  # (B, V, C_width)

        # DiffusionNet blocks
        for block in self.blocks:
            x = block(x, eigenvectors, eigenvalues, mass, gradX, gradY)

        # Output projection
        x = self.last_lin(x)  # (B, V, C_out)

        # Output aggregation
        if self.outputs_at == "global_mean":
            # Mass-weighted mean (discretization-invariant)
            x = torch.einsum("bvc,v->bc", x, mass) / mass.sum()
        elif self.outputs_at != "vertices":
            raise ValueError(f"Unknown outputs_at: {self.outputs_at}")

        # Optional final activation
        if self.last_activation is not None:
            x = self.last_activation(x)

        # Remove batch dim if we added it
        if appended_batch:
            x = x.squeeze(0)

        return x
