"""Loss functions and utilities for DiCMA.

This module implements closed-form squared 2-Wasserstein distance between Gaussians,
covariance Frobenius loss, and a cheap relational-distance fallback for Gromov-Wasserstein.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _matrix_sqrt(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute symmetric matrix square-root via eigendecomposition."""
    # mat: (..., d, d)
    # Ensure full precision for eigendecomposition
    mat_fp32 = mat.float()
    eigenvals, eigenvecs = torch.linalg.eigh(mat_fp32)
    # clamp eigenvalues for numerical stability
    eigenvals_clamped = torch.clamp(eigenvals, min=1e-6)
    sqrt_eig = torch.sqrt(eigenvals_clamped)
    # reconstruct
    return (eigenvecs * sqrt_eig.unsqueeze(-2)) @ eigenvecs.transpose(-1, -2)


def w2_gaussian_squared(
    mu1: torch.Tensor,
    Sigma1: torch.Tensor,
    mu2: torch.Tensor,
    Sigma2: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute squared 2-Wasserstein distance between Gaussians.

    W2^2 = ||mu1 - mu2||^2 + Tr(Sigma1 + Sigma2 - 2*(Sigma2^{1/2} Sigma1 Sigma2^{1/2})^{1/2}).

    Supports broadcasting over leading dims.

    Args:
        mu1: (..., d)
        Sigma1: (..., d, d)
        mu2: (..., d)
        Sigma2: (..., d, d)
    Returns:
        (...,) tensor of squared distances.
    """

    diff = mu1 - mu2
    term_mu = torch.sum(diff * diff, dim=-1)

    # Cast to full precision for numerical stability
    mu1 = mu1.float()
    Sigma1 = Sigma1.float()
    mu2 = mu2.float()
    Sigma2 = Sigma2.float()

    # Ensure symmetric PSD
    Sigma1 = 0.5 * (Sigma1 + Sigma1.transpose(-1, -2))
    Sigma2 = 0.5 * (Sigma2 + Sigma2.transpose(-1, -2))

    # Add regularization for positive definiteness
    dim = Sigma1.size(-1)
    Sigma1 = Sigma1 + eps * torch.eye(dim, device=Sigma1.device, dtype=Sigma1.dtype)
    Sigma2 = Sigma2 + eps * torch.eye(dim, device=Sigma2.device, dtype=Sigma2.dtype)

    sqrt_Sigma2 = _matrix_sqrt(Sigma2, eps=eps)
    inside = sqrt_Sigma2 @ Sigma1 @ sqrt_Sigma2
    sqrt_inside = _matrix_sqrt(inside, eps=eps)

    trace_term = torch.diagonal(Sigma1, dim1=-2, dim2=-1).sum(-1) + torch.diagonal(Sigma2, dim1=-2, dim2=-1).sum(-1) - 2 * torch.diagonal(sqrt_inside, dim1=-2, dim2=-1).sum(-1)

    return term_mu + trace_term


def covariance_frobenius_loss(Sigma1: torch.Tensor, Sigma2: torch.Tensor) -> torch.Tensor:
    """Compute Frobenius norm squared between covariances."""
    return torch.sum((Sigma1 - Sigma2) ** 2, dim=(-2, -1))


def entropic_gromov_wasserstein_loss(
    mu_img: torch.Tensor,
    mu_text: torch.Tensor,
    epsilon: float = 1e-1,
    max_iter: int = 100,
) -> torch.Tensor:
    """Compute entropic Gromov-Wasserstein distance between two point sets.

    Falls back to the cheap relational loss when the POT library is not installed.
    """
    try:
        import numpy as np
        import ot
    except ImportError:
        return relational_gw_loss(mu_img, mu_text)

    n = mu_img.shape[0]
    if n == 0:
        return torch.tensor(0.0, device=mu_img.device)

    # cost matrices (squared distances)
    C1 = torch.cdist(mu_img, mu_img, p=2).pow(2).detach().cpu().numpy()
    C2 = torch.cdist(mu_text, mu_text, p=2).pow(2).detach().cpu().numpy()

    p = np.ones(n, dtype=np.float64) / n
    q = np.ones(n, dtype=np.float64) / n

    # entropic GW (returns loss value)
    gw = ot.gromov.entropic_gromov_wasserstein(C1, C2, p, q, 'square_loss', epsilon, max_iter=max_iter)
    return torch.tensor(gw, device=mu_img.device)


def relational_gw_loss(mu_img: torch.Tensor, mu_text: torch.Tensor) -> torch.Tensor:
    """Cheap relational loss approximating Gromov-Wasserstein.

    Aligns pairwise distance matrices of two sets of prototypes.
    """
    # mu_img: (n, d), mu_text: (n, d)
    # compute pairwise squared Euclidean distances
    D_img = torch.cdist(mu_img, mu_img, p=2).pow(2)
    D_text = torch.cdist(mu_text, mu_text, p=2).pow(2)
    return torch.mean((D_img - D_text) ** 2)
