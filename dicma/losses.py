"""Loss functions and utilities for DiCMA.

This module implements closed-form squared 2-Wasserstein distance between Gaussians,
covariance Frobenius loss, and a cheap relational-distance fallback for Gromov-Wasserstein.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _matrix_sqrt(mat: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Compute symmetric matrix square-root via eigendecomposition."""
    # mat: (..., d, d)
    # Ensure full precision for eigendecomposition
    mat_fp32 = mat.float()
    # Add regularization
    dim = mat_fp32.size(-1)
    mat_fp32 = mat_fp32 + eps * torch.eye(dim, device=mat_fp32.device, dtype=mat_fp32.dtype)
    eigenvals, eigenvecs = torch.linalg.eigh(mat_fp32)
    # clamp eigenvalues for numerical stability
    eigenvals_clamped = torch.clamp(eigenvals, min=1e-4)
    sqrt_eig = torch.sqrt(eigenvals_clamped)
    # reconstruct
    return (eigenvecs * sqrt_eig.unsqueeze(-2)) @ eigenvecs.transpose(-1, -2)


def matrix_sqrt_newton_schulz(A: torch.Tensor, num_iters: int = 10, eps: float = 1e-6) -> torch.Tensor:
    """Robust matrix sqrt via Newton-Schulz iteration (fully differentiable, batched).
    
    Args:
        A: (..., d, d) symmetric PSD matrix
        num_iters: number of iterations (default 10)
        eps: regularization epsilon
        
    Returns:
        sqrt(A): (..., d, d)
    """
    batch_shape = A.shape[:-2]
    d = A.shape[-1]
    
    # Ensure symmetry
    A = 0.5 * (A + A.transpose(-2, -1))
    
    # Add small eps for PD
    I = torch.eye(d, device=A.device, dtype=A.dtype).expand(*batch_shape, d, d)
    A = A + eps * I
    
    # Normalize by Frobenius norm
    normA = torch.norm(A.reshape(*batch_shape, -1), dim=-1, keepdim=True).view(*batch_shape, 1, 1)
    Y = A / normA
    Z = I.clone()
    
    for _ in range(num_iters):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    
    sqrtA = Y * torch.sqrt(normA)
    # Ensure symmetry
    sqrtA = 0.5 * (sqrtA + sqrtA.transpose(-2, -1))
    return sqrtA


def w2_gaussian_squared(
    mu1: torch.Tensor,
    Sigma1: torch.Tensor,
    mu2: torch.Tensor,
    Sigma2: torch.Tensor,
    eps: float = 1e-4,
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
    # Cast to full precision for numerical stability
    mu1 = mu1.float()
    Sigma1 = Sigma1.float()
    mu2 = mu2.float()
    Sigma2 = Sigma2.float()

    diff = mu1 - mu2
    term_mu = torch.sum(diff * diff, dim=-1)

    # Ensure symmetric PSD
    Sigma1 = 0.5 * (Sigma1 + Sigma1.transpose(-1, -2))
    Sigma2 = 0.5 * (Sigma2 + Sigma2.transpose(-1, -2))

    # Stabilization: shrinkage toward isotropic + diagonal floor
    def stabilize_cov(S: torch.Tensor, shrink_alpha: float = 0.1) -> torch.Tensor:
        """Apply shrinkage and regularization to covariance matrix."""
        # S: (..., d, d)
        d = S.size(-1)
        # Compute average trace
        tr = torch.diagonal(S, dim1=-2, dim2=-1).sum(-1) / d
        # Shrinkage: (1-alpha)*S + alpha*tr*I
        I = torch.eye(d, device=S.device, dtype=S.dtype).expand(*S.shape[:-2], d, d)
        S = (1 - shrink_alpha) * S + shrink_alpha * (tr.view(*tr.shape, 1, 1) * I)
        # Diagonal floor
        S = S + eps * I
        return S

    Sigma1 = stabilize_cov(Sigma1)
    Sigma2 = stabilize_cov(Sigma2)

    # Use Newton-Schulz for robust matrix sqrt
    sqrt_Sigma2 = matrix_sqrt_newton_schulz(Sigma2, num_iters=8, eps=eps)
    inside = sqrt_Sigma2 @ Sigma1 @ sqrt_Sigma2
    sqrt_inside = matrix_sqrt_newton_schulz(inside, num_iters=8, eps=eps)

    trace_term = (
        torch.diagonal(Sigma1, dim1=-2, dim2=-1).sum(-1) 
        + torch.diagonal(Sigma2, dim1=-2, dim2=-1).sum(-1) 
        - 2 * torch.diagonal(sqrt_inside, dim1=-2, dim2=-1).sum(-1)
    )

    w2sq = term_mu + trace_term
    
    # Clamp to prevent negative values from numerical errors
    if torch.any(w2sq < 0):
        n_neg = (w2sq < 0).sum().item()
        print(f"Warning: {n_neg} negative W2 values detected, clamping to 0")
    w2sq = torch.clamp(w2sq, min=0.0)
    
    return w2sq


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
