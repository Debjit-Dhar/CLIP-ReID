"""Gaussian prototype module for DiCMA.

This module maintains learnable per-ID Gaussian prototypes and computes losses between
batch empirical Gaussians and the prototypes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

from .losses import (
    w2_gaussian_squared,
    covariance_frobenius_loss,
    entropic_gromov_wasserstein_loss,
)


class GaussianPrototypes(nn.Module):
    def __init__(
        self,
        num_ids: int,
        feat_dim: int,
        rank: Optional[int] = 64,
        eps: float = 1e-6,
        ema_momentum: float = 0.01,
        use_relational_gw: bool = False,
    ):
        """Store per-ID Gaussian prototype moments.

        Args:
            num_ids: number of identities (classes).
            feat_dim: dimensionality of input features.
            rank: projected dimension for covariance (if None, uses full feature dimension).
            eps: numerical stability constant.
            ema_momentum: momentum for running moment estimators when batch size / per-id samples are low.
            use_relational_gw: whether to compute inexpensive relational-GW term.
        """
        super().__init__()
        self.num_ids = num_ids
        self.feat_dim = feat_dim
        self.rank = rank if rank is not None else feat_dim
        self.eps = eps
        self.ema_momentum = ema_momentum
        self.use_relational_gw = use_relational_gw

        # Projection from feature space to low-dimensional space for covariance computation.
        if self.rank != self.feat_dim:
            self.register_buffer("proj_matrix", torch.randn(self.feat_dim, self.rank))
        else:
            self.register_buffer("proj_matrix", torch.eye(self.feat_dim))

        # Prototype means and covariance factors in projected space
        self.mu = nn.Parameter(torch.zeros(num_ids, self.rank))
        # Factor L such that Sigma = L @ L^T + eps I (in projected space)
        self.L = nn.Parameter(torch.randn(num_ids, self.rank, self.rank) * 1e-2)

        # EMA buffers for per-ID statistics (projected space)
        self.register_buffer("running_count", torch.zeros(num_ids, dtype=torch.long))
        self.register_buffer("running_mean", torch.zeros(num_ids, self.rank))
        self.register_buffer("running_cov", torch.eye(self.rank).unsqueeze(0).repeat(num_ids, 1, 1))

    def _project(self, features: torch.Tensor) -> torch.Tensor:
        """Project features into a low-dimensional subspace."""
        # features: (B, feat_dim)
        return features @ self.proj_matrix

    def _get_cov_from_L(self, L: torch.Tensor) -> torch.Tensor:
        """Compute covariance matrix from factor L in projected space."""
        # L: (..., r, r)
        cov = L @ L.transpose(-1, -2)
        # ensure positive definiteness
        eye = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype)
        return cov + self.eps * eye

    def _gather_batch_stats(
        self, features: torch.Tensor, ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-ID empirical mean and covariance for the current batch.

        Returns:
            ids_unique: (M,) unique IDs present in the batch
            mean_batch: (M, r)
            cov_batch: (M, r, r)
            counts: (M,) number of samples per ID
        """
        # Project features into low-dimensional space
        z = self._project(features)
        # Cast to full precision for numerical stability in covariance computation
        z = z.float()
        ids = ids.to(torch.long)
        ids_unique, inverse_indices = torch.unique(ids, return_inverse=True)
        M = ids_unique.shape[0]
        r = z.shape[-1]

        # compute means
        sum_z = torch.zeros(M, r, device=z.device, dtype=z.dtype)
        sum_z.index_add_(0, inverse_indices, z)
        counts = torch.bincount(inverse_indices, minlength=M).to(z.dtype)
        mean = sum_z / counts.view(M, 1).clamp(min=1.0)

        # compute covariances (population estimate)
        cov = torch.zeros(M, r, r, device=z.device, dtype=z.dtype)
        for i in range(M):
            mask = inverse_indices == i
            zi = z[mask]
            if zi.shape[0] > 1:
                centered = zi - mean[i : i + 1]
                cov[i] = centered.t() @ centered / zi.shape[0]
            else:
                cov[i] = torch.zeros(r, r, device=z.device, dtype=z.dtype)

        return ids_unique, mean, cov, counts

    def forward(
        self,
        features: torch.Tensor,
        ids: torch.Tensor,
        batch_minibatch_mode: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Compute DiCMA losses for a minibatch.

        Args:
            features: (B, feat_dim) image features.
            ids: (B,) integer identity labels.
            batch_minibatch_mode: if True, only compute losses for IDs present in the batch.

        Returns:
            dict with keys:
                w2_loss, cov_loss, gw_loss (optional), num_ids, diagnostics
        """
        ids_unique, mean_batch, cov_batch, counts = self._gather_batch_stats(features, ids)
        M = ids_unique.shape[0]

        # Update running statistics with batch estimates
        with torch.no_grad():
            momentum = self.ema_momentum
            for idx, uid in enumerate(ids_unique):
                uid_int = int(uid.item())
                self.running_count[uid_int] = self.running_count[uid_int] + int(counts[idx].item())
                # Only update running moments when there is at least one observation.
                self.running_mean[uid_int] = (1 - momentum) * self.running_mean[uid_int] + momentum * mean_batch[idx]
                if counts[idx] > 1:
                    self.running_cov[uid_int] = (1 - momentum) * self.running_cov[uid_int] + momentum * cov_batch[idx]

        # Use running estimates where batch count is < 2
        use_running = counts < 2
        if use_running.any():
            ran_mean = self.running_mean[ids_unique]
            ran_cov = self.running_cov[ids_unique]
            mean_batch = torch.where(use_running.view(-1, 1), ran_mean, mean_batch)
            cov_batch = torch.where(use_running.view(-1, 1, 1), ran_cov, cov_batch)

        # Prototype moments for this batch
        proto_mu = self.mu[ids_unique]
        proto_cov = self._get_cov_from_L(self.L[ids_unique])

        # Compute losses
        w2 = w2_gaussian_squared(mean_batch, cov_batch, proto_mu, proto_cov, eps=self.eps)
        w2_loss = torch.mean(w2)

        cov_loss = torch.mean(covariance_frobenius_loss(cov_batch, proto_cov))

        result = {
            "w2_loss": w2_loss,
            "cov_loss": cov_loss,
            "num_ids": M,
            "mean_batch": mean_batch,
            "proto_mu": proto_mu,
        }

        if self.use_relational_gw:
            # relational term uses per-ID prototype means in projected space
            # This will fallback to a cheap relational loss if POT is not installed.
            gw_loss = entropic_gromov_wasserstein_loss(mean_batch, proto_mu)
            result["gw_loss"] = gw_loss
        else:
            result["gw_loss"] = torch.tensor(0.0, device=features.device)

        return result

    def get_prototype(self, id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return prototype mean and covariance for a single ID."""
        mu = self.mu[id]
        cov = self._get_cov_from_L(self.L[id])
        return mu, cov
