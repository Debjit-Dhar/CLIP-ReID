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
        eps: float = 1e-4,
        ema_momentum: float = 0.01,
        use_relational_gw: bool = False,
        use_overlapping_patches: bool = False,
        num_patches: int = 16,
        patch_size: int = 16,
        patch_stride: int = 8,
        use_side_embedding: bool = False,
        side_embed_dim: int = 128,
    ):
        """Store per-ID Gaussian prototype moments.

        Args:
            num_ids: number of identities (classes).
            feat_dim: dimensionality of input features.
            rank: projected dimension for covariance (if None, uses full feature dimension).
            eps: numerical stability constant.
            ema_momentum: momentum for running moment estimators when batch size / per-id samples are low.
            use_relational_gw: whether to compute inexpensive relational-GW term.
            use_overlapping_patches: whether to use overlapping patches instead of global features.
            num_patches: number of patches to sample when using overlapping patches.
            patch_size: size of each patch.
            patch_stride: stride for overlapping patches.
            use_side_embedding: whether to use side embeddings (camera/view info).
            side_embed_dim: dimensionality of side embeddings.
        """
        super().__init__()
        self.num_ids = num_ids
        self.feat_dim = feat_dim
        self.rank = rank if rank is not None else feat_dim
        self.eps = eps
        self.ema_momentum = ema_momentum
        self.use_relational_gw = use_relational_gw
        self.use_overlapping_patches = use_overlapping_patches
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.use_side_embedding = use_side_embedding
        self.side_embed_dim = side_embed_dim
        self.normalize = True
        self.repel_lambda = 1e-3
        self.repel_sigma = 1.0

        # Projection from feature space to low-dimensional space for covariance computation.
        if self.rank != self.feat_dim:
            self.register_buffer("proj_matrix", torch.randn(self.feat_dim, self.rank))
        else:
            self.register_buffer("proj_matrix", torch.eye(self.feat_dim))

        # Side embedding for camera/view information
        if self.use_side_embedding:
            self.side_embed = nn.Linear(self.side_embed_dim, self.rank)
            self.side_embed.apply(self._init_weights)

        # Prototype means and covariance factors in projected space
        self.mu = nn.Parameter(torch.zeros(num_ids, self.rank))
        # Factor L such that Sigma = L @ L^T + eps I (in projected space)
        self.L = nn.Parameter(torch.randn(num_ids, self.rank, self.rank) * 1e-2)

        # EMA buffers for per-ID statistics (projected space)
        self.register_buffer("running_count", torch.zeros(num_ids, dtype=torch.long))
        self.register_buffer("running_mean", torch.zeros(num_ids, self.rank))
        self.register_buffer("running_cov", torch.eye(self.rank).unsqueeze(0).repeat(num_ids, 1, 1))

    def _init_weights(self, m):
        """Initialize weights for side embedding."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def _project(self, features: torch.Tensor, side_info: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Project features into a low-dimensional subspace.

        Args:
            features: (B, feat_dim) or (B, num_patches, feat_dim) for patch features.
            side_info: (B, side_embed_dim) side information like camera/view embeddings.
        """
        if self.use_overlapping_patches and features.dim() == 3:
            # features: (B, num_patches, feat_dim)
            B, num_patches, feat_dim = features.shape
            features = features.view(B * num_patches, feat_dim)

        # features: (B, feat_dim) or (B*num_patches, feat_dim)
        projected = features @ self.proj_matrix

        # Add side embedding if provided
        if self.use_side_embedding and side_info is not None:
            side_embedded = self.side_embed(side_info)
            if self.use_overlapping_patches and features.dim() == 3:
                # Broadcast side embedding to all patches
                side_embedded = side_embedded.unsqueeze(1).expand(-1, num_patches, -1).reshape(B * num_patches, -1)
            projected = projected + side_embedded

        if self.use_overlapping_patches and features.dim() == 3:
            # Reshape back to (B, num_patches, rank)
            projected = projected.view(B, num_patches, self.rank)

        return projected

    def _get_cov_from_L(self, L: torch.Tensor) -> torch.Tensor:
        """Compute covariance matrix from factor L in projected space."""
        # L: (..., r, r)
        cov = L @ L.transpose(-1, -2)
        # ensure positive definiteness
        eye = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype)
        return cov + self.eps * eye

    def _gather_batch_stats(
        self, features: torch.Tensor, ids: torch.Tensor, side_info: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-ID empirical mean and covariance for the current batch.

        Args:
            features: (B, feat_dim) or (B, num_patches, feat_dim) for patch features.
            ids: (B,) integer identity labels.
            side_info: (B, side_embed_dim) side information.

        Returns:
            ids_unique: (M,) unique IDs present in the batch
            mean_batch: (M, r)
            cov_batch: (M, r, r)
            counts: (M,) number of samples per ID
        """
        # Project features into low-dimensional space
        z = self._project(features, side_info)

        if self.use_overlapping_patches and z.dim() == 3:
            # Aggregate patch features: use mean pooling across patches
            z = z.mean(dim=1)  # (B, rank)

        # Cast to full precision for numerical stability in covariance computation
        z = z.float()
        if self.normalize:
            z = torch.nn.functional.normalize(z, dim=-1)
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
                # Add shrinkage toward isotropic + regularization
                trace_cov = torch.trace(cov[i]) / r
                cov[i] = 0.9 * cov[i] + 0.1 * trace_cov * torch.eye(r, device=cov.device, dtype=cov.dtype)
                cov[i] = cov[i] + 1e-4 * torch.eye(r, device=cov.device, dtype=cov.dtype)
            else:
                cov[i] = 1e-4 * torch.eye(r, device=z.device, dtype=z.dtype)

        return ids_unique, mean, cov, counts

    def forward(
        self,
        features: torch.Tensor,
        ids: torch.Tensor,
        side_info: Optional[torch.Tensor] = None,
        batch_minibatch_mode: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Compute DiCMA losses for a minibatch.

        Args:
            features: (B, feat_dim) or (B, num_patches, feat_dim) image features.
            ids: (B,) integer identity labels.
            side_info: (B, side_embed_dim) side information like camera/view embeddings.
            batch_minibatch_mode: if True, only compute losses for IDs present in the batch.

        Returns:
            dict with keys:
                w2_loss, cov_loss, gw_loss (optional), num_ids, diagnostics
        """
        ids_unique, mean_batch, cov_batch, counts = self._gather_batch_stats(features, ids, side_info)
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
        if self.normalize:
            proto_mu = torch.nn.functional.normalize(proto_mu, dim=-1)
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

        # Add repulsion loss
        all_proto_mu = self.mu
        if self.normalize:
            all_proto_mu = torch.nn.functional.normalize(all_proto_mu, dim=-1)
        pairwise = torch.cdist(all_proto_mu, all_proto_mu, p=2)
        offdiag = pairwise[~torch.eye(pairwise.size(0), dtype=bool, device=pairwise.device)]
        repulsion = torch.exp(-offdiag.pow(2) / (2 * self.repel_sigma ** 2)).mean()
        result["repulsion_loss"] = self.repel_lambda * repulsion

        return result

    def get_prototype(self, id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return prototype mean and covariance for a single ID."""
        mu = self.mu[id]
        cov = self._get_cov_from_L(self.L[id])
        return mu, cov
