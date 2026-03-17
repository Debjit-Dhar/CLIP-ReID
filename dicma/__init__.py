"""Distributional Cross-Modal Alignment (DiCMA) module.

This module provides tools for aligning image features to learned per-ID Gaussian prototypes
via closed-form Wasserstein-2 and covariance matching losses.
"""

from .gaussian_prototypes import GaussianPrototypes
from .losses import w2_gaussian_squared, covariance_frobenius_loss, relational_gw_loss

__all__ = [
    "GaussianPrototypes",
    "w2_gaussian_squared",
    "covariance_frobenius_loss",
    "relational_gw_loss",
]
