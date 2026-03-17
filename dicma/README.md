# DiCMA (Distributional Cross-Modal Alignment)

This folder contains an implementation of DiCMA for CLIP-ReID that aligns per-ID image feature distributions with learnable Gaussian text prototypes using Wasserstein-2, covariance, and optional relational-GW losses.

## Key components

- `dicma/gaussian_prototypes.py`: Core module implementing per-ID Gaussian prototypes, empirical moment estimation, and DiCMA loss computation.
- `dicma/losses.py`: Numerically stable closed-form `W2^2` between Gaussians, covariance Frobenius loss, and a cheap relational-GW surrogate.
- `dicma/tests/`: Pytest unit tests for key functionality.
- `dicma/analysis.ipynb`: Notebook for quick diagnostics and sanity checks.

## Running with the existing training pipeline

To enable DiCMA in the standard training script, use the new `DICMA` config section or CLI flags.

Example:

```bash
python train.py --config_file configs/person/vit_dicma.yml
```

Or via CLI flags:

```bash
python train.py --config_file configs/person/vit_base.yml --use_dicma --dicma_alpha 1.0 --dicma_beta 0.1
```

## Hyperparameters

DiCMA introduces the following hyperparameters (default values are in `config/defaults_base.py`):

- `DICMA.ENABLED`: enable/disable DiCMA loss.
- `DICMA.RETAIN_BASELINE`: keep the original baseline loss (classification+triplet) when DiCMA is enabled.
- `DICMA.ALPHA`: weight for W2 loss.
- `DICMA.BETA`: weight for covariance loss.
- `DICMA.GAMMA`: weight for relational GW loss.
- `DICMA.RANK`: projected dimension for covariance (low-rank approximation).
- `DICMA.EPS`: stability epsilon for matrix square root.
- `DICMA.EMA_MOMENTUM`: momentum for running moment estimators.

## Notes

- The DiCMA module projects image features to a low-dimensional subspace (rank) and builds Gaussian prototypes in that space, which keeps computation and memory manageable.
- The implementation is modular and can be extended to support more advanced entropic GW solvers if desired.
