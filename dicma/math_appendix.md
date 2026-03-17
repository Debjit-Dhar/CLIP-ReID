# DiCMA Math Appendix

## Closed-form 2-Wasserstein distance between Gaussians

Given two Gaussians:

- $\mathcal{N}(\mu_1, \Sigma_1)$
- $\mathcal{N}(\mu_2, \Sigma_2)$

The squared 2-Wasserstein distance has a closed form:

$$
W_2^2 = \lVert \mu_1 - \mu_2 \rVert_2^2 + \mathrm{Tr}\left(\Sigma_1 + \Sigma_2 - 2\left(\Sigma_2^{1/2} \Sigma_1 \Sigma_2^{1/2}\right)^{1/2}\right)
$$

### Stable computation via eigendecomposition

To compute the matrix square root in a numerically stable manner, we use an eigendecomposition:

1. Compute eigen-decomposition: $A = Q \Lambda Q^T$.
2. Clamp eigenvalues: $\hat{\Lambda} = \mathrm{diag}(\max(\Lambda, \epsilon))$.
3. Compute square root: $A^{1/2} = Q \sqrt{\hat{\Lambda}} Q^T$.

This is implemented in `dicma.losses._matrix_sqrt` and is used in `w2_gaussian_squared`.

## Covariance matching

The covariance Frobenius loss is simply:

$$
L_{\text{cov}} = \lVert \Sigma^I - \Sigma^P \rVert_F^2
$$

where $\Sigma^I$ is the empirical covariance (from the image features) and $\Sigma^P$ is the prototype covariance.

## Sampling and EMA for stability

For identities with very few examples in a minibatch, empirical covariance estimates are noisy. We maintain running estimates via an exponential moving average (EMA):

$$
\mu_{t} = (1 - m) \mu_{t-1} + m \hat{\mu}_t
$$
$$
\Sigma_{t} = (1 - m) \Sigma_{t-1} + m \hat{\Sigma}_t
$$

with momentum $m$ controlled by `DICMA.EMA_MOMENTUM`.

## Relational (cheap GW) loss

A lightweight surrogate for Gromov-Wasserstein alignment is the relational distance between pairwise distance matrices:

$$
L_{\text{rel}} = \lVert D^I - D^P \rVert_F^2
$$

where $D^I_{ij} = \lVert \mu^I_i - \mu^I_j \rVert^2$ and $D^P_{ij}$ is the equivalent matrix for prototype means.

---

*This appendix is meant as a lightweight companion to the implementation in `dicma/`.*
