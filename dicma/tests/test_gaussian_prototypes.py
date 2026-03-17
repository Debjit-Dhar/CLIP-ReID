import torch

from dicma.gaussian_prototypes import GaussianPrototypes
from dicma.losses import w2_gaussian_squared


def test_w2_gaussian_squared_diagonal():
    # Simple 2D diagonal covariances should reduce to closed-form.
    mu1 = torch.tensor([0.0, 0.0])
    mu2 = torch.tensor([1.0, 2.0])
    sigma1 = torch.diag(torch.tensor([1.0, 4.0]))
    sigma2 = torch.diag(torch.tensor([9.0, 16.0]))

    expected = torch.sum((mu1 - mu2) ** 2) + torch.sum((torch.sqrt(torch.tensor([1.0, 4.0])) - torch.sqrt(torch.tensor([9.0, 16.0]))) ** 2)
    out = w2_gaussian_squared(mu1, sigma1, mu2, sigma2)
    assert torch.allclose(out, expected, atol=1e-3)


def test_gaussian_prototypes_forward_and_grad():
    torch.manual_seed(0)
    num_ids = 5
    feat_dim = 16
    prot = GaussianPrototypes(num_ids=num_ids, feat_dim=feat_dim, rank=8, eps=1e-6, ema_momentum=0.1)

    # create synthetic batch with 3 identities and multiple samples
    ids = torch.tensor([0, 0, 1, 2, 2], dtype=torch.long)
    features = torch.randn(ids.shape[0], feat_dim, requires_grad=True)

    out = prot(features, ids)
    loss = out["w2_loss"] + out["cov_loss"]
    loss.backward()

    # ensure gradients are flowing to prototype parameters
    assert prot.mu.grad is not None
    assert prot.L.grad is not None
    assert prot.mu.grad.abs().sum() > 0
    assert prot.L.grad.abs().sum() > 0


def test_entropic_gw_fallback():
    # Should run even if POT (Python Optimal Transport) is not installed.
    x = torch.randn(4, 8)
    y = torch.randn(4, 8)
    from dicma.losses import entropic_gromov_wasserstein_loss

    out = entropic_gromov_wasserstein_loss(x, y)
    assert out >= 0
