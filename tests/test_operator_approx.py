import torch

from ke_drl.operator_approx import compute_G_rff, compute_H_rff, rff_features, sample_matern_rff


def test_rff_operator_shapes_and_finiteness():
    dtype = torch.float64
    R = torch.randn(6, 3, dtype=dtype)
    Z = torch.randn(5, 3, dtype=dtype)
    Gamma = torch.randn(6, 4, dtype=dtype)

    omega, phase, scale = sample_matern_rff(
        num_features=16,
        input_dim=3,
        nu=3.5,
        length_scale=1.0,
        sigma=0.7,
        device=torch.device("cpu"),
        dtype=dtype,
        seed=123,
    )
    G = compute_G_rff(Gamma, 0.8, R, Z, omega=omega, phase=phase, scale=scale)
    H = compute_H_rff(Gamma, 0.8, R, Z, omega=omega, phase=phase, scale=scale, batch_size=2)

    assert G.shape == (4, 5, 5)
    assert H.shape == (4, 5, 5)
    assert torch.isfinite(G).all()
    assert torch.isfinite(H).all()
    assert torch.allclose(G, G.transpose(1, 2), atol=1e-10)


def test_rff_residual_quadratic_is_nonnegative_when_features_are_shared():
    dtype = torch.float64
    R = torch.randn(6, 3, dtype=dtype)
    Z = torch.randn(5, 3, dtype=dtype)
    Gamma = torch.randn(6, 4, dtype=dtype)

    omega, phase, scale = sample_matern_rff(
        num_features=24,
        input_dim=3,
        nu=3.5,
        length_scale=1.0,
        sigma=0.7,
        device=torch.device("cpu"),
        dtype=dtype,
        seed=321,
    )
    feat_Z = rff_features(Z, omega, phase, scale)
    K_Z = feat_Z @ feat_Z.T
    G = compute_G_rff(Gamma, 0.8, R, Z, omega=omega, phase=phase, scale=scale)
    H = compute_H_rff(Gamma, 0.8, R, Z, omega=omega, phase=phase, scale=scale, batch_size=2)

    u = torch.randn(4, 5, dtype=dtype)
    v = torch.randn(4, 5, dtype=dtype)
    residuals = (
        torch.einsum("li,ij,lj->l", u, K_Z, u)
        - 2.0 * torch.einsum("li,lij,lj->l", u, H, v)
        + torch.einsum("li,lij,lj->l", v, G, v)
    )

    assert torch.all(residuals >= -1e-10)
