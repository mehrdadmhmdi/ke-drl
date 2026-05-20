"""Numerical equivalence tests for the vectorized stacked H/G paths.

After the L-stacked refactor of H_sa and G_sa, the new path must produce the
same operators as evaluating per ℓ on the single-Γ path. These tests pin the
agreement at float64 with small toy inputs and at random non-trivial sizes.
"""
import torch

from ke_drl.G_sa import compute_G_pytorch_batched, compute_transformed_grid_pytorch
from ke_drl.H_sa import H_sa


def _params():
    return dict(nu=2.5, length_scale=0.9, sigma=0.7, gamma=0.85)


def test_H_stacked_matches_per_target_loop():
    torch.manual_seed(0)
    dtype = torch.float64
    p = _params()

    n, m, d, L = 12, 7, 3, 5
    R = torch.randn(n, d, dtype=dtype)
    Z = torch.randn(m, d, dtype=dtype)
    Gamma = torch.randn(n, L, dtype=dtype)

    H_stack = H_sa(Gamma, p["gamma"], R, Z,
                   nu=p["nu"], length_scale=p["length_scale"], sigma=p["sigma"],
                   batch_size=3)
    assert H_stack.shape == (L, m, m)

    for ell in range(L):
        H_ref = H_sa(Gamma[:, ell].contiguous(), p["gamma"], R, Z,
                     nu=p["nu"], length_scale=p["length_scale"], sigma=p["sigma"],
                     batch_size=3)
        assert torch.allclose(H_stack[ell], H_ref, atol=1e-12, rtol=1e-10)


def test_G_stacked_matches_per_target_loop():
    torch.manual_seed(1)
    dtype = torch.float64
    p = _params()

    n, m, d, L = 10, 6, 2, 4
    R = torch.randn(n, d, dtype=dtype)
    Z = torch.randn(m, d, dtype=dtype)
    Gamma = torch.randn(n, L, dtype=dtype)

    transformed = compute_transformed_grid_pytorch(Z, R, p["gamma"])

    G_stack = compute_G_pytorch_batched(
        transformed, Gamma,
        nu=p["nu"], length_scale=p["length_scale"], sigma=p["sigma"],
    )
    assert G_stack.shape == (L, m, m)

    for ell in range(L):
        G_ref = compute_G_pytorch_batched(
            transformed, Gamma[:, ell].contiguous(),
            nu=p["nu"], length_scale=p["length_scale"], sigma=p["sigma"],
        )
        assert torch.allclose(G_stack[ell], G_ref, atol=1e-12, rtol=1e-10)


def test_G_stacked_is_symmetric_per_target():
    torch.manual_seed(2)
    dtype = torch.float64
    n, m, d, L = 8, 5, 2, 3
    R = torch.randn(n, d, dtype=dtype)
    Z = torch.randn(m, d, dtype=dtype)
    Gamma = torch.randn(n, L, dtype=dtype)

    transformed = compute_transformed_grid_pytorch(Z, R, 0.9)
    G_stack = compute_G_pytorch_batched(
        transformed, Gamma, nu=1.5, length_scale=1.1, sigma=0.6,
    )
    for ell in range(L):
        assert torch.allclose(G_stack[ell], G_stack[ell].T, atol=1e-12)
