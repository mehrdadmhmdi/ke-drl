import torch

from ke_drl.evaluation_metric import projected_bellman_test_risk


def test_projected_bellman_test_risk_matches_quadratic_form():
    dtype = torch.float64
    k_current = torch.tensor([[1.0, 0.2], [0.5, -0.1], [-0.4, 0.3]], dtype=dtype)
    phi_current = torch.tensor([[0.2, 0.1], [0.1, 0.0], [0.3, -0.2]], dtype=dtype)
    B = torch.tensor([[0.4, -0.2], [0.1, 0.3], [-0.5, 0.7]], dtype=dtype)
    K_Z = torch.tensor([[1.0, 0.25], [0.25, 0.8]], dtype=dtype)

    delta_coeff = (k_current - phi_current).T @ B
    expected = torch.sum((delta_coeff @ K_Z) * delta_coeff, dim=1)

    got_each = projected_bellman_test_risk(
        k_current=k_current,
        phi_current=phi_current,
        B_hat_torch=B,
        K_Z=K_Z,
        reduction="none",
    )
    got_mean = projected_bellman_test_risk(
        k_current=k_current,
        phi_current=phi_current,
        B_hat_torch=B,
        K_Z=K_Z,
    )

    assert torch.allclose(got_each, expected)
    assert torch.allclose(got_mean, expected.mean())


def test_projected_bellman_test_risk_zero_when_delta_is_zero():
    dtype = torch.float64
    k_current = torch.randn(4, 3, dtype=dtype)
    B = torch.randn(4, 5, dtype=dtype)
    K_Z = torch.eye(5, dtype=dtype)

    got = projected_bellman_test_risk(
        k_current=k_current,
        phi_current=k_current.clone(),
        B_hat_torch=B,
        K_Z=K_Z,
    )

    assert float(got) == 0.0
