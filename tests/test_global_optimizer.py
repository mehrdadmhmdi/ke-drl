import torch

from ke_drl.optimize import RKDRL_Optimizer


def test_mass_and_negativity_penalties_prevent_zero_solution():
    dtype = torch.float64
    k_star = torch.eye(3, dtype=dtype)
    phi = torch.zeros_like(k_star)
    K_Z = torch.eye(2, dtype=dtype)
    H = torch.zeros(3, 2, 2, dtype=dtype)
    G = torch.zeros(3, 2, 2, dtype=dtype)
    K_X = torch.eye(3, dtype=dtype)

    opt = RKDRL_Optimizer(device="cpu", dtype=dtype)
    B, _, _ = opt.optimize(
        k_sa=k_star,
        K_Zpi=K_Z,
        H_mat=H,
        Phi=phi,
        G_mat=G,
        K_X=K_X,
        lambda_B=0.0,
        lr=5e-2,
        weight_decay=0.0,
        num_steps=300,
        initial_scale=1e-3,
        random_seed=123,
        mass_anchor_lambda=100.0,
        target_mass=1.0,
        negativity_penalty_lambda=10.0,
        verbose=False,
    )

    beta_targets = k_star.T @ B
    masses = beta_targets.sum(dim=1)
    assert torch.all(masses > 0.95)
    assert float(beta_targets.min()) > -1e-3
    assert torch.linalg.vector_norm(B) > 0.1


def test_frobenius_projection_bounds_B():
    dtype = torch.float64
    k_star = torch.eye(2, dtype=dtype)
    phi = torch.zeros_like(k_star)
    K_Z = torch.eye(2, dtype=dtype)
    H = torch.zeros(2, 2, 2, dtype=dtype)
    G = torch.zeros(2, 2, 2, dtype=dtype)
    K_X = torch.eye(2, dtype=dtype)

    opt = RKDRL_Optimizer(device="cpu", dtype=dtype)
    B, _, _ = opt.optimize(
        k_sa=k_star,
        K_Zpi=K_Z,
        H_mat=H,
        Phi=phi,
        G_mat=G,
        K_X=K_X,
        lr=1e-1,
        num_steps=20,
        initial_scale=10.0,
        random_seed=123,
        max_B_norm=0.5,
        verbose=False,
    )

    assert torch.linalg.vector_norm(B) <= 0.5 + 1e-10
