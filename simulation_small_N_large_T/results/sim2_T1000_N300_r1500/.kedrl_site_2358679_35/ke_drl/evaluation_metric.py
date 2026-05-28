from __future__ import annotations

import torch

from .matern_kernel import matern_kernel


def _as_feature_columns(name: str, x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError(f"{name} must be rank-1 or rank-2.")
    return x


@torch.no_grad()
def predict_embedding_weights(
    X_train: torch.Tensor,
    X_query: torch.Tensor,
    B_hat_torch: torch.Tensor,
    nu: float,
    length_scale: float,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Compute omega_hat(x; B) = B^T k_X(x) for each query input.

    Returns a matrix with shape (n_query, m_Z).
    """
    device = B_hat_torch.device
    dtype = B_hat_torch.dtype
    X_train = X_train.to(device=device, dtype=dtype)
    X_query = X_query.to(device=device, dtype=dtype)
    K_train_query = matern_kernel(X_train, X_query, nu=nu, length_scale=length_scale, sigma=sigma)
    return K_train_query.T @ B_hat_torch


@torch.no_grad()
def projected_bellman_test_risk(
    *,
    k_current: torch.Tensor,
    phi_current: torch.Tensor,
    B_hat_torch: torch.Tensor,
    K_Z: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Projected Bellman diagnostic used for simulation and real-data evaluation.

    For each held-out target point j this computes

        Delta_j^T B K_Z B^T Delta_j,

    where Delta_j = k_j - Phi_j.  The value is zero exactly when the fitted
    projected embedding is Bellman self-consistent at the tested points. Unlike
    the Monte Carlo prediction risk below, this diagnostic has a zero optimum
    and does not include the irreducible sigma_Z^2 term.
    """
    device = B_hat_torch.device
    dtype = B_hat_torch.dtype
    B_hat = B_hat_torch.to(device=device, dtype=dtype)
    K_Z = K_Z.to(device=device, dtype=dtype)
    k_current = _as_feature_columns("k_current", k_current.to(device=device, dtype=dtype))
    phi_current = _as_feature_columns("phi_current", phi_current.to(device=device, dtype=dtype))

    if k_current.shape != phi_current.shape:
        raise ValueError("k_current and phi_current must have the same shape.")
    if k_current.shape[0] != B_hat.shape[0]:
        raise ValueError("feature rows must match B_hat_torch first dimension.")
    if K_Z.shape != (B_hat.shape[1], B_hat.shape[1]):
        raise ValueError("K_Z shape must match the Z-grid dimension of B_hat_torch.")

    K_Z = 0.5 * (K_Z + K_Z.T)
    coeff_delta = (k_current - phi_current).T @ B_hat
    risk_per_point = torch.sum((coeff_delta @ K_Z) * coeff_delta, dim=1)
    # Remove tiny negative roundoff from nearly-zero quadratic forms.
    risk_per_point = torch.clamp_min(risk_per_point, 0.0)

    if reduction == "none":
        return risk_per_point
    if reduction == "sum":
        return risk_per_point.sum()
    if reduction == "mean":
        return risk_per_point.mean()
    raise ValueError("reduction must be one of {'none', 'sum', 'mean'}.")


@torch.no_grad()
def projected_bellman_test_risk_from_inputs(
    *,
    X_train: torch.Tensor,
    X_successor: torch.Tensor,
    X_test: torch.Tensor,
    B_hat_torch: torch.Tensor,
    K_Z: torch.Tensor,
    eta_plus: torch.Tensor,
    lambda_reg: float,
    x_nu: float,
    x_length_scale: float,
    x_sigma: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Convenience wrapper for the projected Bellman diagnostic from raw X inputs.

    Hot simulation code should prefer reusing precomputed K_X/K_plus when
    available; this wrapper is for public API use and tests.
    """
    from .Gamma_sa import Gamma_sa
    from .Phi_sa import Phi_sa

    device = B_hat_torch.device
    dtype = B_hat_torch.dtype
    X_train = X_train.to(device=device, dtype=dtype)
    X_successor = X_successor.to(device=device, dtype=dtype)
    X_test = X_test.to(device=device, dtype=dtype)
    eta_plus = eta_plus.to(device=device, dtype=dtype)

    K_X = matern_kernel(X_train, X_train, nu=x_nu, length_scale=x_length_scale, sigma=x_sigma)
    K_plus = matern_kernel(X_train, X_successor, nu=x_nu, length_scale=x_length_scale, sigma=x_sigma)
    k_test = matern_kernel(X_train, X_test, nu=x_nu, length_scale=x_length_scale, sigma=x_sigma)
    gamma_test = Gamma_sa(K_X, k_test, lambda_reg)
    phi_test = Phi_sa(K_plus, gamma_test, eta_plus)
    return projected_bellman_test_risk(
        k_current=k_test,
        phi_current=phi_test,
        B_hat_torch=B_hat_torch,
        K_Z=K_Z,
        reduction=reduction,
    )


@torch.no_grad()
def embedding_test_risk(
    Z_test: torch.Tensor,
    k_sa_test: torch.Tensor,
    B_hat_torch: torch.Tensor,
    Z_grid: torch.Tensor,
    nu: float,
    length_scale: float,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Oracle Monte Carlo prediction risk.

    This computes E || k_Z(., Z) - mu_hat(x) ||^2 from simulated true returns.
    It contains the irreducible sigma_Z^2 self-kernel term and therefore is
    not a zero-baseline Bellman diagnostic.  It is kept for simulation-only
    comparison and backward compatibility.
    """
    device = B_hat_torch.device
    dtype = B_hat_torch.dtype

    Z_test = Z_test.to(device=device, dtype=dtype)
    Z_grid = Z_grid.to(device=device, dtype=dtype)
    B_hat = B_hat_torch.to(device=device, dtype=dtype)

    if k_sa_test.ndim == 1:
        k_sa = k_sa_test.unsqueeze(0)
    else:
        k_sa = k_sa_test
    k_sa = k_sa.to(device=device, dtype=dtype)

    m = Z_test.shape[0]
    n_sa, _ = B_hat.shape
    if k_sa.shape[1] != n_sa:
        raise ValueError("k_sa_test second dimension must match B_hat_torch first dimension.")

    beta = k_sa @ B_hat
    K_grid_grid = matern_kernel(Z_grid, Z_grid, nu=nu, length_scale=length_scale, sigma=sigma)
    K_grid_grid = 0.5 * (K_grid_grid + K_grid_grid.T)
    K_test_grid = matern_kernel(Z_test, Z_grid, nu=nu, length_scale=length_scale, sigma=sigma)

    term1 = (sigma ** 2) * torch.ones(m, device=device, dtype=dtype)
    term2 = 2.0 * torch.sum(K_test_grid * beta, dim=1)
    term3 = torch.sum(beta * (K_grid_grid @ beta.T).T, dim=1)
    return (term1 - term2 + term3).mean()


@torch.no_grad()
def embedding_test_risk_from_inputs(
    Z_test: torch.Tensor,
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    B_hat_torch: torch.Tensor,
    Z_grid: torch.Tensor,
    *,
    x_nu: float,
    x_length_scale: float,
    z_nu: float,
    z_length_scale: float,
    x_sigma: float = 1.0,
    z_sigma: float = 1.0,
) -> torch.Tensor:
    """
    Oracle Monte Carlo prediction risk using the global map B^T k_X(x_test).
    """
    device = B_hat_torch.device
    dtype = B_hat_torch.dtype
    X_train = X_train.to(device=device, dtype=dtype)
    X_test = X_test.to(device=device, dtype=dtype)
    if Z_test.shape[0] != X_test.shape[0]:
        raise ValueError("Z_test and X_test must have the same number of rows.")
    k_test_train = matern_kernel(
        X_train, X_test, nu=x_nu, length_scale=x_length_scale, sigma=x_sigma
    ).T
    return embedding_test_risk(
        Z_test=Z_test,
        k_sa_test=k_test_train,
        B_hat_torch=B_hat_torch,
        Z_grid=Z_grid,
        nu=z_nu,
        length_scale=z_length_scale,
        sigma=z_sigma,
    )
