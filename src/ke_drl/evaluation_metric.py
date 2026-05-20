import torch
from .matern_kernel import matern_kernel


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
def embedding_test_risk(
        Z_test: torch.Tensor,        # (m, d_z) Monte Carlo Z_j
        k_sa_test: torch.Tensor,     # (m, n_sa) or (n_sa,)  -> k_sa((s,a)_j, (s,a)_train)
        B_hat_torch: torch.Tensor,   # (n_sa, m_Z)  -> embedding operator
        Z_grid: torch.Tensor,        # (m_Z, d_z) support grid in Z-space
        nu: float,
        length_scale: float,
        sigma: float = 1.0,
) -> torch.Tensor:
    """
    Compute:
      R_hat = (1/m) sum_j || k_Z(·, Z_j) - sum_ℓ β_{jℓ} k_Z(·, Z_grid_ℓ) ||_{H_Z}^2,
    where β_j = k_sa_test[j, :] @ B_hat_torch.
    """

    device = B_hat_torch.device
    dtype = B_hat_torch.dtype

    Z_test = Z_test.to(device=device, dtype=dtype)
    Z_grid = Z_grid.to(device=device, dtype=dtype)
    B_hat = B_hat_torch.to(device=device, dtype=dtype)

    # ensure k_sa_test is 2D: (m, n_sa)
    if k_sa_test.ndim == 1:
        k_sa = k_sa_test.unsqueeze(0)      # single (s,a)
    else:
        k_sa = k_sa_test
    k_sa = k_sa.to(device=device, dtype=dtype)

    m, d_z   = Z_test.shape
    n_sa, mZ = B_hat.shape

    assert k_sa.shape[1] == n_sa, "k_sa_test second dim must match B_hat_torch first dim."

    # β_j coefficients: (m, m_Z)
    Beta = k_sa @ B_hat  # each row j corresponds to β_j

    # Gram matrices in Z-space
    K_grid_grid = matern_kernel(Z_grid, Z_grid, nu=nu, length_scale=length_scale, sigma=sigma)   # (m_Z, m_Z)
    K_grid_grid = 0.5 * (K_grid_grid + K_grid_grid.T)  # symmetrize numerically

    K_test_grid = matern_kernel(Z_test, Z_grid, nu=nu, length_scale=length_scale, sigma=sigma)   # (m, m_Z)

    # term1: k(Z_j, Z_j) = sigma^2 for Matérn at zero distance
    term1 = (sigma ** 2) * torch.ones(m, device=device, dtype=dtype)

    # term2: 2 * sum_ℓ β_{jℓ} k(Z_j, Z_grid_ℓ)
    term2 = 2.0 * torch.sum(K_test_grid * Beta, dim=1)  # (m,)

    # term3: β_j^T K_grid_grid β_j, vectorized over j
    K_beta = K_grid_grid @ Beta.T           # (m_Z, m)
    term3 = torch.sum(Beta * K_beta.T, dim=1)  # (m,)

    risk_per_sample = term1 - term2 + term3   # (m,)
    R_hat = risk_per_sample.mean()            # scalar

    return R_hat


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
    Held-out prediction risk using the global map B^T k_X(x_test).
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

###==================================
# # From KE-DRL run:
# B_hat_torch = results["B_hat_torch"]              # (n_sa, m) or (m, n_sa)
# pre = results["pre_computed_matrices"]
# Z_grid = pre["Z_grid"]                            # (m, d_z)
#
# # Build test tensors:
# Z_test = Z_discounted_test_tensor                 # (N_test, d_z)
# k_sa_test = k_sa_test_tensor                      # (N_test, n_sa), same feature map as pre["k_sa"]
#
# risk = embedding_test_risk(
#     Z_test=Z_test,
#     k_sa_test=k_sa_test,
#     B_hat_torch=B_hat_torch,
#     Z_grid=Z_grid,
#     nu=nu_Z, length_scale=ell_Z, sigma=sigma_Z,
# )
# print("Offline embedding test risk:", risk)
