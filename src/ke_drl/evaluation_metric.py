import torch
from .matern_kernel import matern_kernel


@torch.no_grad()
def embedding_test_risk(
    Z_test: torch.Tensor,        # (N_test, d_z), realized returns
    k_sa_test: torch.Tensor,     # (N_test, n_sa), k_sa(s_i, a_i) features
    B_hat_torch: torch.Tensor,   # (n_sa, m) or (m, n_sa), KE-DRL output
    Z_grid: torch.Tensor,        # (m, d_z), Z_grid from pre_computed_matrices["Z_grid"]
    nu: float,
    length_scale: float,
    sigma: float = 1.0,
    batch_size: int = 4096,
    device: torch.device | None = None,
) -> float:
    """
    Compute
        R_hat = (1/N) sum_i || k_Z(·, Z_i) - mu_hat(s_i, a_i) ||_H^2
    where mu_hat(s,a)(·) = sum_j omega_j(s,a) k_Z(·, z_j)
          and omega_i = B^T k_sa(s_i, a_i).

    Shapes:
        Z_test      : (N_test, d_z)
        k_sa_test   : (N_test, n_sa)
        B_hat_torch : (n_sa, m)  OR  (m, n_sa)
        Z_grid      : (m, d_z)
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Z_test = Z_test.to(device)
    k_sa_test = k_sa_test.to(device)
    B_hat_torch = B_hat_torch.to(device)
    Z_grid = Z_grid.to(device)

    N_test, n_sa = k_sa_test.shape

    # ------------------------------------------------------------------
    # Compute omega_test = B^T k_sa for all test points
    # (handle both (n_sa, m) and (m, n_sa) conventions)
    # ------------------------------------------------------------------
    if B_hat_torch.shape[0] == n_sa:
        # B: (n_sa, m)  => omega_i = k_sa_i @ B  (row form of B^T k_sa)
        omega_test = k_sa_test @ B_hat_torch       # (N_test, m)
    elif B_hat_torch.shape[1] == n_sa:
        # B: (m, n_sa)  => omega_i = k_sa_i @ B^T
        omega_test = k_sa_test @ B_hat_torch.t()   # (N_test, m)
    else:
        raise ValueError(
            f"Incompatible shapes: k_sa_test {k_sa_test.shape}, "
            f"B_hat_torch {B_hat_torch.shape}"
        )

    N_test, m = omega_test.shape

    # ------------------------------------------------------------------
    # Precompute K_Z on the grid and self-term
    # ------------------------------------------------------------------
    K_Z = matern_kernel(
        Z_grid, Z_grid,
        nu=nu, length_scale=length_scale, sigma=sigma
    )                                  # (m, m)
    # For this Matérn implementation k_Z(z, z) = sigma^2
    self_term = sigma ** 2

    total_loss = 0.0

    # ------------------------------------------------------------------
    # Batch over test points for memory
    # ------------------------------------------------------------------
    for start in range(0, N_test, batch_size):
        end = min(start + batch_size, N_test)

        Z_batch = Z_test[start:end]          # (B, d_z)
        omega_batch = omega_test[start:end]  # (B, m)

        # k_Z(Z_i, Z_grid): (B, m)
        K_batch = matern_kernel(
            Z_batch, Z_grid,
            nu=nu, length_scale=length_scale, sigma=sigma
        )

        # ||mu_hat||^2_H = omega K_Z omega^T, per sample
        tmp = omega_batch @ K_Z                   # (B, m)
        mu_norm_sq = (tmp * omega_batch).sum(dim=1)  # (B,)

        # -2 <k_Z(·, Z_i), mu_hat> = -2 sum_j omega_j k_Z(Z_i, z_j)
        cross_term = -2.0 * (omega_batch * K_batch).sum(dim=1)  # (B,)

        # L_i = k_Z(Z_i,Z_i) + cross_term + ||mu_hat||^2_H = sigma^2 + ...
        loss_batch = self_term + cross_term + mu_norm_sq        # (B,)
        total_loss += loss_batch.sum().item()

    risk_hat = total_loss / float(N_test)
    return risk_hat


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
