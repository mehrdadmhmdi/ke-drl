from __future__ import annotations

import time
from typing import Optional

import torch

from .Gamma_sa import Gamma_sa
from .G_sa import compute_G_pytorch_batched, compute_transformed_grid_pytorch
from .H_sa import H_sa
from .IS_ULSIF import ULSIFEstimator
from .Phi_sa import Phi_sa
from .ZGrid import ZGrid
from .matern_kernel import matern_kernel
from .optimize import RKDRL_Optimizer


def _kernel_params(nu, length_scale, sigma, alt_nu=None, alt_length_scale=None, alt_sigma=None):
    return {
        "nu": float(nu if alt_nu is None else alt_nu),
        "length_scale": float(length_scale if alt_length_scale is None else alt_length_scale),
        "sigma": float(sigma if alt_sigma is None else alt_sigma),
    }


def _as_2d(name: str, x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x.unsqueeze(0)
    if x.ndim != 2:
        raise ValueError(f"{name} must be rank-1 or rank-2.")
    return x


def KE_DRL(
    *,
    # --- core data ---
    s0: torch.Tensor,
    s1: torch.Tensor,
    a1: torch.Tensor,
    a0: torch.Tensor,
    s_star: torch.Tensor,
    a_star: torch.Tensor,
    r: torch.Tensor,
    discrete_dims: list[int] | None = None,
    # --- policy + kernel/alg params (required by the public API) ---
    target_p_choice: str,
    target_p_params: dict,
    nu: float,
    length_scale: float,
    sigma: float = 1.0,
    gamma_val: float = 0.9,
    lambda_reg: float = 1e-6,
    lambda_B: float = 0.0,
    ratio_lambda_reg: Optional[float] = None,
    x_nu: Optional[float] = None,
    x_length_scale: Optional[float] = None,
    x_sigma: Optional[float] = None,
    ratio_nu: Optional[float] = None,
    ratio_length_scale: Optional[float] = None,
    ratio_sigma: Optional[float] = None,
    num_grid_points: int = 200,
    # --- implementation controls ---
    hull_expand_factor: float = 1.8,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    num_steps: int = 5000,
    target_batch_size: Optional[int] = None,
    random_seed: Optional[int] = None,
    initial_scale: float = 1e-3,
    H_batch_size: int = 10,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
    # --- legacy options kept for API compatibility; ignored by the revised estimator ---
    FP_penalty_lambda: float = 0.0,
    use_low_rank: bool = False,
    rank_for_low_rank: int | None = None,
    B_positive: bool = False,
    fixed_point_constraint: bool = False,
    exact_projection: bool = False,
    ortho_lambda: float = 0.0,
    B_conv: bool = False,
    Sum_one_W: bool = False,
    NonNeg_W: bool = False,
    mass_anchor_lambda: float = 0.0,
    target_mass: float = 1.0,
    B_ridge_penalty: bool = False,
):
    """Fit the global KE-DRL mean-embedding map.

    The returned coefficient matrix B_hat has shape (N, m) and defines the
    conditional embedding weights for any query x through B_hat.T @ k_X(x).
    The `s_star`/`a_star` inputs are interpreted as the target-point set
    X_star used to fit the global objective.
    """
    t0 = time.time()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def TD(x: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(x, dtype=dtype, device=dev)

    s0, s1, a0, a1, s_star, a_star, r = map(TD, (s0, s1, a0, a1, s_star, a_star, r))
    s_star = _as_2d("s_star", s_star)
    a_star = _as_2d("a_star", a_star)

    if s0.ndim != 2 or a0.ndim != 2:
        raise ValueError("s0 and a0 must be 2D: (N, Ds), (N, Da).")
    if s1.ndim != 2 or a1.ndim != 2:
        raise ValueError("s1 and a1 must be 2D: (N, Ds), (N, Da).")
    if s0.shape != s1.shape:
        raise ValueError("s0 and s1 must have the same shape.")
    if a0.shape != a1.shape:
        raise ValueError("a0 and a1 must have the same shape.")
    if s0.shape[0] != a0.shape[0]:
        raise ValueError("s0/a0 row counts must match.")
    if s_star.shape[0] != a_star.shape[0]:
        raise ValueError("s_star/a_star must contain the same number of target points.")
    if s_star.shape[1] != s0.shape[1] or a_star.shape[1] != a0.shape[1]:
        raise ValueError("s_star/a_star feature dimensions must match s0/a0.")
    if r.ndim != 2:
        raise ValueError("r must be 2D: (n_rewards, Dr).")
    if not (0.0 < gamma_val < 1.0):
        raise ValueError("gamma_val must be in (0, 1).")
    if nu <= 1.0:
        raise ValueError("nu must exceed 1.0 for the return-space Matern comparison.")
    if num_grid_points < 2:
        raise ValueError("num_grid_points must be >= 2.")

    x_params = _kernel_params(nu, length_scale, sigma, x_nu, x_length_scale, x_sigma)
    z_params = _kernel_params(nu, length_scale, sigma)
    ratio_params = _kernel_params(
        nu, length_scale, sigma, ratio_nu or x_params["nu"],
        ratio_length_scale or x_params["length_scale"], ratio_sigma or x_params["sigma"],
    )
    ratio_reg = lambda_reg if ratio_lambda_reg is None else ratio_lambda_reg

    if verbose:
        Ds, Da, Dr = s0.shape[1], a0.shape[1], r.shape[1]
        print("=" * 40)
        print("Estimating the global KE-DRL mean embedding")
        print(f"Data dims: N={s0.shape[0]}, L={s_star.shape[0]}, Ds={Ds}, Da={Da}, Dr={Dr}")
        print(f"lambda_Gamma={lambda_reg}, lambda_B={lambda_B}")

    s_a = torch.cat([s0, a0], dim=1)
    s_a_plus = torch.cat([s1, a1], dim=1)
    x_star = torch.cat([s_star, a_star], dim=1)

    ulsif = ULSIFEstimator(
        kernel_func=matern_kernel,
        lambda_reg=ratio_reg,
        **ratio_params,
    )
    alpha = ulsif.fit(s1, a1, target_p_choice, target_p_params, plot=False)
    eta_plus = ulsif.predict(s1, a1).to(dev, dtype).reshape(-1, 1).clamp_min(0.0)
    if verbose:
        try:
            ess_train = ulsif.compute_ess(s1, a1)
            print(f"ESS for eta_plus: {ess_train:.1f} / {s1.shape[0]}")
        except Exception:
            print("ESS unavailable.")

    Z = ZGrid.Z_kmeans(r, n_clusters=int(num_grid_points), constant_factor=float(hull_expand_factor))
    if discrete_dims is not None:
        Z[:, discrete_dims] = torch.round(Z[:, discrete_dims])
        if verbose:
            print(f"Rounded discrete reward dimensions in Z-grid: {discrete_dims}")

    K_X = matern_kernel(s_a, s_a, **x_params)
    K_plus = matern_kernel(s_a, s_a_plus, **x_params)
    k_star = matern_kernel(s_a, x_star, **x_params)
    K_Z = matern_kernel(Z, Z, **z_params)

    Gamma_stack = Gamma_sa(K_X, k_star, lambda_reg)
    Phi_stack = Phi_sa(K_plus, Gamma_stack, eta_plus)

    transformed = compute_transformed_grid_pytorch(Z, r, gamma_val)
    G_stack = compute_G_pytorch_batched(
        transformed, Gamma_stack,
        nu=z_params["nu"], length_scale=z_params["length_scale"], sigma=z_params["sigma"],
    )
    H_stack = H_sa(
        Gamma_stack, gamma_val, r, Z,
        nu=z_params["nu"], length_scale=z_params["length_scale"], sigma=z_params["sigma"],
        batch_size=int(H_batch_size),
    )

    optimizer = RKDRL_Optimizer(device=dev, dtype=dtype)
    B_hat, history_obj, history_be = optimizer.optimize(
        k_sa=k_star,
        K_Zpi=K_Z,
        H_mat=H_stack,
        Phi=Phi_stack,
        G_mat=G_stack,
        K_X=K_X,
        lambda_B=lambda_B,
        target_batch_size=target_batch_size,
        initial_B=None,
        lr=lr,
        weight_decay=weight_decay,
        num_steps=int(num_steps),
        random_seed=random_seed,
        initial_scale=initial_scale,
        FP_penalty_lambda=FP_penalty_lambda,
        use_low_rank=use_low_rank,
        rank=rank_for_low_rank,
        ortho_lambda=ortho_lambda,
        B_positive=B_positive,
        exact_projection=exact_projection,
        fixed_point_constraint=fixed_point_constraint,
        B_conv=B_conv,
        Sum_one_W=Sum_one_W,
        NonNeg_W=NonNeg_W,
        mass_anchor_lambda=mass_anchor_lambda,
        target_mass=target_mass,
        B_ridge_penalty=B_ridge_penalty,
        verbose=verbose,
    )

    B_hat_torch = torch.as_tensor(B_hat, dtype=dtype, device=dev)
    pre_computed_matrices = {
        "Z_grid": Z,
        "X_train": s_a,
        "X_successor": s_a_plus,
        "X_star": x_star,
        "K_X": K_X,
        "K_sa": K_X,          # backward-compatible alias
        "K_plus": K_plus,
        "K_Z": K_Z,
        "k_star": k_star,
        "k_sa": k_star,       # backward-compatible alias
        "Gamma": Gamma_stack,
        "Phi": Phi_stack,
        "H": H_stack,
        "G": G_stack,
        "eta_plus": eta_plus,
        "alpha": torch.as_tensor(alpha, dtype=dtype, device=dev),
        "x_kernel_params": x_params,
        "z_kernel_params": z_params,
        "ratio_kernel_params": ratio_params,
        "lambda_Gamma": torch.as_tensor(lambda_reg, dtype=dtype, device=dev),
        "lambda_B": torch.as_tensor(lambda_B, dtype=dtype, device=dev),
    }

    if verbose:
        print(f"Done in {time.time() - t0:.2f}s.")

    return B_hat_torch, history_obj, history_be, pre_computed_matrices
