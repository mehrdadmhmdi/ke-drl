from __future__ import annotations

import time
from typing import Optional

import torch

from .Gamma_sa import Gamma_sa
from .G_sa import compute_G_pytorch_batched, compute_transformed_grid_pytorch
from .H_sa import H_sa
from .IS_ULSIF import ULSIFEstimator
from .ZGrid import ZGrid
from .conditioning_basis import select_conditioning_basis
from .matern_kernel import matern_kernel
from .operator_approx import compute_G_rff, compute_H_rff, rff_features, sample_matern_rff
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


def _eta_stats(eta: torch.Tensor) -> dict[str, float]:
    x = eta.detach().reshape(-1)
    finite = torch.isfinite(x)
    if not bool(finite.any()):
        return {
            "mean": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "neg_frac": float("nan"),
            "ess": 0.0,
        }
    x = x[finite]
    w = torch.clamp(x, min=0.0)
    sw = w.sum()
    ess = 0.0
    if float(sw.detach().cpu()) > 0.0:
        ess = float(((sw * sw) / (w.pow(2).sum() + 1e-12)).detach().cpu().item())
    return {
        "mean": float(x.mean().detach().cpu().item()),
        "min": float(x.min().detach().cpu().item()),
        "max": float(x.max().detach().cpu().item()),
        "neg_frac": float((x < 0).double().mean().detach().cpu().item()),
        "ess": ess,
    }


def _is_cuda_device(device: str | torch.device) -> bool:
    try:
        return torch.device(device).type == "cuda"
    except Exception:
        return str(device).startswith("cuda")


def _empty_cuda_cache_if_needed(device: str | torch.device) -> None:
    if torch.cuda.is_available() and _is_cuda_device(device):
        torch.cuda.empty_cache()


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
    ratio_alpha_mix: Optional[float] = None,
    ratio_n_basis: Optional[int] = None,
    ratio_basis_source: str = "numerator",
    ratio_basis_seed: Optional[int] = None,
    ratio_target_sample_multiplier: int = 1,
    ratio_nonnegative_alpha: bool = True,
    ratio_calibrate_mean: bool = True,
    mean_embedding_basis_size: Optional[int] = None,
    mean_embedding_basis_method: str = "full",
    mean_embedding_basis_seed: Optional[int] = None,
    mean_embedding_basis_standardize: bool = True,
    mean_embedding_basis_candidate_pool: Optional[int] = None,
    mean_embedding_basis_max_iter: int = 20,
    mean_embedding_basis_batch_size: int = 8192,
    num_grid_points: int = 200,
    # --- implementation controls ---
    hull_expand_factor: float = 1.8,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    num_steps: int = 5000,
    target_batch_size: Optional[int] = None,
    target_weights: Optional[torch.Tensor] = None,
    random_seed: Optional[int] = None,
    initial_scale: float = 1e-3,
    H_batch_size: int = 10,
    operator_method: str = "exact",
    operator_num_features: int = 128,
    operator_seed: Optional[int] = None,
    ridge_mode: str = "rkhs",
    diagnostic_interval: int = 50,
    eta_clip_min: float | None = 0.0,
    eta_clip_max: float | None = None,
    normalize_eta: bool = False,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    optimize_dtype: Optional[torch.dtype] = None,
    offload_operators: str = "auto",
    return_heavy_matrices: bool = True,
    return_best: bool = True,
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
    mass_anchor_lambda: float = 1.0,
    target_mass: float = 1.0,
    negativity_penalty_lambda: float = 0.0,
    max_B_norm: float | None = None,
    B_ridge_penalty: bool = False,
):
    """Fit the global KE-DRL mean-embedding map.

    The returned coefficient matrix B_hat has shape (L, m), where L is the
    chosen mean-embedding conditioning basis size. It defines the conditional
    embedding weights for any query x through B_hat.T @ psi_L(x), with
    psi_L(x) = k_X(U, x) for the selected basis U.
    The `s_star`/`a_star` inputs are interpreted as the target-point set
    X_star used to fit the global objective. By default the finite-grid
    coefficient mass is anchored at one, matching the non-degeneracy penalty in
    the revised global-B objective.

    Large-scale controls
    --------------------
    optimize_dtype : torch.dtype or None
        Run the Adam loop in a lighter dtype (e.g. ``torch.float32``) while
        keeping all kernel/Cholesky work in the native ``dtype``.  The
        returned B_hat is always cast back to ``dtype``.
    offload_operators : str
        ``"auto"`` (default) keeps H_stack and G_stack on GPU when they fit,
        but moves them to CPU when their combined size exceeds 4 GB.
        ``"cpu"`` always offloads; ``"never"`` keeps on GPU unconditionally.
    return_heavy_matrices : bool
        When true, return the historical full ``pre`` payload, including
        training kernels and Bellman operator stacks.  When false, return only
        the matrices needed for risk evaluation and density recovery and release
        large intermediates before returning.
    ratio_alpha_mix : float or None
        Deprecated for the Bellman IS path.  The Bellman weight is the ordinary
        target-to-logged-data ratio fit directly from samples, so this function
        always fits the uLSIF ratio with ``alpha_mix=0.0`` for ``D_eta``.
        Nonzero values are recorded in diagnostics as requested-but-ignored.
    ratio_basis_source : str
        Kernel-center source for the direct ratio fit. The default
        ``"numerator"`` samples centers from target-policy draws at the logged
        successor states, while the denominator is only the observed
        ``(S_plus, A_plus)`` sample. No behavior-policy model is fit.
    ratio_target_sample_multiplier : int
        Number of target-policy action draws per observed successor state used
        to form numerator samples for direct uLSIF. This does not fit or require
        a behavior policy model.
    ratio_nonnegative_alpha : bool
        Clamp uLSIF kernel coefficients to zero after the linear solve.
    ratio_calibrate_mean : bool
        For the ordinary ratio, rescale the fitted ratio to satisfy the
        empirical density-ratio identity ``mean_data eta = 1``.
    """
    t0 = time.time()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def log_stage(name: str, started: float) -> float:
        if verbose:
            if dev == "cuda" or (hasattr(dev, "type") and dev.type == "cuda"):
                torch.cuda.synchronize()
            print(f"[timing] {name}: {time.time() - started:.2f}s", flush=True)
        return time.time()

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
        print(f"torch device={dev}, dtype={dtype}")
        print(f"Data dims: N={s0.shape[0]}, targets={s_star.shape[0]}, Ds={Ds}, Da={Da}, Dr={Dr}")
        print(f"lambda_Gamma={lambda_reg}, lambda_B={lambda_B}")
        print(
            "return-operator construction: "
            f"method={operator_method}, rff_features={operator_num_features}"
        )
        if mass_anchor_lambda <= 0.0:
            print(
                "Warning: mass_anchor_lambda <= 0 leaves the Bellman objective "
                "homogeneous, so B=0 may be a degenerate minimizer."
            )

    s_a = torch.cat([s0, a0], dim=1)
    s_a_plus = torch.cat([s1, a1], dim=1)
    x_star = torch.cat([s_star, a_star], dim=1)

    requested_alpha_mix = None if ratio_alpha_mix is None else float(ratio_alpha_mix)
    if requested_alpha_mix not in (None, 0.0) and verbose:
        print(
            "Warning: ratio_alpha_mix is ignored for Bellman IS weights. "
            "The Bellman continuation ratio is the ordinary target-to-logged-data "
            "density ratio fit directly from target-policy samples versus logged "
            "samples, so KE_DRL fits plain uLSIF with alpha_mix=0.0 for D_eta.",
            flush=True,
        )

    ulsif = ULSIFEstimator(
        kernel_func=matern_kernel,
        lambda_reg=ratio_reg,
        alpha_mix=0.0,
        **ratio_params,
    )
    alpha = ulsif.fit(
        s1,
        a1,
        target_p_choice,
        target_p_params,
        n_basis=ratio_n_basis,
        basis_source=ratio_basis_source,
        basis_seed=ratio_basis_seed if ratio_basis_seed is not None else random_seed,
        target_sample_multiplier=int(ratio_target_sample_multiplier or 1),
        nonnegative_alpha=bool(ratio_nonnegative_alpha),
        calibrate_mean=bool(ratio_calibrate_mean),
        plot=False,
    )
    eta_plus_raw = ulsif.predict(s1, a1).to(dev, dtype).reshape(-1, 1)
    eta_plus = eta_plus_raw
    if eta_clip_min is not None:
        eta_plus = eta_plus.clamp_min(float(eta_clip_min))
    if eta_clip_max is not None:
        eta_plus = eta_plus.clamp_max(float(eta_clip_max))
    if normalize_eta:
        eta_mean = eta_plus.mean()
        if torch.isfinite(eta_mean) and eta_mean > torch.finfo(dtype).eps:
            eta_plus = eta_plus / eta_mean
    alpha_mix_value = float(getattr(ulsif, "alpha_mix", 0.0))
    raw_eta_stats = _eta_stats(eta_plus_raw)
    used_eta_stats = _eta_stats(eta_plus)
    eta_diagnostics = {
        "ratio_type": "ordinary_target_to_logged_data_direct",
        "bellman_is_weight": (
            "eta(x_plus)=p_target(x_plus)/p_logged_data(x_plus), fit directly "
            "from target-policy samples versus logged samples; no behavior-policy model"
        ),
        "ratio_lambda_reg": float(ratio_reg),
        "requested_ratio_alpha_mix": requested_alpha_mix,
        "used_ratio_alpha_mix": float(alpha_mix_value),
        "ignored_ratio_alpha_mix_for_bellman": bool(requested_alpha_mix not in (None, 0.0)),
        "ratio_alpha_mix": float(alpha_mix_value),
        "ratio_alpha_bound": None,
        "ratio_n_basis": None if ratio_n_basis is None else int(ratio_n_basis),
        "ratio_basis_source": str(ratio_basis_source),
        "eta_clip_min": None if eta_clip_min is None else float(eta_clip_min),
        "eta_clip_max": None if eta_clip_max is None else float(eta_clip_max),
        "normalize_eta": bool(normalize_eta),
        "ratio_target_sample_multiplier": int(ratio_target_sample_multiplier or 1),
        "ratio_nonnegative_alpha": bool(ratio_nonnegative_alpha),
        "ratio_calibrate_mean": bool(ratio_calibrate_mean),
        "fits_behavior_policy_model": False,
        "direct_ratio_estimator": "uLSIF",
        "direct_ratio_fit_diagnostics": dict(getattr(ulsif, "fit_diagnostics", {}) or {}),
        "raw_mean": raw_eta_stats["mean"],
        "raw_min": raw_eta_stats["min"],
        "raw_max": raw_eta_stats["max"],
        "raw_neg_frac": raw_eta_stats["neg_frac"],
        "raw_ess": raw_eta_stats["ess"],
        "used_mean": used_eta_stats["mean"],
        "used_min": used_eta_stats["min"],
        "used_max": used_eta_stats["max"],
        "used_neg_frac": used_eta_stats["neg_frac"],
        "used_ess": used_eta_stats["ess"],
        "n": int(s1.shape[0]),
    }
    if verbose:
        print(
            "ESS for eta_plus: "
            f"raw={eta_diagnostics['raw_ess']:.1f}, "
            f"used={eta_diagnostics['used_ess']:.1f} / {s1.shape[0]}"
        )
        print(
            "eta_plus diagnostics: "
            f"ratio_type={eta_diagnostics['ratio_type']}, "
            f"basis={eta_diagnostics['ratio_basis_source']}, "
            f"target_draws/state={eta_diagnostics['ratio_target_sample_multiplier']}, "
            f"no_behavior_model={not eta_diagnostics['fits_behavior_policy_model']}, "
            f"used_alpha_mix={eta_diagnostics['used_ratio_alpha_mix']:.3g}, "
            f"requested_alpha_mix={eta_diagnostics['requested_ratio_alpha_mix']}, "
            f"clip=[{eta_diagnostics['eta_clip_min']}, {eta_diagnostics['eta_clip_max']}], "
            f"normalize={eta_diagnostics['normalize_eta']}; "
            f"raw_mean={eta_diagnostics['raw_mean']:.3g}, "
            f"raw_min={eta_diagnostics['raw_min']:.3g}, "
            f"raw_max={eta_diagnostics['raw_max']:.3g}, "
            f"raw_neg%={100.0 * eta_diagnostics['raw_neg_frac']:.2f}; "
            f"used_mean={eta_diagnostics['used_mean']:.3g}, "
            f"used_min={eta_diagnostics['used_min']:.3g}, "
            f"used_max={eta_diagnostics['used_max']:.3g}"
        )
    del ulsif

    stage_t = time.time()
    Z = ZGrid.Z_kmeans(r, n_clusters=int(num_grid_points), constant_factor=float(hull_expand_factor))
    if discrete_dims is not None:
        Z[:, discrete_dims] = torch.round(Z[:, discrete_dims])
        if verbose:
            print(f"Rounded discrete reward dimensions in Z-grid: {discrete_dims}")
    stage_t = log_stage("Z-grid", stage_t)

    K_X = matern_kernel(s_a, s_a, **x_params)
    stage_t = log_stage("K_X", stage_t)
    K_plus = matern_kernel(s_a, s_a_plus, **x_params)
    stage_t = log_stage("K_plus", stage_t)
    k_star_full = matern_kernel(s_a, x_star, **x_params)
    stage_t = log_stage("k_star_full", stage_t)
    Gamma_stack = Gamma_sa(K_X, k_star_full, lambda_reg)
    stage_t = log_stage("Gamma_stack", stage_t)

    X_basis, basis_indices, basis_meta = select_conditioning_basis(
        s_a,
        n_basis=mean_embedding_basis_size,
        method=mean_embedding_basis_method,
        seed=mean_embedding_basis_seed if mean_embedding_basis_seed is not None else random_seed,
        standardize=bool(mean_embedding_basis_standardize),
        candidate_pool=mean_embedding_basis_candidate_pool,
        max_iter=int(mean_embedding_basis_max_iter),
        batch_size=int(mean_embedding_basis_batch_size),
        device=torch.device(dev),
        verbose=verbose,
    )
    stage_t = log_stage("mean embedding conditioning basis", stage_t)
    basis_is_full_train = X_basis.shape[0] == s_a.shape[0] and torch.equal(
        basis_indices.to(device=s_a.device), torch.arange(s_a.shape[0], device=s_a.device)
    )
    if basis_is_full_train:
        K_basis = K_X
        K_basis_plus = K_plus
        k_star = k_star_full
    else:
        K_basis = matern_kernel(X_basis, X_basis, **x_params)
        stage_t = log_stage("K_basis", stage_t)
        K_basis_plus = matern_kernel(X_basis, s_a_plus, **x_params)
        stage_t = log_stage("K_basis_plus", stage_t)
        k_star = matern_kernel(X_basis, x_star, **x_params)
        stage_t = log_stage("k_star_basis", stage_t)

    Phi_stack = K_basis_plus @ (Gamma_stack * eta_plus)
    stage_t = log_stage("Phi_stack_basis", stage_t)
    if verbose:
        print(
            "Mean-embedding parameterization: "
            f"B rows L={X_basis.shape[0]} over current X=(S,A) basis; "
            f"Z-grid columns m={Z.shape[0]}; raw transition rows N={s_a.shape[0]}"
        )

    # ---- Auto-scaling for large problems --------------------------------
    n_data, m_grid = r.shape[0], Z.shape[0]
    L_targets = Gamma_stack.shape[1] if Gamma_stack.ndim == 2 else 1
    _elem_bytes = 8 if dtype == torch.float64 else 4
    _stack_bytes = 2 * L_targets * m_grid * m_grid * _elem_bytes  # H + G combined
    _stack_gb = _stack_bytes / (1024 ** 3)

    # Auto-select RFF when exact is infeasible (G alone is O(m^2 * N^2 * L))
    operator_method_l = str(operator_method).lower()
    _exact_ops = float(m_grid ** 2) * float(n_data ** 2) * float(L_targets)
    if operator_method_l in {"exact", "full"} and _exact_ops > 1e11:
        if verbose:
            print(
                f"Warning: exact G operator would require ~{_exact_ops:.1e} kernel "
                f"evaluations. Switching to operator_method='rff' with "
                f"{operator_num_features} features for feasibility."
            )
        operator_method_l = "rff"

    # Auto-tune H_batch_size for GPU utilization
    if H_batch_size <= 10 and m_grid > 50:
        H_batch_size = max(10, min(m_grid // 5, 100))
        if verbose:
            print(f"Auto-tuned H_batch_size={H_batch_size} for m={m_grid}")

    # Decide whether to offload H/G stacks to CPU
    offload_l = str(offload_operators).lower()
    _GPU_BUDGET_GB = 4.0
    _do_offload = (
        offload_l == "cpu"
        or (offload_l == "auto" and _stack_gb > _GPU_BUDGET_GB)
    )
    if _do_offload and verbose:
        print(
            f"H+G stack size ~{_stack_gb:.1f} GB (L={L_targets}, m={m_grid}); "
            "offloading to CPU — optimizer will stream mini-batch slices."
        )

    if operator_method_l in {"exact", "full"}:
        K_Z = matern_kernel(Z, Z, **z_params)
        stage_t = log_stage("K_Z exact", stage_t)
        if verbose:
            n, m, L = n_data, m_grid, L_targets
            print(
                "Exact return-operator work estimate: "
                f"H ~ {L * m * m * n:.3e} kernel terms, "
                f"G ~ {L * m * m * n * n:.3e} kernel terms",
                flush=True,
            )
        transformed = compute_transformed_grid_pytorch(Z, r, gamma_val)
        stage_t = log_stage("transformed reward grid", stage_t)
        G_stack = compute_G_pytorch_batched(
            transformed, Gamma_stack,
            nu=z_params["nu"], length_scale=z_params["length_scale"], sigma=z_params["sigma"],
        )
        stage_t = log_stage("G_stack exact", stage_t)
        H_stack = H_sa(
            Gamma_stack, gamma_val, r, Z,
            nu=z_params["nu"], length_scale=z_params["length_scale"], sigma=z_params["sigma"],
            batch_size=int(H_batch_size),
        )
        stage_t = log_stage("H_stack exact", stage_t)
    elif operator_method_l in {"rff", "random_fourier", "random-fourier"}:
        omega, phase, rff_scale = sample_matern_rff(
            num_features=int(operator_num_features),
            input_dim=r.shape[1],
            nu=z_params["nu"],
            length_scale=z_params["length_scale"],
            sigma=z_params["sigma"],
            device=torch.device(dev),
            dtype=dtype,
            seed=operator_seed,
        )
        stage_t = log_stage("RFF feature sample", stage_t)
        feat_Z = rff_features(Z, omega, phase, rff_scale)
        K_Z = feat_Z @ feat_Z.transpose(0, 1)
        stage_t = log_stage("K_Z RFF", stage_t)
        G_stack = compute_G_rff(
            Gamma_stack, gamma_val, r, Z,
            omega=omega, phase=phase, scale=rff_scale,
        )
        stage_t = log_stage("G_stack RFF", stage_t)
        H_stack = compute_H_rff(
            Gamma_stack, gamma_val, r, Z,
            omega=omega, phase=phase, scale=rff_scale,
            batch_size=int(H_batch_size),
        )
        stage_t = log_stage("H_stack RFF", stage_t)
    else:
        raise ValueError(f"Unknown operator_method={operator_method!r}. Use 'exact' or 'rff'.")

    # Offload H/G to CPU to free GPU memory for the optimizer.
    if _do_offload and H_stack.is_cuda:
        H_stack = H_stack.cpu()
        G_stack = G_stack.cpu()
        _empty_cuda_cache_if_needed(dev)
        stage_t = log_stage("offload H/G to CPU", stage_t)

    optimizer = RKDRL_Optimizer(device=dev, dtype=dtype)
    B_hat, history_obj, history_be = optimizer.optimize(
        k_sa=k_star,
        K_Zpi=K_Z,
        H_mat=H_stack,
        Phi=Phi_stack,
        G_mat=G_stack,
        K_X=K_basis,
        lambda_B=lambda_B,
        target_batch_size=target_batch_size,
        initial_B=None,
        lr=lr,
        weight_decay=weight_decay,
        num_steps=int(num_steps),
        random_seed=random_seed,
        initial_scale=initial_scale,
        target_weights=target_weights,
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
        negativity_penalty_lambda=negativity_penalty_lambda,
        max_B_norm=max_B_norm,
        B_ridge_penalty=B_ridge_penalty,
        ridge_mode=ridge_mode,
        diagnostic_interval=diagnostic_interval,
        return_best=return_best,
        optimize_dtype=optimize_dtype,
        verbose=verbose,
    )

    B_hat_torch = B_hat  # already on correct device/dtype from optimizer
    pre_computed_matrices = {
        "Z_grid": Z,
        "X_basis": X_basis,
        "basis_indices": basis_indices,
        "mean_embedding_basis": basis_meta,
        "K_sa": K_basis,      # backward-compatible alias for the fitted feature Gram
        "K_basis": K_basis,
        "K_Z": K_Z,
        "k_star": k_star,
        "k_sa": k_star,       # backward-compatible alias
        "Phi": Phi_stack,
        "eta_plus": eta_plus,
        "eta_plus_raw": eta_plus_raw,
        "eta_diagnostics": eta_diagnostics,
        "optimizer_diagnostics": optimizer.last_diagnostics,
        "x_kernel_params": x_params,
        "z_kernel_params": z_params,
        "ratio_kernel_params": ratio_params,
        "lambda_Gamma": torch.as_tensor(lambda_reg, dtype=dtype, device=dev),
        "lambda_B": torch.as_tensor(lambda_B, dtype=dtype, device=dev),
    }
    if return_heavy_matrices:
        pre_computed_matrices.update({
            "X_train": s_a,
            "X_successor": s_a_plus,
            "X_star": x_star,
            "K_X": K_X,
            "K_plus": K_plus,
            "K_basis_plus": K_basis_plus,
            "k_star_full": k_star_full,
            "Gamma": Gamma_stack,
            "H": H_stack,
            "G": G_stack,
            "alpha": torch.as_tensor(alpha, dtype=dtype, device=dev),
        })
    else:
        del s0, s1, a0, a1, s_star, a_star, r
        del s_a, s_a_plus, x_star, K_X, K_plus, K_basis_plus
        del k_star_full, Gamma_stack, H_stack, G_stack, alpha
        try:
            del transformed
        except UnboundLocalError:
            pass
        try:
            del omega, phase, rff_scale, feat_Z
        except UnboundLocalError:
            pass
        _empty_cuda_cache_if_needed(dev)

    if verbose:
        print(f"Done in {time.time() - t0:.2f}s.")

    return B_hat_torch, history_obj, history_be, pre_computed_matrices
