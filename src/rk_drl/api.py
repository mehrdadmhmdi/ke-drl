import torch
from .matern_kernel import matern_kernel
from .Gamma_sa import Gamma_sa as solve_gamma
from .H_sa import H_sa
from .G_sa import compute_transformed_grid_pytorch, compute_G_pytorch_batched
from .Phi_sa import Phi_sa
from .IS_ULSIF import ULSIFEstimator
from .ZGrid import ZGrid

def _cat_sa(s, a):
    if s.ndim == 1: s = s.unsqueeze(0)
    if a.ndim == 1: a = a.unsqueeze(0)
    return torch.cat([s, a], dim=1)

@torch.inference_mode(False)
def estimate_embedding(
    *,
    s0, s1, a0, a1,
    s_star, a_star,
    r,                            # r0
    target_p_choice, target_p_params,
    nu, length_scale, sigma,
    gamma_val, lambda_reg,
    num_grid_points, hull_expand_factor,
    lr, weight_decay, num_steps,
    FP_penalty_lambda,
    use_low_rank, rank_for_low_rank,
    B_positive, fixed_point_constraint, exact_projection,
    ortho_lambda, B_conv, Sum_one_W, NonNeg_W,
    mass_anchor_lambda, target_mass,
    B_ridge_penalty,
    H_batch_size,
    device=None, dtype=torch.float64,
):
    """
    Returns: dict with {"B": B, "Z_grid": Z, "hist_obj": hist_obj, "hist_be": hist_be, "weights": w}
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    toT = lambda x: torch.as_tensor(x, device=dev, dtype=dtype)

    # --- tensors
    s0, s1, a0, a1 = map(toT, (s0, s1, a0, a1))
    s_star, a_star = map(toT, (s_star, a_star))
    r = toT(r)

    # ========== 1) Behavior SA kernels ==========
    Xb = _cat_sa(s0, a0)                      # (n, d_s+d_a)
    x_star = _cat_sa(s_star, a_star)          # (1, d)
    K_sa = matern_kernel(Xb, Xb, nu, length_scale, sigma)         # (n,n)
    k_sa = matern_kernel(Xb, x_star, nu, length_scale, sigma).squeeze(1)  # (n,)

    # ========== 2) Importance weights via uLSIF (policy shift) ==========
    ul = ULSIFEstimator(lambda_reg=lambda_reg, nu=nu, length_scale=length_scale, sigma=sigma)
    alpha = ul.fit(S=s0, A=a0, target_p_choice=target_p_choice, target_p_params=target_p_params, plot=False)
    # save ESS diagnostic if helpful
    # ess = ul.compute_ess(s0, a0)

    # ========== 3) Γ(s,a) solve (K + nλI)^{-1} k ==========
    Gamma = solve_gamma(K_sa, k_sa, lambda_reg)                  # (n,)

    # ========== 4) Z-grid from rewards ==========
    Z_grid = ZGrid.Z_kmeans(r, n_clusters=num_grid_points, constant_factor=hull_expand_factor)  # (m,d_r)

    # ========== 5) H, G, K_Z ==========
    H = H_sa(Gamma, gamma_val, r, Z_grid, nu, length_scale, sigma, batch_size=int(H_batch_size))
    T = compute_transformed_grid_pytorch(Z_grid, r, gamma_val)   # (m,n,d_r)
    G = compute_G_pytorch_batched(T, Gamma, nu, length_scale, sigma)  # (m,m)
    K_Z = matern_kernel(Z_grid, Z_grid, nu, length_scale, sigma)      # (m,m)

    # ========== 6) Φ vector (use K_sa as K_sa_prime here) ==========
    Phi = Phi_sa(K_sa, Gamma, alpha.squeeze(-1))                 # (n,1)

    # ========== 7) Optimize B ==========
    from .optimize import RKDRL_Optimizer
    opt = RKDRL_Optimizer(device=dev, dtype=dtype)
    B, hist_obj, hist_be = opt.optimize(
        k_sa=k_sa, K_Zpi=K_Z, H_mat=H, Phi=Phi.squeeze(1), G_mat=G,
        initial_B=None,
        lr=lr, weight_decay=weight_decay, num_steps=num_steps,
        FP_penalty_lambda=FP_penalty_lambda,
        use_low_rank=use_low_rank, rank=(rank_for_low_rank or 300),
        ortho_lambda=ortho_lambda, B_positive=B_positive,
        fixed_point_constraint=fixed_point_constraint, exact_projection=exact_projection,
        B_conv=B_conv, Sum_one_W=Sum_one_W, NonNeg_W=NonNeg_W,
        mass_anchor_lambda=mass_anchor_lambda, target_mass=target_mass,
        B_ridge_penalty=B_ridge_penalty, verbose=True,
    )

    # weights over Z: w = Bᵀ k  (and optionally project in optimizer)
    w = (B.t() @ k_sa.view(-1,1)).detach()

    return {"B": B.cpu(), "Z_grid": Z_grid.cpu(), "hist_obj": hist_obj, "hist_be": hist_be, "weights": w.cpu()}

def cli():
    import json, sys
    cfg = json.load(sys.stdin)
    out = estimate_distribution(**cfg)
    print("OK:", {k: (tuple(v.shape) if hasattr(v, 'shape') else len(v)) for k,v in out.items()})
