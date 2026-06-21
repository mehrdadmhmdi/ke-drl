# ke_drl/api.py
from __future__ import annotations
import os, sys, json
from typing import Any, Dict, Optional, Tuple
import torch

from .KE_DRL import KE_DRL
from .get_dataset import get_dataset
from .evaluation_metric import (
    embedding_test_risk,
    embedding_test_risk_from_inputs,
    predict_embedding_weights,
    projected_bellman_test_risk,
    projected_bellman_test_risk_from_inputs,
)
from .rank_diagnostics import matrix_rank_diagnostics


def _recover_tool(config: Dict[str, Any] | None = None):
    from .density_recovery import RecoverAndPlot
    return RecoverAndPlot(config or {})

# ----------------- FIT -----------------
def estimate_embedding(
    *, s0, s1, a0, a1, s_star, a_star, r,
    discrete_dims=None,
    target_p_choice, target_p_params,
    nu, length_scale, sigma,
    gamma_val, lambda_reg,
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
    round_discrete_z_grid: bool = True,
    # passthrough options identical to KE_DRL defaults
    hull_expand_factor: float = 1.0,
    lr: float = 1e-3, weight_decay: float = 0.0, num_steps: int = 5000,
    target_batch_size: Optional[int] = None,
    target_weights: Optional[torch.Tensor] = None,
    random_seed: Optional[int] = None,
    initial_scale: float = 1e-3,
    operator_method: str = "exact",
    operator_num_features: int = 128,
    operator_seed: Optional[int] = None,
    ridge_mode: str = "rkhs",
    diagnostic_interval: int = 50,
    eta_clip_min: Optional[float] = 0.0,
    eta_clip_max: Optional[float] = None,
    normalize_eta: bool = False,
    FP_penalty_lambda: float = 0.0,
    use_low_rank: bool = False, rank_for_low_rank: Optional[int] = None,
    B_positive: bool = False, fixed_point_constraint: bool = False, exact_projection: bool = False,
    ortho_lambda: float = 0.0, B_conv: bool = False, Sum_one_W: bool = False, NonNeg_W: bool = False,
    mass_anchor_lambda: float = 1.0, target_mass: float = 1.0,
    negativity_penalty_lambda: float = 0.0,
    max_B_norm: Optional[float] = None,
    B_ridge_penalty: bool = False,
    H_batch_size: int = 10,
    device: Optional[str] = None, dtype: torch.dtype = torch.float64,
    return_heavy_matrices: bool = True,
    return_best: bool = True,
    verbose: bool = True,
) -> Tuple[torch.Tensor, list, list, Dict[str, torch.Tensor]]:
    return KE_DRL(
        s0=s0, s1=s1, a0=a0, a1=a1,
        s_star=s_star, a_star=a_star, r=r, discrete_dims=discrete_dims,
        target_p_choice=target_p_choice, target_p_params=target_p_params,
        nu=nu, length_scale=length_scale, sigma=sigma,
        gamma_val=gamma_val, lambda_reg=lambda_reg,
        lambda_B=lambda_B, ratio_lambda_reg=ratio_lambda_reg,
        x_nu=x_nu, x_length_scale=x_length_scale, x_sigma=x_sigma,
        ratio_nu=ratio_nu, ratio_length_scale=ratio_length_scale, ratio_sigma=ratio_sigma,
        ratio_alpha_mix=ratio_alpha_mix,
        ratio_n_basis=ratio_n_basis, ratio_basis_source=ratio_basis_source,
        ratio_basis_seed=ratio_basis_seed,
        ratio_target_sample_multiplier=ratio_target_sample_multiplier,
        ratio_nonnegative_alpha=ratio_nonnegative_alpha,
        ratio_calibrate_mean=ratio_calibrate_mean,
        mean_embedding_basis_size=mean_embedding_basis_size,
        mean_embedding_basis_method=mean_embedding_basis_method,
        mean_embedding_basis_seed=mean_embedding_basis_seed,
        mean_embedding_basis_standardize=mean_embedding_basis_standardize,
        mean_embedding_basis_candidate_pool=mean_embedding_basis_candidate_pool,
        mean_embedding_basis_max_iter=mean_embedding_basis_max_iter,
        mean_embedding_basis_batch_size=mean_embedding_basis_batch_size,
        num_grid_points=num_grid_points,
        round_discrete_z_grid=round_discrete_z_grid,
        hull_expand_factor=hull_expand_factor,
        lr=lr, weight_decay=weight_decay, num_steps=num_steps,
        target_batch_size=target_batch_size, target_weights=target_weights,
        random_seed=random_seed, initial_scale=initial_scale,
        operator_method=operator_method, operator_num_features=operator_num_features,
        operator_seed=operator_seed, ridge_mode=ridge_mode,
        diagnostic_interval=diagnostic_interval,
        eta_clip_min=eta_clip_min, eta_clip_max=eta_clip_max, normalize_eta=normalize_eta,
        FP_penalty_lambda=FP_penalty_lambda,
        use_low_rank=use_low_rank, rank_for_low_rank=rank_for_low_rank,
        B_positive=B_positive, fixed_point_constraint=fixed_point_constraint, exact_projection=exact_projection,
        ortho_lambda=ortho_lambda, B_conv=B_conv, Sum_one_W=Sum_one_W, NonNeg_W=NonNeg_W,
        mass_anchor_lambda=mass_anchor_lambda, target_mass=target_mass,
        negativity_penalty_lambda=negativity_penalty_lambda, max_B_norm=max_B_norm,
        B_ridge_penalty=B_ridge_penalty,
        H_batch_size=H_batch_size,
        device=device, dtype=dtype, return_heavy_matrices=return_heavy_matrices,
        return_best=return_best, verbose=verbose,
    )

# ------------- PLOT CONFIG -------------
def build_plot_config(
    *, lr: float, fixed_point_constraint: bool, FP_penalty_lambda: float,
    Sum_one_W: bool, NonNeg_W: bool, mass_anchor_lambda: float, target_mass: float,
    num_steps: int, nu: float, length_scale: float, sigma_k: float, gamma_val: float,
    num_grid_points: int, hull_expand_factor: float, lambda_reg: float,
    bandwidth: float, lambda_rec: float, method: str,
    state_dim: int, reward_dim: int, action_dim: int,
    s_star, a_star, target_policy: str,
) -> Dict[str, Any]:
    return {
        "lr": lr, "fixed_point_constraint": fixed_point_constraint, "FP_penalty_lambda": FP_penalty_lambda,
        "Sum_one_W": Sum_one_W, "NonNeg_W": NonNeg_W, "mass_anchor_lambda": mass_anchor_lambda, "target_mass": target_mass,
        "num_steps": int(num_steps), "nu": float(nu), "length_scale": float(length_scale), "sigma_k": float(sigma_k),
        "gamma_val": float(gamma_val), "num_grid_points": int(num_grid_points),
        "hull_expand_factor": float(hull_expand_factor), "lambda_reg": float(lambda_reg),
        "bandwidth": float(bandwidth), "lambda_rec": float(lambda_rec), "method": str(method),
        "state_dim": int(state_dim), "reward_dim": int(reward_dim), "action_dim": int(action_dim),
        "s_star": (s_star.detach().cpu().tolist() if isinstance(s_star, torch.Tensor) else s_star),
        "a_star": (a_star.detach().cpu().tolist() if isinstance(a_star, torch.Tensor) else a_star),
        "target_policy": str(target_policy),
    }

# ------------- SIMPLE PLOTS -------------
def plot_bellman_error(history_be: list, *, config: Dict[str, Any] | None = None, outdir: str = "./plots/", steps=None):
    tool = _recover_tool(config)
    os.makedirs(outdir, exist_ok=True)
    tool.plot_bellman_error(history_be, outdir=outdir, steps=steps)

def plot_total_loss(history_obj: list, *, config: Dict[str, Any] | None = None, outdir: str = "./plots/", steps=None):
    tool = _recover_tool(config)
    os.makedirs(outdir, exist_ok=True)
    tool.plot_total_loss(history_obj, outdir=outdir, steps=steps)


# ------------- RECOVERY / EVAL -------------
def recover_joint_beta(
    *, B: torch.Tensor, k_sa: torch.Tensor, Z_grid: torch.Tensor, Phi: torch.Tensor, K_sa: torch.Tensor,
    config: Dict[str, Any],
):
    tool = _recover_tool(config)
    return tool.recover_joint_beta(
        B, k_sa, Z_grid, Phi, K_sa,
        nu=config["nu"], length_scale=config["length_scale"], sigma_k=config["sigma_k"],
        method=config["method"], lambda_reg=config["lambda_reg"],
    )

def compute_marginals_from_beta(
    *, beta_full: torch.Tensor, Z_grid: torch.Tensor, config: Dict[str, Any],
    n_grid: int = 400, margin_factor: float = 0.25
):
    tool = _recover_tool(config)
    return tool.marginals_from_beta(
        beta_full, Z_grid, reward_dim=config["reward_dim"],
        nu=config["nu"], length_scale=config["length_scale"], sigma_k=config["sigma_k"],
        lambda_rec=config["lambda_rec"], bandwidth=config["bandwidth"],
        n_grid=n_grid, margin_factor=margin_factor,
    )

def plot_densities(
    *, fz: torch.Tensor, grid_dict: Dict[str, Any], config: Dict[str, Any], outdir: str = "./plots/"
):
    tool = _recover_tool(config)
    os.makedirs(outdir, exist_ok=True)
    tool.plot_densities(fz, grid_dict, outdir=outdir)

def mean_embedding_all(
    *, beta_full: torch.Tensor, Z_grid: torch.Tensor, config: Dict[str, Any],
    do_joint_dims=(0, 1), n1: int = 120, n2: int = 120, outdir: str = "./plots/"
):
    tool = _recover_tool(config)
    return tool.mean_embedding_all(
        beta_full, Z_grid,
        nu=config["nu"], length_scale=config["length_scale"], sigma_k=config["sigma_k"],
        do_joint_dims=do_joint_dims, n1=n1, n2=n2, outdir=outdir,
    )

def plot_operator_check_2d(cache: Dict[str, Any], *, r_obs: torch.Tensor | None,
                           gamma: float, dims=(0,1), outdir: str = "./plots/",
                           config: Dict[str, Any] | None = None):
    tool = _recover_tool(config)
    tool.plot_operator_check_2d(cache, R=r_obs, gamma=gamma, dims=dims, outdir=outdir)

def save_weights_and_grid(beta_full: torch.Tensor, Z_grid: torch.Tensor, run_id: int, mu_dir="./mu", data_dir="./data"):
    os.makedirs(mu_dir, exist_ok=True); os.makedirs(data_dir, exist_ok=True)
    torch.save(Z_grid, os.path.join(data_dir, f"Zgrid_{run_id}.pt"))
    import numpy as np
    np.savetxt(os.path.join(mu_dir, f"weights_{run_id}.csv"),
               beta_full.detach().cpu().view(-1).numpy(), delimiter=",", fmt="%.8e")

# ------------- CLI (optional) -------------
def _shape(x): return tuple(x.shape) if hasattr(x, "shape") else x

def cli():
    """
    stdin JSON:
    {
      "fit": { ... KE_DRL kwargs ... },
      "plots": {
        "config": { ... build_plot_config kwargs ... },
        "r_obs": null or array,
        "what": ["bellman","loss","beta","marginal","mean","op2d"]
      }
    }
    """
    cfg = json.load(sys.stdin)
    B, hist_obj, hist_be, pre = estimate_embedding(**cfg["fit"])
    print("OK fit:", {"B": _shape(B), "hist_obj": len(hist_obj), "hist_be": len(hist_be),
                      **{k: _shape(v) for k, v in pre.items()}})

    if "plots" in cfg:
        pc = cfg["plots"]
        config = build_plot_config(**pc["config"])
        r_obs = (torch.as_tensor(pc["r_obs"]) if pc.get("r_obs") is not None else None)

        # --- CLI fragment ---
        what = set(pc.get("what", []))
        if "bellman" in what: plot_bellman_error(hist_be, config=config)
        if "loss" in what: plot_total_loss(hist_obj, config=config)

        if {"beta","marginal","mean","op2d"} & what:
            beta, Zg = recover_joint_beta(B=B, k_sa=pre["k_sa"], Z_grid=pre["Z_grid"], Phi=pre["Phi"], K_sa=pre["K_sa"], config=config)
            print("OK beta:", {"beta": _shape(beta), "Zg": _shape(Zg)})

            if "marginal" in what:
                fz, grid = compute_marginals_from_beta(beta_full=beta, Z_grid=Zg, config=config)
                plot_densities(fz=fz, grid_dict=grid, config=config)

            if "mean" in what or "op2d" in what:
                cache, _ = mean_embedding_all(beta_full=beta, Z_grid=Zg, config=config)
                if "op2d" in what:
                    plot_operator_check_2d(cache, r_obs=r_obs, gamma=config["gamma_val"])



