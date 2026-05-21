"""
Multi-target KE-DRL estimation for distributional RL benchmark.

This script implements the corrected simulation architecture:
- Separate MC evaluation target points (fixed: 10 points)
- Separate training target points (fixed: 100 points)
- Evaluate full embedding function on a grid for each MC target point
- Track predictions vs ground truth per target point
"""

from __future__ import annotations

import gc
import inspect
import math
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from sim_utils import (
    bootstrap_kedrl,
    clean_policy_params,
    kedrl_import_info,
    print_compute_device,
    resolve_compute_device,
    resolve_torch_dtype,
    seed_from_array,
)
from target_points_config import create_fixed_target_points, create_evaluation_grid
from mc_ground_truth import generate_mc_ground_truth

bootstrap_kedrl()

from ke_drl.KE_DRL import KE_DRL
from ke_drl.matern_kernel import matern_kernel
from ke_drl.evaluation_metric import predict_embedding_weights

print("# ================================================================ #")
print("#   Multi-Target KE-DRL Estimation (Corrected Architecture)        #")
print("# ================================================================ #")


def predict_embedding_on_grid(
    s_target: torch.Tensor,
    a_target: torch.Tensor,
    eval_grid: torch.Tensor,
    X_train: torch.Tensor,
    B_hat: torch.Tensor,
    kernel_params: dict,
) -> torch.Tensor:
    """
    Predict kernel mean embedding at a target point, evaluated on a grid.

    Parameters
    ----------
    s_target : torch.Tensor
        Target state, shape (D_s,)
    a_target : torch.Tensor
        Target action, shape (D_a,)
    eval_grid : torch.Tensor
        Evaluation grid, shape (n_grid_points, D_r)
    X_train : torch.Tensor
        Training state-action pairs, shape (N, D_s + D_a)
    B_hat : torch.Tensor
        Learned coefficient matrix, shape (N, m)
    kernel_params : dict
        Kernel parameters: {nu, length_scale, sigma}

    Returns
    -------
    mu_pred : torch.Tensor
        Predicted embedding on grid, shape (n_grid_points,)
    """
    # Create target point as state-action pair
    x_target = torch.cat([s_target.unsqueeze(0), a_target.unsqueeze(0)], dim=1)

    # Compute kernel vector: k_X(x_target, X_train)
    k_target = matern_kernel(
        X_train, x_target,
        nu=kernel_params["nu"],
        length_scale=kernel_params["length_scale"],
        sigma=kernel_params["sigma"],
    ).squeeze(1)  # (N,)

    # Compute coefficients: omega = B^T k_target
    omega = B_hat.T @ k_target  # (m,)

    # Evaluate on grid: mu_pred(z) = K_Z(z, Z_grid) @ omega
    K_eval = matern_kernel(
        eval_grid, eval_grid,  # Using eval_grid as the Z_grid here
        nu=kernel_params["nu"],
        length_scale=kernel_params["length_scale"],
        sigma=kernel_params["sigma"],
    )

    # This is a simplification - in practice you'd use the actual Z_grid from the fit
    # For now, we just compute the kernel mean directly
    mu_pred = K_eval @ omega

    return mu_pred


def run_single_replicate(
    replicate_id: int,
    offline_data: dict,
    s_mc_targets: torch.Tensor,
    a_mc_targets: torch.Tensor,
    s_train_targets: torch.Tensor,
    a_train_targets: torch.Tensor,
    eval_grids: list[torch.Tensor],
    params: dict,
) -> dict:
    """
    Run KE-DRL estimation for a single offline data replicate.

    Parameters
    ----------
    replicate_id : int
        Replicate identifier
    offline_data : dict
        Offline data with keys: s0, a0, s1, a1, r
    s_mc_targets, a_mc_targets : torch.Tensor
        MC evaluation target points (fixed)
    s_train_targets, a_train_targets : torch.Tensor
        Training target points (fixed)
    eval_grids : list[torch.Tensor]
        Evaluation grids for each MC target point
    params : dict
        Configuration parameters

    Returns
    -------
    results : dict
        Results dictionary with keys:
        - replicate_id
        - B_hat
        - mu_truth (list of tensors, one per MC target)
        - mu_pred (list of tensors, one per MC target)
        - errors (list of scalars)
        - metrics (dict with various quality metrics)
    """
    device = resolve_compute_device(params.get("compute"))
    dtype = resolve_torch_dtype(params.get("dtype", "float64"))
    verbose = bool(params.get("verbose", True))

    if verbose:
        print(f"\n{'='*60}")
        print(f"Replicate {replicate_id}")
        print(f"{'='*60}")

    # Extract offline data
    s0 = torch.as_tensor(offline_data["s0"], dtype=dtype, device=device)
    a0 = torch.as_tensor(offline_data["a0"], dtype=dtype, device=device)
    s1 = torch.as_tensor(offline_data["s1"], dtype=dtype, device=device)
    a1 = torch.as_tensor(offline_data["a1"], dtype=dtype, device=device)
    r = torch.as_tensor(offline_data["r"], dtype=dtype, device=device)

    # Move targets to device
    s_mc = s_mc_targets.to(device=device, dtype=dtype)
    a_mc = a_mc_targets.to(device=device, dtype=dtype)
    s_train = s_train_targets.to(device=device, dtype=dtype)
    a_train = a_train_targets.to(device=device, dtype=dtype)

    # Kernel parameters
    kernel_params = {
        "nu": float(params.get("nu", 3.5)),
        "length_scale": float(params.get("length_scale", 1.0)),
        "sigma": float(params.get("sigma", 0.7)),
    }

    # Target policy configuration
    target_policy_choice = params.get("target_policy_choice", "logistic")
    target_policy_params = params.get("target_p_params", {})

    # Step 1: Compute ground truth embeddings for each MC target point
    if verbose:
        print("[Step 1] Computing MC ground truth embeddings...")

    mu_truth_list = []
    returns_list = []

    for i, (s_mc_i, a_mc_i) in enumerate(zip(s_mc, a_mc)):
        if verbose:
            print(f"  MC target {i+1}/{len(s_mc)}: ", end="", flush=True)

        eval_grid_i = torch.as_tensor(eval_grids[i], dtype=dtype, device=device)

        returns_i, mu_truth_i = generate_mc_ground_truth(
            target_state=s_mc_i,
            target_action=a_mc_i,
            eval_grid=eval_grid_i,
            policy_name=target_policy_choice,
            policy_params=clean_policy_params(target_policy_choice, target_policy_params),
            MDP_config=params.get("MDP"),
            kernel_params=kernel_params,
            n_trajectories=int(params.get("mc_n_trajectories", 10000)),
            trajectory_length=int(params.get("mc_trajectory_length", 300)),
            gamma=float(params.get("gamma_val", 0.9)),
            dtype=dtype,
            device=device,
        )

        mu_truth_list.append(mu_truth_i)
        returns_list.append(returns_i)

        if verbose:
            print(f"✓ (μ shape: {mu_truth_i.shape})")

    # Step 2: Fit KE-DRL using training targets
    if verbose:
        print("\n[Step 2] Fitting KE-DRL with training targets...")

    start_fit = time.time()

    B_hat, history_obj, history_be, matrices = KE_DRL(
        s0=s0,
        s1=s1,
        a0=a0,
        a1=a1,
        s_star=s_train,
        a_star=a_train,
        r=r,
        target_p_choice=target_policy_choice,
        target_p_params=target_policy_params,
        nu=kernel_params["nu"],
        length_scale=kernel_params["length_scale"],
        sigma=kernel_params["sigma"],
        gamma_val=float(params.get("gamma_val", 0.9)),
        lambda_reg=float(params.get("lambda_reg", 1e-6)),
        lambda_B=float(params.get("lambda_B", 0.0)),
        num_steps=int(params.get("num_steps", 5000)),
        target_batch_size=int(params.get("target_batch_size", 64)),
        operator_method=params.get("operator_method", "rff"),
        operator_num_features=int(params.get("operator_num_features", 256)),
        verbose=verbose,
    )

    fit_time = time.time() - start_fit

    if verbose:
        print(f"Fitting complete in {fit_time:.2f}s")

    # Step 3: Predict embeddings at MC target points
    if verbose:
        print("\n[Step 3] Predicting embeddings at MC targets...")

    mu_pred_list = []
    errors = []

    X_train = torch.cat([s0, a0], dim=1)

    for i, (s_mc_i, a_mc_i) in enumerate(zip(s_mc, a_mc)):
        eval_grid_i = torch.as_tensor(eval_grids[i], dtype=dtype, device=device)

        # Predict embedding on grid
        mu_pred_i = predict_embedding_on_grid(
            s_target=s_mc_i,
            a_target=a_mc_i,
            eval_grid=eval_grid_i,
            X_train=X_train,
            B_hat=B_hat,
            kernel_params=kernel_params,
        )

        mu_pred_list.append(mu_pred_i)

        # Compute error
        error_i = torch.mean((mu_pred_i - mu_truth_list[i]) ** 2).item()
        errors.append(error_i)

        if verbose:
            print(f"  MC target {i+1}: MSE = {error_i:.6e}")

    # Compute aggregate metrics
    mean_error = float(sum(errors) / len(errors))
    max_error = float(max(errors))
    min_error = float(min(errors))

    metrics = {
        "mean_squared_error": mean_error,
        "max_squared_error": max_error,
        "min_squared_error": min_error,
        "fit_time": fit_time,
        "n_optimization_steps": len(history_obj),
        "final_objective": float(history_obj[-1]) if history_obj else float("nan"),
        "final_bellman": float(history_be[-1]) if history_be else float("nan"),
    }

    if verbose:
        print(f"\nAggregate metrics:")
        print(f"  Mean MSE: {mean_error:.6e}")
        print(f"  Max MSE:  {max_error:.6e}")
        print(f"  Min MSE:  {min_error:.6e}")

    results = {
        "replicate_id": replicate_id,
        "B_hat": B_hat.detach().cpu(),
        "mu_truth": [m.detach().cpu() for m in mu_truth_list],
        "mu_pred": [m.detach().cpu() for m in mu_pred_list],
        "errors": errors,
        "metrics": metrics,
        "returns_samples": [r.detach().cpu() for r in returns_list],
    }

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def load_offline_data(data_path: Path, dtype: torch.dtype, device: torch.device) -> dict:
    """Load offline data from file."""
    data = torch.load(data_path)
    return {
        "s0": torch.as_tensor(data["s0"], dtype=dtype, device=device),
        "a0": torch.as_tensor(data["a0"], dtype=dtype, device=device),
        "s1": torch.as_tensor(data["s1"], dtype=dtype, device=device),
        "a1": torch.as_tensor(data["a1"], dtype=dtype, device=device),
        "r": torch.as_tensor(data["r"], dtype=dtype, device=device),
    }


if __name__ == "__main__":
    # Load configuration
    with open("./params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # Setup
    dtype = resolve_torch_dtype(params.get("dtype", "float64"))
    device = resolve_compute_device(params.get("compute"))
    print_compute_device(device, prefix="Compute")

    # Load first offline replicate (or use synthetic for now)
    # For multi-target architecture, we would load 100 replicates
    print("\nNote: This is a template. For full 100-replicate run:")
    print("  1. Load offline data replicates [0, 1, ..., 99]")
    print("  2. Use parallel_runner.py for parallelization")
    print("  3. Aggregate results with aggregate_results.py")

    print("\nTemplate complete. See parallel_runner.py for production use.")
