"""Fixed target points configuration for reproducible benchmark evaluation."""

from __future__ import annotations

import torch
from typing import Optional


def create_fixed_target_points(
    s0: torch.Tensor,
    a0: torch.Tensor,
    n_mc_targets: int = 10,
    n_train_targets: int = 100,
    mc_seed: int = 20260512,
    train_seed: int = 20260513,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create fixed target points for MC evaluation and training.

    These points are deterministically selected from the offline data and
    remain fixed across all 100 replicates for reproducibility.

    Parameters
    ----------
    s0 : torch.Tensor
        Training states, shape (N, D_s)
    a0 : torch.Tensor
        Training actions, shape (N, D_a)
    n_mc_targets : int
        Number of target points for MC ground truth evaluation (default: 10)
    n_train_targets : int
        Number of target points for training (default: 100)
    mc_seed : int
        Random seed for MC target point selection
    train_seed : int
        Random seed for training target point selection

    Returns
    -------
    s_mc : torch.Tensor
        States for MC evaluation, shape (n_mc_targets, D_s)
    a_mc : torch.Tensor
        Actions for MC evaluation, shape (n_mc_targets, D_a)
    s_train : torch.Tensor
        States for training, shape (n_train_targets, D_s)
    a_train : torch.Tensor
        Actions for training, shape (n_train_targets, D_a)
    """
    N = s0.shape[0]

    # MC target points: deterministically select from data
    gen_mc = torch.Generator()
    gen_mc.manual_seed(mc_seed)
    mc_indices = torch.randperm(N, generator=gen_mc)[:n_mc_targets]

    # Training target points: deterministically select from data
    # (different seed to ensure no overlap with MC points when possible)
    gen_train = torch.Generator()
    gen_train.manual_seed(train_seed)
    train_indices = torch.randperm(N, generator=gen_train)[:n_train_targets]

    s_mc = s0[mc_indices]
    a_mc = a0[mc_indices]
    s_train = s0[train_indices]
    a_train = a0[train_indices]

    return s_mc, a_mc, s_train, a_train


def create_evaluation_grid(
    return_samples: torch.Tensor,
    n_grid_points: int = 100,
    expand_factor: float = 1.1,
) -> torch.Tensor:
    """
    Create a deterministic evaluation grid for embedding evaluation.

    The grid is created based on the range of return samples to ensure
    coverage of the data space.

    Parameters
    ----------
    return_samples : torch.Tensor
        Sampled return values, shape (n_samples, D_r)
    n_grid_points : int
        Number of grid points per dimension
    expand_factor : float
        Factor to expand grid beyond data range (default: 1.1)

    Returns
    -------
    eval_grid : torch.Tensor
        Evaluation grid for embedding, shape (n_grid_points, D_r)
    """
    D_r = return_samples.shape[1]

    if D_r == 1:
        # 1D case: create a line grid
        min_val = return_samples.min().item()
        max_val = return_samples.max().item()
        range_val = max_val - min_val
        expanded_min = min_val - expand_factor * range_val / 2
        expanded_max = max_val + expand_factor * range_val / 2

        grid = torch.linspace(expanded_min, expanded_max, n_grid_points, dtype=return_samples.dtype)
        return grid.unsqueeze(1)

    elif D_r == 2:
        # 2D case: create a grid
        mins = return_samples.min(dim=0).values
        maxs = return_samples.max(dim=0).values
        ranges = maxs - mins

        expanded_mins = mins - expand_factor * ranges / 2
        expanded_maxs = maxs + expand_factor * ranges / 2

        x = torch.linspace(expanded_mins[0], expanded_maxs[0], n_grid_points, dtype=return_samples.dtype)
        y = torch.linspace(expanded_mins[1], expanded_maxs[1], n_grid_points, dtype=return_samples.dtype)

        xx, yy = torch.meshgrid(x, y, indexing='ij')
        grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        return grid

    else:
        # High-dimensional case: create a grid along principal components
        # For now, just use quantile-based grid per dimension
        quantiles = torch.linspace(0.05, 0.95, n_grid_points, dtype=return_samples.dtype)
        grid_list = []

        for d in range(D_r):
            dim_vals = return_samples[:, d]
            grid_d = torch.quantile(dim_vals, quantiles)
            grid_list.append(grid_d)

        # Stack into (n_grid_points, D_r) - simple approach
        grid = torch.stack(grid_list, dim=1)
        return grid
