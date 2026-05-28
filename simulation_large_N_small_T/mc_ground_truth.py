"""Monte Carlo ground truth computation for distributional RL evaluation."""

from __future__ import annotations

import torch
from typing import Optional

from sim_utils import bootstrap_kedrl, sample_policy_actions

bootstrap_kedrl()

from ke_drl.matern_kernel import matern_kernel


def generate_mc_trajectories(
    target_state: torch.Tensor,
    target_action: torch.Tensor,
    policy_name: str,
    policy_params: dict,
    MDP_config: dict,
    n_trajectories: int = 10000,
    trajectory_length: int = 300,
    gamma: float = 0.9,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Generate Monte Carlo trajectories from a target point under a policy.

    For each trajectory, computes the cumulative discounted reward (return)
    and returns the distribution of returns.

    Parameters
    ----------
    target_state : torch.Tensor
        Initial state for MC trajectories, shape (D_s,)
    target_action : torch.Tensor
        Initial action for MC trajectories, shape (D_a,)
    policy_name : str
        Name of the target policy (e.g., "logistic")
    policy_params : dict
        Parameters for the target policy
    MDP_config : dict
        MDP configuration with keys: W_s, b_s, sigma_s, W_r, b_r, sigma_r
    n_trajectories : int
        Number of MC trajectories to generate
    trajectory_length : int
        Length of each trajectory (time steps)
    gamma : float
        Discount factor
    dtype : torch.dtype
        PyTorch data type
    device : torch.device or str
        Compute device

    Returns
    -------
    returns : torch.Tensor
        Cumulative discounted returns from all trajectories, shape (n_trajectories, D_r)
    """
    W_r = torch.as_tensor(MDP_config["W_r"], dtype=dtype, device=device)
    b_r = torch.as_tensor(MDP_config["b_r"], dtype=dtype, device=device)
    sigma_r = torch.as_tensor(MDP_config["sigma_r"], dtype=dtype, device=device)
    W_s = torch.as_tensor(MDP_config["W_s"], dtype=dtype, device=device)
    b_s = torch.as_tensor(MDP_config["b_s"], dtype=dtype, device=device)
    sigma_s = torch.as_tensor(MDP_config["sigma_s"], dtype=dtype, device=device)

    target_state = torch.as_tensor(target_state, dtype=dtype, device=device).reshape(1, -1)
    target_action = torch.as_tensor(target_action, dtype=dtype, device=device).reshape(1, -1)

    D_r = W_r.shape[0]
    returns = torch.zeros((n_trajectories, D_r), dtype=dtype, device=device)

    # Repeat initial state for all trajectories
    states = target_state.repeat(n_trajectories, 1)  # (n_trajectories, D_s)
    actions = target_action.repeat(n_trajectories, 1)  # (n_trajectories, D_a)

    gamma_power = 1.0

    for t in range(trajectory_length):
        # Compute rewards at current (state, action) pair
        # r = W_r @ [s, a] + b_r + noise
        sa = torch.cat([states, actions], dim=1)
        r_mean = W_r @ sa.T + b_r.unsqueeze(1)  # (D_r, n_trajectories)
        noise_r = torch.randn_like(r_mean) * sigma_r.unsqueeze(1)
        rewards = (r_mean + noise_r).T  # (n_trajectories, D_r)

        # Accumulate discounted returns
        returns += gamma_power * rewards

        # Transition to next state
        # s' = W_s @ [s, a] + b_s + noise
        s_next_mean = W_s @ sa.T + b_s.unsqueeze(1)  # (D_s, n_trajectories)
        noise_s = torch.randn_like(s_next_mean) * sigma_s.unsqueeze(1)
        states = (s_next_mean + noise_s).T  # (n_trajectories, D_s)

        # Sample actions from target policy at next states
        actions = sample_policy_actions(
            policy_name, policy_params, states, states.shape[1]
        )

        gamma_power *= float(gamma)

    return returns


def compute_ground_truth_embedding_on_grid(
    returns: torch.Tensor,
    eval_grid: torch.Tensor,
    *,
    nu: float,
    length_scale: float,
    sigma: float = 1.0,
    batch_size: int = 500,
) -> torch.Tensor:
    """
    Compute kernel mean embedding evaluated on a grid.

    Given a set of return samples (from MC trajectories), compute the kernel
    mean embedding evaluated at each point in eval_grid.

    μ(z) = (1/m) Σ_{i=1}^m k(z, Z_i)

    Parameters
    ----------
    returns : torch.Tensor
        MC return samples, shape (n_samples, D_r)
    eval_grid : torch.Tensor
        Evaluation grid, shape (n_grid_points, D_r)
    nu : float
        Matérn kernel nu parameter
    length_scale : float
        Matérn kernel length scale
    sigma : float
        Matérn kernel amplitude
    batch_size : int
        Batch size for kernel computation

    Returns
    -------
    mu : torch.Tensor
        Kernel mean embedding evaluated on grid, shape (n_grid_points,)
    """
    eval_grid = torch.as_tensor(eval_grid, dtype=returns.dtype, device=returns.device)
    mu = torch.zeros(eval_grid.shape[0], dtype=returns.dtype, device=returns.device)

    # Compute kernel mean: μ(z) = (1/m) Σ k(z, Z_i)
    for start in range(0, returns.shape[0], batch_size):
        end = min(start + batch_size, returns.shape[0])
        chunk = returns[start:end]
        K = matern_kernel(eval_grid, chunk, nu=nu, length_scale=length_scale, sigma=sigma)
        mu += K.sum(dim=1)

    mu /= float(returns.shape[0])
    return mu


def generate_mc_ground_truth(
    target_state: torch.Tensor,
    target_action: torch.Tensor,
    eval_grid: torch.Tensor,
    policy_name: str,
    policy_params: dict,
    MDP_config: dict,
    kernel_params: dict,
    n_trajectories: int = 10000,
    trajectory_length: int = 300,
    gamma: float = 0.9,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate MC ground truth: return samples and kernel mean embedding on grid.

    This is the "oracle" ground truth that we will compare our estimated
    embeddings against.

    Parameters
    ----------
    target_state : torch.Tensor
        Initial state for MC, shape (D_s,)
    target_action : torch.Tensor
        Initial action for MC, shape (D_a,)
    eval_grid : torch.Tensor
        Grid for evaluating the embedding, shape (n_grid_points, D_r)
    policy_name : str
        Target policy name
    policy_params : dict
        Target policy parameters
    MDP_config : dict
        MDP configuration
    kernel_params : dict
        Kernel parameters: {nu, length_scale, sigma}
    n_trajectories : int
        Number of MC trajectories
    trajectory_length : int
        Length per trajectory
    gamma : float
        Discount factor
    dtype : torch.dtype
        Data type
    device : torch.device or str
        Compute device

    Returns
    -------
    returns : torch.Tensor
        MC return samples, shape (n_trajectories, D_r)
    mu_true : torch.Tensor
        Kernel mean embedding on grid, shape (n_grid_points,)
    """
    # Generate MC returns
    returns = generate_mc_trajectories(
        target_state=target_state,
        target_action=target_action,
        policy_name=policy_name,
        policy_params=policy_params,
        MDP_config=MDP_config,
        n_trajectories=n_trajectories,
        trajectory_length=trajectory_length,
        gamma=gamma,
        dtype=dtype,
        device=device,
    )

    # Compute embedding on grid
    mu_true = compute_ground_truth_embedding_on_grid(
        returns,
        eval_grid,
        nu=kernel_params["nu"],
        length_scale=kernel_params["length_scale"],
        sigma=kernel_params["sigma"],
    )

    return returns, mu_true
