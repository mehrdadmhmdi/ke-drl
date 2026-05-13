from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from sim_utils import clean_policy_params, sample_policy_actions


REQUIRED = {
    "beta": ["theta_alpha", "theta_beta"],
    "gaussian": ["theta_mean", "theta_std"],
    "uniform": ["theta_lower", "theta_upper"],
    "logistic": ["theta_loc", "theta_scale"],
}


def _as_list(x: Any) -> list[Any]:
    if isinstance(x, list):
        return x
    return [x]


def _check_policy_block(P: dict[str, Any], block_key: str, *, state_dim: int) -> tuple[str, dict[str, Any]]:
    policy_name = P["policy"][block_key]
    if policy_name not in P["policy"]:
        raise ValueError(f"policy.{block_key}={policy_name!r}, but policy.{policy_name} is not defined.")
    block = dict(P["policy"][policy_name])
    declared = block.get("name", policy_name)
    if declared != policy_name:
        raise ValueError(f"policy.{policy_name}.name={declared!r}, expected {policy_name!r}.")
    missing = [name for name in REQUIRED[policy_name] if name not in block]
    if missing:
        raise ValueError(f"policy.{policy_name} is missing required fields: {missing}")
    for theta_name in REQUIRED[policy_name]:
        theta = _as_list(block[theta_name])
        if len(theta) != state_dim:
            raise ValueError(f"policy.{policy_name}.{theta_name} has length {len(theta)}, expected {state_dim}.")
    return policy_name, block


def _summarize_tensor(name: str, x: torch.Tensor) -> None:
    x = torch.as_tensor(x, dtype=torch.float64).reshape(-1)
    qs = torch.quantile(x, torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], dtype=torch.float64))
    print(
        f"{name}: mean={x.mean().item():.4g}, sd={x.std().item():.4g}, "
        f"q0/q10/q50/q90/q100={qs.tolist()}"
    )


def _uniform_bounds(policy_block: dict[str, Any], s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    theta_lower = torch.as_tensor(policy_block["theta_lower"], dtype=s.dtype, device=s.device)
    theta_upper = torch.as_tensor(policy_block["theta_upper"], dtype=s.dtype, device=s.device)
    eps_lower = torch.as_tensor(policy_block.get("epsilon_lower", 0.0), dtype=s.dtype, device=s.device)
    eps_upper = torch.as_tensor(policy_block.get("epsilon_upper", 0.0), dtype=s.dtype, device=s.device)
    lower = s @ theta_lower + eps_lower
    upper = s @ theta_upper + eps_upper
    upper = torch.where(upper <= lower, lower + 1.0, upper)
    return lower.reshape(-1), upper.reshape(-1)


def _location(policy_name: str, policy_block: dict[str, Any], s: torch.Tensor) -> torch.Tensor | None:
    if policy_name == "logistic":
        theta = torch.as_tensor(policy_block["theta_loc"], dtype=s.dtype, device=s.device)
        eps = torch.as_tensor(policy_block.get("epsilon_loc", 0.0), dtype=s.dtype, device=s.device)
        return (s @ theta + eps).reshape(-1)
    if policy_name == "gaussian":
        theta = torch.as_tensor(policy_block["theta_mean"], dtype=s.dtype, device=s.device)
        eps = torch.as_tensor(policy_block.get("epsilon_mean", 0.0), dtype=s.dtype, device=s.device)
        return (s @ theta + eps).reshape(-1)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--data", default=None)
    args = parser.parse_args()

    with open(args.params, "r", encoding="utf-8") as f:
        P = yaml.safe_load(f)

    state_dim = int(P["state_dim"])
    action_dim = int(P["action_dim"])
    behavior_name, behavior_block = _check_policy_block(P, "Behvaioral_policy", state_dim=state_dim)
    target_name, target_block = _check_policy_block(P, "evaluation_Target_policy", state_dim=state_dim)
    print(f"Policy config OK: behavior={behavior_name}, target={target_name}, state_dim={state_dim}, action_dim={action_dim}")
    if target_name in {"logistic", "gaussian"}:
        print(f"Note: policy.{target_name}.theta_scale/theta_std are log-scale coefficients in Probability_Densities.")

    n_rep = int((P.get("experiment") or {}).get("num_replicates", 1))
    bench_points = int((P.get("benchmark") or {}).get("num_points", 1))
    target_cfg = dict(P.get("target_set") or {})
    target_points = int(target_cfg.get("num_points", 1))
    if n_rep < 1:
        raise ValueError("experiment.num_replicates must be at least 1.")
    if bench_points != 1:
        raise ValueError("This simulation architecture expects benchmark.num_points: 1.")
    if target_points < 1 and str(target_cfg.get("mode", "train_subset")).lower() not in {"all", "train_all"}:
        raise ValueError("target_set.num_points must be at least 1.")
    print(f"Replicate config OK: num_replicates={n_rep}, benchmark points=1, loss target points={target_points}")

    if args.data is None:
        return

    blob = torch.load(Path(args.data), map_location="cpu")
    s0 = torch.as_tensor(blob["s0"], dtype=torch.float64)
    a0 = torch.as_tensor(blob["a0"], dtype=torch.float64)
    available_targets = s0.shape[0] - (1 if bool(target_cfg.get("exclude_benchmark", False)) and s0.shape[0] > 1 else 0)
    if str(target_cfg.get("mode", "train_subset")).lower() not in {"all", "train_all"} and target_points > available_targets:
        raise ValueError(
            f"target_set.num_points={target_points} exceeds available target candidates "
            f"after exclusions ({available_targets})."
        )
    _summarize_tensor("observed behavior actions", a0)

    target_params = clean_policy_params(target_name, target_block)
    target_sample = sample_policy_actions(target_name, target_params, s0, action_dim)
    _summarize_tensor("target-policy sampled actions", target_sample)

    if behavior_name == "uniform" and action_dim == 1:
        lower, upper = _uniform_bounds(behavior_block, s0)
        _summarize_tensor("behavior uniform lower", lower)
        _summarize_tensor("behavior uniform upper", upper)
        loc = _location(target_name, target_block, s0)
        if loc is not None:
            inside = ((loc >= lower) & (loc <= upper)).double().mean().item()
            print(f"target location inside behavior uniform support: {inside:.3f}")
            if inside < 0.80:
                raise ValueError(
                    "Target policy has weak overlap with the behavior support. "
                    "Adjust target location/scale before a main run."
                )


if __name__ == "__main__":
    main()
