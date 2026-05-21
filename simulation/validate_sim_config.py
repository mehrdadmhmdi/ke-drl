from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from sim_utils import clean_policy_params, kedrl_import_info, sample_policy_actions


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


def _check_kedrl_package_api() -> None:
    try:
        import inspect
        from ke_drl.KE_DRL import KE_DRL
        from ke_drl.evaluation_metric import predict_embedding_weights  # noqa: F401
        from ke_drl.operator_approx import compute_G_rff, compute_H_rff  # noqa: F401
        from ke_drl.optimize import RKDRL_Optimizer
        from ke_drl.rank_diagnostics import matrix_rank_diagnostics  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Installed ke_drl package is incompatible with these simulation scripts. "
            "Reinstall the current package before running: "
            'python -m pip install --no-cache-dir --force-reinstall '
            '"git+https://github.com/mehrdadmhmdi/ke-drl.git@main"'
        ) from exc
    missing = []
    if "return_best" not in inspect.signature(KE_DRL).parameters:
        missing.append("KE_DRL(return_best=...)")
    if "return_best" not in inspect.signature(RKDRL_Optimizer.optimize).parameters:
        missing.append("RKDRL_Optimizer.optimize(return_best=...)")
    if missing:
        raise ImportError(
            "Installed ke_drl package is stale for the current simulation scripts; missing "
            + ", ".join(missing)
            + ". Use the current source with KEDRL_SRC=/path/to/kedrl_git/src or reinstall the package."
        )
    print("ke_drl package API OK: prediction weights, RFF operators, rank diagnostics, and best-checkpoint optimizer available")


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


def _policy_mean_coefficients(policy_name: str, policy_block: dict[str, Any], state_dim: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    if policy_name == "uniform":
        theta_lower = torch.as_tensor(policy_block["theta_lower"], dtype=torch.float64)
        theta_upper = torch.as_tensor(policy_block["theta_upper"], dtype=torch.float64)
        eps_lower = torch.as_tensor(policy_block.get("epsilon_lower", [0.0]), dtype=torch.float64).reshape(-1)
        eps_upper = torch.as_tensor(policy_block.get("epsilon_upper", [0.0]), dtype=torch.float64).reshape(-1)
        return 0.5 * (theta_lower + theta_upper), 0.5 * (eps_lower + eps_upper)
    if policy_name == "gaussian":
        theta = torch.as_tensor(policy_block["theta_mean"], dtype=torch.float64)
        eps = torch.as_tensor(policy_block.get("epsilon_mean", [0.0]), dtype=torch.float64).reshape(-1)
        return theta, eps
    if policy_name == "logistic":
        theta = torch.as_tensor(policy_block["theta_loc"], dtype=torch.float64)
        eps = torch.as_tensor(policy_block.get("epsilon_loc", [0.0]), dtype=torch.float64).reshape(-1)
        return theta, eps
    return None


def _print_stationarity_diagnostic(P: dict[str, Any], policy_name: str, policy_block: dict[str, Any]) -> None:
    state_dim = int(P["state_dim"])
    action_dim = int(P["action_dim"])
    W_s = torch.as_tensor(P["MDP"]["W_s"], dtype=torch.float64)
    if W_s.shape != (state_dim, state_dim + action_dim):
        print("Stationarity diagnostic skipped: MDP.W_s has unexpected shape.")
        return
    coeff = _policy_mean_coefficients(policy_name, policy_block, state_dim)
    if coeff is None or action_dim != 1:
        print("Stationarity diagnostic skipped: no affine behavior-policy mean available.")
        return
    theta, eps = coeff
    del eps
    theta = theta.reshape(1, state_dim)
    A_eff = W_s[:, :state_dim] + W_s[:, state_dim:state_dim + 1] @ theta
    spectral_radius = float(torch.linalg.eigvals(A_eff).abs().max())
    burn_in = int(P.get("offline_burn_in", 0))
    print(
        "Behavior-policy linearized state-process diagnostic: "
        f"spectral_radius={spectral_radius:.3f}, offline_burn_in={burn_in}"
    )
    if spectral_radius >= 1.0:
        raise ValueError(
            "The linearized behavior-policy state process is not stable "
            f"(spectral radius {spectral_radius:.3f} >= 1). Adjust MDP/policy parameters."
        )
    if burn_in < 20:
        print("Warning: offline_burn_in is small; recorded transitions may still reflect transient initialization.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--data", default=None)
    args = parser.parse_args()

    with open(args.params, "r", encoding="utf-8") as f:
        P = yaml.safe_load(f)

    print(f"ke_drl import source: {kedrl_import_info()}")
    _check_kedrl_package_api()

    state_dim = int(P["state_dim"])
    action_dim = int(P["action_dim"])
    behavior_name, behavior_block = _check_policy_block(P, "Behvaioral_policy", state_dim=state_dim)
    target_name, target_block = _check_policy_block(P, "evaluation_Target_policy", state_dim=state_dim)
    print(f"Policy config OK: behavior={behavior_name}, target={target_name}, state_dim={state_dim}, action_dim={action_dim}")
    if target_name in {"logistic", "gaussian"}:
        print(f"Note: policy.{target_name}.theta_scale/theta_std are log-scale coefficients in Probability_Densities.")
    _print_stationarity_diagnostic(P, behavior_name, behavior_block)

    n_rep = int((P.get("experiment") or {}).get("num_replicates", 1))
    bench_cfg = dict(P.get("benchmark") or {})
    bench_points = int(bench_cfg.get("num_points", 1))
    target_cfg = dict(P.get("target_set") or {})
    target_points = int(target_cfg.get("num_points", 1))
    if n_rep < 1:
        raise ValueError("experiment.num_replicates must be at least 1.")
    if bench_points < 1:
        raise ValueError("benchmark.num_points must be at least 1.")
    if ("s_star" in bench_cfg) != ("a_star" in bench_cfg):
        raise ValueError("benchmark.s_star and benchmark.a_star must either both be present or both be omitted.")
    if "s_star" in bench_cfg:
        s_cfg = torch.as_tensor(bench_cfg["s_star"], dtype=torch.float64)
        a_cfg = torch.as_tensor(bench_cfg["a_star"], dtype=torch.float64)
        if s_cfg.ndim == 1:
            s_cfg = s_cfg.reshape(1, -1)
        if a_cfg.ndim == 1:
            a_cfg = a_cfg.reshape(1, -1)
        if s_cfg.ndim != 2 or s_cfg.shape[1] != state_dim:
            raise ValueError(f"benchmark.s_star must have shape ({state_dim},) or (n,{state_dim}), got {tuple(s_cfg.shape)}.")
        if a_cfg.ndim != 2 or a_cfg.shape[1] != action_dim:
            raise ValueError(f"benchmark.a_star must have shape ({action_dim},) or (n,{action_dim}), got {tuple(a_cfg.shape)}.")
        if s_cfg.shape[0] != a_cfg.shape[0]:
            raise ValueError("benchmark.s_star and benchmark.a_star must have the same number of rows.")
    if target_points < 1 and str(target_cfg.get("mode", "train_subset")).lower() not in {"all", "train_all"}:
        raise ValueError("target_set.num_points must be at least 1.")
    print(
        "Replicate config OK: "
        f"num_replicates={n_rep}, benchmark points={bench_points} independent of D_i, "
        f"loss target points={target_points}"
    )
    n_train = int(P.get("n_ids", 1)) * max(1, int(P.get("n_timepoints", 2)) - 1)
    m_grid = int(P.get("num_grid_points", 1))
    op_cfg = dict(P.get("operator_approximation") or {})
    op_method = str(op_cfg.get("method", "exact")).lower()
    if op_method in {"rff", "random_fourier", "random-fourier"}:
        print(
            "Return-operator approximation OK: "
            f"method=rff, features={int(op_cfg.get('num_features', 128))}, "
            f"exact G avoided for estimated N={n_train}, m={m_grid}, L={target_points}"
        )
    else:
        g_terms = target_points * m_grid * m_grid * n_train * n_train
        h_terms = target_points * m_grid * m_grid * n_train
        print(f"Exact return-operator work estimate: H~{h_terms:.3e}, G~{g_terms:.3e} kernel terms")
        if g_terms > 1e11:
            raise ValueError(
                "Exact G construction is computationally infeasible at this size. "
                "Set operator_approximation.method: rff or reduce n_ids/num_grid_points/target_set.num_points."
            )
    if "s_star" in bench_cfg:
        s_bench = torch.as_tensor(bench_cfg["s_star"], dtype=torch.float64)
        a_bench = torch.as_tensor(bench_cfg["a_star"], dtype=torch.float64)
        if s_bench.ndim == 1:
            s_bench = s_bench.reshape(1, -1)
        if a_bench.ndim == 1:
            a_bench = a_bench.reshape(1, -1)
        print(
            f"Fixed benchmark config rows={s_bench.shape[0]} "
            f"(benchmark.num_points={bench_points}; additional points are independent target-policy draws if needed)"
        )
        print(f"Fixed benchmark point 0: s_star={s_bench[0].reshape(-1).tolist()}, a_star={a_bench[0].reshape(-1).tolist()}")
        if behavior_name == "uniform" and action_dim == 1:
            lower, upper = _uniform_bounds(behavior_block, s_bench)
            print(
                "Fixed benchmark behavior-support interval for configured rows: "
                f"min_lower={lower.min().item():.4g}, max_upper={upper.max().item():.4g}"
            )
            inside_config = ((a_bench.reshape(-1) >= lower) & (a_bench.reshape(-1) <= upper)).double().mean().item()
            if inside_config < 1.0:
                raise ValueError("At least one configured benchmark.a_star is outside the behavior-policy support.")
        loc = _location(target_name, target_block, s_bench)
        if loc is not None:
            print(f"Fixed benchmark target-policy location range: [{loc.min().item():.4g}, {loc.max().item():.4g}]")

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
