from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import torch


def _kedrl_src_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_src = os.environ.get("KEDRL_SRC")
    if env_src:
        candidates.append(Path(env_src))

    here = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    for base in (here, cwd, *here.parents, *cwd.parents):
        candidates.extend(
            [
                base / "src",
                base / "kedrl_git" / "src",
                base / ".." / "kedrl_git" / "src",
                base / ".." / ".." / "kedrl_git" / "src",
            ]
        )
    return candidates


def bootstrap_kedrl() -> None:
    """Prefer the checked-out package source over any installed ke_drl wheel."""
    seen: set[str] = set()
    for cand in _kedrl_src_candidates():
        src = cand.resolve()
        src_s = str(src)
        if src_s in seen:
            continue
        seen.add(src_s)
        if (src / "ke_drl").is_dir():
            if src_s in sys.path:
                sys.path.remove(src_s)
            sys.path.insert(0, src_s)
            return
    if importlib.util.find_spec("ke_drl") is None:
        raise ModuleNotFoundError(
            "Could not find local src/ke_drl or an installed ke_drl package. "
            "Set KEDRL_SRC to the repository src directory before running simulations."
        )


def kedrl_import_info() -> str:
    """Return the package path that Python will use for ke_drl."""
    spec = importlib.util.find_spec("ke_drl")
    if spec is None:
        return "ke_drl not importable"
    if spec.origin:
        return str(Path(spec.origin).resolve())
    locations = list(spec.submodule_search_locations or [])
    if locations:
        return str(Path(locations[0]).resolve())
    return "ke_drl import path unknown"


bootstrap_kedrl()

from ke_drl.Probability_Densities import Probability_Densities


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def resolve_compute_device(config: dict[str, Any] | None = None, *, purpose: str = "computation") -> torch.device:
    cfg = dict(config or {})
    requested = os.environ.get("KEDRL_DEVICE") or cfg.get("device")
    require_cuda = _as_bool(os.environ.get("KEDRL_REQUIRE_CUDA", cfg.get("require_cuda", False)))
    if requested is None:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(str(requested))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"{purpose} requested CUDA, but torch.cuda.is_available() is False. "
            "Check the Slurm partition/modules and the installed PyTorch build."
        )
    if require_cuda and device.type != "cuda":
        raise RuntimeError(
            f"{purpose} requires CUDA, but resolved device is {device}. "
            "Set compute.device: cuda or fix the CUDA/PyTorch environment."
        )
    return device


def print_compute_device(device: torch.device, *, prefix: str = "Compute") -> None:
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        major, minor = torch.cuda.get_device_capability(idx)
        print(f"{prefix} device: {device} ({name}, capability {major}.{minor}); torch CUDA={torch.version.cuda}", flush=True)
    else:
        print(f"{prefix} device: {device}; CUDA available={torch.cuda.is_available()}", flush=True)


def resolve_torch_dtype(name: Any) -> torch.dtype:
    value = str(name or "float64").strip().lower()
    if value in {"float32", "single", "fp32", "torch.float32"}:
        return torch.float32
    if value in {"float64", "double", "fp64", "torch.float64"}:
        return torch.float64
    raise ValueError(f"Unsupported dtype={name!r}; use float32 or float64.")


def seed_from_array(base_seed: int, array_id: str | int | None) -> int:
    try:
        offset = int(array_id) if array_id is not None else 0
    except (TypeError, ValueError):
        offset = 0
    seed = int(base_seed) + offset
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def clean_policy_params(policy: str, params: dict) -> dict:
    """Return params in the nested shape expected by Probability_Densities."""
    if policy in params and isinstance(params[policy], dict):
        raw = dict(params[policy])
    else:
        raw = dict(params)
    raw.pop("name", None)
    return {policy: raw}


def make_policy_sampler(policy: str, params: dict) -> Probability_Densities:
    return Probability_Densities(**clean_policy_params(policy, params))


def sample_policy_actions(
    policy: str,
    params: dict,
    states: torch.Tensor,
    action_dim: int,
    *,
    sampler: Probability_Densities | None = None,
) -> torch.Tensor:
    sampler = sampler or make_policy_sampler(policy, params)
    actions = sampler.sample_pdf(policy, states)
    if actions is None:
        raise RuntimeError(f"Policy sampler returned None for policy={policy!r}.")
    actions = torch.as_tensor(actions, dtype=states.dtype, device=states.device).reshape(states.shape[0], -1)
    if actions.shape[1] == action_dim:
        return actions
    if actions.shape[1] == 1:
        return actions.repeat(1, action_dim)
    reps = (action_dim + actions.shape[1] - 1) // actions.shape[1]
    return actions.repeat(1, reps)[:, :action_dim]


def _cov_cholesky(cov: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    cov = cov.to(device=device, dtype=dtype)
    dim = cov.shape[0]
    eye = torch.eye(dim, dtype=dtype, device=device)
    try:
        return torch.linalg.cholesky(cov)
    except RuntimeError:
        jitter = 1e-6 * torch.trace(cov).clamp_min(1.0) / float(dim)
        return torch.linalg.cholesky(cov + jitter * eye)


def _mvn_noise_from_chol(n: int, chol: torch.Tensor) -> torch.Tensor:
    eps = torch.randn((n, chol.shape[0]), dtype=chol.dtype, device=chol.device)
    return eps @ chol.transpose(0, 1)


def _linear_gaussian(
    states: torch.Tensor,
    actions: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    chol: torch.Tensor,
) -> torch.Tensor:
    x = torch.cat([states, actions], dim=1)
    mean = x @ W.T + b
    return mean + _mvn_noise_from_chol(mean.shape[0], chol)


def synthetic_data_generation_torch(
    n_ids: int,
    n_timepoints: int,
    state_dim: int,
    reward_dim: int,
    action_dim: int,
    policy: str,
    policy_params: dict,
    W_s: torch.Tensor,
    b_s: torch.Tensor,
    sigma_s: torch.Tensor,
    W_r: torch.Tensor,
    b_r: torch.Tensor,
    sigma_r: torch.Tensor,
    *,
    burn_in: int = 0,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, ...]:
    """Generate pooled offline transitions for the linear-Gaussian MDP."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    n_ids = int(n_ids)
    n_timepoints = int(n_timepoints)
    burn_in = max(0, int(burn_in))
    state_dim = int(state_dim)
    reward_dim = int(reward_dim)
    action_dim = int(action_dim)
    if n_timepoints < 2:
        raise ValueError("n_timepoints must be at least 2.")

    states = torch.empty(n_ids, n_timepoints, state_dim, dtype=dtype, device=dev)
    actions = torch.empty(n_ids, n_timepoints, action_dim, dtype=dtype, device=dev)
    rewards = torch.empty(n_ids, n_timepoints, reward_dim, dtype=dtype, device=dev)
    W_s = W_s.to(device=dev, dtype=dtype)
    b_s = b_s.to(device=dev, dtype=dtype)
    W_r = W_r.to(device=dev, dtype=dtype)
    b_r = b_r.to(device=dev, dtype=dtype)
    chol_s = _cov_cholesky(sigma_s, device=dev, dtype=dtype)
    chol_r = _cov_cholesky(sigma_r, device=dev, dtype=dtype)
    sampler = make_policy_sampler(policy, policy_params)

    state_t = torch.randn(n_ids, state_dim, dtype=dtype, device=dev)
    action_t = sample_policy_actions(policy, policy_params, state_t, action_dim, sampler=sampler)
    for _ in range(burn_in):
        state_t = _linear_gaussian(state_t, action_t, W_s, b_s, chol_s)
        action_t = sample_policy_actions(policy, policy_params, state_t, action_dim, sampler=sampler)

    states[:, 0, :] = state_t
    actions[:, 0, :] = action_t

    for t in range(n_timepoints):
        rewards[:, t, :] = _linear_gaussian(states[:, t, :], actions[:, t, :], W_r, b_r, chol_r)
        if t + 1 < n_timepoints:
            states[:, t + 1, :] = _linear_gaussian(states[:, t, :], actions[:, t, :], W_s, b_s, chol_s)
            actions[:, t + 1, :] = sample_policy_actions(
                policy, policy_params, states[:, t + 1, :], action_dim, sampler=sampler
            )

    s0 = states[:, :-1, :].reshape(-1, state_dim)
    s1 = states[:, 1:, :].reshape(-1, state_dim)
    a0 = actions[:, :-1, :].reshape(-1, action_dim)
    a1 = actions[:, 1:, :].reshape(-1, action_dim)
    r0 = rewards[:, :-1, :].reshape(-1, reward_dim)
    r1 = rewards[:, 1:, :].reshape(-1, reward_dim)
    r = rewards.reshape(-1, reward_dim)
    return tuple(x.detach().cpu() for x in (s0, s1, a0, a1, r0, r1, r))


def monte_carlo_Z(
    n_ids: int,
    n_timepoints: int,
    gamma_val: float,
    s_star: torch.Tensor,
    a_star: torch.Tensor,
    reward_dim: int,
    policy: str,
    policy_params: dict,
    W_s: torch.Tensor,
    b_s: torch.Tensor,
    sigma_s: torch.Tensor,
    W_r: torch.Tensor,
    b_r: torch.Tensor,
    sigma_r: torch.Tensor,
    *,
    plot: bool = False,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device | None = None,
) -> list[torch.Tensor]:
    """Monte Carlo discounted return samples from one or more initial (s,a) pairs."""
    del plot
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    n_ids = int(n_ids)
    n_timepoints = int(n_timepoints)
    reward_dim = int(reward_dim)
    gamma_val = float(gamma_val)

    s_star = torch.as_tensor(s_star, dtype=dtype, device=dev)
    a_star = torch.as_tensor(a_star, dtype=dtype, device=dev)
    if s_star.ndim == 1:
        s_star = s_star.unsqueeze(0)
    if a_star.ndim == 1:
        a_star = a_star.unsqueeze(0)
    if s_star.shape[0] != a_star.shape[0]:
        raise ValueError("s_star and a_star must have the same number of rows.")

    out: list[torch.Tensor] = []
    discounts = torch.pow(
        torch.as_tensor(gamma_val, dtype=dtype, device=dev),
        torch.arange(n_timepoints, dtype=dtype, device=dev),
    )
    W_s = W_s.to(device=dev, dtype=dtype)
    b_s = b_s.to(device=dev, dtype=dtype)
    W_r = W_r.to(device=dev, dtype=dtype)
    b_r = b_r.to(device=dev, dtype=dtype)
    chol_s = _cov_cholesky(sigma_s, device=dev, dtype=dtype)
    chol_r = _cov_cholesky(sigma_r, device=dev, dtype=dtype)
    sampler = make_policy_sampler(policy, policy_params)

    for ell in range(s_star.shape[0]):
        states = s_star[ell : ell + 1, :].repeat(n_ids, 1)
        actions = a_star[ell : ell + 1, :].repeat(n_ids, 1)
        returns = torch.zeros(n_ids, reward_dim, dtype=dtype, device=dev)

        for t in range(n_timepoints):
            reward_t = _linear_gaussian(states, actions, W_r, b_r, chol_r)
            returns += discounts[t] * reward_t
            if t + 1 < n_timepoints:
                states = _linear_gaussian(states, actions, W_s, b_s, chol_s)
                actions = sample_policy_actions(policy, policy_params, states, actions.shape[1], sampler=sampler)

        out.append(returns.detach().cpu())
    return out


def select_target_set(
    s0: torch.Tensor,
    a0: torch.Tensor,
    cfg: dict | None,
    *,
    seed: int,
    fallback_eval_s: torch.Tensor | None = None,
    fallback_eval_a: torch.Tensor | None = None,
    exclude_idx: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Choose the target X set used by the global objective."""
    cfg = dict(cfg or {})
    mode = str(cfg.get("mode", "train_subset")).lower()
    n = s0.shape[0]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(cfg.get("seed_offset", 7919)))

    if mode in {"mc_point", "point", "single"}:
        if fallback_eval_s is None or fallback_eval_a is None:
            raise ValueError("mc_point target mode requires the Monte Carlo evaluation point.")
        return fallback_eval_s.reshape(1, -1), fallback_eval_a.reshape(1, -1), None

    candidates = torch.arange(n)
    if bool(cfg.get("exclude_benchmark", False)) and exclude_idx is not None and n > 1:
        candidates = candidates[candidates != int(exclude_idx)]
    if candidates.numel() == 0:
        raise ValueError("No candidate target points remain after applying target_set exclusions.")

    if mode in {"all", "train_all"}:
        idx = candidates
        return s0[idx], a0[idx], idx

    num_points = int(cfg.get("num_points", min(128, candidates.numel())))
    num_points = max(1, min(num_points, int(candidates.numel())))
    if mode in {"first", "head"}:
        idx = candidates[:num_points]
    elif mode in {"train_subset", "subset", "random"}:
        idx = candidates[torch.randperm(candidates.numel(), generator=generator)[:num_points]]
    else:
        raise ValueError(f"Unknown target_set.mode={mode!r}.")
    return s0[idx], a0[idx], idx
