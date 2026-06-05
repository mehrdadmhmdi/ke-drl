from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from sim_utils import (
    actions_in_uniform_support,
    clean_policy_params,
    kedrl_import_info,
    monte_carlo_Z,
    print_compute_device,
    resolve_compute_device,
    resolve_torch_dtype,
    sample_policy_actions,
    seed_from_array,
)


print("# ================================================== #")
print("#        Monte Carlo Z under Target Policy           #")
print("# ================================================== #")

start = time.time()
job_id = os.environ.get("SLURM_JOB_ID")
array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
print(f"Slurm Job ID: {job_id}")
print(f"Slurm Array ID: {array_id}")
print(f"ke_drl import source: {kedrl_import_info()}")

with open("./params.yaml", "r", encoding="utf-8") as f:
    P = yaml.safe_load(f)

compute_device = resolve_compute_device(P.get("compute"), purpose="Monte Carlo Z")
sim_dtype = resolve_torch_dtype(P.get("dtype", "float64"))
print_compute_device(compute_device, prefix="Monte Carlo")

num_replicates = int(P.get("experiment", {}).get("num_replicates", 1))
bench_cfg = dict(P.get("benchmark") or {})

seed = seed_from_array(int(P.get("random_seed", 20260512)) + 100000, 0)
print(f"Random seed: {seed}")
print(f"Number of offline replicates: {num_replicates}")

to_t = lambda x: torch.as_tensor(x, dtype=sim_dtype)
W_s, b_s, sigma_s = map(to_t, (P["MDP"]["W_s"], P["MDP"]["b_s"], P["MDP"]["sigma_s"]))
W_r, b_r, sigma_r = map(to_t, (P["MDP"]["W_r"], P["MDP"]["b_r"], P["MDP"]["sigma_r"]))

target_policy_name = P["policy"]["evaluation_Target_policy"]
target_policy = P["policy"][target_policy_name]["name"]
target_policy_params = P["policy"][target_policy_name]
behavior_policy_name = P["policy"]["Behvaioral_policy"]
behavior_policy = P["policy"][behavior_policy_name]["name"]
behavior_policy_params = P["policy"][behavior_policy_name]
_empirical_support_bounds: tuple[torch.Tensor, torch.Tensor] | None = None

design_seed = int(P.get("random_seed", 20260512)) + int(bench_cfg.get("seed_offset", 110000))
num_benchmark_points = int(bench_cfg.get("num_points", 1))
if num_benchmark_points < 1:
    raise ValueError("benchmark.num_points must be at least 1.")


def _as_rows(x, dim: int, name: str) -> torch.Tensor:
    out = torch.as_tensor(x, dtype=sim_dtype)
    if out.ndim == 1:
        out = out.reshape(1, -1)
    if out.ndim != 2 or out.shape[1] != dim:
        raise ValueError(f"benchmark.{name} must have shape ({dim},) or (n,{dim}); got {tuple(out.shape)}.")
    return out


def _candidate_states(n_points: int, seed0: int) -> torch.Tensor:
    """Draw candidate evaluation states from the empirical offline support when available."""
    offline_path = Path("data") / "offline_data_0.pt"
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed0)
    if offline_path.exists():
        blob = torch.load(offline_path, map_location="cpu")
        s0 = torch.as_tensor(blob["s0"], dtype=sim_dtype)
        if s0.ndim != 2 or s0.shape[1] != int(P["state_dim"]):
            raise ValueError(f"{offline_path} has invalid s0 shape {tuple(s0.shape)}.")
        idx = torch.randint(s0.shape[0], (n_points,), generator=generator)
        return s0[idx].clone()
    return torch.randn(n_points, int(P["state_dim"]), generator=generator, dtype=sim_dtype)


def _load_empirical_support_bounds() -> tuple[torch.Tensor, torch.Tensor] | None:
    global _empirical_support_bounds
    if _empirical_support_bounds is not None:
        return _empirical_support_bounds
    if not bool(bench_cfg.get("empirical_support_filter", False)):
        return None
    offline_path = Path("data") / "offline_data_0.pt"
    if not offline_path.exists():
        print("Empirical benchmark support filter requested, but data/offline_data_0.pt is not available.", flush=True)
        return None
    blob = torch.load(offline_path, map_location="cpu")
    s0 = torch.as_tensor(blob["s0"], dtype=sim_dtype)
    a0 = torch.as_tensor(blob["a0"], dtype=sim_dtype).reshape(s0.shape[0], -1)
    x0 = torch.cat([s0, a0], dim=1)
    reference_size = int(bench_cfg.get("support_reference_size", 0) or 0)
    if reference_size > 0 and reference_size < x0.shape[0]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(design_seed + 31337)
        idx = torch.randperm(x0.shape[0], generator=generator)[:reference_size]
        x0 = x0[idx]
    q = float(bench_cfg.get("support_quantile", 0.995))
    q = min(max(q, 0.50), 0.9999)
    tail = 0.5 * (1.0 - q)
    probs = torch.tensor([tail, 1.0 - tail], dtype=torch.float64)
    bounds = torch.quantile(x0.to(torch.float64), probs, dim=0).to(dtype=sim_dtype)
    lower, upper = bounds[0], bounds[1]
    expand = float(bench_cfg.get("support_expand_factor", 1.10))
    center = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower).clamp_min(torch.finfo(sim_dtype).eps) * expand
    _empirical_support_bounds = (center - half_width, center + half_width)
    print(
        "Empirical benchmark support filter: "
        f"quantile={q:g}, expand_factor={expand:g}, reference_rows={x0.shape[0]}",
        flush=True,
    )
    return _empirical_support_bounds


def _empirical_support_mask(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    bounds = _load_empirical_support_bounds()
    if bounds is None:
        return torch.ones(s.shape[0], dtype=torch.bool, device=s.device)
    lower, upper = bounds
    x = torch.cat([s.reshape(s.shape[0], -1), a.reshape(a.shape[0], -1)], dim=1)
    lower = lower.to(dtype=x.dtype, device=x.device)
    upper = upper.to(dtype=x.dtype, device=x.device)
    return ((x >= lower) & (x <= upper)).all(dim=1)


def _exact_behavior_support_mask(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    mask = torch.ones(s.shape[0], dtype=torch.bool, device=s.device)
    if behavior_policy == "uniform" and int(P["action_dim"]) == 1:
        mask = mask & actions_in_uniform_support(behavior_policy_params, s, a)
    return mask


def _support_mask(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return _exact_behavior_support_mask(s, a) & _empirical_support_mask(s, a)


def _validate_benchmark_support(s: torch.Tensor, a: torch.Tensor) -> None:
    mask = _exact_behavior_support_mask(s, a).detach().cpu()
    if bool((~mask).any()):
        bad = torch.nonzero(~mask, as_tuple=False).reshape(-1).tolist()
        raise ValueError(
            "Benchmark evaluation point(s) outside behavior-policy support: "
            + ", ".join(str(int(j)) for j in bad[:20])
        )
    empirical_mask = _empirical_support_mask(s, a).detach().cpu()
    if bool((~empirical_mask).any()):
        bad = torch.nonzero(~empirical_mask, as_tuple=False).reshape(-1).tolist()
        print(
            "Warning: fixed benchmark evaluation point(s) outside empirical kernel-support box: "
            + ", ".join(str(int(j)) for j in bad[:20]),
            flush=True,
        )
    print(f"All {s.shape[0]} benchmark evaluation points are inside exact behavior-policy support.")


def _draw_benchmark_points(n_points: int, seed0: int) -> tuple[torch.Tensor, torch.Tensor]:
    if n_points <= 0:
        return (
            torch.empty(0, int(P["state_dim"]), dtype=sim_dtype),
            torch.empty(0, int(P["action_dim"]), dtype=sim_dtype),
        )
    states: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    remaining = n_points
    attempt = 0
    max_attempts = int(bench_cfg.get("max_draw_attempts", 200))
    while remaining > 0 and attempt < max_attempts:
        attempt += 1
        batch_n = max(256, remaining * 32)
        s_batch = _candidate_states(batch_n, seed0 + 1009 * attempt)
        torch.manual_seed(seed0 + 2003 * attempt)
        a_batch = sample_policy_actions(
            target_policy,
            clean_policy_params(target_policy, target_policy_params),
            s_batch,
            int(P["action_dim"]),
        ).reshape(batch_n, int(P["action_dim"]))
        keep = _support_mask(s_batch, a_batch)
        if bool(keep.any()):
            s_keep = s_batch[keep][:remaining]
            a_keep = a_batch[keep][:remaining]
            states.append(s_keep)
            actions.append(a_keep)
            remaining -= int(s_keep.shape[0])
    if remaining > 0:
        raise RuntimeError(
            f"Could not draw {n_points} support-safe benchmark points; "
            f"{remaining} still missing after {attempt} attempts."
        )
    return torch.cat(states, dim=0), torch.cat(actions, dim=0)


if "s_star" in bench_cfg and "a_star" in bench_cfg:
    s_fixed = _as_rows(bench_cfg["s_star"], int(P["state_dim"]), "s_star")
    a_fixed = _as_rows(bench_cfg["a_star"], int(P["action_dim"]), "a_star")
    if s_fixed.shape[0] != a_fixed.shape[0]:
        raise ValueError("benchmark.s_star and benchmark.a_star must have the same number of rows.")
    if s_fixed.shape[0] >= num_benchmark_points:
        s_star = s_fixed[:num_benchmark_points]
        a_star = a_fixed[:num_benchmark_points]
        point_sources = ["fixed_config"] * num_benchmark_points
    else:
        s_extra, a_extra = _draw_benchmark_points(num_benchmark_points - s_fixed.shape[0], design_seed + 1000)
        s_star = torch.cat([s_fixed, s_extra], dim=0)
        a_star = torch.cat([a_fixed, a_extra], dim=0)
        extra_source = (
            "empirical_support_safe_target_policy_draw"
            if bool(bench_cfg.get("empirical_support_filter", False))
            else "support_safe_target_policy_draw"
        )
        point_sources = ["fixed_config"] * int(s_fixed.shape[0]) + [extra_source] * int(s_extra.shape[0])
    point_source = "fixed_config" if len(set(point_sources)) == 1 else "fixed_config_plus_support_safe_target_policy_draws"
else:
    s_star, a_star = _draw_benchmark_points(num_benchmark_points, design_seed)
    point_source = "support_safe_target_policy_draw"
    point_sources = [point_source] * num_benchmark_points
if s_star.shape != (num_benchmark_points, int(P["state_dim"])):
    raise ValueError(f"benchmark s_star has shape {tuple(s_star.shape)}, expected ({num_benchmark_points}, {P['state_dim']}).")
if a_star.shape != (num_benchmark_points, int(P["action_dim"])):
    raise ValueError(f"benchmark a_star has shape {tuple(a_star.shape)}, expected ({num_benchmark_points}, {P['action_dim']}).")
_validate_benchmark_support(s_star, a_star)
print(f"Fixed MC benchmark point source: {point_source}")
print(f"num_benchmark_points={num_benchmark_points}")
print(f"s_star={s_star.tolist()}")
print(f"a_star={a_star.tolist()}")

csv = Path("./data") / "benchmark_point.csv"
csv.parent.mkdir(parents=True, exist_ok=True)
rows = []
for j in range(num_benchmark_points):
    rows.append(
        {
            "benchmark_id": j,
            "point_source": point_sources[j],
            **{f"s{i}": v for i, v in enumerate(s_star[j].detach().cpu().flatten().tolist())},
            **{f"a{i}": v for i, v in enumerate(a_star[j].detach().cpu().flatten().tolist())},
        }
    )
pd.DataFrame(rows).to_csv(csv, index=False)

Z_true = monte_carlo_Z(
    P["Z_sim"]["n_ids"],
    P["Z_sim"]["n_timepoints"],
    P["gamma_val"],
    s_star,
    a_star,
    P["reward_dim"],
    target_policy,
    target_policy_params,
    W_s,
    b_s,
    sigma_s,
    W_r,
    b_r,
    sigma_r,
    plot=False,
    dtype=sim_dtype,
    device=compute_device,
)
print("len(Z_true) =", len(Z_true), "shape each =", tuple(Z_true[0].shape))

out_dir = Path("data")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / str(bench_cfg.get("output", "Z_true.pt"))

torch.save(
    {
        "Z_true": Z_true,
        "metadata": {
            "s_star": s_star,
            "a_star": a_star,
            "point_source": point_source,
            "point_sources": point_sources,
            "num_benchmark_points": num_benchmark_points,
            "policy": target_policy,
            "policy_params": {target_policy: target_policy_params},
            "params_file": "params.yaml",
            "stamp": "fixed",
            "random_seed": seed,
            "design_seed": design_seed,
        },
    },
    out_path,
)

elapsed = time.time() - start
print(f"Target Policy: {target_policy}")
print(f"Target Policy Parameters: {target_policy_params}")
print(f"Saved Z_true tensors as: {out_path.resolve()}")
print(f"Monte Carlo time: {int(elapsed // 60)} minutes and {int(elapsed % 60)} seconds")
print("=" * 70)
