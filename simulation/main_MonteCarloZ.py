from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from sim_utils import (
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


def _draw_benchmark_points(n_points: int, seed0: int) -> tuple[torch.Tensor, torch.Tensor]:
    if n_points <= 0:
        return (
            torch.empty(0, int(P["state_dim"]), dtype=sim_dtype),
            torch.empty(0, int(P["action_dim"]), dtype=sim_dtype),
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed0)
    s = torch.randn(n_points, int(P["state_dim"]), generator=generator, dtype=sim_dtype)
    torch.manual_seed(seed0 + 1)
    a = sample_policy_actions(
        target_policy,
        clean_policy_params(target_policy, target_policy_params),
        s,
        int(P["action_dim"]),
    ).reshape(n_points, int(P["action_dim"]))
    return s, a


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
        point_sources = ["fixed_config"] * int(s_fixed.shape[0]) + ["independent_target_policy_draw"] * int(s_extra.shape[0])
    point_source = "fixed_config" if len(set(point_sources)) == 1 else "fixed_config_plus_independent_target_policy_draws"
else:
    s_star, a_star = _draw_benchmark_points(num_benchmark_points, design_seed)
    point_source = "independent_target_policy_draw"
    point_sources = [point_source] * num_benchmark_points
if s_star.shape != (num_benchmark_points, int(P["state_dim"])):
    raise ValueError(f"benchmark s_star has shape {tuple(s_star.shape)}, expected ({num_benchmark_points}, {P['state_dim']}).")
if a_star.shape != (num_benchmark_points, int(P["action_dim"])):
    raise ValueError(f"benchmark a_star has shape {tuple(a_star.shape)}, expected ({num_benchmark_points}, {P['action_dim']}).")
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
