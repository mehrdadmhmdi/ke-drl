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
print_compute_device(compute_device, prefix="Monte Carlo")

num_replicates = int(P.get("experiment", {}).get("num_replicates", 1))
bench_cfg = dict(P.get("benchmark") or {})

seed = seed_from_array(int(P.get("random_seed", 20260512)) + 100000, 0)
print(f"Random seed: {seed}")
print(f"Number of offline replicates: {num_replicates}")

to_t = lambda x: torch.as_tensor(x, dtype=torch.float64)
W_s, b_s, sigma_s = map(to_t, (P["MDP"]["W_s"], P["MDP"]["b_s"], P["MDP"]["sigma_s"]))
W_r, b_r, sigma_r = map(to_t, (P["MDP"]["W_r"], P["MDP"]["b_r"], P["MDP"]["sigma_r"]))

target_policy_name = P["policy"]["evaluation_Target_policy"]
target_policy = P["policy"][target_policy_name]["name"]
target_policy_params = P["policy"][target_policy_name]

design_seed = int(P.get("random_seed", 20260512)) + int(bench_cfg.get("seed_offset", 110000))
if "s_star" in bench_cfg and "a_star" in bench_cfg:
    s_star = torch.as_tensor(bench_cfg["s_star"], dtype=torch.float64).reshape(-1)
    a_star = torch.as_tensor(bench_cfg["a_star"], dtype=torch.float64).reshape(-1)
    point_source = "fixed_config"
else:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(design_seed)
    s_star = torch.randn(int(P["state_dim"]), generator=generator, dtype=torch.float64)
    torch.manual_seed(design_seed + 1)
    a_star = sample_policy_actions(
        target_policy,
        clean_policy_params(target_policy, target_policy_params),
        s_star.reshape(1, -1),
        int(P["action_dim"]),
    ).reshape(-1)
    point_source = "independent_target_policy_draw"
if s_star.numel() != int(P["state_dim"]):
    raise ValueError(f"benchmark.s_star has length {s_star.numel()}, expected {P['state_dim']}.")
if a_star.numel() != int(P["action_dim"]):
    raise ValueError(f"benchmark.a_star has length {a_star.numel()}, expected {P['action_dim']}.")
print(f"Fixed MC benchmark point source: {point_source}")
print(f"s_star={s_star.tolist()}")
print(f"a_star={a_star.tolist()}")

csv = Path("./data") / "benchmark_point.csv"
csv.parent.mkdir(parents=True, exist_ok=True)
row = {
    "point_source": point_source,
    **{f"s{i}": v for i, v in enumerate(s_star.detach().cpu().flatten().tolist())},
    **{f"a{i}": v for i, v in enumerate(a_star.detach().cpu().flatten().tolist())},
}
pd.DataFrame([row]).to_csv(csv, index=False)

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
    dtype=torch.float64,
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
            "benchmark_id": 0,
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
