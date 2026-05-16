from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from sim_utils import kedrl_import_info, monte_carlo_Z, seed_from_array


print("# ================================================== #")
print("#        Monte Carlo Z under Target Policy           #")
print("# ================================================== #")

start = time.time()
job_id = os.environ.get("SLURM_JOB_ID")
array_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
print(f"Slurm Job ID: {job_id}")
print(f"Slurm Array ID: {array_id} -- used as the offline-replicate id")
print(f"ke_drl import source: {kedrl_import_info()}")

with open("./params.yaml", "r", encoding="utf-8") as f:
    P = yaml.safe_load(f)

offline_data_id = int(os.environ.get("OFFLINE_DATA_ID", array_id))
num_replicates = int(P.get("experiment", {}).get("num_replicates", 1))
if offline_data_id < 0 or offline_data_id >= num_replicates:
    raise ValueError(f"Offline replicate id {offline_data_id} is outside 0,...,{num_replicates - 1}.")

seed = seed_from_array(int(P.get("random_seed", 20260512)) + 100000, offline_data_id)
print(f"Random seed: {seed}")
print(f"Offline data id: {offline_data_id}")
print(f"Number of offline replicates: {num_replicates}")

to_t = lambda x: torch.as_tensor(x, dtype=torch.float64)
W_s, b_s, sigma_s = map(to_t, (P["MDP"]["W_s"], P["MDP"]["b_s"], P["MDP"]["sigma_s"]))
W_r, b_r, sigma_r = map(to_t, (P["MDP"]["W_r"], P["MDP"]["b_r"], P["MDP"]["sigma_r"]))

target_policy_name = P["policy"]["evaluation_Target_policy"]
target_policy = P["policy"][target_policy_name]["name"]
target_policy_params = P["policy"][target_policy_name]

offline_path = Path("data") / f"offline_data_{offline_data_id}.pt"
if not offline_path.exists():
    raise FileNotFoundError(f"Missing offline data file: {offline_path}. Run Job_data.sbatch first.")
saved = torch.load(offline_path, map_location="cpu")
s0 = torch.as_tensor(saved["s0"], dtype=torch.float64)
a0 = torch.as_tensor(saved["a0"], dtype=torch.float64)

bench_cfg = dict(P.get("benchmark") or {})
design_seed = int(P.get("random_seed", 20260512)) + int(bench_cfg.get("seed_offset", 110000)) + offline_data_id
generator = torch.Generator(device="cpu")
generator.manual_seed(design_seed)
idx = torch.randperm(s0.size(0), generator=generator)[0].item()
s_star = s0[idx]
a_star = a0[idx]
print(f"Selected MC benchmark row: {idx}")

csv = Path("./data") / f"benchmark_point_{offline_data_id}.csv"
csv.parent.mkdir(parents=True, exist_ok=True)
row = {
    "offline_data_id": offline_data_id,
    "offline_row": idx,
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
)
print("len(Z_true) =", len(Z_true), "shape each =", tuple(Z_true[0].shape))

out_dir = Path("data")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"Z_true_{offline_data_id}.pt"

torch.save(
    {
        "Z_true": Z_true,
        "metadata": {
            "s_star": s_star,
            "a_star": a_star,
            "offline_row": idx,
            "offline_data_id": offline_data_id,
            "benchmark_id": 0,
            "policy": target_policy,
            "policy_params": {target_policy: target_policy_params},
            "params_file": "params.yaml",
            "stamp": str(offline_data_id),
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
