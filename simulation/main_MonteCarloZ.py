from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from sim_utils import monte_carlo_Z, seed_from_array


print("# ================================================== #")
print("#        Monte Carlo Z under Target Policy           #")
print("# ================================================== #")

start = time.time()
job_id = os.environ.get("SLURM_JOB_ID")
array_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
print(f"Slurm Job ID: {job_id}")
print(f"Slurm Array ID: {array_id} -- used as the data replicate id")

with open("./params.yaml", "r", encoding="utf-8") as f:
    P = yaml.safe_load(f)

seed = seed_from_array(int(P.get("random_seed", 20260512)) + 100000, array_id)
print(f"Random seed: {seed}")

to_t = lambda x: torch.as_tensor(x, dtype=torch.float64)
W_s, b_s, sigma_s = map(to_t, (P["MDP"]["W_s"], P["MDP"]["b_s"], P["MDP"]["sigma_s"]))
W_r, b_r, sigma_r = map(to_t, (P["MDP"]["W_r"], P["MDP"]["b_r"], P["MDP"]["sigma_r"]))

target_policy_name = P["policy"]["evaluation_Target_policy"]
target_policy = P["policy"][target_policy_name]["name"]
target_policy_params = P["policy"][target_policy_name]

saved = torch.load(f"data/offline_data_{array_id}.pt", map_location="cpu")
s0 = torch.as_tensor(saved["s0"], dtype=torch.float64)
a0 = torch.as_tensor(saved["a0"], dtype=torch.float64)

idx = torch.randint(0, s0.size(0), (1,)).item()
s_star = s0[idx]
a_star = a0[idx]
print(f"Selected MC benchmark row: {idx}")

csv = Path("./data/sa_star.csv")
csv.parent.mkdir(parents=True, exist_ok=True)
row = {
    "Point ID": array_id,
    "offline_row": idx,
    **{f"s{i}": v for i, v in enumerate(s_star.detach().cpu().flatten().tolist())},
    **{f"a{i}": v for i, v in enumerate(a_star.detach().cpu().flatten().tolist())},
}
pd.DataFrame([row]).to_csv(csv, mode="a", header=not csv.exists(), index=False)

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
out_path = out_dir / f"Z_true_{array_id}.pt"

torch.save(
    {
        "Z_true": Z_true,
        "metadata": {
            "s_star": s_star,
            "a_star": a_star,
            "offline_row": idx,
            "policy": target_policy,
            "policy_params": {target_policy: target_policy_params},
            "params_file": "params.yaml",
            "stamp": str(array_id),
            "random_seed": seed,
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
