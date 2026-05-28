from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import yaml

from sim_utils import (
    kedrl_import_info,
    print_compute_device,
    resolve_compute_device,
    resolve_torch_dtype,
    seed_from_array,
    synthetic_data_generation_torch,
)


print("# =================================================== #")
print("#  Offline Data Generation for Algorithm Simulation   #")
print("# =================================================== #")

start = time.time()
job_id = os.environ.get("SLURM_JOB_ID")
array_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
print(f"Slurm Job ID: {job_id}")
print(f"Slurm Array ID: {array_id} -- used as the data replicate id")
print(f"ke_drl import source: {kedrl_import_info()}")

with open("./params.yaml", "r", encoding="utf-8") as f:
    P = yaml.safe_load(f)

compute_device = resolve_compute_device(P.get("compute"), purpose="offline data generation")
sim_dtype = resolve_torch_dtype(P.get("dtype", "float64"))
print_compute_device(compute_device, prefix="Offline data")

seed = seed_from_array(int(P.get("random_seed", 20260512)), array_id)
print(f"Random seed: {seed}")

to_t = lambda x: torch.as_tensor(x, dtype=sim_dtype)
W_s, b_s, sigma_s = map(to_t, (P["MDP"]["W_s"], P["MDP"]["b_s"], P["MDP"]["sigma_s"]))
W_r, b_r, sigma_r = map(to_t, (P["MDP"]["W_r"], P["MDP"]["b_r"], P["MDP"]["sigma_r"]))

policy_name = P["policy"]["Behvaioral_policy"]
beh_policy = P["policy"][policy_name]["name"]
beh_policy_params = P["policy"][policy_name]
offline_burn_in = int(P.get("offline_burn_in", 0))
print(f"Offline trajectory burn-in under behavior policy: {offline_burn_in}")

s0, s1, a0, a1, r0, r1, r = synthetic_data_generation_torch(
    P["n_ids"],
    P["n_timepoints"],
    P["state_dim"],
    P["reward_dim"],
    P["action_dim"],
    beh_policy,
    beh_policy_params,
    W_s,
    b_s,
    sigma_s,
    W_r,
    b_r,
    sigma_r,
    burn_in=offline_burn_in,
    dtype=sim_dtype,
    device=compute_device,
)

for name, tensor in {
    "s0": s0,
    "a0": a0,
    "s1": s1,
    "a1": a1,
    "r0": r0,
    "r1": r1,
    "r": r,
}.items():
    print(f"{name} shape: {tuple(tensor.shape)} dtype={tensor.dtype}")

data_folder = Path("data")
data_folder.mkdir(parents=True, exist_ok=True)
out_path = data_folder / f"offline_data_{array_id}.pt"

torch.save(
    {
        "s0": s0,
        "a0": a0,
        "s1": s1,
        "a1": a1,
        "r0": r0,
        "r1": r1,
        "r": r,
        "metadata": {
            "policy": beh_policy,
            "policy_params": beh_policy_params,
            "offline_burn_in": offline_burn_in,
            "params_file": "params.yaml",
            "stamp": str(array_id),
            "random_seed": seed,
        },
    },
    out_path,
)

elapsed = time.time() - start
print(f"Saved tensors as: {out_path.resolve()}")
print(f"Data generation time: {int(elapsed // 60)} minutes and {int(elapsed % 60)} seconds")
print("=" * 70)
