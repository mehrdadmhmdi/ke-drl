from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from sim_utils import bootstrap_kedrl, clean_policy_params, seed_from_array, select_target_set
from sim_eval import mean_embedding_hat, mean_embedding_true, save_mu_outputs


bootstrap_kedrl()

from ke_drl.KE_DRL import KE_DRL
from ke_drl.Gamma_sa import Gamma_sa
from ke_drl.Phi_sa import Phi_sa
from ke_drl.evaluation_metric import predict_embedding_weights
from ke_drl.matern_kernel import matern_kernel

try:
    from ke_drl.density_recovery import RecoverAndPlot
except ModuleNotFoundError as exc:
    RecoverAndPlot = None
    print(f"Density plotting disabled because an optional plotting dependency is unavailable: {exc}")


print("# ================================================================ #")
print("#   Algorithm Simulation: KE-DRL estimation and Density Recovery    #")
print("# ================================================================ #")


def as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(x)


def as_float(x: Any) -> float:
    return float(x.item() if isinstance(x, torch.Tensor) else x)


def as_int(x: Any) -> int:
    return int(float(x.item() if isinstance(x, torch.Tensor) else x))


def first_or_stamped_z_path(data_dir: Path, stamp: str) -> Path:
    preferred = data_dir / "Z_true.pt"
    if preferred.exists():
        return preferred
    stamped = data_dir / f"Z_true_{stamp}.pt"
    if stamped.exists():
        return stamped
    raise FileNotFoundError(f"Could not find {preferred} or {stamped}.")


@torch.no_grad()
def beta_for_evaluation_point(
    *,
    method: str,
    B_hat: torch.Tensor,
    pre: dict[str, Any],
    s_eval: torch.Tensor,
    a_eval: torch.Tensor,
    lambda_reg: float,
) -> torch.Tensor:
    x_eval = torch.cat([s_eval.reshape(1, -1), a_eval.reshape(1, -1)], dim=1).to(
        device=B_hat.device, dtype=B_hat.dtype
    )
    method_l = method.lower()
    if method_l == "song":
        return predict_embedding_weights(
            pre["X_train"], x_eval, B_hat, **pre["x_kernel_params"]
        ).reshape(-1)
    if method_l == "bellman":
        k_eval = matern_kernel(pre["X_train"], x_eval, **pre["x_kernel_params"])
        gamma_eval = Gamma_sa(pre["K_X"], k_eval, lambda_reg)
        phi_eval = Phi_sa(pre["K_plus"], gamma_eval, pre["eta_plus"])
        return (phi_eval.T @ B_hat).reshape(-1)

    print(f"Warning: method={method!r} is not supported in the global simulation wrapper; using song weights.")
    return predict_embedding_weights(pre["X_train"], x_eval, B_hat, **pre["x_kernel_params"]).reshape(-1)


start = time.time()
job_id = os.environ.get("SLURM_JOB_ID")
array_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
print(f"Slurm Job ID: {job_id}")
print(f"Slurm Array ID: {array_id} -- used as the offline-data replicate id")

with open("./params.yaml", "r", encoding="utf-8") as f:
    P = yaml.safe_load(f)

seed = seed_from_array(int(P.get("random_seed", 20260512)) + 200000, array_id)
print(f"Random seed: {seed}")

data_dir = Path("data")
df_path = data_dir / f"offline_data_{array_id}.pt"
blob = torch.load(df_path, map_location="cpu")

s0 = torch.as_tensor(blob["s0"], dtype=torch.float64)
a0 = torch.as_tensor(blob["a0"], dtype=torch.float64)
s1 = torch.as_tensor(blob["s1"], dtype=torch.float64)
a1 = torch.as_tensor(blob["a1"], dtype=torch.float64)
r0 = torch.as_tensor(blob["r0"], dtype=torch.float64)
r = torch.as_tensor(blob["r"], dtype=torch.float64)

meta = blob["metadata"]
beh_policy = meta["policy"]
beh_policy_params = meta["policy_params"]

z_path = first_or_stamped_z_path(data_dir, str(array_id))
blob_z = torch.load(z_path, map_location="cpu")
Z_true = torch.stack([torch.as_tensor(z, dtype=torch.float64) for z in blob_z["Z_true"]], dim=0)
Z_true_tensor = Z_true[0][:, : as_int(P["reward_dim"])]
meta_z = blob_z["metadata"]
s_eval = torch.as_tensor(meta_z["s_star"], dtype=torch.float64)
a_eval = torch.as_tensor(meta_z["a_star"], dtype=torch.float64)
target_policy = meta_z["policy"]
target_policy_params = clean_policy_params(target_policy, meta_z["policy_params"][target_policy])

nu = as_float(P["kernel"]["nu"])
length_scale = as_float(P["kernel"]["length_scale"])
sigma_k = as_float(P["kernel"]["sigma"])
gamma_val = as_float(P["gamma_val"])
lambda_reg = as_float(P["lambda_reg"])
lambda_B = as_float(P.get("lambda_B", P.get("optimization", {}).get("lambda_B", 0.0)))
hull_expand_factor = as_float(P["hull_expand_factor"])
bandwidth = as_float(P["bandwidth"])
lambda_rec = as_float(P["lambda_rec"])
d_r_method = str(P["d_r_method"])

opt = P["optimization"]
lr = as_float(opt["lr"])
weight_decay = as_float(opt["weight_decay"])
num_steps = as_int(opt["num_steps"])
target_batch_size = opt.get("target_batch_size")
target_batch_size = None if target_batch_size in (None, "None", "none", 0, "0") else as_int(target_batch_size)
initial_scale = as_float(opt.get("initial_scale", 1e-3))
H_batch_size = as_int(P["H_batch_size"])

use_low_rank = as_bool(opt.get("use_low_rank", False))
B_positive = as_bool(opt.get("B_positive", False))
fixed_point_constraint = as_bool(opt.get("fixed_point_constraint", False))
B_conv = as_bool(opt.get("B_conv", False))
Sum_one_W = as_bool(opt.get("Sum_one_W", False))
B_ridge_penalty = as_bool(opt.get("B_ridge_penalty", False))
exact_projection = as_bool(opt.get("exact_projection", False))
NonNeg_W = as_bool(opt.get("NonNeg_W", False))
FP_penalty_lambda = as_float(opt.get("FP_penalty_lambda", 0.0))
ortho_lambda = as_float(opt.get("ortho_lambda", 0.0))
mass_anchor_lambda = as_float(opt.get("mass_anchor_lambda", 0.0))
target_mass = as_float(opt.get("target_mass", 1.0))

s_star, a_star, target_idx = select_target_set(
    s0,
    a0,
    P.get("target_set", P.get("x_star", {})),
    seed=seed,
    fallback_eval_s=s_eval,
    fallback_eval_a=a_eval,
)

print("Data and parameters loaded.")
print(f"offline shapes: s0={tuple(s0.shape)}, a0={tuple(a0.shape)}, r0={tuple(r0.shape)}")
print(f"MC truth shape: {tuple(Z_true_tensor.shape)} from {z_path}")
print(f"MC benchmark s*: {s_eval.tolist()}")
print(f"MC benchmark a*: {a_eval.tolist()}")
print(f"global target set size L={s_star.shape[0]}; target mode={P.get('target_set', P.get('x_star', {})).get('mode', 'train_subset')}")
print(f"target policy: {target_policy}")
print(f"lambda_Gamma={lambda_reg}, lambda_B={lambda_B}, d_r_method={d_r_method}")

B_hat, history_obj, history_be, pre = KE_DRL(
    s0=s0,
    s1=s1,
    a1=a1,
    a0=a0,
    s_star=s_star,
    a_star=a_star,
    r=r0,
    target_p_choice=target_policy,
    target_p_params=target_policy_params,
    nu=nu,
    length_scale=length_scale,
    sigma=sigma_k,
    gamma_val=gamma_val,
    lambda_reg=lambda_reg,
    lambda_B=lambda_B,
    num_grid_points=as_int(P["num_grid_points"]),
    hull_expand_factor=hull_expand_factor,
    lr=lr,
    weight_decay=weight_decay,
    num_steps=num_steps,
    target_batch_size=target_batch_size,
    random_seed=seed,
    initial_scale=initial_scale,
    FP_penalty_lambda=FP_penalty_lambda,
    use_low_rank=use_low_rank,
    rank_for_low_rank=None,
    B_positive=B_positive,
    fixed_point_constraint=fixed_point_constraint,
    exact_projection=exact_projection,
    ortho_lambda=ortho_lambda,
    B_conv=B_conv,
    Sum_one_W=Sum_one_W,
    NonNeg_W=NonNeg_W,
    mass_anchor_lambda=mass_anchor_lambda,
    target_mass=target_mass,
    B_ridge_penalty=B_ridge_penalty,
    H_batch_size=H_batch_size,
    device=None,
    dtype=torch.float64,
    verbose=True,
)

print("KE-DRL estimation is done.")
print("B_hat shape:", tuple(B_hat.shape))

Z_grid = pre["Z_grid"]
torch.save(Z_grid.detach().cpu(), data_dir / f"Zgrid_{array_id}.pt")
torch.save(
    {
        "B_hat": B_hat.detach().cpu(),
        "history_obj": history_obj,
        "history_be": history_be,
        "target_index": None if target_idx is None else target_idx.detach().cpu(),
        "target_set_size": int(s_star.shape[0]),
        "z_path": str(z_path),
    },
    data_dir / f"fit_{array_id}.pt",
)
print("Z grid shape:", tuple(Z_grid.shape))

config = {
    "job_id": str(job_id),
    "data_ID": str(array_id),
    "lr": lr,
    "fixed_point_constraint": fixed_point_constraint,
    "FP_penalty_lambda": FP_penalty_lambda,
    "Sum_one_W": Sum_one_W,
    "NonNeg_W": NonNeg_W,
    "mass_anchor_lambda": mass_anchor_lambda,
    "target_mass": target_mass,
    "num_steps": num_steps,
    "nu": nu,
    "length_scale": length_scale,
    "sigma_k": sigma_k,
    "gamma_val": gamma_val,
    "n_ids": as_int(P["n_ids"]),
    "n_timepoints": as_int(P["n_timepoints"]),
    "num_grid_points": as_int(P["num_grid_points"]),
    "hull_expand_factor": hull_expand_factor,
    "lambda_reg": lambda_reg,
    "lambda_B": lambda_B,
    "bandwidth": bandwidth,
    "lambda_rec": lambda_rec,
    "method": d_r_method,
    "state_dim": as_int(P["state_dim"]),
    "reward_dim": as_int(P["reward_dim"]),
    "action_dim": as_int(P["action_dim"]),
    "s_star": s_eval.tolist(),
    "a_star": a_eval.tolist(),
    "behavioral_policy": beh_policy,
    "target_policy": target_policy,
    "target_set_size": int(s_star.shape[0]),
}

beta_eval = beta_for_evaluation_point(
    method=d_r_method,
    B_hat=B_hat,
    pre=pre,
    s_eval=s_eval,
    a_eval=a_eval,
    lambda_reg=lambda_reg,
)
print("Evaluation beta shape:", tuple(beta_eval.shape))

plot_dir = Path("plots") / f"run_{array_id}"
tool = RecoverAndPlot(config) if RecoverAndPlot is not None else None
if tool is not None:
    fz, grid_dict = tool.marginals_from_beta(
        beta_eval,
        Z_grid,
        reward_dim=config["reward_dim"],
        nu=nu,
        length_scale=length_scale,
        sigma_k=sigma_k,
        lambda_rec=lambda_rec,
        bandwidth=bandwidth,
        n_grid=400,
        margin_factor=0.25,
    )
    tool.plot_densities(fz, grid_dict, outdir=str(plot_dir))
    if history_be:
        tool.plot_bellman_error(history_be, outdir=str(plot_dir))
    if history_obj:
        tool.plot_total_loss(history_obj, outdir=str(plot_dir))
else:
    print("Density and loss plots skipped; mean-embedding metrics will still be saved.")

mu_hat = mean_embedding_hat(beta_eval, Z_grid, nu=nu, length_scale=length_scale, sigma=sigma_k)
mu_true = mean_embedding_true(Z_grid, Z_true_tensor.to(device=Z_grid.device, dtype=Z_grid.dtype), nu=nu, length_scale=length_scale, sigma=sigma_k)
metrics = save_mu_outputs(run_id=array_id, mu_hat=mu_hat, mu_true=mu_true, beta=beta_eval)
print("Run metrics:", metrics)

if tool is not None:
    cache, _ = tool.mean_embedding_all(
        beta_eval,
        Z_grid,
        nu=nu,
        length_scale=length_scale,
        sigma_k=sigma_k,
        do_joint_dims=(0, 1),
        n1=120,
        n2=120,
        outdir=str(plot_dir / "ind_plots"),
    )
    try:
        tool.plot_operator_check_2d(cache, R=r0.to(device=Z_grid.device, dtype=Z_grid.dtype), gamma=gamma_val, dims=(0, 1), outdir=str(plot_dir))
    except Exception as exc:
        print(f"Operator-check plot skipped: {exc!r}")
    del cache

del pre
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

elapsed = time.time() - start
print("ALL DONE!")
print(f"Computation time: {int(elapsed // 60)} minutes and {int(elapsed % 60)} seconds")
print("=" * 70)
