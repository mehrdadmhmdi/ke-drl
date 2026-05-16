from __future__ import annotations

import gc
import math
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from sim_utils import bootstrap_kedrl, clean_policy_params, kedrl_import_info, seed_from_array, select_target_set
from sim_eval import common_eval_grid, mean_embedding_hat, mean_embedding_true, save_mu_outputs


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
print("#   Global KE-DRL estimation for one offline replicate              #")
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


def maybe_int(x: Any) -> int | None:
    if x is None:
        return None
    return as_int(x)


def _exp_safe(x: float) -> float:
    return float(math.exp(max(min(float(x), 700.0), -700.0)))


def summarize_risk_history(history_obj: list[float], history_be: list[float]) -> dict[str, float]:
    """Summarize optimizer risks from the full-target history.

    `history_obj` stores log regularized objective values. `history_be` stores
    log square-root Bellman-risk values, so squaring its exponential returns the
    empirical Bellman risk.
    """
    out: dict[str, float] = {"risk_n_steps_recorded": float(len(history_obj))}
    if history_obj:
        first = float(history_obj[0])
        final = float(history_obj[-1])
        best = float(min(history_obj))
        out.update(
            {
                "risk_log_obj_initial": first,
                "risk_log_obj_final": final,
                "risk_log_obj_min": best,
                "risk_obj_initial": _exp_safe(first),
                "risk_obj_final": _exp_safe(final),
                "risk_obj_min": _exp_safe(best),
                "risk_log_obj_drop": first - final,
            }
        )
    if history_be:
        first = float(history_be[0])
        final = float(history_be[-1])
        best = float(min(history_be))
        out.update(
            {
                "risk_log_bellman_root_initial": first,
                "risk_log_bellman_root_final": final,
                "risk_log_bellman_root_min": best,
                "risk_bellman_root_initial": _exp_safe(first),
                "risk_bellman_root_final": _exp_safe(final),
                "risk_bellman_root_min": _exp_safe(best),
                "risk_bellman_initial": _exp_safe(2.0 * first),
                "risk_bellman_final": _exp_safe(2.0 * final),
                "risk_bellman_min": _exp_safe(2.0 * best),
                "risk_log_bellman_root_drop": first - final,
            }
        )
    return out


def load_benchmark_truth(data_dir: Path, offline_data_id: int) -> dict[str, Any]:
    path = data_dir / f"Z_true_{offline_data_id}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing benchmark true-Z file: {path}. Run Job_Z.sbatch after Job_data.sbatch."
        )
    blob = torch.load(path, map_location="cpu")
    Z_list = blob.get("Z_true")
    if not Z_list:
        raise ValueError(f"{path} does not contain a nonempty Z_true list.")
    meta = blob.get("metadata", {})
    if "s_star" not in meta or "a_star" not in meta:
        raise ValueError(f"{path} metadata must contain the benchmark s_star and a_star.")
    truth_offline_id = maybe_int(meta.get("offline_data_id"))
    if truth_offline_id is not None and truth_offline_id != offline_data_id:
        raise ValueError(
            f"{path} was generated for offline_data_id={truth_offline_id}, "
            f"but this estimation job is offline_data_id={offline_data_id}."
        )
    return {
        "path": path,
        "Z_true": torch.as_tensor(Z_list[0], dtype=torch.float64),
        "metadata": meta,
        "s_eval": torch.as_tensor(meta["s_star"], dtype=torch.float64).reshape(1, -1),
        "a_eval": torch.as_tensor(meta["a_star"], dtype=torch.float64).reshape(1, -1),
    }


def write_target_point_table(
    *,
    out_path: Path,
    offline_data_id: int,
    s_star: torch.Tensor,
    a_star: torch.Tensor,
    target_idx: torch.Tensor | None,
    benchmark_row: int | None,
) -> None:
    rows = []
    if target_idx is None:
        idx_values = [None] * int(s_star.shape[0])
    else:
        idx_values = [int(x) for x in target_idx.detach().cpu().reshape(-1).tolist()]
    for j, idx in enumerate(idx_values):
        s = s_star[j].detach().cpu().reshape(-1).tolist()
        a = a_star[j].detach().cpu().reshape(-1).tolist()
        rows.append(
            {
                "offline_data_id": offline_data_id,
                "target_id": j,
                "offline_row": idx,
                "is_benchmark_row": bool(idx is not None and benchmark_row is not None and idx == benchmark_row),
                **{f"s{i}": value for i, value in enumerate(s)},
                **{f"a{i}": value for i, value in enumerate(a)},
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


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

    print(f"Warning: method={method!r} is not supported in the simulation wrapper; using song weights.")
    return predict_embedding_weights(pre["X_train"], x_eval, B_hat, **pre["x_kernel_params"]).reshape(-1)


start = time.time()
job_id = os.environ.get("SLURM_JOB_ID")
array_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
offline_data_id = int(os.environ.get("OFFLINE_DATA_ID", array_id))
print(f"Slurm Job ID: {job_id}")
print(f"Slurm Array ID: {array_id} -- used as the offline-replicate id")
print(f"Offline data id: {offline_data_id}")
print(f"ke_drl import source: {kedrl_import_info()}")

with open("./params.yaml", "r", encoding="utf-8") as f:
    P = yaml.safe_load(f)

num_replicates = as_int(P.get("experiment", {}).get("num_replicates", 1))
if offline_data_id < 0 or offline_data_id >= num_replicates:
    raise ValueError(f"Offline replicate id {offline_data_id} is outside 0,...,{num_replicates - 1}.")

seed = seed_from_array(int(P.get("random_seed", 20260512)) + 200000, offline_data_id)
target_seed = int(P.get("random_seed", 20260512)) + offline_data_id
print(f"Random seed: {seed}")
print(f"Number of offline replicates: {num_replicates}")

data_dir = Path("data")
df_path = data_dir / f"offline_data_{offline_data_id}.pt"
if not df_path.exists():
    raise FileNotFoundError(f"Missing offline data file: {df_path}. Run Job_data.sbatch first.")
blob = torch.load(df_path, map_location="cpu")

s0 = torch.as_tensor(blob["s0"], dtype=torch.float64)
a0 = torch.as_tensor(blob["a0"], dtype=torch.float64)
s1 = torch.as_tensor(blob["s1"], dtype=torch.float64)
a1 = torch.as_tensor(blob["a1"], dtype=torch.float64)
r0 = torch.as_tensor(blob["r0"], dtype=torch.float64)

meta = blob["metadata"]
beh_policy = meta["policy"]
truth = load_benchmark_truth(data_dir, offline_data_id)
meta_z = truth["metadata"]
benchmark_row = maybe_int(meta_z.get("offline_row"))
s_eval = truth["s_eval"]
a_eval = truth["a_eval"]

target_policy = meta_z.get("policy")
if not target_policy:
    target_policy_name = P["policy"]["evaluation_Target_policy"]
    target_policy = P["policy"][target_policy_name]["name"]
policy_params_blob = dict(meta_z.get("policy_params") or {})
if target_policy in policy_params_blob:
    target_policy_params = clean_policy_params(target_policy, policy_params_blob[target_policy])
else:
    target_policy_params = clean_policy_params(target_policy, P["policy"][P["policy"]["evaluation_Target_policy"]])

s_star, a_star, target_idx = select_target_set(
    s0,
    a0,
    P.get("target_set"),
    seed=target_seed,
    fallback_eval_s=s_eval,
    fallback_eval_a=a_eval,
    exclude_idx=benchmark_row,
)
write_target_point_table(
    out_path=data_dir / f"target_points_{offline_data_id}.csv",
    offline_data_id=offline_data_id,
    s_star=s_star,
    a_star=a_star,
    target_idx=target_idx,
    benchmark_row=benchmark_row,
)

requested_targets = int((P.get("target_set") or {}).get("num_points", s_star.shape[0]))
if s_star.shape[0] != requested_targets and str((P.get("target_set") or {}).get("mode", "")).lower() != "all":
    print(f"Warning: requested {requested_targets} target points but selected {s_star.shape[0]}.")

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
negativity_penalty_lambda = as_float(opt.get("negativity_penalty_lambda", opt.get("lambda_neg", 0.0)))
max_B_norm = opt.get("max_B_norm")
max_B_norm = None if max_B_norm in (None, "None", "none", 0, "0") else as_float(max_B_norm)
eta_clip_min = opt.get("eta_clip_min", 0.0)
eta_clip_min = None if eta_clip_min in (None, "None", "none") else as_float(eta_clip_min)
eta_clip_max = opt.get("eta_clip_max")
eta_clip_max = None if eta_clip_max in (None, "None", "none") else as_float(eta_clip_max)
normalize_eta = as_bool(opt.get("normalize_eta", False))

print("Data and parameters loaded.")
print(f"offline path: {df_path}")
print(f"benchmark true-Z path: {truth['path']}")
print(f"offline shapes: s0={tuple(s0.shape)}, a0={tuple(a0.shape)}, r0={tuple(r0.shape)}")
print(f"benchmark row: {benchmark_row}")
print(f"global loss target set shape: s_star={tuple(s_star.shape)}, a_star={tuple(a_star.shape)}")
print(f"benchmark true-Z shape: {tuple(truth['Z_true'].shape)}")
print(f"target policy: {target_policy}")
print(f"behavior policy: {beh_policy}")
print(f"lambda_Gamma={lambda_reg}, lambda_B={lambda_B}, d_r_method={d_r_method}")
print(f"mass_anchor_lambda={mass_anchor_lambda}, target_mass={target_mass}")
print(
    "optimizer stabilizers: "
    f"lambda_neg={negativity_penalty_lambda}, max_B_norm={max_B_norm}, "
    f"eta_clip_min={eta_clip_min}, eta_clip_max={eta_clip_max}, normalize_eta={normalize_eta}"
)

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
    eta_clip_min=eta_clip_min,
    eta_clip_max=eta_clip_max,
    normalize_eta=normalize_eta,
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
    negativity_penalty_lambda=negativity_penalty_lambda,
    max_B_norm=max_B_norm,
    B_ridge_penalty=B_ridge_penalty,
    H_batch_size=H_batch_size,
    device=None,
    dtype=torch.float64,
    verbose=True,
)

print("KE-DRL global estimation is done.")
print("B_hat shape:", tuple(B_hat.shape))

with torch.no_grad():
    k_star_fit = pre["k_star"].to(device=B_hat.device, dtype=B_hat.dtype)
    target_beta = k_star_fit.transpose(0, 1) @ B_hat
    target_masses = target_beta.sum(dim=1).detach().cpu()
    target_neg_frac = (target_beta < 0).double().mean(dim=1).detach().cpu()
    target_mass_diag = {
        "target_mass_mean": float(target_masses.mean()),
        "target_mass_min": float(target_masses.min()),
        "target_mass_max": float(target_masses.max()),
        "target_mass_sd": float(target_masses.std(unbiased=True)) if target_masses.numel() > 1 else 0.0,
        "target_mass_rmse_to_target": float(torch.sqrt(torch.mean((target_masses - target_mass) ** 2))),
        "target_beta_min": float(target_beta.min().detach().cpu()),
        "target_beta_max": float(target_beta.max().detach().cpu()),
        "target_neg_frac_mean": float(target_neg_frac.mean()),
    }
print("Global-loss target mass diagnostics:", target_mass_diag)

Z_grid = pre["Z_grid"]
torch.save(Z_grid.detach().cpu(), data_dir / f"Zgrid_{offline_data_id}.pt")
torch.save(
    {
        "B_hat": B_hat.detach().cpu(),
        "history_obj": history_obj,
        "history_be": history_be,
        "offline_data_id": offline_data_id,
        "benchmark_row": benchmark_row,
        "benchmark_z_path": str(truth["path"]),
        "target_set_size": int(s_star.shape[0]),
        "target_indices": None if target_idx is None else [int(x) for x in target_idx.detach().cpu().reshape(-1).tolist()],
        "target_policy": target_policy,
        "target_policy_params": target_policy_params,
        "target_mass_diagnostics": target_mass_diag,
    },
    data_dir / f"fit_{offline_data_id}.pt",
)
print("Return dictionary Z_grid shape:", tuple(Z_grid.shape))

Path("metrics").mkdir(parents=True, exist_ok=True)
risk_metrics = summarize_risk_history(history_obj, history_be)
risk_metrics.update(
    {
        "offline_data_id": offline_data_id,
        "benchmark_row": benchmark_row,
        "target_set_size": int(s_star.shape[0]),
        "lambda_reg": lambda_reg,
        "lambda_B": lambda_B,
        "num_steps": num_steps,
        **target_mass_diag,
    }
)
pd.DataFrame([risk_metrics]).to_csv(f"metrics/risk_metrics_{offline_data_id}.csv", index=False)
print(f"Replicate {offline_data_id} risk metrics:", risk_metrics)

config = {
    "job_id": str(job_id),
    "offline_data_id": str(offline_data_id),
    "lr": lr,
    "fixed_point_constraint": fixed_point_constraint,
    "FP_penalty_lambda": FP_penalty_lambda,
    "Sum_one_W": Sum_one_W,
    "NonNeg_W": NonNeg_W,
    "mass_anchor_lambda": mass_anchor_lambda,
    "target_mass": target_mass,
    "negativity_penalty_lambda": negativity_penalty_lambda,
    "max_B_norm": max_B_norm,
    "eta_clip_min": eta_clip_min,
    "eta_clip_max": eta_clip_max,
    "normalize_eta": normalize_eta,
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
    "behavioral_policy": beh_policy,
    "target_policy": target_policy,
    "target_set_size": int(s_star.shape[0]),
}

Z_true_tensor = truth["Z_true"][:, : config["reward_dim"]].to(device=Z_grid.device, dtype=Z_grid.dtype)
Z_eval = common_eval_grid(Z_true_tensor, as_int(P["num_grid_points"])).to(device=Z_grid.device, dtype=Z_grid.dtype)
torch.save(Z_eval.detach().cpu(), data_dir / f"Zeval_{offline_data_id}.pt")
print("Benchmark evaluation grid shape:", tuple(Z_eval.shape))

tool = RecoverAndPlot(config) if RecoverAndPlot is not None and offline_data_id == 0 else None
if tool is not None and history_be:
    tool.plot_bellman_error(history_be, outdir="plots")
if tool is not None and history_obj:
    tool.plot_total_loss(history_obj, outdir="plots")

beta_eval = beta_for_evaluation_point(
    method=d_r_method,
    B_hat=B_hat,
    pre=pre,
    s_eval=s_eval,
    a_eval=a_eval,
    lambda_reg=lambda_reg,
)
mu_hat = mean_embedding_hat(
    beta_eval,
    Z_grid,
    nu=nu,
    length_scale=length_scale,
    sigma=sigma_k,
    eval_grid=Z_eval,
)
mu_true = mean_embedding_true(
    Z_eval,
    Z_true_tensor,
    nu=nu,
    length_scale=length_scale,
    sigma=sigma_k,
)
metrics = save_mu_outputs(run_id=offline_data_id, mu_hat=mu_hat, mu_true=mu_true, beta=beta_eval)
metrics.update(
    {
        "offline_data_id": offline_data_id,
        "benchmark_row": benchmark_row,
        "target_set_size": int(s_star.shape[0]),
        "benchmark_z_path": str(truth["path"]),
    }
)
pd.DataFrame([metrics]).to_csv(f"metrics/global_eval_metrics_{offline_data_id}.csv", index=False)
print(f"Replicate {offline_data_id} benchmark metrics:", metrics)

if tool is not None:
    plot_dir = Path("plots") / f"replicate_{offline_data_id}"
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
        tool.plot_operator_check_2d(
            cache,
            R=r0.to(device=Z_grid.device, dtype=Z_grid.dtype),
            gamma=gamma_val,
            dims=(0, 1),
            outdir=str(plot_dir),
        )
    except Exception as exc:
        print(f"Operator-check plot skipped for replicate {offline_data_id}: {exc!r}")
    del cache
elif RecoverAndPlot is None:
    print("Density and loss plots skipped; mean-embedding metrics were still saved.")
else:
    print("Density and loss plots are only generated for offline replicate 0.")

del pre
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

elapsed = time.time() - start
print("ALL DONE!")
print(f"Computation time: {int(elapsed // 60)} minutes and {int(elapsed % 60)} seconds")
print("=" * 70)
