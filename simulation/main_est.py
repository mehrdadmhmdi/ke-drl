from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from sim_utils import bootstrap_kedrl, clean_policy_params, seed_from_array
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
print("#   Global KE-DRL estimation over all evaluation target points      #")
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


def evaluation_config(P: dict[str, Any]) -> dict[str, int]:
    cfg = dict(P.get("evaluation") or {})
    target_cfg = dict(P.get("target_set") or {})
    return {
        "offline_data_id": as_int(cfg.get("offline_data_id", 0)),
        "num_points": as_int(cfg.get("num_points", target_cfg.get("num_points", 30))),
    }


def load_z_truths(data_dir: Path, expected_count: int) -> list[dict[str, Any]]:
    truths: list[dict[str, Any]] = []
    missing = []
    for eval_id in range(expected_count):
        path = data_dir / f"Z_true_{eval_id}.pt"
        if not path.exists():
            missing.append(str(path))
            continue
        blob = torch.load(path, map_location="cpu")
        Z_list = blob.get("Z_true")
        if not Z_list:
            raise ValueError(f"{path} does not contain a nonempty Z_true list.")
        Z_tensor = torch.as_tensor(Z_list[0], dtype=torch.float64)
        meta = blob.get("metadata", {})
        truths.append(
            {
                "eval_id": eval_id,
                "path": path,
                "Z_true": Z_tensor,
                "metadata": meta,
                "s_star": torch.as_tensor(meta["s_star"], dtype=torch.float64).reshape(1, -1),
                "a_star": torch.as_tensor(meta["a_star"], dtype=torch.float64).reshape(1, -1),
            }
        )
    if missing:
        preview = "\n  ".join(missing[:10])
        more = "" if len(missing) <= 10 else f"\n  ... and {len(missing) - 10} more"
        raise FileNotFoundError(
            "The global estimator needs one Monte Carlo truth file per evaluation point. "
            f"Expected {expected_count} files named data/Z_true_0.pt through "
            f"data/Z_true_{expected_count - 1}.pt, but these are missing:\n  {preview}{more}"
        )
    return truths


def check_truth_policy_consistency(truths: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    first = truths[0]["metadata"]
    target_policy = first["policy"]
    target_policy_params = first["policy_params"][target_policy]
    for item in truths[1:]:
        meta = item["metadata"]
        if meta.get("policy") != target_policy:
            raise ValueError(
                f"Inconsistent target policy in {item['path']}: "
                f"{meta.get('policy')!r} != {target_policy!r}"
            )
    return target_policy, clean_policy_params(target_policy, target_policy_params)


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


def write_evaluation_point_table(truths: list[dict[str, Any]], out_path: Path) -> None:
    rows = []
    for item in truths:
        meta = item["metadata"]
        s = item["s_star"].reshape(-1).tolist()
        a = item["a_star"].reshape(-1).tolist()
        rows.append(
            {
                "eval_id": item["eval_id"],
                "offline_row": meta.get("offline_row"),
                "z_path": str(item["path"]),
                **{f"s{i}": value for i, value in enumerate(s)},
                **{f"a{i}": value for i, value in enumerate(a)},
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)


start = time.time()
job_id = os.environ.get("SLURM_JOB_ID")
array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
print(f"Slurm Job ID: {job_id}")
print(f"Slurm Array ID: {array_id} -- ignored; this job fits one global B")

with open("./params.yaml", "r", encoding="utf-8") as f:
    P = yaml.safe_load(f)

eval_cfg = evaluation_config(P)
offline_data_id = eval_cfg["offline_data_id"]
expected_eval_points = eval_cfg["num_points"]

seed = seed_from_array(int(P.get("random_seed", 20260512)) + 200000, offline_data_id)
print(f"Random seed: {seed}")
print(f"Offline data id: {offline_data_id}")
print(f"Expected evaluation points: {expected_eval_points}")

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

truths = load_z_truths(data_dir, expected_eval_points)
target_policy, target_policy_params = check_truth_policy_consistency(truths)
s_star = torch.cat([item["s_star"] for item in truths], dim=0)
a_star = torch.cat([item["a_star"] for item in truths], dim=0)
write_evaluation_point_table(truths, data_dir / "evaluation_points.csv")

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

print("Data and parameters loaded.")
print(f"offline path: {df_path}")
print(f"offline shapes: s0={tuple(s0.shape)}, a0={tuple(a0.shape)}, r0={tuple(r0.shape)}")
print(f"evaluation target set shape: s_star={tuple(s_star.shape)}, a_star={tuple(a_star.shape)}")
print(f"first true-Z shape: {tuple(truths[0]['Z_true'].shape)} from {truths[0]['path']}")
print(f"target policy: {target_policy}")
print(f"behavior policy: {beh_policy}")
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

print("KE-DRL global estimation is done.")
print("B_hat shape:", tuple(B_hat.shape))

Z_grid = pre["Z_grid"]
torch.save(Z_grid.detach().cpu(), data_dir / "Zgrid_global.pt")
torch.save(
    {
        "B_hat": B_hat.detach().cpu(),
        "history_obj": history_obj,
        "history_be": history_be,
        "offline_data_id": offline_data_id,
        "eval_ids": [item["eval_id"] for item in truths],
        "target_set_size": int(s_star.shape[0]),
        "z_paths": [str(item["path"]) for item in truths],
    },
    data_dir / "fit_global.pt",
)
print("Return dictionary Z_grid shape:", tuple(Z_grid.shape))

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

all_truth_samples = torch.cat([item["Z_true"][:, : config["reward_dim"]] for item in truths], dim=0)
Z_eval = common_eval_grid(all_truth_samples, as_int(P["num_grid_points"])).to(device=Z_grid.device, dtype=Z_grid.dtype)
torch.save(Z_eval.detach().cpu(), data_dir / "Zeval_global.pt")
print("Common evaluation grid shape:", tuple(Z_eval.shape))

tool = RecoverAndPlot(config) if RecoverAndPlot is not None else None
if tool is not None and history_be:
    tool.plot_bellman_error(history_be, outdir="plots")
if tool is not None and history_obj:
    tool.plot_total_loss(history_obj, outdir="plots")

metrics_rows = []
for item in truths:
    eval_id = item["eval_id"]
    s_eval = item["s_star"].reshape(-1)
    a_eval = item["a_star"].reshape(-1)
    Z_true_tensor = item["Z_true"][:, : config["reward_dim"]].to(device=Z_grid.device, dtype=Z_grid.dtype)
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
    metrics = save_mu_outputs(run_id=eval_id, mu_hat=mu_hat, mu_true=mu_true, beta=beta_eval)
    metrics["offline_data_id"] = offline_data_id
    metrics_rows.append(metrics)
    print(f"Evaluation point {eval_id} metrics:", metrics)

    if tool is not None and eval_id == 0:
        plot_dir = Path("plots") / f"eval_{eval_id}"
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
            print(f"Operator-check plot skipped for eval point {eval_id}: {exc!r}")
        del cache
if tool is None:
    print("Density and loss plots skipped; mean-embedding metrics were still saved.")

pd.DataFrame(metrics_rows).to_csv("metrics/global_eval_metrics.csv", index=False)

del pre
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

elapsed = time.time() - start
print("ALL DONE!")
print(f"Computation time: {int(elapsed // 60)} minutes and {int(elapsed % 60)} seconds")
print("=" * 70)
