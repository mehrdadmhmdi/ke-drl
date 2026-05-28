from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from main_tune_global import (
    deep_update,
    expected_curve_count,
    run_step,
    stage_shared_data,
    write_combo_metadata,
    write_result,
    write_yaml_atomic,
)
from parallel_offlinedata import run_parallel_offline_data


FINAL_COMBO_ID = 0
FINAL_COMBO_NAME = "final_kernel_sigma_0.7"
FINAL_OVERRIDES: dict[str, Any] = {
    "kernel": {"nu": 5.5, "length_scale": 1.0, "sigma": 0.7},
    "lambda_B": 0.02,
    "lambda_reg": 0.005,
    "optimization": {
        "mass_anchor_lambda": 1.0,
        "negativity_penalty_lambda": 0.0,
    },
}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def env_float_text(name: str, default: str) -> str:
    return os.environ.get(name, default)


def final_size_overrides() -> dict[str, Any]:
    """Large reportable defaults, all overridable from the sbatch environment."""
    n_eval = env_int("KEDRL_FINAL_BENCHMARK_POINTS", 10)
    return {
        "n_ids": env_int("KEDRL_FINAL_N_IDS", 10000),
        "Z_sim": {
            "n_ids": env_int("KEDRL_FINAL_Z_IDS", 100000),
            "n_timepoints": env_int("KEDRL_FINAL_Z_TIMEPOINTS", 500),
        },
        "experiment": {"num_replicates": env_int("KEDRL_FINAL_NUM_REPLICATES", 200)},
        "benchmark": {"num_points": n_eval},
        "target_set": {"mode": "train_subset", "num_points": env_int("KEDRL_FINAL_TRAIN_TARGETS", 100), "exclude_benchmark": True},
        "num_grid_points": env_int("KEDRL_FINAL_GRID_POINTS", 1000),
        "optimization": {
            "num_steps": env_int("KEDRL_FINAL_NUM_STEPS", 5000),
            "diagnostic_interval": env_int("KEDRL_FINAL_DIAGNOSTIC_INTERVAL", 100),
            "target_batch_size": env_int("KEDRL_FINAL_TARGET_BATCH_SIZE", 100),
            "lr": env_float_text("KEDRL_FINAL_LR", "3e-4"),
            "weight_decay": env_float_text("KEDRL_FINAL_WEIGHT_DECAY", "1e-4"),
            "mass_anchor_lambda": float(os.environ.get("KEDRL_FINAL_MASS_ANCHOR", "1.0")),
            "negativity_penalty_lambda": float(os.environ.get("KEDRL_FINAL_NEGATIVITY", "0.0")),
            "eta_clip_min": float(os.environ.get("KEDRL_FINAL_ETA_CLIP_MIN", "0.0")),
            "eta_clip_max": float(os.environ.get("KEDRL_FINAL_ETA_CLIP_MAX", "5.0")),
        },
    }


def resolved_params(base_params: str) -> dict[str, Any]:
    with open(base_params, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    params = deep_update(params, FINAL_OVERRIDES.copy())
    params = deep_update(params, final_size_overrides())
    return params


def write_final_params(base_params: str = "params_tune.yaml") -> dict[str, Any]:
    params = resolved_params(base_params)
    write_yaml_atomic(Path("params.yaml"), params)
    write_yaml_atomic(Path("params_final_resolved.yaml"), params)
    n_rep = int(params.get("experiment", {}).get("num_replicates", 1))
    write_combo_metadata(FINAL_COMBO_ID, FINAL_COMBO_NAME, FINAL_OVERRIDES, n_rep)
    return params


def prepare_shared_data(base_params: str) -> None:
    params = write_final_params(base_params)
    n_rep = int(params.get("experiment", {}).get("num_replicates", 1))
    workers = int(
        os.environ.get("OFFLINE_DATA_WORKERS")
        or os.environ.get("SLURM_CPUS_PER_TASK")
        or os.environ.get("SLURM_CPUS_ON_NODE")
        or 1
    )
    print("Final shared-data parameters:")
    print(
        {
            "n_ids": params["n_ids"],
            "Z_sim": params["Z_sim"],
            "experiment": params["experiment"],
            "benchmark": {"num_points": params["benchmark"]["num_points"]},
            "target_set": params["target_set"],
            "num_grid_points": params["num_grid_points"],
            "optimization_steps": params["optimization"]["num_steps"],
            "kernel": params["kernel"],
            "lambda_B": params["lambda_B"],
            "lambda_reg": params.get("lambda_reg"),
        }
    )
    run_step([sys.executable, "validate_sim_config.py", "--params", "params.yaml"])
    run_parallel_offline_data(n_rep, workers=max(1, min(n_rep, workers)), validate=True)
    run_step([sys.executable, "main_MonteCarloZ.py"])


def fit_one(base_params: str, shared_data_dir: Path, offline_id: int) -> None:
    params = write_final_params(base_params)
    n_rep = int(params.get("experiment", {}).get("num_replicates", 1))
    run_step([sys.executable, "validate_sim_config.py", "--params", "params.yaml"])
    stage_shared_data(shared_data_dir, params, n_rep, offline_id=offline_id)
    run_step(
        [
            sys.executable,
            "validate_sim_config.py",
            "--params",
            "params.yaml",
            "--data",
            f"data/offline_data_{offline_id}.pt",
        ]
    )
    env = os.environ.copy()
    env["SLURM_ARRAY_TASK_ID"] = str(offline_id)
    env["OFFLINE_DATA_ID"] = str(offline_id)
    run_step([sys.executable, "main_est.py"], env=env)
    print(f"Final fit finished for offline replicate {offline_id}.")


def aggregate_final(base_params: str, shared_data_dir: Path) -> None:
    start = time.time()
    params = write_final_params(base_params)
    n_rep = int(params.get("experiment", {}).get("num_replicates", 1))
    stage_shared_data(shared_data_dir, params, n_rep, offline_id=None)
    expected = expected_curve_count(params)
    actual = len(list(Path("mu").glob("mu_hat_*.csv")))
    if actual < expected:
        print(
            f"Warning: aggregating final run with {actual}/{expected} expected evaluation-target curves. "
            "Wait for all array jobs if this is unintended.",
            flush=True,
        )
    run_step([sys.executable, "mu_plot.py"])
    write_result(FINAL_COMBO_ID, FINAL_COMBO_NAME, FINAL_OVERRIDES, time.time() - start)


def show_commands() -> None:
    n_rep = env_int("KEDRL_FINAL_NUM_REPLICATES", 200)
    last = n_rep - 1
    print(
        "\nRecommended Slurm sequence:\n"
        "  jid_prep=$(sbatch --parsable --export=ALL,FINAL_STAGE=prepare Job_final_reportable.sbatch)\n"
        f"  jid_fit=$(sbatch --parsable --dependency=afterok:$jid_prep --array=0-{last} --export=ALL,FINAL_STAGE=fit Job_final_reportable.sbatch)\n"
        "  sbatch --dependency=afterok:$jid_fit --export=ALL,FINAL_STAGE=aggregate Job_final_reportable.sbatch\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prepare", "fit", "aggregate", "commands"], required=True)
    parser.add_argument("--base-params", default="params_tune.yaml")
    parser.add_argument("--shared-data-dir", default=os.environ.get("KEDRL_FINAL_SHARED_DATA_DIR", "data"))
    parser.add_argument("--offline-id", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "commands":
        show_commands()
        return
    if args.mode == "prepare":
        prepare_shared_data(args.base_params)
        return

    shared_data_dir = Path(args.shared_data_dir)
    if args.mode == "fit":
        offline_id = args.offline_id
        if offline_id is None:
            raw = os.environ.get("SLURM_ARRAY_TASK_ID")
            if raw is None:
                raise ValueError("--offline-id or SLURM_ARRAY_TASK_ID is required in fit mode.")
            offline_id = int(raw)
        fit_one(args.base_params, shared_data_dir, offline_id)
        return

    if args.mode == "aggregate":
        aggregate_final(args.base_params, shared_data_dir)


if __name__ == "__main__":
    main()
