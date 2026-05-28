from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from main_tune_global import (
    expected_curve_count,
    run_step,
    stage_shared_data,
    write_combo_metadata,
    write_result,
    write_yaml_atomic,
)
from parallel_offlinedata import run_parallel_offline_data


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw in (None, "") else int(raw)


def deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def scenario_overrides() -> dict[str, Any]:
    n_timepoints = env_int("SIM2_TIMEPOINTS", 500)
    return {
        "n_ids": env_int("SIM2_N_IDS", 300),
        "n_timepoints": n_timepoints,
        "offline_burn_in": env_int("SIM2_BURN_IN", 100),
        "experiment": {"num_replicates": env_int("SIM2_NUM_REPLICATES", 50)},
        "num_grid_points": env_int("SIM2_GRID_POINTS", 400),
        "Z_sim": {
            "n_ids": env_int("SIM2_Z_IDS", 10000),
            "n_timepoints": env_int("SIM2_Z_TIMEPOINTS", n_timepoints),
        },
        "transition_reduction": {
            "enabled": True,
            "method": os.environ.get("SIM2_REDUCTION_METHOD", "kmeans"),
            "n_basis": env_int("SIM2_REDUCED_N", 1500),
            "candidate_pool": env_int("SIM2_CANDIDATE_POOL", 50000),
            "max_iter": env_int("SIM2_KMEANS_ITER", 20),
            "batch_size": env_int("SIM2_REDUCTION_BATCH", 8192),
        },
    }


def scenario_tag(params: dict[str, Any]) -> str:
    if os.environ.get("SIM2_TAG"):
        return os.environ["SIM2_TAG"]
    red = params.get("transition_reduction") or {}
    return "T{}_N{}_r{}".format(
        int(params["n_timepoints"]),
        int(params["n_ids"]),
        int(red.get("n_basis", 0)),
    )


def write_params(base_params: str) -> tuple[dict[str, Any], str]:
    with open(base_params, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    params = deep_update(params, scenario_overrides())
    tag = scenario_tag(params)
    write_yaml_atomic(Path("params.yaml"), params)
    write_yaml_atomic(Path(f"params_sim2_{tag}.yaml"), params)
    write_combo_metadata(0, f"sim2_{tag}", {"simulation_2": scenario_overrides()}, int(params["experiment"]["num_replicates"]))
    return params, tag


def prepare(base_params: str) -> None:
    params, tag = write_params(base_params)
    n_rep = int(params["experiment"]["num_replicates"])
    print("Simulation-2 shared-data parameters:")
    print(
        {
            "tag": tag,
            "n_ids": params["n_ids"],
            "n_timepoints": params["n_timepoints"],
            "raw_transition_rows": int(params["n_ids"]) * (int(params["n_timepoints"]) - 1),
            "transition_reduction": params.get("transition_reduction"),
            "Z_sim": params.get("Z_sim"),
            "num_replicates": n_rep,
            "num_grid_points": params.get("num_grid_points"),
        }
    )
    run_step([sys.executable, "validate_sim_config.py", "--params", "params.yaml"])
    workers = int(
        os.environ.get("OFFLINE_DATA_WORKERS")
        or params.get("offline_data_workers")
        or os.environ.get("SLURM_CPUS_PER_TASK")
        or os.environ.get("SLURM_CPUS_ON_NODE")
        or 1
    )
    run_parallel_offline_data(n_rep, workers=max(1, min(n_rep, workers)), validate=True)
    run_step([sys.executable, "main_MonteCarloZ.py"])


def fit(base_params: str, shared_data_dir: Path, offline_id: int) -> None:
    params, tag = write_params(base_params)
    n_rep = int(params["experiment"]["num_replicates"])
    print(f"Simulation-2 fit tag={tag}, offline_id={offline_id}")
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


def aggregate(base_params: str, shared_data_dir: Path) -> None:
    start = time.time()
    params, tag = write_params(base_params)
    n_rep = int(params["experiment"]["num_replicates"])
    print(f"Simulation-2 aggregate tag={tag}")
    stage_shared_data(shared_data_dir, params, n_rep, offline_id=None)
    expected = expected_curve_count(params)
    actual = len(list(Path("mu").glob("mu_hat_*.csv")))
    if actual < expected:
        print(f"Warning: aggregating {actual}/{expected} expected curves.", flush=True)
    run_step([sys.executable, "mu_plot.py"])
    write_result(0, f"sim2_{tag}", {"simulation_2": scenario_overrides()}, time.time() - start)


def commands() -> None:
    reps = env_int("SIM2_NUM_REPLICATES", 50)
    last = reps - 1
    print(
        "\nExample for T=500:\n"
        "  jid_prep=$(sbatch --parsable --export=ALL,SIM2_STAGE=prepare,SIM2_TIMEPOINTS=500 Job_sim2.sbatch)\n"
        f"  jid_fit=$(sbatch --parsable --dependency=afterok:$jid_prep --array=0-{last} --export=ALL,SIM2_STAGE=fit,SIM2_TIMEPOINTS=500 Job_sim2.sbatch)\n"
        "  sbatch --dependency=afterok:$jid_fit --export=ALL,SIM2_STAGE=aggregate,SIM2_TIMEPOINTS=500 Job_sim2.sbatch\n"
        "\nExample for T=1000:\n"
        "  jid_prep=$(sbatch --parsable --export=ALL,SIM2_STAGE=prepare,SIM2_TIMEPOINTS=1000,SIM2_TAG=T1000_N300_r1500 Job_sim2.sbatch)\n"
        f"  jid_fit=$(sbatch --parsable --dependency=afterok:$jid_prep --array=0-{last} --export=ALL,SIM2_STAGE=fit,SIM2_TIMEPOINTS=1000,SIM2_TAG=T1000_N300_r1500 Job_sim2.sbatch)\n"
        "  sbatch --dependency=afterok:$jid_fit --export=ALL,SIM2_STAGE=aggregate,SIM2_TIMEPOINTS=1000,SIM2_TAG=T1000_N300_r1500 Job_sim2.sbatch\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prepare", "fit", "aggregate", "commands"], required=True)
    parser.add_argument("--base-params", default="params_tune.yaml")
    parser.add_argument("--shared-data-dir", default=os.environ.get("SIM2_SHARED_DATA_DIR", "data"))
    parser.add_argument("--offline-id", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "commands":
        commands()
        return
    if args.mode == "prepare":
        prepare(args.base_params)
        return
    if args.mode == "fit":
        offline_id = args.offline_id
        if offline_id is None:
            raw = os.environ.get("SLURM_ARRAY_TASK_ID")
            if raw is None:
                raise ValueError("--offline-id or SLURM_ARRAY_TASK_ID is required for fit mode.")
            offline_id = int(raw)
        fit(args.base_params, Path(args.shared_data_dir), offline_id)
        return
    if args.mode == "aggregate":
        aggregate(args.base_params, Path(args.shared_data_dir))


if __name__ == "__main__":
    main()
