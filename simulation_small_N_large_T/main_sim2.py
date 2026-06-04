from __future__ import annotations

import argparse
from copy import deepcopy
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


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


POLICY_CODE_TO_NAME = {
    "U": "uniform",
    "G": "gaussian",
    "L": "logistic",
}

UG_CENTER = [0.1, -0.1, 0.15, -0.45, 0.0]
UL_CENTER = [0.08, -0.12, 0.16, -0.42, -0.02]
GAUSSIAN_CENTER = [0.12, -0.08, 0.12, -0.35, 0.04]
LOGISTIC_CENTER = [0.05, -0.12, 0.18, -0.4, -0.03]


def gaussian_policy(theta_mean: list[float], epsilon_mean: float, *, log_std: float) -> dict[str, Any]:
    return {
        "name": "gaussian",
        "theta_mean": theta_mean,
        "theta_std": [0.0, 0.0, 0.0, 0.0, 0.0],
        "epsilon_mean": [epsilon_mean],
        "epsilon_std": [log_std],
    }


def logistic_policy(theta_loc: list[float], epsilon_loc: float, *, log_scale: float) -> dict[str, Any]:
    return {
        "name": "logistic",
        "theta_loc": theta_loc,
        "theta_scale": [0.0, 0.0, 0.0, 0.0, 0.0],
        "epsilon_loc": [epsilon_loc],
        "epsilon_scale": [log_scale],
    }


def uniform_centered_policy(theta_center: list[float], epsilon_center: float, *, half_width: float) -> dict[str, Any]:
    return {
        "name": "uniform",
        "theta_lower": theta_center,
        "theta_upper": theta_center,
        "epsilon_lower": [epsilon_center - half_width],
        "epsilon_upper": [epsilon_center + half_width],
    }


POLICY_PAIR_CONFIGS: dict[str, dict[str, Any]] = {
    "UG": {
        "uniform": uniform_centered_policy(UG_CENTER, 0.025, half_width=0.75),
        "gaussian": gaussian_policy(UG_CENTER, 0.025, log_std=-2.2),
    },
    "UL": {
        "uniform": uniform_centered_policy(UL_CENTER, 0.03, half_width=0.75),
        "logistic": logistic_policy(UL_CENTER, 0.03, log_scale=-2.4),
    },
    "GU": {
        "gaussian": gaussian_policy(GAUSSIAN_CENTER, 0.02, log_std=-2.0),
        "uniform": uniform_centered_policy(GAUSSIAN_CENTER, 0.02, half_width=0.25),
    },
    "GL": {
        "gaussian": gaussian_policy(GAUSSIAN_CENTER, 0.02, log_std=-2.0),
        "logistic": logistic_policy([0.1, -0.06, 0.14, -0.32, 0.02], 0.03, log_scale=-2.4),
    },
    "LU": {
        "logistic": logistic_policy(LOGISTIC_CENTER, 0.03, log_scale=-2.2),
        "uniform": uniform_centered_policy(LOGISTIC_CENTER, 0.03, half_width=0.25),
    },
    "LG": {
        "logistic": logistic_policy(LOGISTIC_CENTER, 0.03, log_scale=-2.2),
        "gaussian": gaussian_policy([0.07, -0.1, 0.16, -0.38, -0.01], 0.02, log_std=-2.2),
    },
}


def supported_policy_pairs() -> list[str]:
    return [
        f"{behavior}{target}"
        for behavior in POLICY_CODE_TO_NAME
        for target in POLICY_CODE_TO_NAME
        if behavior != target
    ]


def scenario_policy_pair() -> str:
    pair = os.environ.get("SIM2_POLICY_PAIR", "UL").strip().upper()
    if len(pair) != 2 or any(code not in POLICY_CODE_TO_NAME for code in pair):
        allowed = ", ".join(supported_policy_pairs())
        raise ValueError(f"SIM2_POLICY_PAIR must be one of {allowed}; got {pair!r}.")
    if pair[0] == pair[1]:
        allowed = ", ".join(supported_policy_pairs())
        raise ValueError(f"SIM2_POLICY_PAIR must use different behavior and target policies ({allowed}); got {pair!r}.")
    return pair


def scenario_policies(pair: str | None = None) -> tuple[str, str]:
    pair = scenario_policy_pair() if pair is None else pair
    return POLICY_CODE_TO_NAME[pair[0]], POLICY_CODE_TO_NAME[pair[1]]


def scenario_policy_config(pair: str) -> dict[str, Any]:
    behavior_policy, target_policy = scenario_policies(pair)
    cfg = {
        "Behvaioral_policy": behavior_policy,
        "evaluation_Target_policy": target_policy,
    }
    cfg.update(deepcopy(POLICY_PAIR_CONFIGS[pair]))
    return cfg


def scenario_overrides() -> dict[str, Any]:
    n_ids = env_int("SIM2_N_IDS", 300)
    n_timepoints = env_int("SIM2_TIMEPOINTS", 50)
    raw_transition_rows = n_ids * max(1, n_timepoints - 1)
    basis_n = env_int("SIM2_BASIS_N", env_int("SIM2_REDUCED_N", 200))
    policy_pair = scenario_policy_pair()
    return {
        "n_ids": n_ids,
        "n_timepoints": n_timepoints,
        "offline_burn_in": env_int("SIM2_BURN_IN", 100),
        "experiment": {"num_replicates": env_int("SIM2_NUM_REPLICATES", 100)},
        "num_grid_points": env_int("SIM2_GRID_POINTS", 100),
        "Z_sim": {
            "n_ids": env_int("SIM2_Z_IDS", 10000),
            "n_timepoints": env_int("SIM2_Z_TIMEPOINTS", 500),
        },
        "target_set": {
            "mode": os.environ.get("SIM2_TARGET_MODE", "all"),
            "num_points": env_int("SIM2_TARGET_POINTS", raw_transition_rows),
            "seed_offset": 7919,
            "exclude_benchmark": False,
        },
        "mean_embedding_basis": {
            "method": os.environ.get("SIM2_BASIS_METHOD", "kmeans"),
            "n_basis": basis_n,
            "candidate_pool": env_int("SIM2_CANDIDATE_POOL", 50000),
            "max_iter": env_int("SIM2_KMEANS_ITER", 20),
            "batch_size": env_int("SIM2_REDUCTION_BATCH", 8192),
        },
        "transition_reduction": {
            "enabled": True,
            "method": os.environ.get("SIM2_REDUCTION_METHOD", "kmeans"),
            "n_basis": env_int("SIM2_OPERATOR_REDUCED_N", basis_n),
            "candidate_pool": env_int("SIM2_CANDIDATE_POOL", 50000),
            "max_iter": env_int("SIM2_KMEANS_ITER", 20),
            "batch_size": env_int("SIM2_REDUCTION_BATCH", 8192),
        },
        "policy": scenario_policy_config(policy_pair),
    }


def scenario_tag(params: dict[str, Any]) -> str:
    if os.environ.get("SIM2_TAG"):
        return os.environ["SIM2_TAG"]
    basis = params.get("mean_embedding_basis") or {}
    return "T{}_N{}_r{}".format(
        int(params["n_timepoints"]),
        int(params["n_ids"]),
        int(basis.get("n_basis", 0)),
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
    policy = params.get("policy") or {}
    print("Simulation-2 shared-data parameters:")
    print(
        {
            "tag": tag,
            "policy_pair": scenario_policy_pair(),
            "behavior_policy": policy.get("Behvaioral_policy"),
            "target_policy": policy.get("evaluation_Target_policy"),
            "n_ids": params["n_ids"],
            "n_timepoints": params["n_timepoints"],
            "raw_transition_rows": int(params["n_ids"]) * (int(params["n_timepoints"]) - 1),
            "mean_embedding_basis": params.get("mean_embedding_basis"),
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
    policy = params.get("policy") or {}
    print(
        f"Simulation-2 fit tag={tag}, policy_pair={scenario_policy_pair()}, "
        f"behavior={policy.get('Behvaioral_policy')}, target={policy.get('evaluation_Target_policy')}, "
        f"offline_id={offline_id}"
    )
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
    reuse_existing_params = env_bool("SIM2_AGGREGATE_EXISTING_PARAMS") and Path("params.yaml").exists()
    if reuse_existing_params:
        with open("params.yaml", "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
        tag = scenario_tag(params)
        print("Simulation-2 aggregate is reusing existing params.yaml.", flush=True)
    else:
        params, tag = write_params(base_params)
    n_rep = int(params["experiment"]["num_replicates"])
    policy = params.get("policy") or {}
    print(
        f"Simulation-2 aggregate tag={tag}, policy_pair={scenario_policy_pair()}, "
        f"behavior={policy.get('Behvaioral_policy')}, target={policy.get('evaluation_Target_policy')}"
    )
    stage_shared_data(shared_data_dir, params, n_rep, offline_id=None)
    expected = expected_curve_count(params)
    actual = len(list(Path("mu").glob("mu_hat_*.csv")))
    if actual < expected:
        print(f"Warning: aggregating {actual}/{expected} expected curves.", flush=True)
    run_step([sys.executable, "mu_plot.py"])
    if reuse_existing_params:
        metadata = {
            "simulation_2": {
                "params_source": "existing params.yaml",
                "policy": params.get("policy"),
                "partial_aggregate": actual < expected,
                "actual_curves": actual,
                "expected_curves": expected,
            }
        }
    else:
        metadata = {"simulation_2": scenario_overrides()}
    write_result(0, f"sim2_{tag}", metadata, time.time() - start)


def commands() -> None:
    reps = env_int("SIM2_NUM_REPLICATES", 100)
    last = reps - 1
    pairs = " ".join(supported_policy_pairs())
    print(
        "\nExample for T=50, N=300, L=200, m=100, all off-diagonal policy pairs:\n"
        f"  for PAIR in {pairs}; do\n"
        "    jid_prep=$(sbatch --parsable --job-name=${PAIR}_prepare --export=ALL,SIM2_POLICY_PAIR=$PAIR,SIM2_STAGE=prepare Job_sim2.sbatch)\n"
        f"    jid_fit=$(sbatch --parsable --dependency=afterok:$jid_prep --array=0-{last} --job-name=${{PAIR}}_fit --export=ALL,SIM2_POLICY_PAIR=$PAIR,SIM2_STAGE=fit Job_sim2.sbatch)\n"
        "    sbatch --dependency=afterok:$jid_fit --job-name=${PAIR}_aggregate --export=ALL,SIM2_POLICY_PAIR=$PAIR,SIM2_STAGE=aggregate Job_sim2.sbatch\n"
        "  done\n"
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
