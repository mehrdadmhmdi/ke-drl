from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def run_step(args: list[str], *, env: dict[str, str] | None = None) -> None:
    print("RUN:", " ".join(args), flush=True)
    subprocess.run(args, check=True, env=env)


def load_combo(grid_path: Path, combo_id: int) -> tuple[str, dict[str, Any]]:
    with open(grid_path, "r", encoding="utf-8") as f:
        grid = yaml.safe_load(f)
    combos = list(grid.get("combos") or [])
    if combo_id < 0 or combo_id >= len(combos):
        raise IndexError(f"combo_id={combo_id} outside tuning grid 0,...,{len(combos) - 1}.")
    combo = combos[combo_id]
    return str(combo.get("name", f"combo_{combo_id}")), dict(combo.get("overrides") or {})


def aggregate_risk_metrics() -> dict[str, float]:
    paths = sorted(Path("metrics").glob("risk_metrics_*.csv"))
    if not paths:
        return {}
    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    out: dict[str, float] = {"risk_n_replicates": float(len(df))}
    for col in [
        "risk_log_obj_final",
        "risk_log_obj_min",
        "risk_obj_final",
        "risk_obj_min",
        "risk_log_bellman_root_final",
        "risk_log_bellman_root_min",
        "risk_bellman_final",
        "risk_bellman_min",
        "risk_log_obj_drop",
        "risk_log_bellman_root_drop",
        "target_mass_mean",
        "target_mass_min",
        "target_mass_max",
        "target_mass_sd",
        "target_mass_rmse_to_target",
        "target_beta_min",
        "target_beta_max",
        "target_neg_frac_mean",
    ]:
        if col in df:
            out[f"{col}_mean"] = float(df[col].mean())
            out[f"{col}_sd"] = float(df[col].std(ddof=1)) if len(df) > 1 else 0.0
    return out


def write_result(combo_id: int, combo_name: str, overrides: dict[str, Any], elapsed: float) -> None:
    agg = pd.read_csv("metrics/aggregate_metrics.csv").iloc[0].to_dict()
    cal = pd.read_csv("metrics/calibration_deming.csv").iloc[0].to_dict()
    risk = aggregate_risk_metrics()
    score_true_z = (
        float(agg["RMSE_mean"])
        + 0.25 * float(agg["MAE_mean"])
        + 0.05 * float(agg["SupNorm_mean"])
        + 0.02 * abs(float(cal["deming_slope"]) - 1.0)
    )
    score_risk = float(risk.get("risk_log_obj_final_mean", float("nan")))
    score_mass = float(risk.get("target_mass_rmse_to_target_mean", float("nan")))
    row = {
        "combo_id": combo_id,
        "combo_name": combo_name,
        "score": score_true_z,
        "score_true_z": score_true_z,
        "score_risk": score_risk,
        "score_mass": score_mass,
        "elapsed_sec": elapsed,
        "overrides_json": json.dumps(overrides, sort_keys=True),
        **agg,
        **cal,
        **risk,
    }
    Path("metrics").mkdir(exist_ok=True)
    pd.DataFrame([row]).to_csv("metrics/tuning_result.csv", index=False)
    print("Tuning result:", row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combo-id", type=int, required=True)
    parser.add_argument("--base-params", default="params_tune.yaml")
    parser.add_argument("--grid", default="tuning_grid.yaml")
    args = parser.parse_args()

    start = time.time()
    combo_name, overrides = load_combo(Path(args.grid), args.combo_id)
    print(f"Tuning combo {args.combo_id}: {combo_name}")
    print("Overrides:", overrides)

    with open(args.base_params, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    params = deep_update(params, overrides)
    with open("params.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(params, f, sort_keys=False)

    run_step([sys.executable, "validate_sim_config.py", "--params", "params.yaml"])

    n_rep = int(params.get("experiment", {}).get("num_replicates", 1))
    for rep_id in range(n_rep):
        env = os.environ.copy()
        env["SLURM_ARRAY_TASK_ID"] = str(rep_id)
        env["OFFLINE_DATA_ID"] = str(rep_id)
        run_step([sys.executable, "main_offlinedata.py"], env=env)
        run_step(
            [
                sys.executable,
                "validate_sim_config.py",
                "--params",
                "params.yaml",
                "--data",
                f"data/offline_data_{rep_id}.pt",
            ]
        )

    for rep_id in range(n_rep):
        env = os.environ.copy()
        env["SLURM_ARRAY_TASK_ID"] = str(rep_id)
        env["OFFLINE_DATA_ID"] = str(rep_id)
        run_step([sys.executable, "main_MonteCarloZ.py"], env=env)

    for rep_id in range(n_rep):
        env = os.environ.copy()
        env["SLURM_ARRAY_TASK_ID"] = str(rep_id)
        env["OFFLINE_DATA_ID"] = str(rep_id)
        run_step([sys.executable, "main_est.py"], env=env)

    run_step([sys.executable, "mu_plot.py"])
    write_result(args.combo_id, combo_name, overrides, time.time() - start)


if __name__ == "__main__":
    main()
