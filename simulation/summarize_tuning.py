from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    rows = []
    for path in sorted(Path("runs").glob("tune_global_*/metrics/tuning_result.csv")):
        df = pd.read_csv(path)
        df["run_dir"] = str(path.parents[1])
        rows.append(df)
    if not rows:
        raise FileNotFoundError("No tuning_result.csv files found under runs/tune_global_*/metrics/.")
    out = pd.concat(rows, ignore_index=True)
    if "score_true_z" not in out and "score" in out:
        out["score_true_z"] = out["score"]
    if "score_risk" in out and out["score_risk"].notna().any():
        out["true_z_rank"] = out["score_true_z"].rank(method="min", ascending=True)
        out["risk_rank"] = out["score_risk"].rank(method="min", ascending=True)
        out["combined_rank"] = out["true_z_rank"] + out["risk_rank"]
        out = out.sort_values(["combined_rank", "score_true_z", "score_risk"])
    else:
        out = out.sort_values("score_true_z")
    out.to_csv("runs/tuning_summary.csv", index=False)
    cols = [
        "combo_id",
        "combo_name",
        "score_true_z",
        "score_risk",
        "score_mass",
        "true_z_rank",
        "risk_rank",
        "combined_rank",
        "RMSE_mean",
        "MAE_mean",
        "SupNorm_mean",
        "Bias_mean",
        "Corr_mean",
        "deming_slope",
        "risk_log_obj_final_mean",
        "risk_bellman_final_mean",
        "target_mass_mean_mean",
        "target_mass_rmse_to_target_mean",
        "run_dir",
    ]
    cols = [c for c in cols if c in out.columns]
    print(out[cols].to_string(index=False))
    print("Saved runs/tuning_summary.csv")


if __name__ == "__main__":
    main()
