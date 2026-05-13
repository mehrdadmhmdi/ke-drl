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
    out = pd.concat(rows, ignore_index=True).sort_values("score")
    out.to_csv("runs/tuning_summary.csv", index=False)
    cols = [
        "combo_id",
        "combo_name",
        "score",
        "RMSE_mean",
        "MAE_mean",
        "SupNorm_mean",
        "Bias_mean",
        "Corr_mean",
        "deming_slope",
        "run_dir",
    ]
    print(out[cols].to_string(index=False))
    print("Saved runs/tuning_summary.csv")


if __name__ == "__main__":
    main()
