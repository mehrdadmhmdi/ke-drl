#!/usr/bin/env python3
import glob, pandas as pd, numpy as np, os
import sys

rows = []
for md in glob.glob("tunning/runs/*/metrics"):
    try:
        agg = pd.read_csv(f"{md}/aggregate_metrics.csv")
        cal = pd.read_csv(f"{md}/calibration_deming.csv")
        rows.append({
            "metrics_dir": md,
            "rmse": float(agg.RMSE_mean.iloc[0]),
            "supn": float(agg.SupNorm_mean.iloc[0]),
            "dslope": float(cal.deming_slope.iloc[0]),
            "dint": float(cal.deming_intercept.iloc[0])
        })
    except:
        continue

if not rows:
    print("No metrics found", file=sys.stderr)
    sys.exit(1)

df = pd.DataFrame(rows)
# for c in ["rmse", "supn", "dslope", "dint"]:
#     z = (df[c] - df[c].mean()) / (df[c].std(ddof=1) or 1)
#     df[f"{c}_z"] = z
#


# Select columns for CSV
output_df = df[["metrics_dir", "rmse", "supn", "dslope", "dint"]]

# Write to CSV
output_path = "tunning/TUNING_SUMMARY.csv"
output_df.to_csv(output_path, index=False)
print(f"Written results to {output_path}", file=sys.stderr)