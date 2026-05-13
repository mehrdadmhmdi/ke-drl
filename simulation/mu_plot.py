from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import yaml

from sim_eval import export_metrics_tables, plot_mu_summary


print("# ================================================================ #")
print("#   Aggregating simulation mean-embedding outputs                  #")
print("# ================================================================ #")

start = time.time()
job_id = os.environ.get("SLURM_JOB_ID")
array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
print(f"Slurm Job ID: {job_id}")
print(f"Slurm Array ID: {array_id}")

parser = argparse.ArgumentParser()
parser.add_argument("--mu-dir", default="./mu")
parser.add_argument("--metrics-dir", default="./metrics")
parser.add_argument("--plots-dir", default="./plots")
parser.add_argument("--params", default="params.yaml")
args = parser.parse_args()

with open(args.params, "r", encoding="utf-8") as f:
    P = yaml.safe_load(f)
print(
    "Aggregation params:",
    {
        "evaluation": P.get("evaluation"),
        "lambda_reg": P.get("lambda_reg"),
        "lambda_B": P.get("lambda_B", P.get("optimization", {}).get("lambda_B")),
        "kernel": P.get("kernel"),
        "target_set": P.get("target_set", P.get("x_star", {})),
    },
)

df = export_metrics_tables(mu_dir=args.mu_dir, metrics_dir=args.metrics_dir)
plot_mu_summary(mu_dir=args.mu_dir, outdir=args.plots_dir)
summary_path = Path(args.plots_dir) / "mu_summary_UG.png"
if not summary_path.exists():
    raise RuntimeError(
        f"Expected {summary_path} was not created. Check that matplotlib is installed in the cluster environment."
    )

print(f"Aggregated {len(df)} evaluation points")
print("ALL DONE!")
elapsed = time.time() - start
print(f"Plotting time: {int(elapsed // 60)} minutes and {int(elapsed % 60)} seconds")
print("=" * 70)
