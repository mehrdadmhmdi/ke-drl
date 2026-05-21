from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import yaml

from sim_utils import kedrl_import_info
from sim_eval import export_metrics_tables, plot_all_replicate_mu_diagnostics, plot_mu_summary


print("# ================================================================ #")
print("#   Aggregating simulation mean-embedding outputs                  #")
print("# ================================================================ #")

start = time.time()
job_id = os.environ.get("SLURM_JOB_ID")
array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
print(f"Slurm Job ID: {job_id}")
print(f"Slurm Array ID: {array_id}")
print(f"ke_drl import source: {kedrl_import_info()}")

parser = argparse.ArgumentParser()
parser.add_argument("--mu-dir", default="./mu")
parser.add_argument("--metrics-dir", default="./metrics")
parser.add_argument("--plots-dir", default="./plots")
parser.add_argument("--params", default="params.yaml")
parser.add_argument("--skip-replicate-plots", action="store_true")
args = parser.parse_args()

with open(args.params, "r", encoding="utf-8") as f:
    P = yaml.safe_load(f)
print(
    "Aggregation params:",
    {
        "experiment": P.get("experiment"),
        "benchmark": P.get("benchmark"),
        "lambda_reg": P.get("lambda_reg"),
        "lambda_B": P.get("lambda_B", P.get("optimization", {}).get("lambda_B")),
        "kernel": P.get("kernel"),
        "target_set": P.get("target_set", {}),
    },
)

df = export_metrics_tables(mu_dir=args.mu_dir, metrics_dir=args.metrics_dir)
plot_mu_summary(mu_dir=args.mu_dir, metrics_dir=args.metrics_dir, outdir=args.plots_dir)
rep_plot_count = 0
if not args.skip_replicate_plots:
    rep_plot_count = plot_all_replicate_mu_diagnostics(mu_dir=args.mu_dir, outdir=args.plots_dir)
summary_path = Path(args.plots_dir) / "mu_summary_UG.png"
if not summary_path.exists():
    raise RuntimeError(
        f"Expected {summary_path} was not created. Check that matplotlib is installed in the cluster environment."
    )

if "benchmark_id" in df.columns:
    n_rep = df["offline_data_id"].nunique() if "offline_data_id" in df.columns else "unknown"
    n_bench = df["benchmark_id"].nunique()
    print(f"Aggregated {len(df)} benchmark-replicate curves ({n_rep} offline replicates x {n_bench} benchmarks)")
else:
    print(f"Aggregated {len(df)} offline replicates")
if not args.skip_replicate_plots:
    print(f"Created {rep_plot_count} per-replicate mean-vs-truth plots")
print("ALL DONE!")
elapsed = time.time() - start
print(f"Plotting time: {int(elapsed // 60)} minutes and {int(elapsed % 60)} seconds")
print("=" * 70)
