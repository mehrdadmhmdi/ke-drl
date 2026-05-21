"""
Aggregate results from 100 replicates and create visualizations.

This script:
1. Loads results from all 100 replicates
2. Aggregates mu_truth and mu_pred for each MC target point
3. Computes statistics (mean, std) across replicates
4. Creates 10-subplot figure with truth vs prediction curves
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

print("# ================================================================ #")
print("#   Aggregate Multi-Target Results & Visualization                 #")
print("# ================================================================ #")


def aggregate_results(results_list: list) -> dict:
    """
    Aggregate results across replicates.

    Parameters
    ----------
    results_list : list
        Results from all 100 replicates

    Returns
    -------
    aggregated : dict
        Aggregated results with structure:
        {
            target_point_i: {
                'mu_truth_mean': np.array (n_grid,),
                'mu_truth_std': np.array (n_grid,),
                'mu_pred_mean': np.array (n_grid,),
                'mu_pred_std': np.array (n_grid,),
                'errors': list of floats (one per replicate),
                'error_mean': float,
                'error_std': float,
            },
            ...
        }
    """
    # Filter out errors
    valid_results = [r for r in results_list if "error" not in r]
    print(f"\nAggregating {len(valid_results)} valid replicates...")

    if not valid_results:
        raise ValueError("No valid results to aggregate!")

    # Number of MC target points
    n_targets = len(valid_results[0]["mu_truth"])
    n_grid = valid_results[0]["mu_truth"][0].shape[0]

    print(f"  MC target points: {n_targets}")
    print(f"  Grid points per target: {n_grid}")

    aggregated = {}

    for target_idx in range(n_targets):
        # Collect mu_truth and mu_pred across all replicates
        mu_truth_list = []
        mu_pred_list = []
        errors = []

        for result in valid_results:
            mu_truth = result["mu_truth"][target_idx].numpy()
            mu_pred = result["mu_pred"][target_idx].numpy()
            error = result["errors"][target_idx]

            mu_truth_list.append(mu_truth)
            mu_pred_list.append(mu_pred)
            errors.append(error)

        # Stack: shape (n_replicates, n_grid)
        mu_truth_stack = np.array(mu_truth_list)
        mu_pred_stack = np.array(mu_pred_list)

        # Compute statistics
        mu_truth_mean = mu_truth_stack.mean(axis=0)
        mu_truth_std = mu_truth_stack.std(axis=0)
        mu_pred_mean = mu_pred_stack.mean(axis=0)
        mu_pred_std = mu_pred_stack.std(axis=0)

        error_mean = float(np.mean(errors))
        error_std = float(np.std(errors))

        aggregated[target_idx] = {
            "mu_truth_mean": mu_truth_mean,
            "mu_truth_std": mu_truth_std,
            "mu_pred_mean": mu_pred_mean,
            "mu_pred_std": mu_pred_std,
            "errors": errors,
            "error_mean": error_mean,
            "error_std": error_std,
        }

        print(f"  Target {target_idx:2d}: error_mean = {error_mean:.6e} ± {error_std:.6e}")

    return aggregated


def create_multi_target_figure(
    aggregated: dict,
    n_cols: int = 5,
    figsize: tuple = (20, 8),
    output_path: Path | str = "./multi_target_comparison.png",
):
    """
    Create multi-subplot figure comparing truth vs prediction for each target.

    Parameters
    ----------
    aggregated : dict
        Aggregated results from aggregate_results()
    n_cols : int
        Number of columns in subplot grid
    figsize : tuple
        Figure size (width, height)
    output_path : Path or str
        Path to save figure
    """
    n_targets = len(aggregated)
    n_rows = (n_targets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=figsize,
        constrained_layout=True,
    )

    # Flatten axes for easier iteration
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.reshape(n_rows, -1)

    axes_flat = axes.flatten()

    # Color palette
    color_truth = "#FF8C00"  # Bold orange
    color_pred = "#0088FF"  # Blue

    # Plot each target
    for target_idx in range(n_targets):
        ax = axes_flat[target_idx]
        data = aggregated[target_idx]

        grid_idx = np.arange(data["mu_truth_mean"].shape[0])

        # Ground truth (orange, bold)
        ax.plot(
            grid_idx, data["mu_truth_mean"],
            color=color_truth,
            linewidth=2.5,
            label="Ground Truth (MC)",
            zorder=10,
        )
        ax.fill_between(
            grid_idx,
            data["mu_truth_mean"] - data["mu_truth_std"],
            data["mu_truth_mean"] + data["mu_truth_std"],
            color=color_truth,
            alpha=0.2,
            label="±1 std (Truth)",
        )

        # Prediction (blue)
        ax.plot(
            grid_idx, data["mu_pred_mean"],
            color=color_pred,
            linewidth=2,
            label="KE-DRL Estimate",
            zorder=9,
        )
        ax.fill_between(
            grid_idx,
            data["mu_pred_mean"] - data["mu_pred_std"],
            data["mu_pred_mean"] + data["mu_pred_std"],
            color=color_pred,
            alpha=0.2,
            label="±1 std (Pred)",
        )

        # Labels and title
        ax.set_xlabel("Evaluation Grid Index", fontsize=10)
        ax.set_ylabel("Kernel Mean Embedding", fontsize=10)
        error_mean = data["error_mean"]
        error_std = data["error_std"]
        ax.set_title(
            f"Target Point {target_idx} | MSE = {error_mean:.2e} ± {error_std:.2e}",
            fontsize=11,
            fontweight="bold",
        )

        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best", fontsize=8)

    # Hide unused subplots
    for idx in range(n_targets, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Multi-Target KE-DRL: Ground Truth vs Predictions (100 Replicates)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved figure to {output_path}")
    plt.close()


def create_error_summary_figure(
    aggregated: dict,
    output_path: Path | str = "./error_summary.png",
):
    """
    Create a summary figure of errors across all targets.

    Parameters
    ----------
    aggregated : dict
        Aggregated results
    output_path : Path or str
        Path to save figure
    """
    n_targets = len(aggregated)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # Extract error statistics
    error_means = [aggregated[i]["error_mean"] for i in range(n_targets)]
    error_stds = [aggregated[i]["error_std"] for i in range(n_targets)]
    target_indices = list(range(n_targets))

    # Bar plot of mean errors
    ax = axes[0]
    bars = ax.bar(target_indices, error_means, yerr=error_stds, capsize=5, color="#0088FF", alpha=0.7)
    ax.set_xlabel("Target Point Index", fontsize=11)
    ax.set_ylabel("Mean Squared Error", fontsize=11)
    ax.set_title("MSE per Target Point (±std across 100 replicates)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.set_yscale("log")

    # Box plot of error distributions
    ax = axes[1]
    error_distributions = [aggregated[i]["errors"] for i in range(n_targets)]
    bp = ax.boxplot(error_distributions, labels=target_indices, patch_artist=True)

    for patch in bp["boxes"]:
        patch.set_facecolor("#FF8C00")
        patch.set_alpha(0.7)

    ax.set_xlabel("Target Point Index", fontsize=11)
    ax.set_ylabel("Squared Error", fontsize=11)
    ax.set_title("Error Distribution per Target (100 replicates)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.set_yscale("log")

    fig.suptitle("Error Summary Across All Targets", fontsize=14, fontweight="bold", y=0.995)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved error summary to {output_path}")
    plt.close()


def create_metrics_table(aggregated: dict, output_path: Path | str = "./metrics_table.csv"):
    """
    Create a CSV table with error metrics for all targets.

    Parameters
    ----------
    aggregated : dict
        Aggregated results
    output_path : Path or str
        Path to save table
    """
    rows = []

    for target_idx in range(len(aggregated)):
        data = aggregated[target_idx]
        rows.append({
            "target_point": target_idx,
            "error_mean": data["error_mean"],
            "error_std": data["error_std"],
            "error_min": float(np.min(data["errors"])),
            "error_max": float(np.max(data["errors"])),
            "error_median": float(np.median(data["errors"])),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"✓ Saved metrics table to {output_path}")

    # Print summary
    print("\nMetrics Summary:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    # Load configuration
    with open("./params.yaml", "r") as f:
        params = yaml.safe_load(f)

    data_dir = Path(params.get("data_dir", "./data"))
    results_path = data_dir / "all_results.pt"

    if not results_path.exists():
        print(f"ERROR: Results file not found at {results_path}")
        print("Run parallel_runner_multi_target.py first")
        exit(1)

    # Load results
    print(f"\nLoading results from {results_path}...")
    results_list = torch.load(results_path)
    print(f"Loaded {len(results_list)} replicate results")

    # Aggregate
    aggregated = aggregate_results(results_list)

    # Create visualizations
    print("\nGenerating visualizations...")

    # Main figure: 10 subplots
    create_multi_target_figure(
        aggregated,
        n_cols=5,
        output_path=data_dir / "multi_target_comparison.png",
    )

    # Error summary figure
    create_error_summary_figure(
        aggregated,
        output_path=data_dir / "error_summary.png",
    )

    # Metrics table
    create_metrics_table(
        aggregated,
        output_path=data_dir / "metrics_table.csv",
    )

    print("\n" + "="*60)
    print("✓ Aggregation and visualization complete!")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  - {data_dir}/multi_target_comparison.png (10-subplot figure)")
    print(f"  - {data_dir}/error_summary.png (error analysis)")
    print(f"  - {data_dir}/metrics_table.csv (detailed metrics)")
