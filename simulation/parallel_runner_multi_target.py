"""
Parallel executor for multi-target KE-DRL simulation across 100 replicates.

This script:
1. Creates fixed target points (10 MC eval + 100 training)
2. Loads or generates 100 offline data replicates
3. Runs KE-DRL for each replicate in parallel
4. Aggregates results for visualization
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed

from sim_utils import (
    bootstrap_kedrl,
    resolve_compute_device,
    resolve_torch_dtype,
    seed_from_array,
)
from target_points_config import create_fixed_target_points, create_evaluation_grid
from main_est_multi_target import run_single_replicate, load_offline_data

bootstrap_kedrl()

print("# ================================================================ #")
print("#   Parallel Multi-Target KE-DRL (100 Replicates)                  #")
print("# ================================================================ #")


def setup_fixed_targets(s0: torch.Tensor, a0: torch.Tensor, params: dict) -> tuple:
    """
    Create and save fixed target points.

    Returns
    -------
    s_mc, a_mc, s_train, a_train : torch.Tensor
        Fixed target points for all replicates
    eval_grids : list[torch.Tensor]
        Evaluation grids for each MC target point
    """
    print("\n[Setup] Creating fixed target points...")

    s_mc, a_mc, s_train, a_train = create_fixed_target_points(
        s0, a0,
        n_mc_targets=int(params.get("n_mc_targets", 10)),
        n_train_targets=int(params.get("n_train_targets", 100)),
        mc_seed=int(params.get("mc_seed", 20260512)),
        train_seed=int(params.get("train_seed", 20260513)),
    )

    print(f"  MC target points:    {s_mc.shape[0]}")
    print(f"  Training targets:    {s_train.shape[0]}")

    # Create evaluation grids for each MC target
    # (in practice, these would come from MC return samples)
    # For now, use dummy grids - will be replaced with real ones during MC
    eval_grids = [
        create_evaluation_grid(
            torch.randn(1000, int(params.get("reward_dim", 1))),
            n_grid_points=int(params.get("n_eval_grid_points", 100)),
        )
        for _ in range(s_mc.shape[0])
    ]

    return s_mc, a_mc, s_train, a_train, eval_grids


def worker_task(
    replicate_id: int,
    data_path: Path,
    s_mc_targets: torch.Tensor,
    a_mc_targets: torch.Tensor,
    s_train_targets: torch.Tensor,
    a_train_targets: torch.Tensor,
    eval_grids: list,
    params: dict,
) -> dict:
    """
    Worker task for one replicate (runs in separate process).

    Parameters
    ----------
    replicate_id : int
        Replicate identifier [0, 1, ..., 99]
    data_path : Path
        Path to offline data file
    s_mc_targets, a_mc_targets, s_train_targets, a_train_targets : torch.Tensor
        Fixed target points
    eval_grids : list
        Evaluation grids
    params : dict
        Configuration

    Returns
    -------
    results : dict
        Results from run_single_replicate
    """
    # Load offline data
    dtype = resolve_torch_dtype(params.get("dtype", "float64"))
    device = resolve_compute_device(params.get("compute"))

    try:
        offline_data = load_offline_data(data_path, dtype, device)
    except Exception as e:
        print(f"ERROR: Failed to load data for replicate {replicate_id}: {e}")
        return {"replicate_id": replicate_id, "error": str(e)}

    # Run estimation
    try:
        results = run_single_replicate(
            replicate_id=replicate_id,
            offline_data=offline_data,
            s_mc_targets=s_mc_targets,
            a_mc_targets=a_mc_targets,
            s_train_targets=s_train_targets,
            a_train_targets=a_train_targets,
            eval_grids=eval_grids,
            params=params,
        )
        return results
    except Exception as e:
        print(f"ERROR: Estimation failed for replicate {replicate_id}: {e}")
        return {"replicate_id": replicate_id, "error": str(e)}


def run_parallel_estimation(
    params: dict,
    n_replicates: int = 100,
    n_workers: int = 8,
):
    """
    Run KE-DRL estimation in parallel across multiple replicates.

    Parameters
    ----------
    params : dict
        Configuration parameters
    n_replicates : int
        Number of offline data replicates (default: 100)
    n_workers : int
        Number of parallel workers (default: 8)
    """
    dtype = resolve_torch_dtype(params.get("dtype", "float64"))
    device = "cpu"  # Workers use CPU to avoid GPU contention

    # Setup
    print("\n[Setup] Loading/creating offline data...")

    data_dir = Path(params.get("data_dir", "./data"))
    data_dir.mkdir(exist_ok=True)

    # For first replicate: load or generate
    first_data_path = data_dir / "offline_data_0.pt"

    if first_data_path.exists():
        print(f"Loading offline data from {data_dir}")
        offline_data_0 = torch.load(first_data_path)
        s0 = offline_data_0["s0"]
        a0 = offline_data_0["a0"]
    else:
        print(f"ERROR: No offline data found at {first_data_path}")
        print(f"Run main_offlinedata.py first to generate offline data")
        return

    # Create fixed targets
    s0_torch = torch.as_tensor(s0, dtype=dtype)
    a0_torch = torch.as_tensor(a0, dtype=dtype)
    s_mc, a_mc, s_train, a_train, eval_grids = setup_fixed_targets(
        s0_torch, a0_torch, params
    )

    # Save target points for reference
    target_points_path = data_dir / "fixed_target_points.pt"
    torch.save(
        {
            "s_mc": s_mc,
            "a_mc": a_mc,
            "s_train": s_train,
            "a_train": a_train,
        },
        target_points_path,
    )
    print(f"Saved fixed target points to {target_points_path}")

    # Run parallel estimation
    print(f"\n[Execution] Running {n_replicates} replicates in parallel ({n_workers} workers)...")
    print(f"{'='*60}")

    results_list = []
    start_time = time.time()
    completed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}

        # Submit all tasks
        for replicate_id in range(n_replicates):
            data_path = data_dir / f"offline_data_{replicate_id}.pt"

            if not data_path.exists():
                print(f"Warning: Offline data not found for replicate {replicate_id}")
                continue

            future = executor.submit(
                worker_task,
                replicate_id=replicate_id,
                data_path=data_path,
                s_mc_targets=s_mc,
                a_mc_targets=a_mc,
                s_train_targets=s_train,
                a_train_targets=a_train,
                eval_grids=eval_grids,
                params=params,
            )
            futures[future] = replicate_id

        # Collect results as they complete
        for future in as_completed(futures):
            replicate_id = futures[future]
            try:
                result = future.result()
                results_list.append(result)
                completed += 1

                if "error" not in result:
                    print(f"[{completed}/{n_replicates}] Replicate {replicate_id}: ✓")
                else:
                    print(f"[{completed}/{n_replicates}] Replicate {replicate_id}: ✗ {result['error']}")
            except Exception as e:
                print(f"[{completed}/{n_replicates}] Replicate {replicate_id}: ✗ {e}")

    total_time = time.time() - start_time

    print(f"{'='*60}")
    print(f"Completed {completed}/{n_replicates} replicates in {total_time:.1f}s")

    # Save results
    results_path = data_dir / "all_results.pt"
    torch.save(results_list, results_path)
    print(f"Saved results to {results_path}")

    return results_list


if __name__ == "__main__":
    # Load configuration
    with open("./params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # Run parallel estimation
    results = run_parallel_estimation(
        params,
        n_replicates=int(params.get("n_replicates", 100)),
        n_workers=int(params.get("n_workers", 8)),
    )

    print("\nNext step: Run aggregate_results_multi_target.py to visualize results")
