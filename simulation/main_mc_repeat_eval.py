from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from sim_eval import mean_embedding_hat, mean_embedding_true, metrics_from_mu
from sim_utils import (
    kedrl_import_info,
    monte_carlo_Z,
    print_compute_device,
    resolve_compute_device,
    resolve_torch_dtype,
    seed_from_array,
)


print("# ================================================================ #")
print("#   Repeated Monte Carlo truth comparison for fitted embeddings     #")
print("# ================================================================ #")


@dataclass
class RunSpec:
    offline_data_id: int
    benchmark_id: int
    run_id: str
    point_source: str
    eval_grid: torch.Tensor
    grid_key: str
    mu_hat: np.ndarray


def as_int(x: Any) -> int:
    return int(float(x.item() if isinstance(x, torch.Tensor) else x))


def parse_int_list(raw: str | None) -> list[int] | None:
    if raw is None or str(raw).strip() == "":
        return None
    out: list[int] = []
    for piece in str(raw).split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            lo_s, hi_s = piece.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            step = 1 if hi >= lo else -1
            out.extend(list(range(lo, hi + step, step)))
        else:
            out.append(int(piece))
    return sorted(dict.fromkeys(out))


def sample_sd(x: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    arr = np.asarray(x, dtype=float)
    n = arr.size if axis is None else arr.shape[axis]
    if n <= 1:
        if axis is None:
            return 0.0
        shape = list(arr.shape)
        del shape[axis]
        return np.zeros(shape, dtype=float)
    return np.nanstd(arr, axis=axis, ddof=1)


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def tensor_digest(x: torch.Tensor) -> str:
    arr = x.detach().cpu().contiguous().numpy()
    h = hashlib.sha1()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(str(arr.shape).encode("ascii"))
    h.update(arr.tobytes())
    return h.hexdigest()[:16]


def run_id_for(offline_id: int, benchmark_id: int, n_benchmark: int) -> str:
    return f"{offline_id}_b{benchmark_id}" if n_benchmark > 1 else str(offline_id)


def load_benchmark_points(
    *,
    data_dir: Path,
    params: dict[str, Any],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    bench_cfg = dict(params.get("benchmark") or {})
    z_path = data_dir / str(bench_cfg.get("output", "Z_true.pt"))
    if z_path.exists():
        blob = torch_load(z_path)
        meta = blob.get("metadata", {})
        if "s_star" in meta and "a_star" in meta:
            s_star = torch.as_tensor(meta["s_star"], dtype=dtype)
            a_star = torch.as_tensor(meta["a_star"], dtype=dtype)
            if s_star.ndim == 1:
                s_star = s_star.reshape(1, -1)
            if a_star.ndim == 1:
                a_star = a_star.reshape(1, -1)
            point_sources = list(meta.get("point_sources") or [meta.get("point_source", "unknown")] * s_star.shape[0])
            return s_star, a_star, point_sources

    csv_path = data_dir / "benchmark_point.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        s_cols = sorted([c for c in df.columns if re.fullmatch(r"s\d+", c)], key=lambda c: int(c[1:]))
        a_cols = sorted([c for c in df.columns if re.fullmatch(r"a\d+", c)], key=lambda c: int(c[1:]))
        if s_cols and a_cols:
            s_star = torch.as_tensor(df[s_cols].to_numpy(dtype=float), dtype=dtype)
            a_star = torch.as_tensor(df[a_cols].to_numpy(dtype=float), dtype=dtype)
            point_sources = (
                df["point_source"].astype(str).tolist()
                if "point_source" in df.columns
                else ["benchmark_point_csv"] * len(df)
            )
            return s_star, a_star, point_sources

    if "s_star" not in bench_cfg or "a_star" not in bench_cfg:
        raise FileNotFoundError(
            f"Could not find benchmark points in {z_path}, {csv_path}, or params.yaml benchmark.s_star/a_star."
        )
    s_star = torch.as_tensor(bench_cfg["s_star"], dtype=dtype)
    a_star = torch.as_tensor(bench_cfg["a_star"], dtype=dtype)
    if s_star.ndim == 1:
        s_star = s_star.reshape(1, -1)
    if a_star.ndim == 1:
        a_star = a_star.reshape(1, -1)
    n_points = int(bench_cfg.get("num_points", s_star.shape[0]))
    s_star = s_star[:n_points]
    a_star = a_star[:n_points]
    return s_star, a_star, ["fixed_config"] * s_star.shape[0]


def discover_offline_ids(data_dir: Path, mu_dir: Path) -> list[int]:
    ids: set[int] = set()
    for path in data_dir.glob("fit_*.pt"):
        match = re.fullmatch(r"fit_(\d+)", path.stem)
        if match:
            ids.add(int(match.group(1)))
    for path in mu_dir.glob("mu_hat_*.csv"):
        stem = path.stem.replace("mu_hat_", "")
        match = re.match(r"^(\d+)(?:_b\d+)?$", stem)
        if match:
            ids.add(int(match.group(1)))
    if not ids:
        raise FileNotFoundError(
            f"No fitted replicate outputs found in {data_dir} or {mu_dir}. "
            "Run the simulation fit stage first."
        )
    return sorted(ids)


def load_eval_grid(data_dir: Path, run_id: str, offline_id: int, benchmark_id: int) -> torch.Tensor:
    candidates = [data_dir / f"Zeval_{run_id}.pt"]
    if benchmark_id == 0:
        candidates.append(data_dir / f"Zeval_{offline_id}.pt")
    for path in candidates:
        if path.exists():
            return torch.as_tensor(torch_load(path))
    raise FileNotFoundError(
        "Missing evaluation grid for repeated MC comparison. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def load_mu_hat(
    *,
    data_dir: Path,
    mu_dir: Path,
    run_id: str,
    offline_id: int,
    eval_grid: torch.Tensor,
    nu: float,
    length_scale: float,
    sigma: float,
) -> np.ndarray:
    mu_path = mu_dir / f"mu_hat_{run_id}.csv"
    if mu_path.exists():
        mu_hat = np.loadtxt(mu_path, delimiter=",").reshape(-1)
    else:
        weights_path = mu_dir / f"weights_{run_id}.csv"
        zgrid_path = data_dir / f"Zgrid_{offline_id}.pt"
        if not weights_path.exists() or not zgrid_path.exists():
            raise FileNotFoundError(
                f"Missing {mu_path}; fallback also needs {weights_path} and {zgrid_path}."
            )
        beta = torch.as_tensor(np.loadtxt(weights_path, delimiter=",").reshape(-1), dtype=eval_grid.dtype)
        z_grid = torch.as_tensor(torch_load(zgrid_path), dtype=eval_grid.dtype)
        mu_hat = (
            mean_embedding_hat(
                beta,
                z_grid,
                nu=nu,
                length_scale=length_scale,
                sigma=sigma,
                eval_grid=eval_grid,
            )
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
    if mu_hat.shape[0] != eval_grid.shape[0]:
        raise ValueError(
            f"{mu_path} has length {mu_hat.shape[0]}, but evaluation grid has {eval_grid.shape[0]} rows."
        )
    return mu_hat


def build_run_specs(
    *,
    data_dir: Path,
    mu_dir: Path,
    offline_ids: list[int],
    benchmark_ids: list[int],
    point_sources: list[str],
    n_benchmark: int,
    kernel_cfg: dict[str, Any],
    dtype: torch.dtype,
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    nu = float(kernel_cfg.get("nu", 5.5))
    length_scale = float(kernel_cfg.get("length_scale", 1.0))
    sigma = float(kernel_cfg.get("sigma", 1.0))
    for offline_id in offline_ids:
        for benchmark_id in benchmark_ids:
            if benchmark_id < 0 or benchmark_id >= n_benchmark:
                raise ValueError(f"benchmark_id={benchmark_id} outside 0,...,{n_benchmark - 1}.")
            rid = run_id_for(offline_id, benchmark_id, n_benchmark)
            eval_grid = load_eval_grid(data_dir, rid, offline_id, benchmark_id).to(dtype=dtype)
            mu_hat = load_mu_hat(
                data_dir=data_dir,
                mu_dir=mu_dir,
                run_id=rid,
                offline_id=offline_id,
                eval_grid=eval_grid,
                nu=nu,
                length_scale=length_scale,
                sigma=sigma,
            )
            source = point_sources[benchmark_id] if benchmark_id < len(point_sources) else "unknown"
            specs.append(
                RunSpec(
                    offline_data_id=offline_id,
                    benchmark_id=benchmark_id,
                    run_id=rid,
                    point_source=source,
                    eval_grid=eval_grid,
                    grid_key=tensor_digest(eval_grid),
                    mu_hat=mu_hat,
                )
            )
    if not specs:
        raise ValueError("No fitted run/benchmark combinations selected.")
    return specs


def generate_truth_mu_repeats(
    *,
    params: dict[str, Any],
    specs: list[RunSpec],
    s_star: torch.Tensor,
    a_star: torch.Tensor,
    mc_repeats: int,
    repeat_start: int,
    seed_offset: int,
    z_ids: int,
    z_timepoints: int,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    progress_every: int,
) -> tuple[dict[tuple[int, str], np.ndarray], list[int]]:
    reward_dim = as_int(params["reward_dim"])
    kernel_cfg = dict(params.get("kernel") or {})
    nu = float(kernel_cfg.get("nu", 5.5))
    length_scale = float(kernel_cfg.get("length_scale", 1.0))
    sigma_k = float(kernel_cfg.get("sigma", 1.0))

    to_t = lambda x: torch.as_tensor(x, dtype=dtype)
    W_s, b_s, sigma_s = map(to_t, (params["MDP"]["W_s"], params["MDP"]["b_s"], params["MDP"]["sigma_s"]))
    W_r, b_r, sigma_r = map(to_t, (params["MDP"]["W_r"], params["MDP"]["b_r"], params["MDP"]["sigma_r"]))

    target_policy_name = params["policy"]["evaluation_Target_policy"]
    target_policy = params["policy"][target_policy_name]["name"]
    target_policy_params = params["policy"][target_policy_name]

    grids_by_benchmark: dict[int, dict[str, torch.Tensor]] = {}
    for spec in specs:
        grids_by_benchmark.setdefault(spec.benchmark_id, {})
        grids_by_benchmark[spec.benchmark_id].setdefault(spec.grid_key, spec.eval_grid.to(dtype=dtype))

    truth_lists: dict[tuple[int, str], list[np.ndarray]] = {
        (benchmark_id, grid_key): []
        for benchmark_id, grids in grids_by_benchmark.items()
        for grid_key in grids
    }
    seeds: list[int] = []
    base_seed = int(params.get("random_seed", 20260512)) + int(seed_offset)
    start_time = time.time()
    for local_repeat in range(mc_repeats):
        mc_repeat = repeat_start + local_repeat
        seed = seed_from_array(base_seed, mc_repeat)
        seeds.append(seed)
        Z_true_list = monte_carlo_Z(
            z_ids,
            z_timepoints,
            float(params["gamma_val"]),
            s_star,
            a_star,
            reward_dim,
            target_policy,
            target_policy_params,
            W_s,
            b_s,
            sigma_s,
            W_r,
            b_r,
            sigma_r,
            plot=False,
            dtype=dtype,
            device=device,
        )
        for benchmark_id, grids in grids_by_benchmark.items():
            Z_true = Z_true_list[benchmark_id][:, :reward_dim].to(dtype=dtype)
            for grid_key, eval_grid in grids.items():
                mu_true = mean_embedding_true(
                    eval_grid,
                    Z_true,
                    nu=nu,
                    length_scale=length_scale,
                    sigma=sigma_k,
                    batch_size=batch_size,
                )
                truth_lists[(benchmark_id, grid_key)].append(mu_true.detach().cpu().numpy().reshape(-1))
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if progress_every > 0 and ((local_repeat + 1) % progress_every == 0 or local_repeat + 1 == mc_repeats):
            elapsed = time.time() - start_time
            print(
                f"Completed MC repeat {local_repeat + 1}/{mc_repeats} "
                f"(seed={seed}, elapsed={elapsed:.1f}s)",
                flush=True,
            )

    truth_mu = {key: np.vstack(values) for key, values in truth_lists.items()}
    return truth_mu, seeds


def summarize_repeated_differences(
    *,
    specs: list[RunSpec],
    truth_mu: dict[tuple[int, str], np.ndarray],
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repeat_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    pointwise_rows: list[dict[str, Any]] = []

    metric_cols = [
        "RMSE",
        "MAE",
        "SupNorm",
        "Bias",
        "Corr",
        "deming_slope",
        "deming_intercept",
        "diff_mean",
        "diff_sd",
        "diff_abs_mean",
        "diff_min",
        "diff_max",
    ]

    for spec in specs:
        mu_true_repeats = truth_mu[(spec.benchmark_id, spec.grid_key)]
        if mu_true_repeats.shape[1] != spec.mu_hat.shape[0]:
            raise ValueError(
                f"Truth matrix for run_id={spec.run_id} has width {mu_true_repeats.shape[1]}, "
                f"but mu_hat has length {spec.mu_hat.shape[0]}."
            )
        diffs = spec.mu_hat.reshape(1, -1) - mu_true_repeats
        run_metric_rows: list[dict[str, Any]] = []
        for local_repeat in range(mu_true_repeats.shape[0]):
            diff = diffs[local_repeat]
            metrics = metrics_from_mu(spec.mu_hat, mu_true_repeats[local_repeat])
            row = {
                "run_id": spec.run_id,
                "offline_data_id": spec.offline_data_id,
                "benchmark_id": spec.benchmark_id,
                "benchmark_point_source": spec.point_source,
                "mc_repeat": local_repeat,
                "mc_seed": seeds[local_repeat],
                **metrics,
                "diff_mean": float(np.nanmean(diff)),
                "diff_sd": float(sample_sd(diff)),
                "diff_abs_mean": float(np.nanmean(np.abs(diff))),
                "diff_min": float(np.nanmin(diff)),
                "diff_max": float(np.nanmax(diff)),
            }
            repeat_rows.append(row)
            run_metric_rows.append(row)

        run_df = pd.DataFrame(run_metric_rows)
        pointwise_mean = np.nanmean(diffs, axis=0)
        pointwise_sd = sample_sd(diffs, axis=0)
        flat_diff = diffs.reshape(-1)
        summary: dict[str, Any] = {
            "run_id": spec.run_id,
            "offline_data_id": spec.offline_data_id,
            "benchmark_id": spec.benchmark_id,
            "benchmark_point_source": spec.point_source,
            "mc_repeats": int(mu_true_repeats.shape[0]),
            "grid_points": int(spec.mu_hat.shape[0]),
            "grid_key": spec.grid_key,
            "diff_mean_all": float(np.nanmean(flat_diff)),
            "diff_sd_all": float(sample_sd(flat_diff)),
            "diff_abs_mean_all": float(np.nanmean(np.abs(flat_diff))),
            "diff_pointwise_mean_abs_mean": float(np.nanmean(np.abs(pointwise_mean))),
            "diff_pointwise_sd_mean": float(np.nanmean(pointwise_sd)),
            "diff_pointwise_sd_max": float(np.nanmax(pointwise_sd)),
            "mu_hat_mean": float(np.nanmean(spec.mu_hat)),
            "mu_hat_sd": float(sample_sd(spec.mu_hat)),
            "mu_true_mean_over_mc_grid": float(np.nanmean(mu_true_repeats)),
            "mu_true_sd_over_mc_grid": float(sample_sd(mu_true_repeats.reshape(-1))),
        }
        for col in metric_cols:
            values = run_df[col].to_numpy(dtype=float)
            summary[f"{col}_mean"] = float(np.nanmean(values))
            summary[f"{col}_sd"] = float(sample_sd(values))
        summary_rows.append(summary)

        eval_grid_np = spec.eval_grid.detach().cpu().numpy()
        mu_true_mean = np.nanmean(mu_true_repeats, axis=0)
        mu_true_sd = sample_sd(mu_true_repeats, axis=0)
        for grid_idx in range(spec.mu_hat.shape[0]):
            row = {
                "run_id": spec.run_id,
                "offline_data_id": spec.offline_data_id,
                "benchmark_id": spec.benchmark_id,
                "grid_idx": grid_idx,
                "mu_hat": float(spec.mu_hat[grid_idx]),
                "mu_true_mean": float(mu_true_mean[grid_idx]),
                "mu_true_sd": float(mu_true_sd[grid_idx]),
                "diff_mean": float(pointwise_mean[grid_idx]),
                "diff_sd": float(pointwise_sd[grid_idx]),
            }
            for dim in range(eval_grid_np.shape[1]):
                row[f"z{dim}"] = float(eval_grid_np[grid_idx, dim])
            pointwise_rows.append(row)

    return pd.DataFrame(repeat_rows), pd.DataFrame(summary_rows), pd.DataFrame(pointwise_rows)


def aggregate_summary(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    id_cols = {"run_id", "grid_key", "benchmark_point_source"}
    numeric_cols = [
        col
        for col in summary_df.columns
        if col not in id_cols and pd.api.types.is_numeric_dtype(summary_df[col])
    ]
    aggregate: dict[str, Any] = {"n_run_summaries": int(len(summary_df))}
    for col in numeric_cols:
        if col in {"offline_data_id", "benchmark_id"}:
            continue
        values = summary_df[col].to_numpy(dtype=float)
        aggregate[f"{col}_mean"] = float(np.nanmean(values))
        aggregate[f"{col}_sd"] = float(sample_sd(values))
    aggregate_df = pd.DataFrame([aggregate])

    grouped_rows: list[dict[str, Any]] = []
    for benchmark_id, group in summary_df.groupby("benchmark_id", dropna=False):
        row: dict[str, Any] = {"benchmark_id": benchmark_id, "n_run_summaries": int(len(group))}
        for col in numeric_cols:
            if col in {"offline_data_id", "benchmark_id"}:
                continue
            values = group[col].to_numpy(dtype=float)
            row[f"{col}_mean"] = float(np.nanmean(values))
            row[f"{col}_sd"] = float(sample_sd(values))
        grouped_rows.append(row)
    by_benchmark_df = pd.DataFrame(grouped_rows)
    return aggregate_df, by_benchmark_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate repeated Monte Carlo truth samples and compare fitted embeddings against each repeat."
    )
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--mu-dir", default="mu")
    parser.add_argument("--metrics-dir", default="metrics")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--offline-ids", default=os.environ.get("SIM2_MC_OFFLINE_IDS"))
    parser.add_argument("--benchmark-ids", default=os.environ.get("SIM2_MC_BENCHMARK_IDS"))
    parser.add_argument("--mc-repeats", type=int, default=int(os.environ.get("SIM2_MC_REPEATS", "100")))
    parser.add_argument("--repeat-start", type=int, default=int(os.environ.get("SIM2_MC_REPEAT_START", "0")))
    parser.add_argument("--seed-offset", type=int, default=int(os.environ.get("SIM2_MC_SEED_OFFSET", "300000")))
    parser.add_argument("--z-ids", type=int, default=None)
    parser.add_argument("--z-timepoints", type=int, default=None)
    parser.add_argument("--device", default=os.environ.get("KEDRL_DEVICE"))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("SIM2_MC_BATCH_SIZE", "2000")))
    parser.add_argument("--progress-every", type=int, default=int(os.environ.get("SIM2_MC_PROGRESS_EVERY", "5")))
    args = parser.parse_args()

    start = time.time()
    print(f"Slurm Job ID: {os.environ.get('SLURM_JOB_ID')}")
    print(f"Slurm Array ID: {os.environ.get('SLURM_ARRAY_TASK_ID')}")
    print(f"ke_drl import source: {kedrl_import_info()}")

    with open(args.params, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    if args.device:
        params = dict(params)
        compute_cfg = dict(params.get("compute") or {})
        compute_cfg["device"] = args.device
        if str(args.device).lower() == "cpu":
            compute_cfg["require_cuda"] = False
        params["compute"] = compute_cfg

    data_dir = Path(args.data_dir)
    mu_dir = Path(args.mu_dir)
    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.output_dir) if args.output_dir else metrics_dir / "mc_repeats"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    dtype = resolve_torch_dtype(params.get("dtype", "float64"))
    compute_device = resolve_compute_device(params.get("compute"), purpose="repeated Monte Carlo truth")
    print_compute_device(compute_device, prefix="Repeated MC")

    if args.mc_repeats < 1:
        raise ValueError("--mc-repeats must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    s_star, a_star, point_sources = load_benchmark_points(data_dir=data_dir, params=params, dtype=dtype)
    n_benchmark = int(s_star.shape[0])
    offline_ids = parse_int_list(args.offline_ids) or discover_offline_ids(data_dir, mu_dir)
    benchmark_ids = parse_int_list(args.benchmark_ids) or list(range(n_benchmark))
    kernel_cfg = dict(params.get("kernel") or {})
    specs = build_run_specs(
        data_dir=data_dir,
        mu_dir=mu_dir,
        offline_ids=offline_ids,
        benchmark_ids=benchmark_ids,
        point_sources=point_sources,
        n_benchmark=n_benchmark,
        kernel_cfg=kernel_cfg,
        dtype=dtype,
    )

    unique_grids = sorted({(spec.benchmark_id, spec.grid_key) for spec in specs})
    z_ids = args.z_ids if args.z_ids is not None else as_int(params["Z_sim"]["n_ids"])
    z_timepoints = args.z_timepoints if args.z_timepoints is not None else as_int(params["Z_sim"]["n_timepoints"])
    print(
        "Repeated MC setup:",
        {
            "offline_ids": offline_ids,
            "benchmark_ids": benchmark_ids,
            "run_specs": len(specs),
            "unique_benchmark_grids": len(unique_grids),
            "mc_repeats": args.mc_repeats,
            "repeat_start": args.repeat_start,
            "seed_offset": args.seed_offset,
            "z_ids": z_ids,
            "z_timepoints": z_timepoints,
            "gamma_val": params.get("gamma_val"),
            "kernel": kernel_cfg,
        },
    )

    truth_mu, seeds = generate_truth_mu_repeats(
        params=params,
        specs=specs,
        s_star=s_star,
        a_star=a_star,
        mc_repeats=args.mc_repeats,
        repeat_start=args.repeat_start,
        seed_offset=args.seed_offset,
        z_ids=z_ids,
        z_timepoints=z_timepoints,
        device=compute_device,
        dtype=dtype,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
    )

    repeat_df, summary_df, pointwise_df = summarize_repeated_differences(
        specs=specs,
        truth_mu=truth_mu,
        seeds=seeds,
    )
    aggregate_df, by_benchmark_df = aggregate_summary(summary_df)

    repeat_path = out_dir / "mc_repeat_metrics.csv"
    summary_path = out_dir / "mc_repeat_run_summary.csv"
    pointwise_path = out_dir / "mc_repeat_pointwise_diff_summary.csv"
    aggregate_path = metrics_dir / "mc_repeat_aggregate_metrics.csv"
    by_benchmark_path = metrics_dir / "mc_repeat_per_benchmark_aggregate_metrics.csv"
    repeat_df.to_csv(repeat_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    pointwise_df.to_csv(pointwise_path, index=False)
    aggregate_df.to_csv(aggregate_path, index=False)
    by_benchmark_df.to_csv(by_benchmark_path, index=False)

    manifest = {
        "params": str(Path(args.params).resolve()),
        "data_dir": str(data_dir.resolve()),
        "mu_dir": str(mu_dir.resolve()),
        "output_dir": str(out_dir.resolve()),
        "mc_repeats": args.mc_repeats,
        "repeat_start": args.repeat_start,
        "seed_offset": args.seed_offset,
        "seeds": seeds,
        "offline_ids": offline_ids,
        "benchmark_ids": benchmark_ids,
        "n_run_specs": len(specs),
        "n_unique_benchmark_grids": len(unique_grids),
        "z_ids": z_ids,
        "z_timepoints": z_timepoints,
        "device": str(compute_device),
        "dtype": str(dtype),
        "outputs": {
            "repeat_metrics": str(repeat_path),
            "run_summary": str(summary_path),
            "pointwise_diff_summary": str(pointwise_path),
            "aggregate_metrics": str(aggregate_path),
            "per_benchmark_aggregate_metrics": str(by_benchmark_path),
        },
    }
    with open(out_dir / "mc_repeat_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    elapsed = time.time() - start
    print("Repeated MC outputs:")
    for path in [repeat_path, summary_path, pointwise_path, aggregate_path, by_benchmark_path, out_dir / "mc_repeat_manifest.json"]:
        print(f"  {path.resolve()}")
    print(f"Repeated MC comparison finished in {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
