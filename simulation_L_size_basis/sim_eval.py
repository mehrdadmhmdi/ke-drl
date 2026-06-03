from __future__ import annotations

import json
import math
import os
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import torch
import yaml

from sim_utils import bootstrap_kedrl


bootstrap_kedrl()

from ke_drl.matern_kernel import matern_kernel
from ke_drl.embedding_metrics import (
    embedding_explained_signal_from_true_samples,
    embedding_r2_from_true_samples,
    empirical_embedding_mmd2,
    normalized_bellman_error,
)


def _configure_times_fonts(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman",
                "Nimbus Roman No9 L",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def mean_embedding_true(
    Z_grid: torch.Tensor,
    Z_true: torch.Tensor,
    *,
    nu: float,
    length_scale: float,
    sigma: float,
    batch_size: int = 2000,
) -> torch.Tensor:
    """Compute m^{-1} sum_j k_Z(z, Z_j) without building a huge dense matrix."""
    Z_grid = torch.as_tensor(Z_grid)
    Z_true = torch.as_tensor(Z_true, dtype=Z_grid.dtype, device=Z_grid.device)
    out = torch.zeros(Z_grid.shape[0], dtype=Z_grid.dtype, device=Z_grid.device)
    for start in range(0, Z_true.shape[0], batch_size):
        chunk = Z_true[start : start + batch_size]
        out += matern_kernel(Z_grid, chunk, nu=nu, length_scale=length_scale, sigma=sigma).sum(dim=1)
    return out / float(Z_true.shape[0])


def mean_embedding_hat(
    beta: torch.Tensor,
    Z_grid: torch.Tensor,
    *,
    nu: float,
    length_scale: float,
    sigma: float,
    eval_grid: torch.Tensor | None = None,
) -> torch.Tensor:
    if eval_grid is None:
        eval_grid = Z_grid
    eval_grid = torch.as_tensor(eval_grid, dtype=Z_grid.dtype, device=Z_grid.device)
    Kzz = matern_kernel(eval_grid, Z_grid, nu=nu, length_scale=length_scale, sigma=sigma)
    return (Kzz @ beta.reshape(-1)).contiguous()


def fixed_point_embedding_risk(
    beta: torch.Tensor,
    Z_grid: torch.Tensor,
    Z_test: torch.Tensor,
    *,
    nu: float,
    length_scale: float,
    sigma: float,
    batch_size: int = 2000,
) -> torch.Tensor:
    """Compute the simulation-only oracle risk E||k(., Z)-mu_hat||^2.

    This includes the irreducible self-kernel term sigma^2, so it is not the
    zero-baseline projected Bellman diagnostic used for tuning.
    """
    beta = torch.as_tensor(beta, dtype=Z_grid.dtype, device=Z_grid.device).reshape(-1)
    Z_grid = torch.as_tensor(Z_grid, dtype=beta.dtype, device=beta.device)
    Z_test = torch.as_tensor(Z_test, dtype=beta.dtype, device=beta.device)
    K_grid = matern_kernel(Z_grid, Z_grid, nu=nu, length_scale=length_scale, sigma=sigma)
    quad = beta @ (K_grid @ beta)
    cross = torch.zeros((), dtype=beta.dtype, device=beta.device)
    for start in range(0, Z_test.shape[0], batch_size):
        chunk = Z_test[start : start + batch_size]
        cross = cross + (matern_kernel(chunk, Z_grid, nu=nu, length_scale=length_scale, sigma=sigma) @ beta).sum()
    return (sigma**2) - 2.0 * cross / float(Z_test.shape[0]) + quad


def common_eval_grid(Z_true: torch.Tensor, n_points: int) -> torch.Tensor:
    """Deterministic benchmark grid for one replicate's Monte Carlo truth sample."""
    Z_true = torch.as_tensor(Z_true)
    n = min(int(n_points), Z_true.shape[0])
    order = torch.argsort(Z_true[:, 0])
    idx = torch.linspace(0, Z_true.shape[0] - 1, n, device=Z_true.device).round().long()
    return Z_true[order[idx]].contiguous()


def _deming(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    xm, ym = x.mean(), y.mean()
    x0, y0 = x - xm, y - ym
    sxx = float(np.mean(x0 * x0))
    syy = float(np.mean(y0 * y0))
    sxy = float(np.mean(x0 * y0))
    if abs(sxy) < 1e-15:
        return float("nan"), float("nan")
    slope = (syy - sxx + math.sqrt((syy - sxx) ** 2 + 4.0 * sxy ** 2)) / (2.0 * sxy)
    intercept = ym - slope * xm
    return float(slope), float(intercept)


def metrics_from_mu(mu_hat: np.ndarray, mu_true: np.ndarray) -> dict[str, float]:
    mu_hat = np.asarray(mu_hat, dtype=float).reshape(-1)
    mu_true = np.asarray(mu_true, dtype=float).reshape(-1)
    diff = mu_hat - mu_true
    slope, intercept = _deming(mu_true, mu_hat)
    corr = float(np.corrcoef(mu_true, mu_hat)[0, 1]) if mu_true.size > 1 else float("nan")
    return {
        "RMSE": float(np.sqrt(np.mean(diff * diff))),
        "MAE": float(np.mean(np.abs(diff))),
        "SupNorm": float(np.max(np.abs(diff))),
        "Bias": float(np.mean(diff)),
        "Corr": corr,
        "deming_slope": slope,
        "deming_intercept": intercept,
    }


def save_mu_outputs(
    *,
    run_id: str | int,
    mu_hat: torch.Tensor,
    mu_true: torch.Tensor,
    beta: torch.Tensor,
    extra_metrics: dict | None = None,
    mu_dir: str | os.PathLike = "./mu",
    metrics_dir: str | os.PathLike = "./metrics",
) -> dict[str, float]:
    mu_dir = Path(mu_dir)
    metrics_dir = Path(metrics_dir)
    mu_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rid = str(run_id)
    mu_hat_np = mu_hat.detach().cpu().numpy().reshape(-1)
    mu_true_np = mu_true.detach().cpu().numpy().reshape(-1)
    beta_np = beta.detach().cpu().numpy().reshape(-1)

    np.savetxt(mu_dir / f"mu_hat_{rid}.csv", mu_hat_np, delimiter=",", fmt="%.8e")
    np.savetxt(mu_dir / f"mu_true_{rid}.csv", mu_true_np, delimiter=",", fmt="%.8e")
    np.savetxt(mu_dir / f"weights_{rid}.csv", beta_np, delimiter=",", fmt="%.8e")

    metrics = metrics_from_mu(mu_hat_np, mu_true_np)
    metrics.update(
        {
            "beta_sum": float(beta_np.sum()),
            "beta_l1": float(np.abs(beta_np).sum()),
            "beta_l2": float(np.sqrt(np.sum(beta_np * beta_np))),
            "beta_min": float(beta_np.min()),
            "beta_max": float(beta_np.max()),
            "beta_neg_frac": float(np.mean(beta_np < 0.0)),
        }
    )
    if extra_metrics:
        metrics.update(extra_metrics)
    row = {"run_id": rid, **metrics}
    pd.DataFrame([row]).to_csv(metrics_dir / f"run_metrics_{rid}.csv", index=False)
    return row


def export_metrics_tables(
    *,
    mu_dir: str | os.PathLike = "./mu",
    metrics_dir: str | os.PathLike = "./metrics",
    reporting_filter: dict | None = None,
) -> pd.DataFrame:
    mu_dir = Path(mu_dir)
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float]] = []
    for hat_path in sorted(mu_dir.glob("mu_hat_*.csv")):
        run_id = hat_path.stem.replace("mu_hat_", "")
        true_path = mu_dir / f"mu_true_{run_id}.csv"
        if not true_path.exists():
            continue
        mu_hat = np.loadtxt(hat_path, delimiter=",")
        mu_true = np.loadtxt(true_path, delimiter=",")
        row = {"run_id": run_id, **metrics_from_mu(mu_hat, mu_true)}
        run_metrics_path = metrics_dir / f"run_metrics_{run_id}.csv"
        if run_metrics_path.exists():
            saved_row = pd.read_csv(run_metrics_path).iloc[0].to_dict()
            row.update({k: v for k, v in saved_row.items() if k != "run_id"})
        weights_path = mu_dir / f"weights_{run_id}.csv"
        if weights_path.exists():
            beta = np.loadtxt(weights_path, delimiter=",").reshape(-1)
            row.update(
                {
                    "beta_sum": float(beta.sum()),
                    "beta_l1": float(np.abs(beta).sum()),
                    "beta_l2": float(np.sqrt(np.sum(beta * beta))),
                    "beta_min": float(beta.min()),
                    "beta_max": float(beta.max()),
                    "beta_neg_frac": float(np.mean(beta < 0.0)),
                }
            )
        rows.append(row)

    df_all = pd.DataFrame(rows)
    if df_all.empty:
        raise FileNotFoundError(f"No matched mu_hat_*.csv/mu_true_*.csv pairs in {mu_dir}")

    df = _apply_reporting_filter(df_all, metrics_dir=metrics_dir, reporting_filter=reporting_filter)

    df.to_csv(metrics_dir / "per_point_metrics.csv", index=False)
    df.to_csv(metrics_dir / "per_run_metrics.csv", index=False)  # backward-compatible name
    agg = {}
    for col in [
        "RMSE",
        "MAE",
        "SupNorm",
        "Bias",
        "Corr",
        "beta_sum",
        "beta_l1",
        "beta_l2",
        "beta_neg_frac",
        "risk_bellman_final",
        "risk_obj_final",
        "projected_bellman_test_risk",
        "oracle_embedding_risk",
        "benchmark_embedding_risk",
        "embedding_error_mmd2",
        "embedding_truth_signal",
        "relative_embedding_error",
        "explained_embedding_signal",
        "embedding_hat_norm2",
        "embedding_error_mmd2_total",
        "embedding_truth_signal_total",
        "relative_embedding_error_global",
        "explained_embedding_signal_global",
        "embedding_error_mmd2_mean",
        "embedding_truth_signal_mean",
        "relative_embedding_error_mean",
        "explained_embedding_signal_mean",
        "normalized_bellman_error",
        "bellman_fit",
        "bellman_residual",
        "normalized_bellman_error_global",
        "bellman_fit_global",
        "bellman_residual_total",
        "embedding_hat_norm2_total",
        "embedding_mmd2_to_true",
        "embedding_baseline_mmd2",
        "embedding_r2_pointwise",
        "embedding_r2_global",
        "embedding_mmd2_total",
        "embedding_baseline_mmd2_total",
    ]:
        if col not in df:
            continue
        agg[f"{col}_mean"] = float(df[col].mean())
        agg[f"{col}_sd"] = float(df[col].std(ddof=1)) if len(df) > 1 else 0.0
    pd.DataFrame([agg]).to_csv(metrics_dir / "aggregate_metrics.csv", index=False)
    if "benchmark_id" in df.columns:
        numeric_cols = [
            c for c in df.columns
            if c not in {"run_id", "benchmark_point_source", "benchmark_z_path"}
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        grouped_rows = []
        for bid, g in df.groupby("benchmark_id", dropna=False):
            row = {"benchmark_id": bid, "n_rows": int(len(g))}
            for col in numeric_cols:
                if col == "benchmark_id":
                    continue
                row[f"{col}_mean"] = float(g[col].mean())
                row[f"{col}_sd"] = float(g[col].std(ddof=1)) if len(g) > 1 else 0.0
            grouped_rows.append(row)
        pd.DataFrame(grouped_rows).to_csv(metrics_dir / "per_benchmark_aggregate_metrics.csv", index=False)

    cal = {
        "deming_slope": float(df["deming_slope"].mean()),
        "deming_intercept": float(df["deming_intercept"].mean()),
        "deming_slope_sd": float(df["deming_slope"].std(ddof=1)) if len(df) > 1 else 0.0,
        "deming_intercept_sd": float(df["deming_intercept"].std(ddof=1)) if len(df) > 1 else 0.0,
    }
    pd.DataFrame([cal]).to_csv(metrics_dir / "calibration_deming.csv", index=False)
    return df


def _reporting_filter_is_enabled(reporting_filter: dict | None) -> bool:
    if not reporting_filter:
        return False
    enabled = reporting_filter.get("enabled", True)
    if isinstance(enabled, str):
        return enabled.strip().lower() not in {"0", "false", "no", "off"}
    return bool(enabled)


def _comparison_mask(values: pd.Series, comparison: str, threshold: float) -> pd.Series:
    comparison = str(comparison or "lt").strip().lower()
    if comparison in {"lt", "<"}:
        return values < threshold
    if comparison in {"le", "<="}:
        return values <= threshold
    if comparison in {"gt", ">"}:
        return values > threshold
    if comparison in {"ge", ">="}:
        return values >= threshold
    raise ValueError(f"Unknown reporting_filter.comparison={comparison!r}. Use lt/le/gt/ge.")


def _metric_series(df: pd.DataFrame, metric: str) -> pd.Series:
    metric = str(metric)
    metric_l = metric.strip().lower()
    if metric in df.columns:
        return pd.to_numeric(df[metric], errors="coerce")
    if metric_l in {"abs_bias", "|bias|", "absolute_bias"} and "Bias" in df.columns:
        return pd.to_numeric(df["Bias"], errors="coerce").abs()
    raise KeyError(
        f"reporting_filter metric {metric!r} was requested, but it is not in the metrics table. "
        f"Available columns include: {', '.join(map(str, df.columns[:25]))}"
    )


def _filter_metric_thresholds(cfg: dict, default_metric: str, default_threshold: float) -> dict[str, float]:
    raw = cfg.get("metrics") or cfg.get("diagnostic_metrics")
    if raw is None:
        return {default_metric: default_threshold}
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items()}
    if isinstance(raw, (list, tuple)):
        return {str(k): default_threshold for k in raw}
    raise TypeError("reporting_filter.metrics must be a mapping or a list of metric names.")


def _aggregate_value(values: pd.Series, how: str) -> float:
    values = pd.to_numeric(values, errors="coerce")
    how = str(how or "median").strip().lower()
    if not values.notna().any():
        return float("nan")
    if how == "mean":
        return float(values.mean())
    if how == "max":
        return float(values.max())
    if how in {"q75", "p75"}:
        return float(values.quantile(0.75))
    if how in {"q90", "p90"}:
        return float(values.quantile(0.90))
    if how in {"q95", "p95"}:
        return float(values.quantile(0.95))
    return float(values.median())


def _apply_top_k_target_filter(
    df_all: pd.DataFrame,
    *,
    metrics_dir: Path,
    cfg: dict,
    metric: str,
    threshold: float,
    comparison: str,
) -> pd.DataFrame:
    if "benchmark_id" not in df_all.columns:
        raise KeyError("reporting_filter mode=top_k_targets requires a benchmark_id column.")

    top_k = int(cfg.get("top_k", cfg.get("n_targets", 10)))
    if top_k < 1:
        raise ValueError("reporting_filter.top_k must be at least 1.")
    aggregate = str(cfg.get("aggregate", "median"))
    rank_metric = str(cfg.get("rank_metric", metric))
    metric_thresholds = _filter_metric_thresholds(cfg, metric, threshold)

    rows = []
    for bid, group in df_all.groupby("benchmark_id", dropna=False):
        row: dict[str, float | int | bool] = {"benchmark_id": bid, "total_rows": int(len(group))}
        passed = True
        for name, cutoff in metric_thresholds.items():
            values = _metric_series(group, name)
            agg_value = _aggregate_value(values, aggregate)
            mask = _comparison_mask(pd.Series([agg_value]), comparison, cutoff)
            ok = bool(mask.iloc[0]) and math.isfinite(agg_value)
            row[f"{name}_{aggregate}"] = agg_value
            row[f"{name}_threshold"] = cutoff
            row[f"{name}_passes"] = ok
            passed = passed and ok
        row["passes_all_metrics"] = passed
        row["rank_value"] = _aggregate_value(_metric_series(group, rank_metric), aggregate)
        rows.append(row)

    by_benchmark = pd.DataFrame(rows)
    candidates = by_benchmark[
        by_benchmark["passes_all_metrics"].astype(bool)
        & pd.to_numeric(by_benchmark["rank_value"], errors="coerce").notna()
    ].copy()
    candidates = candidates.sort_values(["rank_value", "benchmark_id"], ascending=[True, True])
    selected_benchmarks = candidates.head(top_k)["benchmark_id"].tolist()

    allow_fewer = bool(cfg.get("allow_fewer", True))
    if len(selected_benchmarks) < top_k and not allow_fewer:
        raise ValueError(
            f"Diagnostic target filter found only {len(selected_benchmarks)} targets passing all thresholds; "
            f"top_k={top_k}. Loosen thresholds or set reporting_filter.allow_fewer: true."
        )
    if not selected_benchmarks:
        raise ValueError(
            f"Diagnostic target filter kept 0 target points. No benchmark had "
            f"{aggregate} {metric_thresholds} with comparison {comparison}. "
            "Loosen reporting_filter thresholds or inspect per_run_metrics_all.csv."
        )

    by_benchmark["selected"] = by_benchmark["benchmark_id"].isin(selected_benchmarks)
    selected_rank = {bid: rank + 1 for rank, bid in enumerate(selected_benchmarks)}
    by_benchmark["selected_rank"] = by_benchmark["benchmark_id"].map(selected_rank)
    by_benchmark.to_csv(metrics_dir / "diagnostic_filter_by_benchmark.csv", index=False)

    df = df_all[df_all["benchmark_id"].isin(selected_benchmarks)].copy()
    total_reps = df_all["offline_data_id"].nunique() if "offline_data_id" in df_all.columns else np.nan
    selected_reps = df["offline_data_id"].nunique() if "offline_data_id" in df.columns else np.nan
    summary = {
        "enabled": True,
        "mode": "top_k_targets",
        "rank_metric": rank_metric,
        "aggregate": aggregate,
        "comparison": comparison,
        "top_k": top_k,
        "total_rows": int(len(df_all)),
        "selected_rows": int(len(df)),
        "dropped_rows": int(len(df_all) - len(df)),
        "selected_fraction": float(len(df) / len(df_all)),
        "total_offline_replicates": total_reps,
        "selected_offline_replicates": selected_reps,
        "total_benchmarks": int(df_all["benchmark_id"].nunique()),
        "selected_benchmarks": int(len(selected_benchmarks)),
        "selected_benchmark_ids": ",".join(map(str, selected_benchmarks)),
        "metric_thresholds_json": json.dumps(metric_thresholds, sort_keys=True),
    }
    pd.DataFrame([summary]).to_csv(metrics_dir / "diagnostic_filter_summary.csv", index=False)

    keep_cols = [
        c
        for c in ["run_id", "offline_data_id", "benchmark_id", rank_metric, "RMSE", "MAE", "Bias"]
        if c in df.columns
    ]
    df[keep_cols].to_csv(metrics_dir / "diagnostic_filter_selected_rows.csv", index=False)
    print(
        "Diagnostic target filter selected {} targets ({}) from {} candidates using {} {} {} and top_k={}.".format(
            len(selected_benchmarks),
            ", ".join(map(str, selected_benchmarks)),
            int(df_all["benchmark_id"].nunique()),
            aggregate,
            rank_metric,
            comparison,
            top_k,
        ),
        flush=True,
    )
    return df


def _apply_reporting_filter(
    df_all: pd.DataFrame,
    *,
    metrics_dir: Path,
    reporting_filter: dict | None,
) -> pd.DataFrame:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    stale_paths = [
        metrics_dir / "diagnostic_filter_summary.csv",
        metrics_dir / "diagnostic_filter_by_benchmark.csv",
        metrics_dir / "diagnostic_filter_selected_rows.csv",
        metrics_dir / "per_point_metrics_all.csv",
        metrics_dir / "per_run_metrics_all.csv",
    ]
    if not _reporting_filter_is_enabled(reporting_filter):
        for path in stale_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        return df_all

    cfg = dict(reporting_filter or {})
    metric = str(cfg.get("metric", "projected_bellman_test_risk"))
    threshold = float(cfg.get("threshold", 0.15))
    comparison = str(cfg.get("comparison", "lt"))
    _ = _metric_series(df_all, metric)

    if bool(cfg.get("save_unfiltered_copy", True)):
        df_all.to_csv(metrics_dir / "per_point_metrics_all.csv", index=False)
        df_all.to_csv(metrics_dir / "per_run_metrics_all.csv", index=False)

    mode = str(cfg.get("mode", "rows")).strip().lower()
    if mode in {"top_k_targets", "top_k_benchmarks", "targets", "benchmarks"}:
        return _apply_top_k_target_filter(
            df_all,
            metrics_dir=metrics_dir,
            cfg=cfg,
            metric=metric,
            threshold=threshold,
            comparison=comparison,
        )

    values = _metric_series(df_all, metric)
    mask = _comparison_mask(values, comparison, threshold) & values.notna()
    df = df_all.loc[mask].copy()
    if df.empty:
        raise ValueError(
            f"Diagnostic reporting filter kept 0/{len(df_all)} rows: {metric} {comparison} {threshold}. "
            "Loosen the threshold or inspect per_run_metrics_all.csv."
        )

    selected_values = _metric_series(df, metric)
    total_reps = df_all["offline_data_id"].nunique() if "offline_data_id" in df_all.columns else np.nan
    selected_reps = df["offline_data_id"].nunique() if "offline_data_id" in df.columns else np.nan
    total_benchmarks = df_all["benchmark_id"].nunique() if "benchmark_id" in df_all.columns else np.nan
    selected_benchmarks = df["benchmark_id"].nunique() if "benchmark_id" in df.columns else np.nan
    summary = {
        "enabled": True,
        "metric": metric,
        "comparison": comparison,
        "threshold": threshold,
        "total_rows": int(len(df_all)),
        "selected_rows": int(len(df)),
        "dropped_rows": int(len(df_all) - len(df)),
        "selected_fraction": float(len(df) / len(df_all)),
        "total_offline_replicates": total_reps,
        "selected_offline_replicates": selected_reps,
        "total_benchmarks": total_benchmarks,
        "selected_benchmarks": selected_benchmarks,
        "selected_metric_min": float(selected_values.min()),
        "selected_metric_median": float(selected_values.median()),
        "selected_metric_max": float(selected_values.max()),
    }
    pd.DataFrame([summary]).to_csv(metrics_dir / "diagnostic_filter_summary.csv", index=False)

    if "benchmark_id" in df_all.columns:
        rows = []
        for bid, group in df_all.groupby("benchmark_id", dropna=False):
            group_values = _metric_series(group, metric)
            group_mask = _comparison_mask(group_values, comparison, threshold) & group_values.notna()
            kept = group.loc[group_mask]
            rows.append(
                {
                    "benchmark_id": bid,
                    "total_rows": int(len(group)),
                    "selected_rows": int(len(kept)),
                    "selected_fraction": float(len(kept) / len(group)) if len(group) else np.nan,
                    f"{metric}_median_all": float(group_values.median()) if group_values.notna().any() else np.nan,
                    f"{metric}_median_selected": (
                        float(_metric_series(kept, metric).median()) if len(kept) else np.nan
                    ),
                }
            )
        pd.DataFrame(rows).to_csv(metrics_dir / "diagnostic_filter_by_benchmark.csv", index=False)

    keep_cols = [c for c in ["run_id", "offline_data_id", "benchmark_id", metric, "RMSE", "MAE", "Bias"] if c in df.columns]
    df[keep_cols].to_csv(metrics_dir / "diagnostic_filter_selected_rows.csv", index=False)
    print(
        f"Diagnostic reporting filter retained {len(df)}/{len(df_all)} rows "
        f"with {metric} {comparison} {threshold}.",
        flush=True,
    )
    return df


def plot_mu_summary(
    *,
    mu_dir: str | os.PathLike = "./mu",
    metrics_dir: str | os.PathLike = "./metrics",
    outdir: str | os.PathLike = "./plots",
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        _configure_times_fonts(plt)
    except ModuleNotFoundError as exc:
        print(f"Plotting skipped because matplotlib is unavailable: {exc}")
        return

    mu_dir = Path(mu_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    metrics_df = None
    per_run_path = Path(metrics_dir) / "per_run_metrics.csv"
    allowed_run_ids: set[str] | None = None
    if per_run_path.exists():
        metrics_df = pd.read_csv(per_run_path)
        if "run_id" in metrics_df.columns:
            allowed_run_ids = set(metrics_df["run_id"].astype(str))

    hats, trues, run_ids = [], [], []
    for hat_path in sorted(mu_dir.glob("mu_hat_*.csv")):
        run_id = hat_path.stem.replace("mu_hat_", "")
        if allowed_run_ids is not None and str(run_id) not in allowed_run_ids:
            continue
        true_path = mu_dir / f"mu_true_{run_id}.csv"
        if true_path.exists():
            hats.append(np.loadtxt(hat_path, delimiter=",").reshape(-1))
            trues.append(np.loadtxt(true_path, delimiter=",").reshape(-1))
            run_ids.append(run_id)
    if not hats:
        raise FileNotFoundError(f"No matched mu outputs in {mu_dir}")

    H = np.vstack(hats)
    T = np.vstack(trues)
    x = np.arange(H.shape[1])
    h_mean, h_sd = H.mean(axis=0), H.std(axis=0, ddof=1) if H.shape[0] > 1 else np.zeros(H.shape[1])
    t_mean, t_sd = T.mean(axis=0), T.std(axis=0, ddof=1) if T.shape[0] > 1 else np.zeros(T.shape[1])

    multi_benchmark = (
        metrics_df is not None
        and "benchmark_id" in metrics_df.columns
        and metrics_df["benchmark_id"].nunique(dropna=True) > 1
    )
    if multi_benchmark:
        caption = _build_summary_caption(
            mu_dir=mu_dir,
            metrics_dir=Path(metrics_dir),
            outdir=outdir,
            metrics_df=metrics_df,
        )
        _plot_multi_benchmark_summary(
            H,
            T,
            run_ids=run_ids,
            metrics_df=metrics_df,
            outdir=outdir,
            plt=plt,
            caption=caption,
        )
    else:
        _plot_four_panel_summary(H, T, run_ids=run_ids, metrics_df=metrics_df, outdir=outdir, plt=plt)

    if metrics_df is not None:
        plot_embedding_quality_diagnostic(metrics_df=metrics_df, outdir=outdir, plt=plt)

    if metrics_df is not None and "benchmark_id" in metrics_df.columns:
        rid_to_benchmark = dict(zip(metrics_df["run_id"].astype(str), metrics_df["benchmark_id"]))
        benchmark_ids = sorted(pd.Series(list(rid_to_benchmark.values())).dropna().unique().tolist())
        if len(benchmark_ids) > 1:
            for bid in benchmark_ids:
                idx = [i for i, rid in enumerate(run_ids) if rid_to_benchmark.get(str(rid)) == bid]
                if not idx:
                    continue
                group_dir = outdir / f"benchmark_{int(bid)}"
                group_dir.mkdir(parents=True, exist_ok=True)
                group_metrics = metrics_df[metrics_df["benchmark_id"] == bid].copy()
                _plot_four_panel_summary(
                    H[np.asarray(idx)],
                    T[np.asarray(idx)],
                    run_ids=[run_ids[i] for i in idx],
                    metrics_df=group_metrics,
                    outdir=group_dir,
                    plt=plt,
                )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, t_mean, lw=1.7, color="black", label="MC truth")
    ax.fill_between(x, t_mean - 1.96 * t_sd, t_mean + 1.96 * t_sd, color="black", alpha=0.08)
    ax.plot(x, h_mean, lw=1.7, color="#1f77b4", label="KE-DRL")
    ax.fill_between(x, h_mean - 1.96 * h_sd, h_mean + 1.96 * h_sd, color="#1f77b4", alpha=0.16)
    ax.set_xlabel("Index on evaluation-target Z grid")
    ax.set_ylabel("Mean embedding")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "mu_summary.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(T.reshape(-1), H.reshape(-1), s=6, alpha=0.25)
    _set_calibration_axes(
        ax,
        [T.reshape(-1)],
        [H.reshape(-1)],
        q_low=2.5,
        q_high=97.5,
        include_zero=False,
        ideal_color="black",
        ideal_lw=1.0,
        ideal_ls="-",
        ideal_label=None,
    )
    ax.text(0.96, 0.06, "axes use separate central 95%", transform=ax.transAxes, ha="right", fontsize=8)
    ax.set_xlabel("MC truth mean embedding")
    ax.set_ylabel("Estimated mean embedding")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "mu_calibration.png", dpi=300)
    plt.close(fig)


def _binned_means(x: np.ndarray, y: np.ndarray, n_bins: int = 10):
    order = np.argsort(x)
    xs = np.asarray(x, dtype=float).reshape(-1)[order]
    ys = np.asarray(y, dtype=float).reshape(-1)[order]
    bins = np.array_split(np.arange(xs.size), min(n_bins, xs.size))
    bx, by, se = [], [], []
    for b in bins:
        if b.size == 0:
            continue
        vals = ys[b]
        bx.append(float(xs[b].mean()))
        by.append(float(vals.mean()))
        se.append(float(vals.std(ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0)
    return np.asarray(bx), np.asarray(by), np.asarray(se)


def _finite_flatten(*arrays: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    vals: list[np.ndarray] = []
    for arr in arrays:
        if arr is None:
            continue
        x = np.asarray(arr, dtype=float).reshape(-1)
        x = x[np.isfinite(x)]
        if x.size:
            vals.append(x)
    return np.concatenate(vals) if vals else np.asarray([], dtype=float)


def _robust_limits(
    *arrays: np.ndarray | list[float] | tuple[float, ...],
    q_low: float = 1.0,
    q_high: float = 99.0,
    pad: float = 0.08,
    min_span: float = 1e-3,
    include_zero: bool = False,
) -> tuple[float, float]:
    vals = _finite_flatten(*arrays)
    if vals.size == 0:
        return -1.0, 1.0
    if vals.size >= 5:
        lo, hi = np.nanpercentile(vals, [q_low, q_high])
    else:
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    lo, hi = float(lo), float(hi)
    if include_zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return -1.0, 1.0
    if hi < lo:
        lo, hi = hi, lo
    if hi - lo < min_span:
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 0.5 * min_span, mid + 0.5 * min_span
    span = hi - lo
    return lo - pad * span, hi + pad * span


def _set_calibration_axes(
    ax,
    x_arrays: list[np.ndarray],
    y_arrays: list[np.ndarray],
    *,
    q_low: float = 0.0,
    q_high: float = 100.0,
    pad: float = 0.10,
    include_zero: bool = False,
    min_span: float = 1e-4,
    ideal_color: str = "0.25",
    ideal_lw: float = 1.2,
    ideal_ls: str = "--",
    ideal_label: str | None = "ideal",
) -> tuple[float, float, float, float]:
    """Set calibration axes independently while keeping the y=x reference visible."""
    xlo, xhi = _robust_limits(
        *x_arrays,
        q_low=q_low,
        q_high=q_high,
        pad=pad,
        include_zero=include_zero,
        min_span=min_span,
    )
    ideal_y = np.asarray([xlo, xhi], dtype=float)
    ylo, yhi = _robust_limits(
        *y_arrays,
        ideal_y,
        q_low=q_low,
        q_high=q_high,
        pad=pad,
        include_zero=include_zero,
        min_span=min_span,
    )
    ax.plot(
        [xlo, xhi],
        [xlo, xhi],
        ideal_ls,
        color=ideal_color,
        lw=ideal_lw,
        label=ideal_label,
    )
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    return xlo, xhi, ylo, yhi


def plot_single_mu_diagnostic(
    *,
    mu_hat,
    mu_true,
    outdir: str | os.PathLike,
    run_id: str | int,
    plt=None,
    filename: str = "mu_hat_vs_truth.png",
) -> None:
    """Plot one replicate's estimated evaluation-target embedding against MC truth."""
    close_plt = False
    if plt is None:
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt  # type: ignore[no-redef]
            _configure_times_fonts(plt)
            close_plt = True
        except ModuleNotFoundError as exc:
            print(f"Replicate mu diagnostic skipped because matplotlib is unavailable: {exc}")
            return

    hat = mu_hat.detach().cpu().numpy() if hasattr(mu_hat, "detach") else np.asarray(mu_hat)
    true = mu_true.detach().cpu().numpy() if hasattr(mu_true, "detach") else np.asarray(mu_true)
    hat = np.asarray(hat, dtype=float).reshape(-1)
    true = np.asarray(true, dtype=float).reshape(-1)
    if hat.shape != true.shape:
        raise ValueError(f"mu_hat and mu_true must have the same shape; got {hat.shape} and {true.shape}.")

    diff = hat - true
    rmse = float(np.sqrt(np.mean(diff * diff)))
    mae = float(np.mean(np.abs(diff)))
    corr = float(np.corrcoef(true, hat)[0, 1]) if true.size > 1 and np.std(true) > 0 and np.std(hat) > 0 else float("nan")

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    x = np.arange(hat.size)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axs[0]
    ax.plot(x, true, color="#0b2a50", lw=1.7, label="MC truth")
    ax.plot(x, hat, color="#ff5f05", lw=1.5, label="KE-DRL")
    lo, hi = _robust_limits(true, hat, q_low=0.5, q_high=99.5, include_zero=True)
    ax.set_ylim(lo, hi)
    ax.set_title(f"Replicate {run_id}: embedding curve")
    ax.set_xlabel("Index on evaluation-target Z grid")
    ax.set_ylabel("Mean embedding")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axs[1]
    ax.scatter(true, hat, s=12, alpha=0.5, color="#ff5f05", edgecolor="none")
    _set_calibration_axes(
        ax,
        [true],
        [hat],
        q_low=0.5,
        q_high=99.5,
        include_zero=True,
        min_span=1e-3,
        ideal_color="#0b2a50",
        ideal_lw=1.2,
        ideal_label=None,
    )
    ax.set_title(f"RMSE={rmse:.3g}, MAE={mae:.3g}, Corr={corr:.3g}")
    ax.set_xlabel("MC truth mean embedding")
    ax.set_ylabel("Estimated mean embedding")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path / filename, dpi=300)
    plt.close(fig)
    if close_plt:
        plt.close("all")


def plot_all_replicate_mu_diagnostics(
    *,
    mu_dir: str | os.PathLike = "./mu",
    outdir: str | os.PathLike = "./plots",
    metrics_dir: str | os.PathLike | None = "./metrics",
) -> int:
    """Regenerate per-replicate mean-vs-truth plots from saved mu CSV files."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        _configure_times_fonts(plt)
    except ModuleNotFoundError as exc:
        print(f"Replicate mu diagnostics skipped because matplotlib is unavailable: {exc}")
        return 0

    mu_dir = Path(mu_dir)
    outdir = Path(outdir)
    allowed_run_ids: set[str] | None = None
    if metrics_dir is not None:
        per_run_path = Path(metrics_dir) / "per_run_metrics.csv"
        if per_run_path.exists():
            metrics_df = pd.read_csv(per_run_path)
            if "run_id" in metrics_df.columns:
                allowed_run_ids = set(metrics_df["run_id"].astype(str))
    count = 0
    for hat_path in sorted(mu_dir.glob("mu_hat_*.csv")):
        run_id = hat_path.stem.replace("mu_hat_", "")
        if allowed_run_ids is not None and str(run_id) not in allowed_run_ids:
            continue
        true_path = mu_dir / f"mu_true_{run_id}.csv"
        if not true_path.exists():
            continue
        plot_single_mu_diagnostic(
            mu_hat=np.loadtxt(hat_path, delimiter=",").reshape(-1),
            mu_true=np.loadtxt(true_path, delimiter=",").reshape(-1),
            outdir=_replicate_plot_dir(outdir, run_id),
            run_id=run_id,
            plt=plt,
        )
        count += 1
    return count


def plot_embedding_quality_diagnostic(
    *,
    metrics_df: pd.DataFrame,
    outdir: str | os.PathLike,
    plt=None,
    filename: str = "embedding_quality_summary.png",
) -> bool:
    """Plot signal-normalized embedding quality diagnostics when available."""
    if metrics_df is None:
        return False

    close_plt = False
    if plt is None:
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt  # type: ignore[no-redef]
            _configure_times_fonts(plt)
            close_plt = True
        except ModuleNotFoundError as exc:
            print(f"Embedding quality diagnostic skipped because matplotlib is unavailable: {exc}")
            return False

    df = metrics_df.copy()
    numeric_cols = [
        "embedding_error_mmd2",
        "embedding_truth_signal",
        "relative_embedding_error",
        "explained_embedding_signal",
        "explained_embedding_signal_global",
        "embedding_hat_norm2",
        "bellman_residual",
        "normalized_bellman_error",
        "normalized_bellman_error_global",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    sim_cols = {"embedding_error_mmd2", "embedding_truth_signal", "explained_embedding_signal"}
    real_cols = {"embedding_hat_norm2", "bellman_residual", "normalized_bellman_error"}
    has_sim = sim_cols.issubset(df.columns) and not df[list(sim_cols)].replace([np.inf, -np.inf], np.nan).dropna(how="all").empty
    has_real = real_cols.issubset(df.columns) and not df[list(real_cols)].replace([np.inf, -np.inf], np.nan).dropna(how="all").empty
    if not has_sim and not has_real:
        return False
    mode = "simulation" if has_sim else "real"

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.6))

    ax = axs[0]
    global_col = "explained_embedding_signal_global" if mode == "simulation" else "normalized_bellman_error_global"
    if global_col in df.columns and df[global_col].notna().any():
        if "offline_data_id" in df.columns:
            global_df = df.groupby("offline_data_id", as_index=False)[global_col].first()
            labels = global_df["offline_data_id"].astype(str).to_numpy()
            vals = global_df[global_col].to_numpy(dtype=float)
        else:
            vals = df[global_col].dropna().drop_duplicates().to_numpy(dtype=float)
            labels = np.arange(vals.size).astype(str)
        finite = np.isfinite(vals)
        vals = vals[finite]
        labels = labels[finite]
        x = np.arange(vals.size)
        ax.axhline(0.0, color="0.45", ls="--", lw=1.0)
        if mode == "simulation":
            ax.axhline(1.0, color="0.25", ls=":", lw=1.0)
        ax.scatter(x, vals, s=28, color="#13294B" if mode == "simulation" else "#CC79A7", alpha=0.85)
        if vals.size:
            if vals.size <= 25:
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=90 if vals.size > 8 else 0, fontsize=8)
            else:
                ax.set_xticks([])
            anchors = [0.0, 1.0] if mode == "simulation" else [0.0]
            ylo, yhi = _robust_limits(vals, anchors, q_low=0, q_high=100, include_zero=True, pad=0.18)
            ax.set_ylim(ylo, yhi)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "Global diagnostic unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(a) Global Explained Embedding Signal" if mode == "simulation" else "(a) Global Normalized Bellman Error")
        ax.set_xlabel("Offline Replicate" if mode == "simulation" else "Offline replicate")
        ax.set_ylabel(
            r"Explained Embedding Signal $\mathrm{EES}_r$"
            if mode == "simulation"
            else "Normalized Bellman error"
        )
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Global diagnostic unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.grid(axis="y", alpha=0.25)

    ax = axs[1]
    if mode == "simulation":
        fitted_cols = ["embedding_hat_norm2", "relative_embedding_error"]
        if set(fitted_cols).issubset(df.columns):
            fitted_df = df[fitted_cols].replace([np.inf, -np.inf], np.nan).dropna()
        else:
            fitted_df = pd.DataFrame(columns=fitted_cols)
        if not fitted_df.empty:
            x = fitted_df["embedding_hat_norm2"].to_numpy(dtype=float)
            y = fitted_df["relative_embedding_error"].to_numpy(dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            x, y = x[finite], y[finite]
        else:
            x, y = np.asarray([], dtype=float), np.asarray([], dtype=float)
        if x.size:
            ax.scatter(x, y, s=18, color="#FF5F05", alpha=0.55, edgecolor="none")
            ax.axhline(0.0, color="0.45", ls=":", lw=1.0)
            ax.axhline(1.0, color="0.25", ls="--", lw=1.1)
            ax.text(
                0.98,
                1.0,
                "Failure Threshold: Error Equals Truth Signal",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=8,
            )
            ax.text(
                0.98,
                0.0,
                "Exact Embedding Match",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=8,
            )
            xlo, xhi = _robust_limits(x, q_low=0, q_high=100, include_zero=True)
            y_min = float(np.nanmin(y))
            y_max = float(np.nanmax(y))
            y_upper = y_max
            clipped = False
            if y.size >= 10:
                y_q99 = float(np.nanpercentile(y, 99.0))
                robust_span = max(y_q99 - y_min, 1e-12)
                clipped = bool(np.isfinite(y_q99) and y_max > y_q99 + 0.5 * robust_span)
                if clipped:
                    y_upper = y_q99
            ylo = min(0.0, y_min)
            yhi = max(1.0, y_upper)
            if yhi - ylo < 1e-12:
                ylo, yhi = ylo - 0.5, yhi + 0.5
            else:
                pad = 0.08 * (yhi - ylo)
                ylo, yhi = ylo - pad, yhi + pad
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(ylo, yhi)
            if clipped:
                ax.text(0.98, 0.94, "y-axis clipped at 99%", transform=ax.transAxes, ha="right", va="top", fontsize=8)
            ax.set_title("(b) Relative Embedding Error vs Estimated Embedding Signal")
            ax.set_xlabel(r"Estimated Embedding Signal $\|\hat{\mu}_i\|_{\mathcal{H}}^2$")
            ax.set_ylabel(
                r"Relative Embedding Error $\mathrm{REE}_i = "
                r"\|\hat{\mu}_i-\mu_i^{MC}\|_{\mathcal{H}}^2 / "
                r"(\|\mu_i^{MC}\|_{\mathcal{H}}^2+\varepsilon)$"
            )
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "Relative residual diagnostic unavailable", ha="center", va="center", transform=ax.transAxes)
    else:
        scatter_df = df[["embedding_hat_norm2", "bellman_residual"]].replace([np.inf, -np.inf], np.nan).dropna()
        scatter_df = scatter_df[(scatter_df["embedding_hat_norm2"] >= 0.0) & (scatter_df["bellman_residual"] >= 0.0)]
        if not scatter_df.empty:
            x = scatter_df["embedding_hat_norm2"].to_numpy(dtype=float)
            y = scatter_df["bellman_residual"].to_numpy(dtype=float)
            ax.scatter(x, y, s=18, color="#009E73", alpha=0.65, edgecolor="none")
            xlo, xhi = _robust_limits(x, q_low=0, q_high=100, include_zero=True)
            ylo, yhi = _robust_limits(y, q_low=0, q_high=100, include_zero=True)
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(ylo, yhi)
            ax.set_title("(b) Bellman Residual vs Embedding Signal")
            ax.set_xlabel(r"Estimated signal $\|\hat{\mu}_i\|_H^2$")
            ax.set_ylabel("Held-out Bellman residual")
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "Bellman diagnostic unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.grid(alpha=0.25)

    ax = axs[2]
    point_col = "explained_embedding_signal" if mode == "simulation" else "normalized_bellman_error"
    if mode == "simulation" and point_col in df.columns and df[point_col].notna().any():
        clip_low = -0.2
        clip_high = 1.05
        lower_outlier_count = 0
        if "benchmark_id" in df.columns:
            box_data, labels, positions = [], [], []
            for pos, (bid, group) in enumerate(df.groupby("benchmark_id", dropna=False), start=1):
                vals = group[point_col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    box_data.append(vals)
                    labels.append(_display_test_target_id(bid) if pd.notna(bid) else "NA")
                    positions.append(pos)
                    lower = vals[vals < clip_low]
                    if lower.size:
                        lower_outlier_count += int(lower.size)
                        jitter = np.linspace(-0.16, 0.16, lower.size) if lower.size > 1 else np.asarray([0.0])
                        ax.scatter(
                            pos + jitter,
                            np.full(lower.size, clip_low),
                            marker="v",
                            s=16,
                            color="#FF5F05",
                            alpha=0.72,
                            edgecolor="none",
                            zorder=3,
                        )
            if box_data:
                bp = ax.boxplot(box_data, labels=labels, showmeans=True, showfliers=False, patch_artist=True)
                for patch, color in zip(bp["boxes"], _benchmark_palette(len(box_data))):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.22)
                ax.tick_params(axis="x", labelrotation=90 if len(labels) > 8 else 0, labelsize=7)
                if len(labels) > 40:
                    for idx, tick in enumerate(ax.get_xticklabels()):
                        tick.set_visible(idx % 5 == 0)
                elif len(labels) > 25:
                    for idx, tick in enumerate(ax.get_xticklabels()):
                        tick.set_visible(idx % 2 == 0)
        else:
            vals = df[point_col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                lower = vals[vals < clip_low]
                lower_outlier_count += int(lower.size)
                if lower.size:
                    jitter = np.linspace(-0.16, 0.16, lower.size) if lower.size > 1 else np.asarray([0.0])
                    ax.scatter(
                        1.0 + jitter,
                        np.full(lower.size, clip_low),
                        marker="v",
                        s=16,
                        color="#FF5F05",
                        alpha=0.72,
                        edgecolor="none",
                        zorder=3,
                    )
                bp = ax.boxplot([vals], labels=["all"], showmeans=True, showfliers=False, patch_artist=True)
                bp["boxes"][0].set_facecolor("#13294B")
                bp["boxes"][0].set_alpha(0.22)
        ax.axhline(0.0, color="0.45", ls="--", lw=1.0)
        ax.axhline(1.0, color="0.25", ls=":", lw=1.0)
        ax.set_ylim(clip_low - 0.03, clip_high)
        if lower_outlier_count:
            ax.text(
                0.98,
                0.06,
                f"{lower_outlier_count} Lower Outliers Clipped At -0.2",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
            )
        ax.set_title("(c) Pointwise Explained Embedding Signal")
        ax.set_xlabel("Evaluation Target")
        ax.set_ylabel(r"Explained Embedding Signal $\mathrm{EES}_i$")
    elif point_col in df.columns and df[point_col].notna().any():
        if "benchmark_id" in df.columns:
            box_data, labels = [], []
            for bid, group in df.groupby("benchmark_id", dropna=False):
                vals = group[point_col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
                if vals.size:
                    box_data.append(vals)
                    labels.append(_display_test_target_id(bid) if pd.notna(bid) else "NA")
            if box_data:
                ax.boxplot(box_data, labels=labels, showmeans=True)
                ax.tick_params(axis="x", labelrotation=90 if len(labels) > 8 else 0, labelsize=7)
        else:
            vals = df[point_col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            ax.boxplot([vals], labels=["all"], showmeans=True)
        ax.axhline(0.0, color="0.45", ls="--", lw=1.0)
        vals = df[point_col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        ylo, yhi = _robust_limits(vals, [0.0], q_low=0, q_high=97.5, include_zero=True, pad=0.18)
        ax.set_ylim(ylo, yhi)
        ax.set_title("(c) Pointwise Normalized Bellman Error")
        ax.set_xlabel("Test target")
        ax.set_ylabel("Normalized Bellman error")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Pointwise diagnostic unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path / filename, dpi=300)
    fig.savefig(out_path / Path(filename).with_suffix(".pdf"), dpi=300)
    plt.close(fig)
    if close_plt:
        plt.close("all")
    return True


def plot_embedding_r2_diagnostic(
    *,
    metrics_df: pd.DataFrame,
    outdir: str | os.PathLike,
    plt=None,
    filename: str = "embedding_quality_summary.png",
) -> bool:
    """Backward-compatible wrapper for the renamed embedding-quality plot."""
    return plot_embedding_quality_diagnostic(metrics_df=metrics_df, outdir=outdir, plt=plt, filename=filename)


def _replicate_plot_dir(outdir: Path, run_id: str) -> Path:
    if "_b" in run_id:
        rep_id, benchmark_id = run_id.rsplit("_b", 1)
        if benchmark_id.isdigit():
            return outdir / f"replicate_{rep_id}" / f"benchmark_{benchmark_id}"
    return outdir / f"replicate_{run_id}"


def _format_num(x, digits: int = 3) -> str:
    try:
        val = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(val):
        return str(x)
    return f"{val:.{digits}g}"


def _display_test_target_id(bid) -> str:
    try:
        val = float(bid)
    except (TypeError, ValueError):
        return str(bid)
    if np.isfinite(val) and val.is_integer():
        return str(int(val) + 1)
    return str(bid)


def _format_vector(values, digits: int = 3) -> str:
    return "(" + ", ".join(_format_num(v, digits=digits) for v in values) + ")"


def _candidate_run_roots(*paths: Path) -> list[Path]:
    roots: list[Path] = [Path.cwd()]
    for path in paths:
        try:
            roots.append(path.resolve().parent)
        except OSError:
            roots.append(path.parent)
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            unique.append(root)
            seen.add(key)
    return unique


def _load_params_for_caption(mu_dir: Path, metrics_dir: Path, outdir: Path) -> dict | None:
    for root in _candidate_run_roots(mu_dir, metrics_dir, outdir):
        for name in ("params.yaml", "params_tune.yaml"):
            path = root / name
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
    return None


def _load_combo_metadata(metrics_dir: Path) -> dict:
    path = metrics_dir / "tuning_combo_metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _load_filter_summary(metrics_dir: Path) -> dict:
    path = metrics_dir / "diagnostic_filter_summary.csv"
    if not path.exists():
        return {}
    try:
        return pd.read_csv(path).iloc[0].to_dict()
    except (OSError, IndexError, pd.errors.EmptyDataError):
        return {}


def _load_evaluation_points(mu_dir: Path, metrics_dir: Path, outdir: Path) -> pd.DataFrame | None:
    for root in _candidate_run_roots(mu_dir, metrics_dir, outdir):
        path = root / "data" / "benchmark_point.csv"
        if path.exists():
            try:
                return pd.read_csv(path)
            except OSError:
                return None
    return None


def _format_evaluation_points(points: pd.DataFrame | None, params: dict | None) -> str:
    if points is not None and not points.empty:
        s_cols = sorted([c for c in points.columns if c.startswith("s")], key=lambda x: int(x[1:]))
        a_cols = sorted([c for c in points.columns if c.startswith("a")], key=lambda x: int(x[1:]))
        pieces = []
        for _, row in points.sort_values("benchmark_id").iterrows():
            bid = int(row["benchmark_id"]) if "benchmark_id" in row and pd.notna(row["benchmark_id"]) else len(pieces)
            pieces.append(
                f"Test Target Point {_display_test_target_id(bid)}: s={_format_vector([row[c] for c in s_cols])}, "
                f"a={_format_vector([row[c] for c in a_cols])}"
            )
        return "; ".join(pieces)

    bench = (params or {}).get("benchmark") or {}
    s_star = bench.get("s_star") or []
    a_star = bench.get("a_star") or []
    pieces = []
    for bid, (s, a) in enumerate(zip(s_star, a_star)):
        pieces.append(f"Test Target Point {_display_test_target_id(bid)}: s={_format_vector(s)}, a={_format_vector(a)}")
    return "; ".join(pieces) if pieces else "Test target point values unavailable"


def _actual_replicate_count(metrics_df: pd.DataFrame | None) -> int | None:
    if metrics_df is None or "run_id" not in metrics_df.columns:
        return None
    reps = set()
    for rid in metrics_df["run_id"].astype(str):
        reps.add(rid.split("_b", 1)[0])
    return len(reps) if reps else None


def _build_summary_caption(
    *,
    mu_dir: Path,
    metrics_dir: Path,
    outdir: Path,
    metrics_df: pd.DataFrame | None,
) -> str | None:
    params = _load_params_for_caption(mu_dir, metrics_dir, outdir)
    if params is None:
        return None

    meta = _load_combo_metadata(metrics_dir)
    filter_summary = _load_filter_summary(metrics_dir)
    points = _load_evaluation_points(mu_dir, metrics_dir, outdir)
    bench = params.get("benchmark") or {}
    target_set = params.get("target_set") or {}
    z_sim = params.get("Z_sim") or {}
    opt = params.get("optimization") or {}
    kernel = params.get("kernel") or {}
    op = params.get("operator_approximation") or {}
    ratio = params.get("ratio") or {}
    basis = params.get("mean_embedding_basis") or {}
    policy = params.get("policy") or {}
    n_rep_actual = _actual_replicate_count(metrics_df)
    n_rep_requested = (params.get("experiment") or {}).get("num_replicates")
    if filter_summary:
        total_reps = filter_summary.get("total_offline_replicates")
        selected_reps = filter_summary.get("selected_offline_replicates")
        try:
            rep_part = (
                f"{int(float(total_reps))} completed offline replicates "
                f"({int(float(selected_reps))} represented after diagnostic filtering)"
            )
        except (TypeError, ValueError, OverflowError):
            rep_part = f"{n_rep_actual} filtered offline replicates represented"
    elif n_rep_actual is None:
        rep_part = f"{n_rep_requested} requested offline replicates"
    else:
        rep_part = f"{n_rep_actual} completed offline replicates"
    if n_rep_requested is not None and n_rep_actual is not None and n_rep_actual != int(n_rep_requested):
        rep_part += f" of {n_rep_requested} requested"

    combo = "configuration"
    if meta:
        combo = f"configuration {meta.get('combo_id', 'NA')} ({meta.get('combo_name', 'unnamed')})"
        overrides = meta.get("overrides") or {}
        if overrides:
            combo += f", overrides={json.dumps(overrides, sort_keys=True)}"

    specs = [
        f"{combo}.",
        f"Offline data: n_ids={params.get('n_ids')}, n_timepoints={params.get('n_timepoints')}, "
        f"burn_in={params.get('offline_burn_in')}, {rep_part}.",
        f"Training targets: {target_set.get('num_points')} points, mode={target_set.get('mode')}, "
        f"exclude_evaluation_targets={target_set.get('exclude_benchmark')}.",
        f"Test targets: {bench.get('num_points')} points; {_format_evaluation_points(points, params)}.",
        f"Monte Carlo truth: {z_sim.get('n_ids')} trajectories, {z_sim.get('n_timepoints')} time points; "
        f"gamma={params.get('gamma_val')}; Z grid={params.get('num_grid_points')}; reward_dim={params.get('reward_dim')}.",
        f"Kernel/operator: {kernel.get('type')} nu={kernel.get('nu')}, length_scale={kernel.get('length_scale')}, "
        f"sigma={kernel.get('sigma')}; operator={op.get('method')} with {op.get('num_features')} features; "
        f"uLSIF basis={ratio.get('n_basis')}.",
        f"Mean-embedding basis: method={basis.get('method', 'full')}, L={basis.get('n_basis', 'full')}; "
        f"B shape is L x {params.get('num_grid_points')}.",
        f"Optimization: steps={opt.get('num_steps')}, lr={opt.get('lr')}, weight_decay={opt.get('weight_decay')}, "
        f"target_batch_size={opt.get('target_batch_size')}, lambda_reg={params.get('lambda_reg')}, "
        f"lambda_B={params.get('lambda_B')}, mass_anchor_lambda={opt.get('mass_anchor_lambda')}, "
        f"eta_clip=[{opt.get('eta_clip_min')}, {opt.get('eta_clip_max')}].",
        f"Policies: behavior={policy.get('Behvaioral_policy')}, test_target={policy.get('evaluation_Target_policy')}.",
    ]
    reporting_filter = params.get("reporting_filter") or params.get("diagnostic_filter") or {}
    if reporting_filter and reporting_filter.get("enabled", True):
        metric = reporting_filter.get("metric", "projected_bellman_test_risk")
        comparison = reporting_filter.get("comparison", "lt")
        threshold = reporting_filter.get("threshold", 0.15)
        filter_text = f"Reporting filter: plotted/exported curves satisfy {metric} {comparison} {threshold}"
        if filter_summary:
            try:
                filter_text += (
                    f"; retained {int(float(filter_summary.get('selected_rows')))}"
                    f"/{int(float(filter_summary.get('total_rows')))} benchmark-replicate curves"
                    f" and {int(float(filter_summary.get('selected_benchmarks')))}"
                    f"/{int(float(filter_summary.get('total_benchmarks')))} test target points"
                )
            except (TypeError, ValueError, OverflowError):
                pass
        specs.append(filter_text + ".")
    return " ".join(specs)


def _benchmark_palette(n: int) -> list[str]:
    base = [
        "#FF5F05",  # Illini orange
        "#13294B",  # Illini blue
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#F0E442",
        "#000000",
        "#882255",
        "#44AA99",
        "#AA4499",
        "#117733",
    ]
    if n <= len(base):
        return base[:n]
    return [base[i % len(base)] for i in range(n)]


def _calibration_distance_by_benchmark(
    H: np.ndarray,
    T: np.ndarray,
    *,
    run_ids: list[str],
    rid_to_benchmark: dict[str, object],
    benchmark_ids: list[object],
) -> dict[object, float]:
    scores: dict[object, float] = {}
    for bid in benchmark_ids:
        idx = [i for i, rid in enumerate(run_ids) if rid_to_benchmark.get(str(rid)) == bid]
        if not idx:
            continue
        H_b = H[np.asarray(idx)]
        T_b = T[np.asarray(idx)]
        t_mean = T_b.mean(axis=0)
        h_mean = H_b.mean(axis=0)
        if t_mean.size == 0:
            continue
        order = np.argsort(t_mean)
        bins = np.array_split(order, min(10, order.size))
        bx = np.asarray([float(t_mean[b].mean()) for b in bins if b.size])
        by_runs = [
            np.asarray([float(row[b].mean()) for b in bins if b.size])
            for row in H_b
        ]
        if bx.size == 0 or not by_runs:
            continue
        by = np.vstack(by_runs).mean(axis=0)
        diff = by - bx
        finite = np.isfinite(diff)
        if finite.any():
            scores[bid] = float(np.sqrt(np.mean(diff[finite] * diff[finite]) / 2.0))
    return scores


def _plot_multi_benchmark_summary(
    H: np.ndarray,
    T: np.ndarray,
    *,
    run_ids: list[str],
    metrics_df: pd.DataFrame,
    outdir: Path,
    plt,
    caption: str | None = None,
    benchmark_ids_subset: list[object] | None = None,
    filename_suffix: str = "",
    make_top10: bool = True,
) -> None:
    metrics = metrics_df.copy()
    metrics["run_id"] = metrics["run_id"].astype(str)
    rid_to_benchmark = dict(zip(metrics["run_id"], metrics["benchmark_id"]))
    all_benchmark_ids = sorted(pd.Series(metrics["benchmark_id"]).dropna().unique().tolist())
    scores = _calibration_distance_by_benchmark(
        H,
        T,
        run_ids=run_ids,
        rid_to_benchmark=rid_to_benchmark,
        benchmark_ids=all_benchmark_ids,
    )
    if benchmark_ids_subset is None:
        benchmark_ids = all_benchmark_ids
    else:
        wanted = set(benchmark_ids_subset)
        benchmark_ids = [bid for bid in all_benchmark_ids if bid in wanted]
    if not benchmark_ids:
        return
    colors = _benchmark_palette(len(benchmark_ids))
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "*", "p"]

    fig, axs = plt.subplots(2, 3, figsize=(19, 12.0))

    ax = axs[0, 0]
    cal_x, cal_y = [], []
    for j, (color, bid) in enumerate(zip(colors, benchmark_ids)):
        idx = [i for i, rid in enumerate(run_ids) if rid_to_benchmark.get(str(rid)) == bid]
        if not idx:
            continue
        H_b = H[np.asarray(idx)]
        T_b = T[np.asarray(idx)]
        t_mean = T_b.mean(axis=0)
        h_mean = H_b.mean(axis=0)
        order = np.argsort(t_mean)
        bins = np.array_split(order, min(10, order.size))
        bx = np.asarray([float(t_mean[b].mean()) for b in bins if b.size])
        by_runs = []
        for row in H_b:
            by_runs.append(np.asarray([float(row[b].mean()) for b in bins if b.size]))
        Y = np.vstack(by_runs)
        by = Y.mean(axis=0)
        ax.plot(
            bx,
            by,
            color=color,
            marker=markers[j % len(markers)],
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            lw=1.8,
            label=f"Test Target Point {_display_test_target_id(bid)}",
        )
        cal_x.append(bx)
        cal_y.append(by)
    _set_calibration_axes(
        ax,
        cal_x,
        cal_y,
        q_low=0,
        q_high=100,
        pad=0.10,
        include_zero=False,
        min_span=1e-4,
        ideal_color="0.25",
        ideal_lw=1.2,
        ideal_label="ideal",
    )
    ax.set_title("(a) Test-Target Calibration")
    ax.set_xlabel("True mean embedding (bin mean)")
    ax.set_ylabel("Estimated mean embedding (bin mean)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2, frameon=True, loc="best")

    def metric_boxplot(ax, column: str, ylabel: str, title: str) -> None:
        from matplotlib.lines import Line2D

        box_data, labels, used_colors, used_bids = [], [], [], []
        if column not in metrics.columns:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{column} not available", ha="center", va="center", transform=ax.transAxes)
            return
        for color, bid in zip(colors, benchmark_ids):
            vals = pd.to_numeric(metrics.loc[metrics["benchmark_id"] == bid, column], errors="coerce").dropna().to_numpy()
            if vals.size:
                box_data.append(vals)
                labels.append(_display_test_target_id(bid))
                used_colors.append(color)
                used_bids.append(bid)
        if not box_data:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{column} not available", ha="center", va="center", transform=ax.transAxes)
            return
        bp = ax.boxplot(
            box_data,
            labels=labels,
            showmeans=True,
            patch_artist=True,
            meanprops={
                "marker": "^",
                "markerfacecolor": "#2ca02c",
                "markeredgecolor": "#2ca02c",
                "markersize": 5,
            },
            medianprops={"color": "#ff7f0e", "linewidth": 1.2},
        )
        for patch, color in zip(bp["boxes"], used_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.25)
        if column == "Bias":
            ax.axhline(0.0, color="0.35", lw=1.0, ls="--", alpha=0.75)
        ax.set_xlabel("Test Target Point")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ylo, yhi = _robust_limits(*box_data, q_low=0, q_high=97.5, include_zero=True)
        if column == "Bias":
            span = max(abs(ylo), abs(yhi), 1e-4)
            ylo, yhi = -span, span
        ax.set_ylim(ylo, yhi)
        if any(float(np.nanmax(vals)) > yhi for vals in box_data if vals.size):
            ax.text(0.96, 0.92, "y-axis clipped at 97.5%", transform=ax.transAxes, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        handles = [
            Line2D(
                [0],
                [0],
                marker="^",
                color="none",
                markerfacecolor="#2ca02c",
                markeredgecolor="#2ca02c",
                markersize=5,
                label="mean",
            ),
            Line2D([0], [0], color="#ff7f0e", lw=1.2, label="median"),
        ]
        ax.legend(handles=handles, fontsize=6.5, ncol=2, frameon=True, loc="best")

    metric_boxplot(
        axs[0, 1],
        "Bias",
        r"Bias across Z grid ($\hat{\mu}-\mu$)",
        "(b) Bias By Test Target Point",
    )

    metric_boxplot(
        axs[0, 2],
        "MAE",
        "MAE across Z grid",
        "(c) MAE By Test Target Point",
    )
    metric_boxplot(
        axs[1, 0],
        "RMSE",
        "RMSE across Z grid",
        "(d) RMSE By Test Target Point",
    )
    metric_boxplot(
        axs[1, 1],
        "projected_bellman_test_risk",
        "Projected Bellman risk",
        "(e) Projected Bellman Risk By Test Target Point",
    )

    ax = axs[1, 2]
    ecdf_abs_values = []
    for color, bid in zip(colors, benchmark_ids):
        idx = [i for i, rid in enumerate(run_ids) if rid_to_benchmark.get(str(rid)) == bid]
        if not idx:
            continue
        abs_err = np.sort(np.abs(H[np.asarray(idx)] - T[np.asarray(idx)]).reshape(-1))
        ecdf_abs_values.append(abs_err)
        ecdf = np.arange(1, abs_err.size + 1) / abs_err.size
        ax.plot(abs_err, ecdf, color=color, lw=1.9, label=f"Test Target Point {_display_test_target_id(bid)}")
    ax.set_title(r"(f) ECDF Of $|\hat{\mu}-\mu|$ By Test Target Point")
    ax.set_xlabel(r"$|\hat{\mu}-\mu|$")
    ax.set_ylabel("ECDF")
    all_abs = np.concatenate(ecdf_abs_values) if ecdf_abs_values else np.asarray([], dtype=float)
    xmax = float(np.nanpercentile(all_abs, 99.5)) if all_abs.size else 1.0
    if np.isfinite(xmax) and xmax > 0:
        ax.set_xlim(0.0, xmax)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    if caption:
        wrapped = textwrap.fill(caption, width=215)
        fig.text(
            0.5,
            0.035,
            wrapped,
            ha="center",
            va="bottom",
            fontsize=7.2,
            linespacing=1.22,
        )
        fig.tight_layout(rect=(0.0, 0.23, 1.0, 0.985))
    else:
        fig.tight_layout()
    fig.savefig(outdir / f"mu_summary_UG{filename_suffix}.png", dpi=300)
    fig.savefig(outdir / f"mu_summary_benchmarks{filename_suffix}.png", dpi=300)
    plt.close(fig)

    if make_top10 and benchmark_ids_subset is None and len(all_benchmark_ids) > 10:
        ranked = [bid for bid in all_benchmark_ids if np.isfinite(scores.get(bid, float("nan")))]
        ranked.sort(key=lambda bid: (scores[bid], int(bid) if float(bid).is_integer() else float(bid)))
        top10 = ranked[:10]
        if top10:
            top_caption = "Top 10 test target points closest to the ideal line in Panel (a): " + ", ".join(
                _display_test_target_id(bid)
                for bid in top10
            )
            _plot_multi_benchmark_summary(
                H,
                T,
                run_ids=run_ids,
                metrics_df=metrics_df,
                outdir=outdir,
                plt=plt,
                caption=top_caption,
                benchmark_ids_subset=top10,
                filename_suffix="_top10",
                make_top10=False,
            )


def _plot_four_panel_summary(
    H: np.ndarray,
    T: np.ndarray,
    *,
    run_ids: list[str],
    metrics_df: pd.DataFrame | None,
    outdir: Path,
    plt,
) -> None:
    x = np.arange(H.shape[1])
    h_mean = H.mean(axis=0)
    t_mean = T.mean(axis=0)
    h_sd = H.std(axis=0, ddof=1) if H.shape[0] > 1 else np.zeros(H.shape[1])
    t_sd = T.std(axis=0, ddof=1) if T.shape[0] > 1 else np.zeros(T.shape[1])
    diff = H - T

    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    ax = axs[0, 0]
    for row in H:
        ax.plot(x, row, color="0.72", lw=0.7, alpha=0.35)
    ax.fill_between(x, t_mean - 1.96 * t_sd, t_mean + 1.96 * t_sd, color="#0b2a50", alpha=0.10)
    ax.fill_between(x, h_mean - 1.96 * h_sd, h_mean + 1.96 * h_sd, color="#ff5f05", alpha=0.20)
    ax.plot(x, h_mean, color="#ff5f05", lw=2.0, label=r"$\hat{\mu}$")
    ax.plot(x, t_mean, color="#0b2a50", lw=2.0, label=r"$\mu$")
    lo, hi = _robust_limits(
        H,
        T,
        h_mean - 1.96 * h_sd,
        h_mean + 1.96 * h_sd,
        t_mean - 1.96 * t_sd,
        t_mean + 1.96 * t_sd,
        q_low=0.5,
        q_high=99.5,
        include_zero=True,
    )
    ax.set_ylim(lo, hi)
    ax.set_title("(a) Mean +/- 1.96 SD Across Offline Samples")
    ax.set_xlabel("Index on evaluation-target Z grid")
    ax.set_ylabel("Mean embedding")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axs[0, 1]
    order = np.argsort(t_mean)
    bins = np.array_split(order, min(10, order.size))
    bx = np.asarray([float(t_mean[b].mean()) for b in bins if b.size])
    line_values = []
    for tr, hr in zip(T, H):
        by_run = np.asarray([float(hr[b].mean()) for b in bins if b.size])
        line_values.append(by_run)
    Y = np.vstack(line_values)
    by = Y.mean(axis=0)
    slope, intercept = _deming(bx, by)
    _set_calibration_axes(
        ax,
        [bx],
        [by],
        q_low=0,
        q_high=100,
        pad=0.15,
        include_zero=False,
        min_span=1e-4,
        ideal_color="#0b2a50",
        ideal_lw=1.5,
        ideal_label="ideal",
    )
    ax.plot(
        bx,
        by,
        color="#ff5f05",
        marker="o",
        lw=2.0,
        label="mean calibration",
    )
    ax.text(0.04, 0.94, f"Binned Deming slope={slope:.3f}, int={intercept:.3f}", transform=ax.transAxes, va="top")
    ax.text(0.96, 0.06, "axes centered on binned means", transform=ax.transAxes, ha="right", fontsize=8)
    ax.set_title("(b) Quantile Calibration")
    ax.set_xlabel("True mean embedding (bin mean)")
    ax.set_ylabel("Estimated mean embedding (bin mean)")
    ax.grid(alpha=0.25)
    ax.legend()

    per_run = pd.DataFrame(
        {
            "|Bias|": np.abs(diff.mean(axis=1)),
            "MAE": np.mean(np.abs(diff), axis=1),
            "RMSE": np.sqrt(np.mean(diff * diff, axis=1)),
        }
    )
    risk_col = None
    risk_label = None
    if metrics_df is not None and "projected_bellman_test_risk" in metrics_df.columns:
        risk_col = "projected_bellman_test_risk"
        risk_label = "Projected Bellman risk"
    elif metrics_df is not None and "benchmark_embedding_risk" in metrics_df.columns:
        risk_col = "benchmark_embedding_risk"
        risk_label = "Oracle embedding risk"
    if metrics_df is not None and risk_col is not None:
        risk_lookup = metrics_df.assign(run_id=metrics_df["run_id"].astype(str)).set_index("run_id")
        values = []
        for rid in run_ids:
            values.append(
                float(risk_lookup.loc[str(rid), risk_col])
                if str(rid) in risk_lookup.index
                else np.nan
            )
        if np.isfinite(values).any():
            risk_vals = np.asarray(values, dtype=float)
            per_run[risk_label] = risk_vals
    ax = axs[1, 0]
    ax.boxplot([per_run[c].to_numpy() for c in per_run.columns], labels=list(per_run.columns), showmeans=True)
    ax.set_title("(c) Per-run Error Summaries")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=10)
    ylo, yhi = _robust_limits(*[per_run[c].to_numpy() for c in per_run.columns], q_low=0, q_high=97.5, include_zero=True)
    ax.set_ylim(ylo, yhi)
    if np.isfinite(per_run.to_numpy(dtype=float)).any():
        raw_max = float(np.nanmax(per_run.to_numpy(dtype=float)))
        if raw_max > yhi:
            ax.text(0.96, 0.92, "y-axis clipped at 97.5%", transform=ax.transAxes, ha="right", fontsize=8)

    ax = axs[1, 1]
    abs_err = np.sort(np.abs(diff).reshape(-1))
    ecdf = np.arange(1, abs_err.size + 1) / abs_err.size
    ax.plot(abs_err, ecdf, color="#ff5f05", lw=2.2)
    ax.set_title(r"(d) Empirical CDF of $|\hat{\mu}-\mu|$")
    ax.set_xlabel(r"$|\hat{\mu}-\mu|$")
    ax.set_ylabel("ECDF")
    if abs_err.size:
        xmax = float(np.nanpercentile(abs_err, 99.5))
        if np.isfinite(xmax) and xmax > 0 and xmax < float(np.nanmax(abs_err)):
            ax.set_xlim(0.0, xmax)
            ax.text(0.96, 0.08, "x-axis clipped at 99.5%", transform=ax.transAxes, ha="right", fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(outdir / "mu_summary_UG.png", dpi=300)
    plt.close(fig)
