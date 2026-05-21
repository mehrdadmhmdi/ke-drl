from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sim_utils import bootstrap_kedrl


bootstrap_kedrl()

from ke_drl.matern_kernel import matern_kernel


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

    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No matched mu_hat_*.csv/mu_true_*.csv pairs in {mu_dir}")

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

    hats, trues, run_ids = [], [], []
    for hat_path in sorted(mu_dir.glob("mu_hat_*.csv")):
        run_id = hat_path.stem.replace("mu_hat_", "")
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

    metrics_df = None
    per_run_path = Path(metrics_dir) / "per_run_metrics.csv"
    if per_run_path.exists():
        metrics_df = pd.read_csv(per_run_path)

    multi_benchmark = (
        metrics_df is not None
        and "benchmark_id" in metrics_df.columns
        and metrics_df["benchmark_id"].nunique(dropna=True) > 1
    )
    if multi_benchmark:
        _plot_multi_benchmark_summary(H, T, run_ids=run_ids, metrics_df=metrics_df, outdir=outdir, plt=plt)
    else:
        _plot_four_panel_summary(H, T, run_ids=run_ids, metrics_df=metrics_df, outdir=outdir, plt=plt)

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
    ax.set_xlabel("Index on benchmark Z grid")
    ax.set_ylabel("Mean embedding")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "mu_summary.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(T.reshape(-1), H.reshape(-1), s=6, alpha=0.25)
    lo, hi = _robust_limits(T, H, q_low=2.5, q_high=97.5, include_zero=False)
    ax.plot([lo, hi], [lo, hi], color="black", lw=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.text(0.96, 0.06, "axes use central 95%", transform=ax.transAxes, ha="right", fontsize=8)
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


def plot_single_mu_diagnostic(
    *,
    mu_hat,
    mu_true,
    outdir: str | os.PathLike,
    run_id: str | int,
    plt=None,
    filename: str = "mu_hat_vs_truth.png",
) -> None:
    """Plot one replicate's estimated benchmark embedding against MC truth."""
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
    ax.set_xlabel("Index on benchmark Z grid")
    ax.set_ylabel("Mean embedding")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axs[1]
    ax.scatter(true, hat, s=12, alpha=0.5, color="#ff5f05", edgecolor="none")
    lo, hi = _robust_limits(true, hat, q_low=0.5, q_high=99.5, include_zero=True)
    ax.plot([lo, hi], [lo, hi], "--", color="#0b2a50", lw=1.2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
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
    count = 0
    for hat_path in sorted(mu_dir.glob("mu_hat_*.csv")):
        run_id = hat_path.stem.replace("mu_hat_", "")
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


def _replicate_plot_dir(outdir: Path, run_id: str) -> Path:
    if "_b" in run_id:
        rep_id, benchmark_id = run_id.rsplit("_b", 1)
        if benchmark_id.isdigit():
            return outdir / f"replicate_{rep_id}" / f"benchmark_{benchmark_id}"
    return outdir / f"replicate_{run_id}"


def _benchmark_palette(n: int) -> list[str]:
    base = [
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


def _plot_multi_benchmark_summary(
    H: np.ndarray,
    T: np.ndarray,
    *,
    run_ids: list[str],
    metrics_df: pd.DataFrame,
    outdir: Path,
    plt,
) -> None:
    metrics = metrics_df.copy()
    metrics["run_id"] = metrics["run_id"].astype(str)
    rid_to_benchmark = dict(zip(metrics["run_id"], metrics["benchmark_id"]))
    benchmark_ids = sorted(pd.Series(metrics["benchmark_id"]).dropna().unique().tolist())
    colors = _benchmark_palette(len(benchmark_ids))
    x = np.arange(H.shape[1])

    fig, axs = plt.subplots(2, 3, figsize=(19, 9.5))

    ax = axs[0, 0]
    y_for_limits = []
    for color, bid in zip(colors, benchmark_ids):
        idx = [i for i, rid in enumerate(run_ids) if rid_to_benchmark.get(str(rid)) == bid]
        if not idx:
            continue
        H_b = H[np.asarray(idx)]
        T_b = T[np.asarray(idx)]
        h_mean = H_b.mean(axis=0)
        h_sd = H_b.std(axis=0, ddof=1) if H_b.shape[0] > 1 else np.zeros(H_b.shape[1])
        t_mean = T_b.mean(axis=0)
        ax.fill_between(x, h_mean - 1.96 * h_sd, h_mean + 1.96 * h_sd, color=color, alpha=0.10)
        ax.plot(x, t_mean, color=color, lw=1.8, ls="--", label=f"truth {int(bid)}")
        ax.plot(x, h_mean, color=color, lw=2.2, label=f"estimate {int(bid)}")
        y_for_limits.extend([t_mean, h_mean, h_mean - 1.96 * h_sd, h_mean + 1.96 * h_sd])
    lo, hi = _robust_limits(*y_for_limits, q_low=0.5, q_high=99.5, include_zero=True)
    ax.set_ylim(lo, hi)
    ax.set_title("(a) Benchmark-specific mean embeddings")
    ax.set_xlabel("Index on benchmark Z grid")
    ax.set_ylabel("Mean embedding")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2, frameon=True)

    ax = axs[0, 1]
    cal_x, cal_y = [], []
    for color, bid in zip(colors, benchmark_ids):
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
        se = Y.std(axis=0, ddof=1) / math.sqrt(Y.shape[0]) if Y.shape[0] > 1 else np.zeros(Y.shape[1])
        ax.errorbar(bx, by, yerr=1.96 * se, color=color, marker="o", lw=1.5, capsize=2, label=f"benchmark {int(bid)}")
        cal_x.append(bx)
        cal_y.append(by)
    lo, hi = _robust_limits(*cal_x, *cal_y, q_low=0, q_high=100, pad=0.15, include_zero=False, min_span=1e-4)
    ax.plot([lo, hi], [lo, hi], "--", color="0.25", lw=1.2, label="ideal")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title("(b) Benchmark-specific calibration")
    ax.set_xlabel("True mean embedding (bin mean)")
    ax.set_ylabel("Estimated mean embedding (bin mean)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    def metric_boxplot(ax, column: str, ylabel: str, title: str) -> None:
        box_data, labels, used_colors = [], [], []
        if column not in metrics.columns:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{column} not available", ha="center", va="center", transform=ax.transAxes)
            return
        for color, bid in zip(colors, benchmark_ids):
            vals = pd.to_numeric(metrics.loc[metrics["benchmark_id"] == bid, column], errors="coerce").dropna().to_numpy()
            if vals.size:
                box_data.append(vals)
                labels.append(str(int(bid)))
                used_colors.append(color)
        if not box_data:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{column} not available", ha="center", va="center", transform=ax.transAxes)
            return
        bp = ax.boxplot(box_data, labels=labels, showmeans=True, patch_artist=True)
        for patch, color in zip(bp["boxes"], used_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.25)
        ax.set_xlabel("Benchmark point")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ylo, yhi = _robust_limits(*box_data, q_low=0, q_high=97.5, include_zero=True)
        ax.set_ylim(ylo, yhi)
        if any(float(np.nanmax(vals)) > yhi for vals in box_data if vals.size):
            ax.text(0.96, 0.92, "y-axis clipped at 97.5%", transform=ax.transAxes, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)

    metric_boxplot(
        axs[0, 2],
        "MAE",
        "MAE across Z grid",
        "(c) MAE by benchmark point",
    )
    metric_boxplot(
        axs[1, 0],
        "RMSE",
        "RMSE across Z grid",
        "(d) RMSE by benchmark point",
    )
    metric_boxplot(
        axs[1, 1],
        "projected_bellman_test_risk",
        "Projected Bellman risk",
        "(e) Projected Bellman risk by benchmark point",
    )

    ax = axs[1, 2]
    for color, bid in zip(colors, benchmark_ids):
        idx = [i for i, rid in enumerate(run_ids) if rid_to_benchmark.get(str(rid)) == bid]
        if not idx:
            continue
        abs_err = np.sort(np.abs(H[np.asarray(idx)] - T[np.asarray(idx)]).reshape(-1))
        ecdf = np.arange(1, abs_err.size + 1) / abs_err.size
        ax.plot(abs_err, ecdf, color=color, lw=1.9, label=f"benchmark {int(bid)}")
    ax.set_title(r"(f) ECDF of $|\hat{\mu}-\mu|$ by benchmark")
    ax.set_xlabel(r"$|\hat{\mu}-\mu|$")
    ax.set_ylabel("ECDF")
    all_abs = np.abs(H - T).reshape(-1)
    xmax = float(np.nanpercentile(all_abs, 99.5)) if all_abs.size else 1.0
    if np.isfinite(xmax) and xmax > 0:
        ax.set_xlim(0.0, xmax)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(outdir / "mu_summary_UG.png", dpi=300)
    fig.savefig(outdir / "mu_summary_benchmarks.png", dpi=300)
    plt.close(fig)


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
    ax.set_xlabel("Index on fixed benchmark Z grid")
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
        ax.plot(bx, by_run, color="0.6", lw=0.8, alpha=0.25)
    Y = np.vstack(line_values)
    by = Y.mean(axis=0)
    se = Y.std(axis=0, ddof=1) / math.sqrt(Y.shape[0]) if Y.shape[0] > 1 else np.zeros(Y.shape[1])
    slope, intercept = _deming(bx, by)
    lo, hi = _robust_limits(bx, by, q_low=0, q_high=100, pad=0.15, include_zero=False, min_span=1e-4)
    ax.plot([lo, hi], [lo, hi], "--", color="#0b2a50", lw=1.5, label="ideal")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.errorbar(
        bx,
        by,
        yerr=1.96 * se,
        color="#ff5f05",
        marker="o",
        lw=2.0,
        capsize=3,
        label="mean calibration +/- 95% CI",
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
