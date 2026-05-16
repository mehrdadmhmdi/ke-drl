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
    """Compute E||k(.,Z)-mu_hat||^2 for one fixed evaluation point."""
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
        "benchmark_embedding_risk",
    ]:
        if col not in df:
            continue
        agg[f"{col}_mean"] = float(df[col].mean())
        agg[f"{col}_sd"] = float(df[col].std(ddof=1)) if len(df) > 1 else 0.0
    pd.DataFrame([agg]).to_csv(metrics_dir / "aggregate_metrics.csv", index=False)

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

    _plot_four_panel_summary(H, T, run_ids=run_ids, metrics_df=metrics_df, outdir=outdir, plt=plt)

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
    lo = min(float(T.min()), float(H.min()))
    hi = max(float(T.max()), float(H.max()))
    ax.plot([lo, hi], [lo, hi], color="black", lw=1.0)
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
    slope, intercept = _deming(T.reshape(-1), H.reshape(-1))
    lo = min(float(np.nanmin(T)), float(np.nanmin(H)))
    hi = max(float(np.nanmax(T)), float(np.nanmax(H)))
    ax.plot([lo, hi], [lo, hi], "--", color="#0b2a50", lw=1.5, label="ideal")
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
    ax.text(0.04, 0.94, f"Deming slope={slope:.3f}, int={intercept:.3f}", transform=ax.transAxes, va="top")
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
    if metrics_df is not None and "risk_bellman_final" in metrics_df.columns:
        risk_lookup = metrics_df.assign(run_id=metrics_df["run_id"].astype(str)).set_index("run_id")
        values = []
        for rid in run_ids:
            values.append(float(risk_lookup.loc[str(rid), "risk_bellman_final"]) if str(rid) in risk_lookup.index else np.nan)
        if np.isfinite(values).any():
            per_run["Bellman risk"] = np.asarray(values, dtype=float)
    ax = axs[1, 0]
    ax.boxplot([per_run[c].to_numpy() for c in per_run.columns], labels=list(per_run.columns), showmeans=True)
    ax.set_title("(c) Per-run Error Summaries")
    ax.grid(axis="y", alpha=0.25)

    ax = axs[1, 1]
    abs_err = np.sort(np.abs(diff).reshape(-1))
    ecdf = np.arange(1, abs_err.size + 1) / abs_err.size
    ax.plot(abs_err, ecdf, color="#ff5f05", lw=2.2)
    ax.set_title(r"(d) Empirical CDF of $|\hat{\mu}-\mu|$")
    ax.set_xlabel(r"$|\hat{\mu}-\mu|$")
    ax.set_ylabel("ECDF")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(outdir / "mu_summary_UG.png", dpi=300)
    plt.close(fig)
