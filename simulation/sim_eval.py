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
) -> torch.Tensor:
    Kzz = matern_kernel(Z_grid, Z_grid, nu=nu, length_scale=length_scale, sigma=sigma)
    return (Kzz @ beta.reshape(-1)).contiguous()


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
        rows.append({"run_id": run_id, **metrics_from_mu(mu_hat, mu_true)})

    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No matched mu_hat_*.csv/mu_true_*.csv pairs in {mu_dir}")

    df.to_csv(metrics_dir / "per_run_metrics.csv", index=False)
    agg = {}
    for col in ["RMSE", "MAE", "SupNorm", "Bias", "Corr"]:
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

    hats, trues = [], []
    for hat_path in sorted(mu_dir.glob("mu_hat_*.csv")):
        run_id = hat_path.stem.replace("mu_hat_", "")
        true_path = mu_dir / f"mu_true_{run_id}.csv"
        if true_path.exists():
            hats.append(np.loadtxt(hat_path, delimiter=",").reshape(-1))
            trues.append(np.loadtxt(true_path, delimiter=",").reshape(-1))
    if not hats:
        raise FileNotFoundError(f"No matched mu outputs in {mu_dir}")

    H = np.vstack(hats)
    T = np.vstack(trues)
    x = np.arange(H.shape[1])
    h_mean, h_sd = H.mean(axis=0), H.std(axis=0, ddof=1) if H.shape[0] > 1 else np.zeros(H.shape[1])
    t_mean, t_sd = T.mean(axis=0), T.std(axis=0, ddof=1) if T.shape[0] > 1 else np.zeros(T.shape[1])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, t_mean, lw=1.7, color="black", label="MC truth")
    ax.fill_between(x, t_mean - 1.96 * t_sd, t_mean + 1.96 * t_sd, color="black", alpha=0.08)
    ax.plot(x, h_mean, lw=1.7, color="#1f77b4", label="KE-DRL")
    ax.fill_between(x, h_mean - 1.96 * h_sd, h_mean + 1.96 * h_sd, color="#1f77b4", alpha=0.16)
    ax.set_xlabel("Z-grid index")
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
