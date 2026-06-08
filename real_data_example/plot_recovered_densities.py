#!/usr/bin/env python3
"""
Plot recovered return distributions from exported KE-DRL embedding coefficients.

This script does NOT use the simplex projection beta -> w.
Instead, it estimates valid probability atom weights w by matching the embedding
induced by the candidate probability law to the exported KE-DRL embedding
coefficients beta_hat.

Expected inputs are copied into a folder named plotting_files by default.
Each policy should have its own subfolder, e.g.
  plotting_files/rev_policy/artifacts.pt
  plotting_files/rev_policy/plot_payload.pt                         [preferred]
  plotting_files/rev_policy/mean_embedding_coefficients_beta.csv    [fallback]
  plotting_files/rev_policy/Z_grid_normalized_optimized.csv         [fallback]

Z_grid_raw is not required as an input. It is constructed from the scaler:
    Z_raw = r_mu + r_sd * Z_norm
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Use the exact Matérn kernel implementation from the KE-DRL package.
# No local fallback is used, so this plotting-only script stays kernel-consistent
# with policy_evaluation9.py and the fitted KE-DRL model.
from ke_drl.matern_kernel import matern_kernel


# ---------------------------
# plotting/font utilities
# ---------------------------
def set_safe_matplotlib_fonts() -> None:
    import logging
    import matplotlib.font_manager as fm

    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Liberation Serif", "Times New Roman"],
        "mathtext.fontset": "dejavuserif",
        "axes.unicode_minus": False,
    })

    if getattr(fm.findfont, "_safe_nimbus_patch", False):
        return
    orig_findfont = fm.findfont

    def safe_findfont(prop, *args, **kwargs):
        try:
            fam = prop.get_family() if hasattr(prop, "get_family") else []
            fam_list = fam if isinstance(fam, (list, tuple)) else [fam]
            if any("Nimbus Roman" in str(x) for x in fam_list):
                prop = fm.FontProperties(family=["DejaVu Serif"])
        except Exception:
            pass
        return orig_findfont(prop, *args, **kwargs)

    safe_findfont._safe_nimbus_patch = True
    fm.findfont = safe_findfont


# ---------------------------
# IO utilities
# ---------------------------
def as_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_csv_array(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def save_csv(path: Path, arr, header: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = as_numpy(arr)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    h = "" if header is None else ",".join(map(str, header))
    np.savetxt(path, x, delimiter=",", header=h, comments="")


def read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ---------------------------
# kernel utilities
# ---------------------------
# Matérn kernel is imported directly from ke_drl.matern_kernel above.


# ---------------------------
# scale/support utilities
# ---------------------------
def denorm(Z_norm: np.ndarray, r_mu: np.ndarray, r_sd: np.ndarray) -> np.ndarray:
    return Z_norm * r_sd.reshape(1, -1) + r_mu.reshape(1, -1)


def zscore(Z_raw: np.ndarray, r_mu: np.ndarray, r_sd: np.ndarray) -> np.ndarray:
    return (Z_raw - r_mu.reshape(1, -1)) / (r_sd.reshape(1, -1) + 1e-12)


def parse_csv_list(s: Optional[str]) -> List[str]:
    if s is None or str(s).strip() == "":
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def parse_float_list(s: Optional[str], d: int, default: float) -> List[float]:
    if s is None or str(s).strip() == "":
        return [float(default)] * d
    vals = [float(x.strip()) for x in str(s).split(",") if x.strip()]
    if len(vals) == 1:
        vals = vals * d
    if len(vals) != d:
        raise ValueError(f"Expected one bandwidth or {d} bandwidths, got {vals}.")
    if any(v <= 0 for v in vals):
        raise ValueError(f"Bandwidths must be positive, got {vals}.")
    return vals


def infer_nonnegative(name: str) -> bool:
    s = name.lower()
    return any(k in s for k in ["revenue", "sales", "price", "click", "count", "booking", "amount"])


def support_adjust_raw(
    Z_raw: np.ndarray,
    reward_cols: Sequence[str],
    discrete_dims: Sequence[int],
    clip_nonnegative: bool = True,
) -> np.ndarray:
    Z = np.array(Z_raw, dtype=float, copy=True)
    for j, name in enumerate(reward_cols):
        if j in set(discrete_dims):
            Z[:, j] = np.round(Z[:, j])
            if clip_nonnegative:
                Z[:, j] = np.maximum(Z[:, j], 0.0)
        elif clip_nonnegative and infer_nonnegative(name):
            Z[:, j] = np.maximum(Z[:, j], 0.0)
    return Z


def pretty_label(name: str) -> str:
    mapping = {
        "gross_revenue_per_night": "Gross Revenue per Night",
        "total_clicks": "Total Clicks",
        "total_sales": "Total Sales",
    }
    return mapping.get(name, name.replace("_", " ").title())


# ---------------------------
# loading policy exports
# ---------------------------
def load_policy_exports(policy_dir: Path, args) -> dict:
    policy_dir = Path(policy_dir)
    artifacts_path = policy_dir / "artifacts.pt"
    plot_payload_path = policy_dir / "plot_payload.pt"
    metrics_path = policy_dir / "metrics.json"

    if not artifacts_path.exists():
        raise FileNotFoundError(f"Missing {artifacts_path}. Needed for r_mu/r_sd scaler.")

    artifacts = torch.load(artifacts_path, map_location="cpu", weights_only=False)
    norm = artifacts.get("normalization", {})
    if "r_mu" not in norm or "r_sd" not in norm:
        raise KeyError(f"{artifacts_path} does not contain normalization['r_mu'] and normalization['r_sd'].")
    r_mu = as_numpy(norm["r_mu"]).reshape(-1).astype(float)
    r_sd = as_numpy(norm["r_sd"]).reshape(-1).astype(float)

    metrics = read_json(metrics_path) if metrics_path.exists() else {}
    reward_cols = args.reward_cols or metrics.get("embedding_config", {}).get("reward_cols") or metrics.get("reward_cols")

    payload = None
    if plot_payload_path.exists():
        payload = torch.load(plot_payload_path, map_location="cpu", weights_only=False)
        reward_cols = reward_cols or payload.get("reward_cols")

    if reward_cols is None:
        # Last fallback: infer dim names.
        d = int(r_mu.size)
        reward_cols = [f"reward_{j}" for j in range(d)]
    reward_cols = list(reward_cols)

    beta = None
    Z_norm = None

    if payload is not None:
        if "beta" in payload:
            beta = as_numpy(payload["beta"]).reshape(-1).astype(float)
        # Use the optimized normalized dictionary matching beta_hat.
        for key in ["Zg_norm_optimized", "density_atoms_Z_grid_normalized_optimized", "Z_grid_normalized_optimized"]:
            if key in payload:
                Z_norm = as_numpy(payload[key]).astype(float)
                break

    if beta is None:
        beta_csv = policy_dir / "mean_embedding_coefficients_beta.csv"
        if not beta_csv.exists():
            raise FileNotFoundError(
                f"Missing beta. Expected {beta_csv} or plot_payload.pt with key 'beta'. "
                "Run policy_evaluation9.py with --do-plots 1 first."
            )
        beta = load_csv_array(beta_csv).reshape(-1).astype(float)

    if Z_norm is None:
        for fname in [
            "density_atoms_Z_grid_normalized_optimized.csv",
            "Z_grid_normalized_optimized.csv",
        ]:
            p = policy_dir / fname
            if p.exists():
                Z_norm = load_csv_array(p).astype(float)
                break

    if Z_norm is None:
        for key in ["Z_grid_normalized_optimized", "Z_grid"]:
            if key in artifacts:
                Z_norm = as_numpy(artifacts[key]).astype(float)
                break

    if Z_norm is None:
        raise FileNotFoundError(f"Could not locate normalized Z-grid in {policy_dir}.")

    if Z_norm.ndim != 2:
        raise ValueError(f"Z_norm must be 2D, got {Z_norm.shape}")
    if beta.size != Z_norm.shape[0]:
        raise ValueError(
            f"beta length {beta.size} does not match Z-grid rows {Z_norm.shape[0]} in {policy_dir}."
        )
    if Z_norm.shape[1] != r_mu.size:
        raise ValueError(
            f"Z-grid dimension {Z_norm.shape[1]} does not match r_mu dimension {r_mu.size}."
        )

    # User requested: construct raw grid using scaler, not by relying on exported raw grid.
    Z_raw_est = denorm(Z_norm, r_mu, r_sd)

    discrete_names = parse_csv_list(args.discrete_reward_cols)
    discrete_dims = [reward_cols.index(x) for x in discrete_names if x in reward_cols]
    Z_raw = support_adjust_raw(
        Z_raw_est,
        reward_cols=reward_cols,
        discrete_dims=discrete_dims,
        clip_nonnegative=bool(args.clip_nonnegative),
    )
    Z_norm_for_density = zscore(Z_raw, r_mu, r_sd)

    label = args.label
    if label is None:
        label = metrics.get("policy_name", policy_dir.name)

    return {
        "policy_dir": policy_dir,
        "label": str(label),
        "reward_cols": reward_cols,
        "beta_hat": beta,
        "Z_norm_dict": Z_norm,
        "Z_raw_est": Z_raw_est,
        "Z_raw": Z_raw,
        "Z_norm_for_density": Z_norm_for_density,
        "r_mu": r_mu,
        "r_sd": r_sd,
        "discrete_dims": discrete_dims,
        "metrics": metrics,
    }


# ---------------------------
# induced embedding optimization
# ---------------------------
def gaussian_quadrature_nodes(n: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    nodes_np, weights_np = np.polynomial.hermite.hermgauss(int(n))
    # For E[f(X)] where X ~ N(mu, h^2): X = mu + sqrt(2) h node, weight / sqrt(pi)
    nodes = torch.as_tensor(nodes_np, dtype=dtype, device=device)
    weights = torch.as_tensor(weights_np / math.sqrt(math.pi), dtype=dtype, device=device)
    return nodes, weights


def build_induced_A_matrix(
    Z_norm_dict: np.ndarray,
    Z_raw_atoms: np.ndarray,
    r_mu: np.ndarray,
    r_sd: np.ndarray,
    bandwidths_raw: Sequence[float],
    discrete_dims: Sequence[int],
    *,
    nu: float,
    ell: float,
    sigma: float,
    quad_points: int = 21,
    device: str = "cpu",
    batch_atoms: int = 64,
) -> torch.Tensor:
    """
    A_{ell,i} = E_{X_i}[ k(Z_norm_dict[ell], X_i_norm) ],
    where X_i is the smoothed atom distribution in raw coordinates.

    Continuous dims: Gaussian smoothing with bandwidth h_j.
    Discrete dims: fixed at the atom's discrete raw value.
    """
    dev = torch.device(device)
    dtype = torch.float64
    Zdict = torch.as_tensor(Z_norm_dict, dtype=dtype, device=dev)
    Zraw = torch.as_tensor(Z_raw_atoms, dtype=dtype, device=dev)
    mu = torch.as_tensor(r_mu, dtype=dtype, device=dev).reshape(1, -1)
    sd = torch.as_tensor(r_sd, dtype=dtype, device=dev).reshape(1, -1)
    bw = torch.as_tensor(bandwidths_raw, dtype=dtype, device=dev).reshape(1, -1)

    m, d = Zraw.shape
    cont_dims = [j for j in range(d) if j not in set(discrete_dims)]
    nodes, qweights = gaussian_quadrature_nodes(quad_points, dev, dtype)

    # Build product quadrature over continuous dimensions.
    # Expedia case usually has one continuous dimension, so this stays small.
    if len(cont_dims) == 0:
        offsets = torch.zeros((1, d), dtype=dtype, device=dev)
        weights = torch.ones(1, dtype=dtype, device=dev)
    else:
        meshes = torch.meshgrid(*([nodes] * len(cont_dims)), indexing="ij")
        wmeshes = torch.meshgrid(*([qweights] * len(cont_dims)), indexing="ij")
        offsets_small = torch.stack([x.reshape(-1) for x in meshes], dim=1)
        weights = torch.ones(offsets_small.shape[0], dtype=dtype, device=dev)
        for wm in wmeshes:
            weights = weights * wm.reshape(-1)
        offsets = torch.zeros((offsets_small.shape[0], d), dtype=dtype, device=dev)
        for k, j in enumerate(cont_dims):
            offsets[:, j] = math.sqrt(2.0) * offsets_small[:, k]

    Q = offsets.shape[0]
    if Q > 2000:
        raise ValueError(
            f"Product quadrature has {Q} points. Reduce --quad-points or number of continuous dims."
        )

    A_cols = []
    for start in range(0, m, int(batch_atoms)):
        end = min(m, start + int(batch_atoms))
        atoms = Zraw[start:end]  # b x d
        # samples: b x Q x d
        samples = atoms[:, None, :] + offsets[None, :, :] * bw[None, :, :]
        # keep discrete dims fixed exactly
        for j in discrete_dims:
            samples[:, :, j] = atoms[:, None, j]
        samples_norm = (samples - mu) / (sd + 1e-12)
        flat = samples_norm.reshape(-1, d)
        K_flat = matern_kernel(Zdict.float(), flat.float(), nu=nu, length_scale=ell, sigma=sigma).to(dtype)
        K_flat = K_flat.reshape(m, end - start, Q)
        A_block = torch.sum(K_flat * weights.reshape(1, 1, Q), dim=2)  # m x b
        A_cols.append(A_block.detach().cpu())
        del samples, samples_norm, flat, K_flat, A_block
    A = torch.cat(A_cols, dim=1)
    return A


def optimize_probability_weights(
    beta_hat: np.ndarray,
    Z_norm_dict: np.ndarray,
    A: torch.Tensor,
    *,
    nu: float,
    ell: float,
    sigma: float,
    ridge: float,
    lr: float,
    steps: int,
    tol: float,
    init: str,
    device: str,
    print_every: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], List[dict]]:
    dev = torch.device(device)
    dtype = torch.float64
    beta = torch.as_tensor(beta_hat, dtype=dtype, device=dev).reshape(-1)
    Zdict = torch.as_tensor(Z_norm_dict, dtype=dtype, device=dev)
    A = A.to(device=dev, dtype=dtype)
    m = beta.numel()
    K = matern_kernel(Zdict.float(), Zdict.float(), nu=nu, length_scale=ell, sigma=sigma).to(dtype)
    K = 0.5 * (K + K.T)
    K = K + float(ridge) * torch.eye(m, dtype=dtype, device=dev)

    # beta_tilde(w) = M w, M = (K + ridge I)^(-1) A
    M = torch.linalg.solve(K, A)

    if init == "abs_beta":
        init_w = torch.abs(beta) + 1e-8
        init_w = init_w / init_w.sum()
        theta = torch.log(init_w).clone().detach().requires_grad_(True)
    elif init == "positive_beta":
        init_w = torch.clamp(beta, min=0.0) + 1e-8
        init_w = init_w / init_w.sum()
        theta = torch.log(init_w).clone().detach().requires_grad_(True)
    else:
        theta = torch.zeros(m, dtype=dtype, device=dev, requires_grad=True)

    opt = torch.optim.Adam([theta], lr=float(lr))
    history: List[dict] = []
    prev_obj = None
    prev_w = None

    for t in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        w = torch.softmax(theta, dim=0)
        beta_tilde = M @ w
        diff = beta_tilde - beta
        obj = diff @ K @ diff
        obj.backward()
        opt.step()

        with torch.no_grad():
            obj_val = float(obj.detach().cpu().item())
            w_now = w.detach().clone()
            rel_change = math.inf if prev_obj is None else abs(prev_obj - obj_val) / max(1.0, abs(prev_obj))
            l1_w_change = math.inf if prev_w is None else float(torch.sum(torch.abs(w_now - prev_w)).cpu().item())
            if (t == 0) or ((t + 1) % int(print_every) == 0) or (t + 1 == int(steps)):
                print(
                    f"iter={t+1:6d} obj={obj_val:.8e} rel_change={rel_change:.3e} "
                    f"w_l1_change={l1_w_change:.3e} w_min={float(w_now.min()):.3e} w_max={float(w_now.max()):.3e}",
                    flush=True,
                )
            history.append({
                "iter": int(t + 1),
                "objective": obj_val,
                "relative_objective_change": float(rel_change),
                "w_l1_change": float(l1_w_change),
            })
            if prev_obj is not None and rel_change < float(tol) and l1_w_change < math.sqrt(float(tol)):
                break
            prev_obj = obj_val
            prev_w = w_now

    with torch.no_grad():
        w = torch.softmax(theta, dim=0)
        beta_tilde = M @ w
        diff = beta_tilde - beta
        obj = diff @ K @ diff
        target_norm_sq = beta @ K @ beta
        induced_norm_sq = beta_tilde @ K @ beta_tilde
        values_target = K @ beta
        values_induced = K @ beta_tilde
        values_diff = values_induced - values_target

        beta_np = beta.detach().cpu().numpy()
        beta_tilde_np = beta_tilde.detach().cpu().numpy()
        w_np = w.detach().cpu().numpy()
        val_t = values_target.detach().cpu().numpy()
        val_i = values_induced.detach().cpu().numpy()
        corr = float(np.corrcoef(val_t, val_i)[0, 1]) if np.std(val_t) > 1e-12 and np.std(val_i) > 1e-12 else float("nan")
        entropy = float(-(w * torch.log(w + 1e-30)).sum().cpu().item())
        diagnostics = {
            "objective_rkhs_sq": float(obj.cpu().item()),
            "objective_rkhs": float(torch.sqrt(torch.clamp(obj, min=0)).cpu().item()),
            "target_rkhs_norm_sq": float(target_norm_sq.cpu().item()),
            "target_rkhs_norm": float(torch.sqrt(torch.clamp(target_norm_sq, min=0)).cpu().item()),
            "induced_rkhs_norm_sq": float(induced_norm_sq.cpu().item()),
            "relative_rkhs_error": float(torch.sqrt(torch.clamp(obj, min=0) / torch.clamp(target_norm_sq, min=1e-30)).cpu().item()),
            "beta_rmse": float(np.sqrt(np.mean((beta_tilde_np - beta_np) ** 2))),
            "beta_l2": float(np.linalg.norm(beta_tilde_np - beta_np)),
            "beta_max_abs": float(np.max(np.abs(beta_tilde_np - beta_np))),
            "embedding_value_rmse_on_atoms": float(np.sqrt(np.mean(as_numpy(values_diff) ** 2))),
            "embedding_value_corr_on_atoms": corr,
            "w_sum": float(w_np.sum()),
            "w_min": float(w_np.min()),
            "w_max": float(w_np.max()),
            "w_entropy": entropy,
            "w_effective_n_exp_entropy": float(np.exp(entropy)),
            "w_effective_n_inverse_hhi": float(1.0 / np.sum(w_np**2)),
            "beta_hat_sum": float(beta_np.sum()),
            "beta_hat_min": float(beta_np.min()),
            "beta_hat_max": float(beta_np.max()),
            "beta_tilde_sum": float(beta_tilde_np.sum()),
            "beta_tilde_min": float(beta_tilde_np.min()),
            "beta_tilde_max": float(beta_tilde_np.max()),
        }
    return w_np, beta_tilde_np, diagnostics, history


# ---------------------------
# density / PMF construction
# ---------------------------
def continuous_kernel_matrix(grid: np.ndarray, atom_vals: np.ndarray, h: float) -> np.ndarray:
    h = max(float(h), 1e-12)
    u = (grid.reshape(-1, 1) - atom_vals.reshape(1, -1)) / h
    return np.exp(-0.5 * u * u) / (math.sqrt(2.0 * math.pi) * h)


def hard_discrete_allocation(atom_vals: np.ndarray, support: np.ndarray) -> np.ndarray:
    vals = atom_vals.reshape(-1)
    sup = support.reshape(-1)
    A = np.zeros((sup.size, vals.size), dtype=float)
    if vals.size == 0 or sup.size == 0:
        return A
    idx = np.abs(vals[:, None] - sup[None, :]).argmin(axis=1)
    A[idx, np.arange(vals.size)] = 1.0
    return A


def normalize_density(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.maximum(np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    if x.size > 1:
        area = float(np.trapezoid(y, x))
        if area > 0 and np.isfinite(area):
            y = y / area
    return y


def normalize_joint(z: np.ndarray, x: np.ndarray, y: np.ndarray, kind0: str, kind1: str) -> np.ndarray:
    z = np.maximum(np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    area = 0.0
    if kind0 == "continuous" and kind1 == "continuous":
        area = float(np.trapezoid(np.trapezoid(z, y, axis=1), x)) if x.size > 1 and y.size > 1 else 0.0
    elif kind0 == "continuous" and kind1 == "discrete":
        area = float(np.trapezoid(z.sum(axis=1), x)) if x.size > 1 else 0.0
    elif kind0 == "discrete" and kind1 == "continuous":
        area = float(np.trapezoid(z.sum(axis=0), y)) if y.size > 1 else 0.0
    else:
        area = float(z.sum())
    if area > 0 and np.isfinite(area):
        z = z / area
    return z


def build_recovery_payload(
    Z_raw: np.ndarray,
    w: np.ndarray,
    reward_cols: Sequence[str],
    discrete_dims: Sequence[int],
    bandwidths_raw: Sequence[float],
    num_points: int,
) -> dict:
    m, d = Z_raw.shape
    w = np.asarray(w, dtype=float).reshape(-1)
    w = np.maximum(w, 0.0)
    w = w / max(w.sum(), 1e-300)

    one_d = {}
    A_mats = []
    grids = []
    kinds = []

    for j in range(d):
        vals = Z_raw[:, j].astype(float)
        name = reward_cols[j]
        if j in set(discrete_dims):
            support = np.unique(np.round(vals).astype(int)).astype(float)
            support = support[support >= 0] if infer_nonnegative(name) else support
            if support.size == 0:
                support = np.asarray([0.0])
            A = hard_discrete_allocation(vals, support)
            pmf = A @ w
            pmf = np.maximum(pmf, 0.0)
            pmf = pmf / max(pmf.sum(), 1e-300)
            one_d[str(j)] = {"kind": "discrete", "grid": support, "density": pmf, "ylabel": "Probability"}
            A_mats.append(A)
            grids.append(support)
            kinds.append("discrete")
        else:
            h = float(bandwidths_raw[j])
            lo = float(np.nanmin(vals) - 4.0 * h)
            hi = float(np.nanmax(vals) + 4.0 * h)
            if infer_nonnegative(name):
                lo = max(0.0, lo)
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = 0.0, 1.0
            grid = np.linspace(lo, hi, int(num_points))
            A = continuous_kernel_matrix(grid, vals, h)
            dens = A @ w
            dens = normalize_density(dens, grid)
            one_d[str(j)] = {"kind": "continuous", "grid": grid, "density": dens, "ylabel": "Density", "bandwidth": h}
            A_mats.append(A)
            grids.append(grid)
            kinds.append("continuous")

    joint = None
    if d >= 2:
        A0, A1 = A_mats[0], A_mats[1]
        x, y = grids[0], grids[1]
        z = (A0 * w.reshape(1, -1)) @ A1.T
        z = normalize_joint(z, x, y, kinds[0], kinds[1])
        joint = {"x": x, "y": y, "z": z, "kind0": kinds[0], "kind1": kinds[1]}

    return {"one_d": one_d, "joint": joint, "weights": w}


# ---------------------------
# plots
# ---------------------------
def plot_individual(payload: dict, label: str, reward_cols: Sequence[str], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    one_d = payload["one_d"]
    for key, item in one_d.items():
        j = int(key)
        grid = np.asarray(item["grid"], dtype=float)
        y = np.asarray(item["density"], dtype=float)
        fig = plt.figure()
        if item["kind"] == "discrete":
            plt.bar(grid, y, width=0.45, alpha=0.75)
            plt.ylabel("Probability")
            title = f"{label}: recovered PMF of {pretty_label(reward_cols[j])}"
        else:
            plt.plot(grid, y, linewidth=2.5)
            plt.ylabel("Density")
            title = f"{label}: recovered density of {pretty_label(reward_cols[j])}"
        plt.xlabel(pretty_label(reward_cols[j]))
        plt.title(title)
        p = out_dir / f"{label}_marginal_dim{j}_{reward_cols[j]}.png"
        fig.savefig(p, bbox_inches="tight",dpi=700)
        plt.close(fig)
        paths[f"marginal_dim{j}"] = str(p)

    joint = payload.get("joint")
    if joint is not None:
        x, y, z = joint["x"], joint["y"], joint["z"]
        X, Y = np.meshgrid(y, x)
        fig = plt.figure()
        if joint["kind1"] == "discrete" or joint["kind0"] == "discrete":
            plt.pcolormesh(X, Y, z, shading="auto")
            plt.colorbar(label="Mixed density / probability")
        else:
            plt.contourf(X, Y, z, levels=20)
            plt.colorbar(label="Density")
        plt.xlabel(pretty_label(reward_cols[1]))
        plt.ylabel(pretty_label(reward_cols[0]))
        if joint["kind1"] == "discrete":
            plt.xticks(y)
        if joint["kind0"] == "discrete":
            plt.yticks(x)
        plt.title(f"{label}: recovered mixed joint display")
        p = out_dir / f"{label}_joint_heatmap.png"
        fig.savefig(p, bbox_inches="tight",dpi=700)
        plt.close(fig)
        paths["joint_heatmap"] = str(p)

        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(X, Y, z, alpha=0.75, linewidth=0, antialiased=True)
            ax.set_xlabel(pretty_label(reward_cols[1]))
            ax.set_ylabel(pretty_label(reward_cols[0]))
            ax.set_zlabel("Recovered display value")
            ax.set_title(f"{label}: recovered joint surface")
            p = out_dir / f"{label}_joint_surface3d.png"
            fig.savefig(p, bbox_inches="tight",dpi=700)
            plt.close(fig)
            paths["joint_surface3d"] = str(p)
        except Exception:
            pass
    return paths


def plot_overlay(payloads: List[dict], labels: List[str], reward_cols: Sequence[str], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    colors = ["#13294B", "#FF5F05", "#2ca02c", "#9467bd"]
    linestyles = ["-", "--", "-.", ":"]

    d = len(reward_cols)
    for j in range(d):
        fig = plt.figure()
        kind = payloads[0]["one_d"][str(j)]["kind"]
        if kind == "discrete":
            # align supports by plotting side-by-side bars
            all_support = np.unique(np.concatenate([p["one_d"][str(j)]["grid"] for p in payloads]))
            width = 0.8 / max(1, len(payloads))
            for k, (payload, label) in enumerate(zip(payloads, labels)):
                item = payload["one_d"][str(j)]
                vals = np.zeros_like(all_support, dtype=float)
                idx_map = {float(x): i for i, x in enumerate(all_support)}
                for gx, px in zip(item["grid"], item["density"]):
                    vals[idx_map[float(gx)]] = px
                offset = (k - (len(payloads) - 1) / 2.0) * width
                plt.bar(all_support + offset, vals, width=width, alpha=0.7, label=label, color=colors[k % len(colors)])
            plt.ylabel("Probability")
            title = f"Overlay recovered PMF: {pretty_label(reward_cols[j])}"
        else:
            for k, (payload, label) in enumerate(zip(payloads, labels)):
                item = payload["one_d"][str(j)]
                plt.plot(
                    item["grid"], item["density"],
                    linewidth=2.5,
                    linestyle=linestyles[k % len(linestyles)],
                    color=colors[k % len(colors)],
                    label=label,
                )
            plt.ylabel("Density")
            title = f"Overlay recovered density: {pretty_label(reward_cols[j])}"
        plt.xlabel(pretty_label(reward_cols[j]))
        plt.title(title)
        plt.legend(frameon=True)
        p = out_dir / f"overlay_marginal_dim{j}_{reward_cols[j]}.png"
        fig.savefig(p, bbox_inches="tight",dpi=700)
        plt.close(fig)
        paths[f"overlay_marginal_dim{j}"] = str(p)

    if all(p.get("joint") is not None for p in payloads):
        fig = plt.figure()
        for k, (payload, label) in enumerate(zip(payloads, labels)):
            joint = payload["joint"]
            x, y, z = joint["x"], joint["y"], joint["z"]
            X, Y = np.meshgrid(y, x)
            zmax = float(np.nanmax(z)) if z.size else 0.0
            if zmax <= 0:
                continue
            levels = np.linspace(0.1 * zmax, 0.9 * zmax, 7)
            plt.contour(
                X, Y, z,
                levels=levels,
                colors=[colors[k % len(colors)]],
                linestyles=linestyles[k % len(linestyles)],
                linewidths=2.0,
                alpha=0.95,
            )
            # dummy handle
            plt.plot([], [], color=colors[k % len(colors)], linestyle=linestyles[k % len(linestyles)], label=label)
        plt.xlabel(pretty_label(reward_cols[1]))
        plt.ylabel(pretty_label(reward_cols[0]))
        if payloads[0]["joint"]["kind1"] == "discrete":
            plt.xticks(payloads[0]["joint"]["y"])
        if payloads[0]["joint"]["kind0"] == "discrete":
            plt.yticks(payloads[0]["joint"]["x"])
        plt.title("Overlay recovered mixed joint contours")
        plt.legend(frameon=True)
        p = out_dir / "overlay_joint_contours.png"
        fig.savefig(p, bbox_inches="tight",dpi=700)
        plt.close(fig)
        paths["overlay_joint_contours"] = str(p)

        # ------------------------------------------------------------
        # Overlay heatmap panel: one heatmap per policy on its own grid.
        # This avoids forcing interpolation when one reward dimension is
        # discrete and the other is continuous.
        # ------------------------------------------------------------
        n_pol = len(payloads)
        fig, axes = plt.subplots(1, n_pol, figsize=(6.8 * n_pol, 5.6), squeeze=False, constrained_layout=False)
        vmax = max(float(np.nanmax(p["joint"]["z"])) for p in payloads if p.get("joint") is not None)
        for k, (payload, label) in enumerate(zip(payloads, labels)):
            joint = payload["joint"]
            x, y, z = joint["x"], joint["y"], joint["z"]
            X, Y = np.meshgrid(y, x)
            ax = axes[0, k]
            im = ax.pcolormesh(X, Y, z, shading="auto", vmin=0.0, vmax=vmax if vmax > 0 else None)
            ax.set_xlabel(pretty_label(reward_cols[1]))
            ax.set_ylabel(pretty_label(reward_cols[0]))
            if joint["kind1"] == "discrete":
                ax.set_xticks(y)
            if joint["kind0"] == "discrete":
                ax.set_yticks(x)
            ax.set_title(f"{label}: recovered joint heatmap")
            fig.colorbar(im, ax=ax, label="Recovered mixed density / probability")
        fig.suptitle("Recovered joint heatmaps by policy")
        p = out_dir / "overlay_joint_heatmaps_by_policy.png"
        fig.subplots_adjust(left=0.06, right=0.97, bottom=0.10, top=0.88, wspace=0.25)
        fig.savefig(p, bbox_inches="tight", pad_inches=0.25, dpi=700)
        plt.close(fig)
        paths["overlay_joint_heatmaps_by_policy"] = str(p)

        # ------------------------------------------------------------
        # Overlay 3D surface: draw both policy surfaces in one 3D axis.
        # Each surface can use its own grid; no interpolation is required.
        # ------------------------------------------------------------
        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            fig = plt.figure(figsize=(12.6, 8.8), constrained_layout=False)
            ax = fig.add_subplot(111, projection="3d")
            ax.set_proj_type("ortho")
            from matplotlib.patches import Patch
            handles = []
            for k, (payload, label) in enumerate(zip(payloads, labels)):
                joint = payload["joint"]
                x, y, z = joint["x"], joint["y"], joint["z"]
                X, Y = np.meshgrid(y, x)
                color = colors[k % len(colors)]
                z_plot = np.asarray(z, dtype=float).copy()
                z_plot[~np.isfinite(z_plot)] = np.nan
                zmax_local = float(np.nanmax(z_plot)) if z_plot.size else 0.0
                if zmax_local > 0:
                    z_plot[z_plot <= max(1e-12, 1e-6 * zmax_local)] = np.nan
                ax.plot_surface(X, Y, z_plot, color=color, alpha=0.48, linewidth=0, antialiased=True)
                handles.append(Patch(facecolor=color, edgecolor=color, alpha=0.48, label=label))
            ax.set_xlabel(pretty_label(reward_cols[1]), labelpad=14)
            ax.set_ylabel(pretty_label(reward_cols[0]), labelpad=16)
            ax.set_zlabel("Recovered mixed density / probability", labelpad=16)
            ax.set_box_aspect((1.60, 1.05, 0.55))
            ax.view_init(elev=24, azim=-55)
            if payloads[0]["joint"]["kind1"] == "discrete":
                ax.set_xticks(payloads[0]["joint"]["y"])
            if payloads[0]["joint"]["kind0"] == "discrete":
                ax.set_yticks(payloads[0]["joint"]["x"])
            ax.set_title("Overlay recovered joint surface")
            ax.legend(handles=handles, loc="upper right")
            p = out_dir / "overlay_joint_surface3d.png"
            fig.subplots_adjust(left=0.03, right=0.96, bottom=0.08, top=0.92)
            fig.savefig(p, bbox_inches="tight", pad_inches=0.35, dpi=700)
            plt.close(fig)
            paths["overlay_joint_surface3d"] = str(p)
        except Exception as e:
            print(f"Warning: overlay 3D surface failed: {e}", flush=True)

        if len(payloads) == 2:
            A, B = payloads[0]["joint"], payloads[1]["joint"]
            # If grids match, plot difference heatmap.
            if np.array_equal(A["x"], B["x"]) and np.array_equal(A["y"], B["y"]):
                diff = A["z"] - B["z"]
                X, Y = np.meshgrid(A["y"], A["x"])
                fig = plt.figure()
                plt.pcolormesh(X, Y, diff, shading="auto")
                plt.colorbar(label=f"{labels[0]} - {labels[1]}")
                plt.xlabel(pretty_label(reward_cols[1]))
                plt.ylabel(pretty_label(reward_cols[0]))
                plt.title("Difference of recovered joint displays")
                if A["kind1"] == "discrete":
                    plt.xticks(A["y"])
                if A["kind0"] == "discrete":
                    plt.yticks(A["x"])
                p = out_dir / "overlay_joint_difference_heatmap.png"
                fig.savefig(p, bbox_inches="tight",dpi=700)
                plt.close(fig)
                paths["overlay_joint_difference_heatmap"] = str(p)
    return paths


# ---------------------------
# high-level recovery function
# ---------------------------
def recovery_plot(policy_dir: Path, args, out_dir: Path, label: Optional[str] = None) -> dict:
    data = load_policy_exports(policy_dir, args)
    if label is not None:
        data["label"] = label
    label = data["label"]
    reward_cols = data["reward_cols"]
    d = len(reward_cols)
    bandwidths_raw = parse_float_list(args.bandwidth_per_dim, d, args.bandwidth)

    print("\n" + "=" * 90)
    print(f"RECOVERY FOR: {label}")
    print("=" * 90)
    print(f"policy_dir     : {data['policy_dir']}")
    print(f"reward_cols    : {reward_cols}")
    print(f"discrete_dims  : {data['discrete_dims']}")
    print(f"Z shape        : {data['Z_norm_dict'].shape}")
    print(f"beta shape     : {data['beta_hat'].shape}")
    print(f"bandwidth_raw  : {bandwidths_raw}")

    A = build_induced_A_matrix(
        Z_norm_dict=data["Z_norm_dict"],
        Z_raw_atoms=data["Z_raw"],
        r_mu=data["r_mu"],
        r_sd=data["r_sd"],
        bandwidths_raw=bandwidths_raw,
        discrete_dims=data["discrete_dims"],
        nu=args.nu_Z,
        ell=args.ell_Z,
        sigma=args.sigma_Z,
        quad_points=args.quad_points,
        device=args.device,
        batch_atoms=args.batch_atoms,
    )

    w, beta_tilde, diagnostics, history = optimize_probability_weights(
        beta_hat=data["beta_hat"],
        Z_norm_dict=data["Z_norm_dict"],
        A=A,
        nu=args.nu_Z,
        ell=args.ell_Z,
        sigma=args.sigma_Z,
        ridge=args.ridge,
        lr=args.lr,
        steps=args.steps,
        tol=args.tol,
        init=args.init,
        device=args.device,
        print_every=args.print_every,
    )

    print("\nEmbedding fit diagnostics")
    for k, v in diagnostics.items():
        print(f"  {k:35s}: {v}")

    payload = build_recovery_payload(
        Z_raw=data["Z_raw"],
        w=w,
        reward_cols=reward_cols,
        discrete_dims=data["discrete_dims"],
        bandwidths_raw=bandwidths_raw,
        num_points=args.num_points,
    )

    policy_out = out_dir / label.replace("/", "_").replace(" ", "_")
    paths = plot_individual(payload, label, reward_cols, policy_out)

    # exports
    save_csv(policy_out / "Z_grid_normalized_used.csv", data["Z_norm_dict"], reward_cols)
    save_csv(policy_out / "Z_grid_raw_constructed_from_scaler.csv", data["Z_raw_est"], reward_cols)
    save_csv(policy_out / "Z_grid_raw_used_for_density.csv", data["Z_raw"], reward_cols)
    save_csv(policy_out / "beta_hat.csv", data["beta_hat"], ["beta_hat"])
    save_csv(policy_out / "beta_tilde_induced.csv", beta_tilde, ["beta_tilde_induced"])
    save_csv(policy_out / "density_weights_induced_embedding_match.csv", w, ["density_weight"])
    atom_table = np.column_stack([np.arange(w.size), data["beta_hat"], beta_tilde, w, data["Z_norm_dict"], data["Z_raw"]])
    header = ["atom_index", "beta_hat", "beta_tilde", "density_weight"]
    header += [f"Z_norm_{c}" for c in reward_cols]
    header += [f"Z_raw_{c}" for c in reward_cols]
    save_csv(policy_out / "atom_recovery_table.csv", atom_table, header)
    write_json(policy_out / "embedding_fit_diagnostics.json", diagnostics)
    write_json(policy_out / "optimization_history.json", {"history": history})

    return {
        "label": label,
        "policy_dir": str(policy_dir),
        "reward_cols": reward_cols,
        "payload": payload,
        "diagnostics": diagnostics,
        "paths": paths,
    }



def discover_policy_dirs(input_root: Path, policy_dirs: Optional[Sequence[str]] = None) -> List[Path]:
    """Resolve policy directories from plotting_files.

    Supported layouts:
      1. plotting_files/<policy_a>/artifacts.pt, plotting_files/<policy_b>/artifacts.pt
      2. plotting_files/artifacts.pt for a single-policy run
      3. Explicit --policy-dirs can be absolute paths or names relative to --input-root.
    """
    input_root = Path(input_root)
    if policy_dirs:
        out = []
        for item in policy_dirs:
            p = Path(item)
            if not p.is_absolute():
                p2 = input_root / p
                p = p2 if p2.exists() else p
            out.append(p)
        return out

    if (input_root / "artifacts.pt").exists():
        return [input_root]

    if not input_root.exists():
        raise FileNotFoundError(
            f"Input folder {input_root} does not exist. Create it or pass --input-root."
        )

    dirs = sorted([p for p in input_root.iterdir() if p.is_dir() and (p / "artifacts.pt").exists()])
    if not dirs:
        raise FileNotFoundError(
            f"No policy folders found under {input_root}. Expected subfolders containing artifacts.pt."
        )
    return dirs

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot KE-DRL recovered distributions via induced-embedding weight matching.")
    parser.add_argument("--input-root", type=str, default="plotting_files", help="Folder containing policy plotting files/subfolders.")
    parser.add_argument("--policy-dirs", nargs="+", default=None, help="Optional policy folders; absolute paths or names relative to --input-root.")
    parser.add_argument("--labels", type=str, default=None, help="Comma-separated labels, same length as resolved policy folders.")
    parser.add_argument("--label", type=str, default=None, help="Optional single-policy label fallback.")
    parser.add_argument("--out-dir", type=str, default="recovery_plots_induced")
    parser.add_argument("--reward-cols", type=lambda s: parse_csv_list(s), default=None)
    parser.add_argument("--discrete-reward-cols", type=str, default="total_clicks")
    parser.add_argument("--clip-nonnegative", type=int, default=1)

    parser.add_argument("--nu-Z", type=float, default=3.5)
    parser.add_argument("--ell-Z", type=float, default=0.8)
    parser.add_argument("--sigma-Z", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1e-4, help="Ridge in beta_tilde=(K+ridge I)^(-1)Aw.")

    parser.add_argument("--bandwidth", type=float, default=40.0)
    parser.add_argument("--bandwidth-per-dim", type=str, default="40,1", help="Raw-scale bandwidths, e.g. '40,1'.")
    parser.add_argument("--quad-points", type=int, default=31)
    parser.add_argument("--batch-atoms", type=int, default=64)
    parser.add_argument("--num-points", type=int, default=500)

    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--init", type=str, default="positive_beta", choices=["uniform", "abs_beta", "positive_beta"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--print-every", type=int, default=500)

    args = parser.parse_args()
    set_safe_matplotlib_fonts()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    policy_dirs = discover_policy_dirs(Path(args.input_root), args.policy_dirs)

    labels = parse_csv_list(args.labels) if args.labels else []
    if labels and len(labels) != len(policy_dirs):
        raise ValueError("--labels must have same length as resolved policy folders.")
    if not labels:
        labels = [Path(p).name for p in policy_dirs]

    print("Resolved policy folders:")
    for p, lab in zip(policy_dirs, labels):
        print(f"  {lab}: {p}")

    results = []
    for p, lab in zip(policy_dirs, labels):
        res = recovery_plot(Path(p), args, out_dir, label=lab)
        results.append(res)

    overlay_paths = {}
    if len(results) >= 2:
        # Require same reward names/order for overlay.
        reward_cols = results[0]["reward_cols"]
        if any(r["reward_cols"] != reward_cols for r in results):
            raise ValueError("All policies must have same reward_cols for overlay plots.")
        overlay_paths = plot_overlay(
            payloads=[r["payload"] for r in results],
            labels=[r["label"] for r in results],
            reward_cols=reward_cols,
            out_dir=out_dir / "overlay",
        )

    summary = {
        "individual": [
            {
                "label": r["label"],
                "policy_dir": r["policy_dir"],
                "diagnostics": r["diagnostics"],
                "paths": r["paths"],
            }
            for r in results
        ],
        "overlay_paths": overlay_paths,
    }
    write_json(out_dir / "recovery_plot_summary.json", summary)
    print(f"\nSaved recovery summary: {out_dir / 'recovery_plot_summary.json'}")


if __name__ == "__main__":
    main()
