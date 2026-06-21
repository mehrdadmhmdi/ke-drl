#!/usr/bin/env python3
"""
Direct reward-weighted linear-Gaussian target-policy construction.

This script intentionally does not use d3rlpy, IQL, neural policies, critic action
selection, or candidate action banks.  For each reward coordinate c it fits

    A^std | S^std=s ~ N( s @ Theta_mu,c + epsilon_mu,c,
                         diag(exp(s @ Theta_sigma,c + epsilon_sigma,c)^2) )

by reward-weighted Gaussian negative log likelihood on the logged state-action
pairs.  The saved artifacts are converted back to raw-scale state/action
coefficients so policy_evaluation.py can load them as the usual
linear_gaussian_policy_<reward>.npz/json files.

The fit is intentionally conservative by default: reward weighting can move the
target policy toward high-reward logged actions, but overlap controls keep the
result close to the logged action support so the downstream direct ratio
estimator is not asked to extrapolate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from expedia_preprocessing import fit_state_encoder, state_encoder_from_metadata


# -----------------------------------------------------------------------------
# plotting defaults
# -----------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Liberation Serif", "Times New Roman"],
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.unicode_minus": False,
})

DATA_COLOR = "#2E8B57"
REV_COLOR = "#13294B"
CLICK_COLOR = "#FF5F05"
AUX_COLOR = "#9467BD"


def pretty_series_label(name: str) -> str:
    s = str(name).strip().lower()
    if s == "data":
        return "data"
    if "click" in s:
        return "click-focused policy"
    if "revenue" in s or "sales" in s:
        return "revenue-focused policy"
    return str(name).replace("_", " ").title()


def series_color(name: str) -> str:
    s = str(name).strip().lower()
    if s == "data":
        return DATA_COLOR
    if "click" in s:
        return CLICK_COLOR
    if "revenue" in s or "sales" in s:
        return REV_COLOR
    return AUX_COLOR


def reward_suggests_click_like(name: str) -> bool:
    s = str(name).lower()
    return any(k in s for k in ["click", "booking", "count", "visit", "impression"])


# -----------------------------------------------------------------------------
# generic IO/utilities
# -----------------------------------------------------------------------------
def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> str:
    if str(device_str).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    return str(device_str)


def parse_csv_list(x: Optional[str]) -> Optional[List[str]]:
    if x is None:
        return None
    x = str(x).strip()
    if x == "":
        return None
    return [c.strip() for c in x.split(",") if c.strip()]


def detect_key(d: dict, candidates: Sequence[str]) -> Optional[str]:
    for k in candidates:
        if k in d:
            return k
    return None


def normalize_blob_payload(blob: dict) -> dict:
    if not isinstance(blob, dict):
        raise TypeError(f"Loaded blob must be a dict, got {type(blob)}")
    if any(k in blob for k in ["s0", "a0", "r0", "r", "s1"]):
        return blob
    if "data" in blob and isinstance(blob["data"], dict):
        out = dict(blob["data"])
        meta = blob.get("meta", {})
        if isinstance(meta, dict):
            for k, v in meta.items():
                if k not in out:
                    out[k] = v
        return out
    raise KeyError(
        "Blob is neither flat nor nested in {'data','meta'} format. "
        f"Top-level keys are: {list(blob.keys())}"
    )


def load_blob(path: Path) -> dict:
    print(f"Loading {path}")
    return normalize_blob_payload(torch.load(path, map_location="cpu"))


def resolve_blob_path(blob_arg: Optional[str], data_base: Optional[str], default_name: Optional[str] = None) -> Optional[Path]:
    blob_name = blob_arg if blob_arg is not None else default_name
    if blob_name is None:
        return None
    p = Path(blob_name)
    if p.is_absolute() or p.exists() or data_base is None:
        return p
    return Path(data_base) / p


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_2d_tensor(blob: dict, key: str) -> torch.Tensor:
    if key not in blob:
        raise KeyError(f"Missing key '{key}'")
    x = blob[key]
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if x.ndim == 1:
        x = x.unsqueeze(1)
    if x.ndim != 2:
        raise ValueError(f"{key} must be 1D or 2D, got shape={tuple(x.shape)}")
    return x.float()


def default_names_for_blob_section(blob: dict, tensor_key: str, names_key: str) -> List[str]:
    x = get_2d_tensor(blob, tensor_key)
    names = blob.get(names_key, None)
    if isinstance(names, (list, tuple)):
        if len(names) != x.shape[1]:
            raise ValueError(
                f"{names_key} has length {len(names)} but {tensor_key} has {x.shape[1]} columns."
            )
        return [str(v) for v in names]
    return [f"{tensor_key}_{j}" for j in range(x.shape[1])]


def select_named_columns(
    blob: dict,
    tensor_key: str,
    names_key: str,
    wanted_cols: Optional[List[str]],
    role_name: str,
) -> Tuple[torch.Tensor, List[str]]:
    x = get_2d_tensor(blob, tensor_key)
    names = default_names_for_blob_section(blob, tensor_key, names_key)
    if wanted_cols is None:
        return x.float(), names
    name_to_idx = {name: j for j, name in enumerate(names)}
    missing = [c for c in wanted_cols if c not in name_to_idx]
    if missing:
        raise ValueError(
            f"Requested {role_name} cols not found: {missing}\n"
            f"Available {role_name} cols are: {sorted(names)}"
        )
    idx = [name_to_idx[c] for c in wanted_cols]
    return x[:, idx].float(), list(wanted_cols)


def maybe_subsample_arrays(
    obs: np.ndarray,
    act: np.ndarray,
    rew: np.ndarray,
    max_n: Optional[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = obs.shape[0]
    if max_n is None or max_n <= 0 or max_n >= n:
        return obs, act, rew
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    idx.sort()
    return obs[idx], act[idx], rew[idx]


def maybe_load_split(
    path: Optional[Path],
    state_cols: Optional[List[str]],
    action_cols: Optional[List[str]],
    reward_cols: Optional[List[str]],
    max_n: Optional[int],
    seed: int,
) -> Optional[Dict[str, np.ndarray]]:
    if path is None:
        return None
    blob = load_blob(path)
    s_raw, raw_state_names = select_named_columns(blob, "s0", "state_cols", state_cols, "state")
    a, action_names = select_named_columns(blob, "a0", "action_cols", action_cols, "action")
    reward_key = "r0" if "r0" in blob else "r"
    r, reward_names = select_named_columns(blob, reward_key, "reward_cols", reward_cols, "reward")

    obs_raw = s_raw.detach().cpu().numpy().astype(np.float64)
    act = a.detach().cpu().numpy().astype(np.float64)
    rew = r.detach().cpu().numpy().astype(np.float64)
    obs_raw, act, rew = maybe_subsample_arrays(obs_raw, act, rew, max_n, seed)
    return {
        "obs_raw": obs_raw,
        "act": act,
        "rew": rew,
        "raw_state_names": raw_state_names,
        "action_names": action_names,
        "reward_names": reward_names,
    }


def apply_state_encoder_to_split(split: Dict[str, np.ndarray], encoder) -> Dict[str, np.ndarray]:
    out = dict(split)
    out["obs"] = encoder.transform(split["obs_raw"]).astype(np.float64)
    out["state_names"] = list(encoder.encoded_state_names)
    out["state_encoder_diagnostics"] = encoder.diagnostics(split["obs_raw"])
    return out


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# -----------------------------------------------------------------------------
# policy representation
# -----------------------------------------------------------------------------
@dataclass
class LinearGaussianPolicy:
    theta_mu: np.ndarray
    epsilon_mu: np.ndarray
    theta_sigma: np.ndarray
    epsilon_sigma: np.ndarray
    action_lows: np.ndarray
    action_highs: np.ndarray
    action_names: List[str]
    state_names: List[str]
    reward_name: str
    integer_idx: Optional[int] = None
    integer_low: Optional[int] = None
    integer_high: Optional[int] = None
    integer_name: Optional[str] = None


def infer_integer_idx(action_names: List[str], integer_action_col: Optional[str]) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[str]]:
    if integer_action_col is None or str(integer_action_col).strip().lower() in {"", "none", "no", "false", "0"}:
        return None, None, None, None
    if integer_action_col not in action_names:
        raise ValueError(f"integer action col '{integer_action_col}' not found in {action_names}")
    return action_names.index(integer_action_col), None, None, integer_action_col


def mean_action(policy: LinearGaussianPolicy, s: np.ndarray, clipped: bool = False) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    mu = s @ policy.theta_mu + policy.epsilon_mu
    if clipped:
        return clip_to_support(policy, mu)
    return mu


def std_action(policy: LinearGaussianPolicy, s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    log_std = s @ policy.theta_sigma + policy.epsilon_sigma
    return np.exp(np.clip(log_std, -50.0, 50.0))


def clip_to_support(policy: LinearGaussianPolicy, a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).copy()
    return np.clip(a, policy.action_lows, policy.action_highs)


def round_integer_action(policy: LinearGaussianPolicy, a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).copy()
    if policy.integer_idx is not None:
        j = int(policy.integer_idx)
        a[..., j] = np.round(a[..., j])
        if policy.integer_low is not None:
            a[..., j] = np.maximum(a[..., j], policy.integer_low)
        if policy.integer_high is not None:
            a[..., j] = np.minimum(a[..., j], policy.integer_high)
    return a


def clip_and_round(policy: LinearGaussianPolicy, a: np.ndarray) -> np.ndarray:
    return round_integer_action(policy, clip_to_support(policy, a))


def greedy_action(policy: LinearGaussianPolicy, s: np.ndarray) -> np.ndarray:
    return clip_and_round(policy, mean_action(policy, s, clipped=False))


def sample_action(policy: LinearGaussianPolicy, s: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    mu = mean_action(policy, s, clipped=True)
    sd = std_action(policy, s)
    a = mu + rng.standard_normal(mu.shape) * sd
    return clip_and_round(policy, a)


def save_linear_gaussian_policy(policy: LinearGaussianPolicy, npz_path: Path, json_path: Path) -> None:
    ensure_dir(npz_path.parent)
    np.savez_compressed(
        npz_path,
        theta_mu=policy.theta_mu,
        epsilon_mu=policy.epsilon_mu,
        theta_sigma=policy.theta_sigma,
        epsilon_sigma=policy.epsilon_sigma,
        action_lows=policy.action_lows,
        action_highs=policy.action_highs,
        integer_idx=-1 if policy.integer_idx is None else int(policy.integer_idx),
        integer_low=-1 if policy.integer_low is None else int(policy.integer_low),
        integer_high=-1 if policy.integer_high is None else int(policy.integer_high),
    )
    meta = {
        "policy_type": "linear_gaussian",
        "training_method": "direct_reward_weighted_gaussian_mle",
        "reward_name": policy.reward_name,
        "state_names": policy.state_names,
        "action_names": policy.action_names,
        "state_dim": len(policy.state_names),
        "action_dim": len(policy.action_names),
        "integer_action_index": policy.integer_idx,
        "integer_action_low": policy.integer_low,
        "integer_action_high": policy.integer_high,
        "integer_action_name": policy.integer_name,
        "action_lows": policy.action_lows.tolist(),
        "action_highs": policy.action_highs.tolist(),
    }
    json_path.write_text(json.dumps(meta, indent=2))


def load_linear_gaussian_policy(npz_path: str | Path, json_path: str | Path) -> LinearGaussianPolicy:
    arr = np.load(npz_path)
    meta = json.loads(Path(json_path).read_text())

    def maybe_none(v):
        if v is None:
            return None
        iv = int(v)
        return None if iv < 0 else iv

    return LinearGaussianPolicy(
        theta_mu=np.asarray(arr["theta_mu"], dtype=np.float64),
        epsilon_mu=np.asarray(arr["epsilon_mu"], dtype=np.float64),
        theta_sigma=np.asarray(arr["theta_sigma"], dtype=np.float64),
        epsilon_sigma=np.asarray(arr["epsilon_sigma"], dtype=np.float64),
        action_lows=np.asarray(arr["action_lows"], dtype=np.float64),
        action_highs=np.asarray(arr["action_highs"], dtype=np.float64),
        action_names=list(meta["action_names"]),
        state_names=list(meta["state_names"]),
        reward_name=str(meta["reward_name"]),
        integer_idx=maybe_none(meta.get("integer_action_index")),
        integer_low=maybe_none(meta.get("integer_action_low")),
        integer_high=maybe_none(meta.get("integer_action_high")),
        integer_name=meta.get("integer_action_name"),
    )


# -----------------------------------------------------------------------------
# direct weighted Gaussian MLE
# -----------------------------------------------------------------------------
def weighted_ridge_mean_init(X: np.ndarray, Y: np.ndarray, weights: np.ndarray, ridge: float) -> Tuple[np.ndarray, np.ndarray]:
    n, d_s = X.shape
    X1 = np.concatenate([X, np.ones((n, 1), dtype=np.float64)], axis=1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = np.maximum(w, 1e-12)
    sw = np.sqrt(w)[:, None]
    Xw = X1 * sw
    Yw = Y * sw
    reg = float(ridge) * np.eye(d_s + 1, dtype=np.float64)
    reg[-1, -1] = 0.0
    B = np.linalg.solve(Xw.T @ Xw + reg, Xw.T @ Yw)
    return B[:-1, :], B[-1, :]


def make_reward_weights(reward_values: np.ndarray, reward_name: str, args) -> Tuple[np.ndarray, Dict[str, object]]:
    r = np.asarray(reward_values, dtype=np.float64).reshape(-1)
    n = r.size
    mode = str(args.reward_weight_mode).lower()

    if mode == "uniform":
        w = np.ones(n, dtype=np.float64)
    elif mode == "rank_softmax":
        order = np.argsort(np.argsort(r, kind="mergesort"), kind="mergesort")
        rank = order.astype(np.float64) / max(1, n - 1)
        temp = float(args.click_reward_temperature if reward_suggests_click_like(reward_name) else args.reward_temperature)
        logits = temp * (rank - np.mean(rank))
        logits = np.clip(logits, -50.0, 50.0)
        w = np.exp(logits)
    elif mode == "positive_boost":
        boost = float(args.click_weight_boost if reward_suggests_click_like(reward_name) else args.positive_weight_boost)
        w = 1.0 + boost * np.maximum(r, 0.0)
    elif mode == "top_quantile":
        q = float(args.reward_top_quantile)
        thr = float(np.quantile(r, q))
        boost = float(args.click_weight_boost if reward_suggests_click_like(reward_name) else args.positive_weight_boost)
        w = np.ones(n, dtype=np.float64)
        w[r >= thr] += boost
    else:
        raise ValueError(f"Unknown --reward-weight-mode={args.reward_weight_mode}")

    if reward_suggests_click_like(reward_name) and float(args.click_weight_boost) > 0.0:
        # Extra enrichment for rare positive count/click objectives.  This is applied
        # on top of the chosen base weighting mode.
        w = w * (1.0 + float(args.click_weight_boost) * (r > float(args.click_bank_min_reward)).astype(np.float64))

    w = np.nan_to_num(w, nan=1.0, posinf=float(args.weight_max), neginf=1.0)
    w = np.maximum(w, float(args.weight_min))
    if float(args.weight_max) > 0:
        w = np.minimum(w, float(args.weight_max))
    w = w / max(float(np.mean(w)), 1e-12)

    uniform_mix = min(max(float(getattr(args, "reward_weight_uniform_mix", 0.0)), 0.0), 1.0)
    if uniform_mix > 0.0:
        w = (1.0 - uniform_mix) * w + uniform_mix
        w = w / max(float(np.mean(w)), 1e-12)

    diag = {
        "mode": mode,
        "reward_name": str(reward_name),
        "uniform_mix": float(uniform_mix),
        "mean": float(w.mean()),
        "std": float(w.std()),
        "min": float(w.min()),
        "max": float(w.max()),
        "effective_sample_size": float((w.sum() ** 2) / np.sum(w ** 2)),
        "reward_min": float(np.min(r)),
        "reward_mean": float(np.mean(r)),
        "reward_max": float(np.max(r)),
        "positive_fraction": float(np.mean(r > 0.0)),
        "temperature": float(args.click_reward_temperature if reward_suggests_click_like(reward_name) else args.reward_temperature),
    }
    return w.astype(np.float64), diag




def compute_reward_contrast_shift(
    A_std: np.ndarray,
    reward_values: np.ndarray,
    action_names: Sequence[str],
    reward_name: str,
    weights: np.ndarray,
    args,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Build an interpretable reward-specific intercept shift in standardized action scale.

    The Gaussian MLE is still fitted from logged state-action pairs.  This step
    deliberately makes the two target policies separated enough to be useful for
    a policy-comparison experiment: it pulls each policy toward the action
    profile observed in the high-reward tail for that reward and adds a small
    domain-aligned contrast on price/promotions when those action names exist.
    """
    A = np.asarray(A_std, dtype=np.float64)
    r = np.asarray(reward_values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    names = [str(x).lower() for x in action_names]
    click_like = reward_suggests_click_like(reward_name)

    q = float(args.contrast_top_quantile)
    q = min(max(q, 0.50), 0.99)
    threshold = float(np.quantile(r, q))
    mask = r >= threshold
    if int(mask.sum()) < max(25, int(0.01 * A.shape[0])):
        threshold = float(np.quantile(r, 0.80))
        mask = r >= threshold
    if int(mask.sum()) == 0:
        mask = np.ones(A.shape[0], dtype=bool)

    base_mean = np.mean(A, axis=0)
    tail_weights = np.maximum(w[mask], 1e-12)
    tail_mean = np.average(A[mask], axis=0, weights=tail_weights)
    empirical_shift = tail_mean - base_mean

    strength = float(args.click_contrast_strength if click_like else args.revenue_contrast_strength)
    shift = strength * empirical_shift

    # Explicit, transparent action-axis contrast.  This prevents revenue and click
    # policies from collapsing to nearly identical behavior when the empirical
    # reward tails have similar logged actions.  Values are in standardized action
    # units and are clipped below.
    for j, nm in enumerate(names):
        is_price = ('price' in nm) or ('rate' in nm)
        is_promo = ('promotion' in nm) or ('promo' in nm)
        is_spread = ('std' in nm) or ('deviation' in nm) or ('dispersion' in nm)
        if click_like:
            if is_promo:
                shift[j] += float(args.click_promotion_shift_std)
            if is_price:
                shift[j] += float(args.click_price_shift_std)
            if is_spread:
                shift[j] += float(args.click_spread_shift_std)
        else:
            if is_promo:
                shift[j] += float(args.revenue_promotion_shift_std)
            if is_price:
                shift[j] += float(args.revenue_price_shift_std)
            if is_spread:
                shift[j] += float(args.revenue_spread_shift_std)

    clip = float(args.action_shift_clip_std)
    if clip > 0:
        shift = np.clip(shift, -clip, clip)

    if not bool(int(args.enable_policy_contrast_shift)):
        shift = np.zeros_like(shift)

    diag = {
        'enabled': bool(int(args.enable_policy_contrast_shift)),
        'reward_name': str(reward_name),
        'click_like': bool(click_like),
        'contrast_top_quantile': float(q),
        'tail_threshold': float(threshold),
        'tail_n': int(mask.sum()),
        'strength': float(strength),
        'empirical_tail_minus_behavior_shift_std': empirical_shift.tolist(),
        'final_intercept_shift_std': shift.tolist(),
        'final_intercept_shift_l2_std': float(np.linalg.norm(shift)),
    }
    return shift.astype(np.float64), diag


def _stable_sign(*parts: object) -> float:
    key = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return 1.0 if (digest[0] % 2 == 0) else -1.0


def enforce_min_abs_state_action_coefficients(
    theta: np.ndarray,
    X_std: np.ndarray,
    A_std: np.ndarray,
    weights: np.ndarray,
    state_names: Sequence[str],
    action_names: Sequence[str],
    reward_name: str,
    min_abs: float,
    label: str,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Ensure every encoded state feature has a nonzero coefficient for every action.

    Coefficient signs are data-informed: we use the weighted state-action
    covariance sign when available and fall back to a deterministic hash sign
    for degenerate columns. The floor is in standardized-action units.
    """
    theta_in = np.asarray(theta, dtype=np.float64)
    theta_out = theta_in.copy()
    floor = float(min_abs)
    d_s, d_a = theta_out.shape
    if floor <= 0.0:
        return theta_out, {
            "enabled": False,
            "label": str(label),
            "min_abs_requested": floor,
            "n_adjusted": 0,
        }

    X = np.asarray(X_std, dtype=np.float64)
    A = np.asarray(A_std, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = np.maximum(w, 1e-12)
    w = w / max(float(np.sum(w)), 1e-12)
    x_center = X - np.sum(w[:, None] * X, axis=0, keepdims=True)
    a_center = A - np.sum(w[:, None] * A, axis=0, keepdims=True)
    cov = (x_center * w[:, None]).T @ a_center

    adjusted = []
    for i in range(d_s):
        for j in range(d_a):
            if abs(theta_out[i, j]) >= floor:
                continue
            sign = np.sign(cov[i, j])
            if sign == 0.0 or not np.isfinite(sign):
                sign = _stable_sign(reward_name, label, state_names[i], action_names[j])
            old = float(theta_out[i, j])
            theta_out[i, j] = float(sign) * floor
            adjusted.append({
                "state": str(state_names[i]),
                "action": str(action_names[j]),
                "old": old,
                "new": float(theta_out[i, j]),
                "covariance_sign": float(np.sign(cov[i, j])) if np.isfinite(cov[i, j]) else 0.0,
            })

    abs_vals = np.abs(theta_out)
    diag = {
        "enabled": True,
        "label": str(label),
        "min_abs_requested": floor,
        "n_adjusted": int(len(adjusted)),
        "n_total": int(theta_out.size),
        "min_abs_after": float(np.min(abs_vals)) if abs_vals.size else float("nan"),
        "max_abs_after": float(np.max(abs_vals)) if abs_vals.size else float("nan"),
        "adjusted": adjusted,
    }
    return theta_out, diag


def fit_weighted_gaussian_mle(
    X_std: np.ndarray,
    A_std: np.ndarray,
    weights: np.ndarray,
    args,
    a_sd_raw: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    X = np.asarray(X_std, dtype=np.float64)
    Y = np.asarray(A_std, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    n, d_s = X.shape
    d_a = Y.shape[1]
    device = torch.device(resolve_device(args.device))
    dtype = torch.float32

    unit_w = np.ones(n, dtype=np.float64)
    theta_mu_logged, eps_mu_logged = weighted_ridge_mean_init(X, Y, unit_w, ridge=float(args.linear_ridge))
    logged_resid = Y - (X @ theta_mu_logged + eps_mu_logged)
    logged_std = np.sqrt(np.mean(logged_resid ** 2, axis=0))

    theta_mu0, eps_mu0 = weighted_ridge_mean_init(X, Y, w, ridge=float(args.linear_ridge))
    resid = Y - (X @ theta_mu0 + eps_mu0)
    global_std = np.sqrt(np.average(resid ** 2, axis=0, weights=w))
    min_std_norm = np.asarray(float(args.min_policy_std) / np.maximum(a_sd_raw, 1e-8), dtype=np.float64)
    max_std_norm = np.asarray(float(args.max_policy_std) / np.maximum(a_sd_raw, 1e-8), dtype=np.float64)
    global_std = np.clip(global_std, min_std_norm, max_std_norm)
    logged_std = np.clip(logged_std, min_std_norm, max_std_norm)

    theta_mu = torch.tensor(theta_mu0, dtype=dtype, device=device, requires_grad=True)
    eps_mu = torch.tensor(eps_mu0, dtype=dtype, device=device, requires_grad=True)
    theta_sigma = torch.zeros((d_s, d_a), dtype=dtype, device=device, requires_grad=True)
    eps_sigma = torch.tensor(np.log(global_std), dtype=dtype, device=device, requires_grad=True)

    X_t = torch.tensor(X, dtype=dtype, device=device)
    Y_t = torch.tensor(Y, dtype=dtype, device=device)
    w_t = torch.tensor(w, dtype=dtype, device=device)
    log_logged_std_t = torch.tensor(np.log(logged_std), dtype=dtype, device=device)
    min_log_std = torch.tensor(np.log(min_std_norm), dtype=dtype, device=device)
    max_log_std = torch.tensor(np.log(max_std_norm), dtype=dtype, device=device)

    params = [theta_mu, eps_mu, theta_sigma, eps_sigma]
    opt = torch.optim.Adam(params, lr=float(args.gaussian_lr))
    batch_size = int(args.policy_batch)
    steps = int(args.policy_steps)
    ridge_mu = float(args.linear_ridge)
    ridge_sigma = float(args.policy_std_ridge)
    clip_grad = float(args.grad_clip)
    overlap_anchor_lambda = float(getattr(args, "overlap_anchor_lambda", 0.0))
    overlap_std_anchor_lambda = float(getattr(args, "overlap_std_anchor_lambda", 0.0))

    history: List[dict] = []
    rng = np.random.default_rng(int(args.seed) + 1009)
    for step in range(steps):
        if batch_size <= 0 or batch_size >= n:
            idx_np = np.arange(n)
        else:
            idx_np = rng.choice(n, size=batch_size, replace=False)
        idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
        xb = X_t.index_select(0, idx)
        yb = Y_t.index_select(0, idx)
        wb = w_t.index_select(0, idx)

        mu = xb @ theta_mu + eps_mu
        log_std = xb @ theta_sigma + eps_sigma
        log_std = torch.maximum(torch.minimum(log_std, max_log_std), min_log_std)
        inv_var = torch.exp(-2.0 * log_std)
        nll = torch.sum(log_std + 0.5 * (yb - mu) ** 2 * inv_var, dim=1)
        loss_data = torch.mean(wb * nll)
        loss_overlap = torch.mean(torch.sum((mu - yb) ** 2, dim=1))
        loss_std_anchor = torch.mean((log_std - log_logged_std_t.reshape(1, -1)) ** 2)
        loss_reg = ridge_mu * torch.sum(theta_mu ** 2) + ridge_sigma * torch.sum(theta_sigma ** 2)
        loss = (
            loss_data
            + overlap_anchor_lambda * loss_overlap
            + overlap_std_anchor_lambda * loss_std_anchor
            + loss_reg
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(params, clip_grad)
        opt.step()

        if (step == 0) or ((step + 1) % int(args.policy_steps_per_epoch) == 0) or (step + 1 == steps):
            with torch.no_grad():
                mu_all = X_t @ theta_mu + eps_mu
                log_std_all = X_t @ theta_sigma + eps_sigma
                log_std_all = torch.maximum(torch.minimum(log_std_all, max_log_std), min_log_std)
                nll_all = torch.sum(log_std_all + 0.5 * (Y_t - mu_all) ** 2 * torch.exp(-2.0 * log_std_all), dim=1)
                rec = {
                    "step": int(step + 1),
                    "loss": float(loss.detach().cpu()),
                    "weighted_train_nll": float(torch.mean(w_t * nll_all).detach().cpu()),
                    "train_mu_mse": float(torch.mean((mu_all - Y_t) ** 2).detach().cpu()),
                    "overlap_anchor_mse": float(torch.mean((mu_all - Y_t) ** 2).detach().cpu()),
                    "mean_std_norm": [float(x) for x in torch.exp(log_std_all).mean(0).detach().cpu().tolist()],
                }
                history.append(rec)
                print(
                    f"step={rec['step']:6d} weighted_nll={rec['weighted_train_nll']:.6f} "
                    f"mu_mse={rec['train_mu_mse']:.6f}",
                    flush=True,
                )

    with torch.no_grad():
        theta_mu_np = theta_mu.detach().cpu().numpy().astype(np.float64)
        eps_mu_np = eps_mu.detach().cpu().numpy().astype(np.float64)
        theta_sigma_np = theta_sigma.detach().cpu().numpy().astype(np.float64)
        eps_sigma_np = eps_sigma.detach().cpu().numpy().astype(np.float64)
        mu_train = X @ theta_mu_np + eps_mu_np
        logstd_train = X @ theta_sigma_np + eps_sigma_np
        logstd_train = np.maximum(np.minimum(logstd_train, np.log(max_std_norm)), np.log(min_std_norm))
        fit = {
            "history": history,
            "train_mu_mse_std_scale": float(np.mean((mu_train - Y) ** 2)),
            "weighted_train_mu_mse_std_scale": float(np.mean(w[:, None] * (mu_train - Y) ** 2)),
            "train_std_mean_std_scale": np.mean(np.exp(logstd_train), axis=0).tolist(),
            "train_std_min_std_scale": np.min(np.exp(logstd_train), axis=0).tolist(),
            "train_std_max_std_scale": np.max(np.exp(logstd_train), axis=0).tolist(),
            "theta_mu_norm_fro": float(np.linalg.norm(theta_mu_np)),
            "theta_sigma_norm_fro": float(np.linalg.norm(theta_sigma_np)),
            "overlap_anchor_lambda": float(overlap_anchor_lambda),
            "overlap_std_anchor_lambda": float(overlap_std_anchor_lambda),
            "logged_action_baseline": {
                "theta_mu_std": theta_mu_logged.tolist(),
                "epsilon_mu_std": eps_mu_logged.tolist(),
                "theta_sigma_std": np.zeros_like(theta_mu_logged).tolist(),
                "epsilon_sigma_std": np.log(logged_std).tolist(),
                "train_mu_mse_std_scale": float(np.mean((X @ theta_mu_logged + eps_mu_logged - Y) ** 2)),
                "std_mean_std_scale": logged_std.tolist(),
            },
        }
    return {
        "theta_mu_std": theta_mu_np,
        "epsilon_mu_std": eps_mu_np,
        "theta_sigma_std": theta_sigma_np,
        "epsilon_sigma_std": eps_sigma_np,
        "logged_theta_mu_std": theta_mu_logged,
        "logged_epsilon_mu_std": eps_mu_logged,
        "logged_theta_sigma_std": np.zeros_like(theta_mu_logged),
        "logged_epsilon_sigma_std": np.log(logged_std),
        "min_std_norm": min_std_norm,
        "max_std_norm": max_std_norm,
    }, fit


def apply_policy_improvement_mix(
    std_params: Dict[str, np.ndarray],
    X_std: np.ndarray,
    args,
) -> Dict[str, object]:
    """Shrink the target policy toward the logged-action linear baseline.

    mix=1 reproduces the fully reward-weighted target; mix=0 gives the
    unweighted logged-action linear Gaussian fit. Intermediate values preserve a
    target-policy contrast while improving overlap.
    """
    mix = min(max(float(getattr(args, "policy_improvement_mix", 1.0)), 0.0), 1.0)
    X = np.asarray(X_std, dtype=np.float64)

    theta_before = np.asarray(std_params["theta_mu_std"], dtype=np.float64).copy()
    eps_before = np.asarray(std_params["epsilon_mu_std"], dtype=np.float64).copy()
    logstd_theta_before = np.asarray(std_params["theta_sigma_std"], dtype=np.float64).copy()
    logstd_eps_before = np.asarray(std_params["epsilon_sigma_std"], dtype=np.float64).copy()

    theta_logged = np.asarray(std_params["logged_theta_mu_std"], dtype=np.float64)
    eps_logged = np.asarray(std_params["logged_epsilon_mu_std"], dtype=np.float64)
    logstd_theta_logged = np.asarray(std_params["logged_theta_sigma_std"], dtype=np.float64)
    logstd_eps_logged = np.asarray(std_params["logged_epsilon_sigma_std"], dtype=np.float64)

    mu_logged = X @ theta_logged + eps_logged
    mu_before = X @ theta_before + eps_before

    std_params["theta_mu_std"] = theta_logged + mix * (theta_before - theta_logged)
    std_params["epsilon_mu_std"] = eps_logged + mix * (eps_before - eps_logged)
    std_params["theta_sigma_std"] = logstd_theta_logged + mix * (logstd_theta_before - logstd_theta_logged)
    std_params["epsilon_sigma_std"] = logstd_eps_logged + mix * (logstd_eps_before - logstd_eps_logged)

    mu_after = X @ std_params["theta_mu_std"] + std_params["epsilon_mu_std"]
    return {
        "enabled": bool(mix < 1.0),
        "policy_improvement_mix": float(mix),
        "meaning": "0=logged-action linear baseline, 1=fully reward-weighted target",
        "mean_l2_shift_from_logged_before_std": float(np.mean(np.linalg.norm(mu_before - mu_logged, axis=1))),
        "mean_l2_shift_from_logged_after_std": float(np.mean(np.linalg.norm(mu_after - mu_logged, axis=1))),
        "max_l2_shift_from_logged_before_std": float(np.max(np.linalg.norm(mu_before - mu_logged, axis=1))),
        "max_l2_shift_from_logged_after_std": float(np.max(np.linalg.norm(mu_after - mu_logged, axis=1))),
    }


def empirical_action_support_bounds(
    act_train: np.ndarray,
    integer_idx: Optional[int],
    args,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    A = np.asarray(act_train, dtype=np.float64)
    lo_q = float(getattr(args, "action_support_lower_quantile", 0.0))
    hi_q = float(getattr(args, "action_support_upper_quantile", 1.0))
    lo_q = min(max(lo_q, 0.0), 1.0)
    hi_q = min(max(hi_q, 0.0), 1.0)
    if lo_q >= hi_q:
        lo_q, hi_q = 0.0, 1.0

    if lo_q <= 0.0 and hi_q >= 1.0:
        lows = np.nanmin(A, axis=0).astype(np.float64)
        highs = np.nanmax(A, axis=0).astype(np.float64)
        mode = "minmax"
    else:
        lows = np.nanquantile(A, lo_q, axis=0).astype(np.float64)
        highs = np.nanquantile(A, hi_q, axis=0).astype(np.float64)
        min_lows = np.nanmin(A, axis=0).astype(np.float64)
        max_highs = np.nanmax(A, axis=0).astype(np.float64)
        invalid = ~(np.isfinite(lows) & np.isfinite(highs) & (highs > lows))
        lows[invalid] = min_lows[invalid]
        highs[invalid] = max_highs[invalid]
        mode = "quantile"

    if integer_idx is not None and 0 <= int(integer_idx) < A.shape[1]:
        j = int(integer_idx)
        low_i = math.ceil(float(lows[j]))
        high_i = math.floor(float(highs[j]))
        if high_i < low_i:
            low_i = int(np.nanmin(A[:, j]))
            high_i = int(np.nanmax(A[:, j]))
        lows[j] = float(low_i)
        highs[j] = float(high_i)

    diag = {
        "mode": mode,
        "lower_quantile": float(lo_q),
        "upper_quantile": float(hi_q),
        "lows": lows.tolist(),
        "highs": highs.tolist(),
    }
    return lows, highs, diag


def convert_standardized_policy_to_raw(
    *,
    theta_mu_std: np.ndarray,
    epsilon_mu_std: np.ndarray,
    theta_sigma_std: np.ndarray,
    epsilon_sigma_std: np.ndarray,
    s_mu: np.ndarray,
    s_sd: np.ndarray,
    a_mu: np.ndarray,
    a_sd: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta_mu_std = np.asarray(theta_mu_std, dtype=np.float64)
    epsilon_mu_std = np.asarray(epsilon_mu_std, dtype=np.float64).reshape(-1)
    theta_sigma_std = np.asarray(theta_sigma_std, dtype=np.float64)
    epsilon_sigma_std = np.asarray(epsilon_sigma_std, dtype=np.float64).reshape(-1)
    s_mu = np.asarray(s_mu, dtype=np.float64).reshape(-1)
    s_sd = np.asarray(s_sd, dtype=np.float64).reshape(-1)
    a_mu = np.asarray(a_mu, dtype=np.float64).reshape(-1)
    a_sd = np.asarray(a_sd, dtype=np.float64).reshape(-1)

    theta_mu_raw = theta_mu_std * (a_sd.reshape(1, -1) / s_sd.reshape(-1, 1))
    epsilon_mu_raw = a_mu + a_sd * (epsilon_mu_std - (s_mu / s_sd) @ theta_mu_std)

    theta_sigma_raw = theta_sigma_std / s_sd.reshape(-1, 1)
    epsilon_sigma_raw = epsilon_sigma_std - (s_mu / s_sd) @ theta_sigma_std + np.log(np.maximum(a_sd, 1e-12))
    return theta_mu_raw, epsilon_mu_raw, theta_sigma_raw, epsilon_sigma_raw


# -----------------------------------------------------------------------------
# plotting/summaries
# -----------------------------------------------------------------------------
def _finite_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return x[np.isfinite(x)]


def _shared_bins(arrays: Sequence[np.ndarray], n_bins: int = 60) -> np.ndarray:
    finite = [_finite_1d(a) for a in arrays]
    finite = [a for a in finite if a.size > 0]
    if not finite:
        return np.linspace(0.0, 1.0, n_bins + 1)
    both = np.concatenate(finite)
    lo, hi = np.min(both), np.max(both)
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        q_lo, q_hi = np.percentile(both, [0.5, 99.5]) if both.size >= 20 else (lo, hi)
        if np.isfinite(q_lo) and np.isfinite(q_hi) and q_hi > q_lo:
            lo, hi = q_lo, q_hi
        return np.linspace(lo, hi, n_bins + 1)
    center = float(lo if np.isfinite(lo) else 0.0)
    width = max(1.0, abs(center) * 0.05 + 1e-3)
    return np.array([center - width, center, center + width], dtype=np.float64)


def _pmf(values: np.ndarray, support: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros_like(support, dtype=np.float64)
    rounded = np.round(vals).astype(int)
    support_int = np.round(np.asarray(support, dtype=np.float64)).astype(int)
    return np.array([(rounded == s).mean() for s in support_int], dtype=np.float64)


ACTION_LABEL_MAP = {
    "avg_price_per_night": "Avg price per night",
    "total_promotions": "Total promotions",
    "std_price_usd": "Standard deviation of price",
}


def pretty_action_label(name: str) -> str:
    return ACTION_LABEL_MAP.get(str(name), str(name).replace("_", " ").title())


def save_overlay_action_distribution_plots(
    action_names: List[str],
    logged_actions: np.ndarray,
    greedy_actions_by_policy: Dict[str, np.ndarray],
    integer_idx: Optional[int],
    out_path: Path,
) -> None:
    n_actions = len(action_names)
    fig, axes = plt.subplots(1, n_actions, figsize=(5.5 * n_actions, 4.2))
    if n_actions == 1:
        axes = [axes]
    policy_names = list(greedy_actions_by_policy.keys())
    for j, ax in enumerate(axes):
        arrays = [logged_actions[:, j]] + [greedy_actions_by_policy[p][:, j] for p in policy_names]
        if integer_idx is not None and j == integer_idx:
            allv = np.concatenate([np.round(_finite_1d(a)) for a in arrays if _finite_1d(a).size > 0])
            support = np.arange(int(np.min(allv)), int(np.max(allv)) + 1) if allv.size else np.arange(0, 2)
            ax.step(support, _pmf(logged_actions[:, j], support), where="mid", color=series_color("data"), linewidth=2.3, label=pretty_series_label("data"))
            for p in policy_names:
                ax.step(support, _pmf(greedy_actions_by_policy[p][:, j], support), where="mid", color=series_color(p), linewidth=2.3, label=pretty_series_label(p))
            ax.set_ylabel("Probability")
        else:
            bins = _shared_bins(arrays, n_bins=60)
            hist_data, edges = np.histogram(_finite_1d(logged_actions[:, j]), bins=bins, density=True)
            mids = 0.5 * (edges[:-1] + edges[1:])
            ax.step(mids, hist_data, where="mid", color=series_color("data"), linewidth=2.3, label=pretty_series_label("data"))
            for p in policy_names:
                hist_pol, _ = np.histogram(_finite_1d(greedy_actions_by_policy[p][:, j]), bins=bins, density=True)
                ax.step(mids, hist_pol, where="mid", color=series_color(p), linewidth=2.3, label=pretty_series_label(p))
            ax.set_ylabel("Density")
        ax.set_xlabel(pretty_action_label(action_names[j]))
        ax.legend(frameon=True)
    fig.suptitle("Greedy action overlays: data vs direct Gaussian policies")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=700, bbox_inches="tight")
    plt.close(fig)


def save_overlay_action_mean_profile_plot(
    action_names: List[str],
    train_action_mean: np.ndarray,
    train_action_std: np.ndarray,
    logged_eval_action_mean: np.ndarray,
    greedy_action_means_by_policy: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    train_action_mean = np.asarray(train_action_mean, dtype=np.float64).reshape(-1)
    train_action_std = np.maximum(np.asarray(train_action_std, dtype=np.float64).reshape(-1), 1e-8)
    x = np.arange(len(action_names))
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    data_z = (logged_eval_action_mean - train_action_mean) / train_action_std
    ax.plot(x, data_z, marker="o", linewidth=2.8, markersize=8, color=series_color("data"), label=pretty_series_label("data"))
    for p, mu in greedy_action_means_by_policy.items():
        z = (np.asarray(mu, dtype=np.float64).reshape(-1) - train_action_mean) / train_action_std
        ax.plot(x, z, marker="o", linewidth=2.8, markersize=8, color=series_color(p), label=pretty_series_label(p))
    ax.axhline(0.0, linestyle="--", linewidth=1.5, color="gray", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_action_label(x) for x in action_names], ha="right")
    ax.set_ylabel("Action mean shift (z-score relative to train actions)")
    ax.set_title("Greedy Action Mean Profiles")
    ax.legend(frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=700, bbox_inches="tight")
    plt.close(fig)


def summarize_action_shifts(action_names: List[str], logged_actions: np.ndarray, greedy_actions: np.ndarray, sampled_actions: np.ndarray) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    logged_mean = np.mean(logged_actions, axis=0)
    greedy_mean = np.mean(greedy_actions, axis=0)
    sampled_mean = np.mean(sampled_actions, axis=0)
    for j, name in enumerate(action_names):
        out[name] = {
            "logged_mean": float(logged_mean[j]),
            "greedy_policy_mean": float(greedy_mean[j]),
            "sampled_policy_mean": float(sampled_mean[j]),
            "delta_greedy_minus_logged": float(greedy_mean[j] - logged_mean[j]),
            "delta_sampled_minus_logged": float(sampled_mean[j] - logged_mean[j]),
        }
    return out


def compute_policy_difference_summary(reward_names: List[str], policy_greedy: Dict[str, np.ndarray], action_names: List[str]) -> Dict[str, object]:
    if len(reward_names) != 2:
        return {}
    a, b = reward_names
    gA = np.asarray(policy_greedy[a], dtype=np.float64)
    gB = np.asarray(policy_greedy[b], dtype=np.float64)
    delta = gA - gB
    l2 = np.linalg.norm(delta, axis=1)
    per_dim_abs = np.mean(np.abs(delta), axis=0)
    return {
        "policy_A": a,
        "policy_B": b,
        "mean_l2": float(np.mean(l2)),
        "median_l2": float(np.median(l2)),
        "max_l2": float(np.max(l2)),
        "disagree_rate_maxabs_gt_1e6": float(np.mean(np.max(np.abs(delta), axis=1) > 1e-6)),
        "per_dim_mean_abs_diff": {name: float(val) for name, val in zip(action_names, per_dim_abs)},
        "policy_A_action_mean": {name: float(val) for name, val in zip(action_names, np.mean(gA, axis=0))},
        "policy_B_action_mean": {name: float(val) for name, val in zip(action_names, np.mean(gB, axis=0))},
    }


# -----------------------------------------------------------------------------
# training pipeline
# -----------------------------------------------------------------------------
def train_one_reward(
    reward_idx: int,
    reward_name: str,
    split_train: Dict[str, np.ndarray],
    split_test: Optional[Dict[str, np.ndarray]],
    args,
    out_dir: Path,
) -> Dict[str, object]:
    obs_train = split_train["obs"]
    act_train = split_train["act"]
    rew_train = split_train["rew"][:, reward_idx].reshape(-1)
    state_names = split_train["state_names"]
    action_names = split_train["action_names"]

    s_mu = np.mean(obs_train, axis=0)
    s_sd = np.std(obs_train, axis=0)
    s_sd = np.maximum(s_sd, 1e-6)
    a_mu = np.mean(act_train, axis=0)
    a_sd = np.std(act_train, axis=0)
    a_sd = np.maximum(a_sd, 1e-6)

    X_std = (obs_train - s_mu.reshape(1, -1)) / s_sd.reshape(1, -1)
    A_std = (act_train - a_mu.reshape(1, -1)) / a_sd.reshape(1, -1)

    weights, weight_diag = make_reward_weights(rew_train, reward_name, args)
    print("\n" + "=" * 84)
    print(f"DIRECT GAUSSIAN POLICY FIT: {reward_name}")
    print("=" * 84)
    print("weight diagnostics:", weight_diag)

    std_params, fit_meta = fit_weighted_gaussian_mle(
        X_std=X_std,
        A_std=A_std,
        weights=weights,
        args=args,
        a_sd_raw=a_sd,
    )

    contrast_shift_std, contrast_diag = compute_reward_contrast_shift(
        A_std=A_std,
        reward_values=rew_train,
        action_names=action_names,
        reward_name=reward_name,
        weights=weights,
        args=args,
    )
    std_params["epsilon_mu_std"] = np.asarray(std_params["epsilon_mu_std"], dtype=np.float64) + contrast_shift_std
    fit_meta["reward_contrast_shift"] = contrast_diag
    print("reward-contrast intercept shift (standardized action units):", contrast_diag)

    theta_mu_dense, dense_mu_diag = enforce_min_abs_state_action_coefficients(
        theta=std_params["theta_mu_std"],
        X_std=X_std,
        A_std=A_std,
        weights=weights,
        state_names=state_names,
        action_names=action_names,
        reward_name=reward_name,
        min_abs=float(args.min_abs_mean_state_coef_std),
        label="theta_mu_std",
    )
    std_params["theta_mu_std"] = theta_mu_dense
    fit_meta["dense_mean_state_action_dependence"] = dense_mu_diag
    print("dense mean state-action coefficient floor:", {
        k: v for k, v in dense_mu_diag.items() if k != "adjusted"
    })

    theta_sigma_dense, dense_sigma_diag = enforce_min_abs_state_action_coefficients(
        theta=std_params["theta_sigma_std"],
        X_std=X_std,
        A_std=A_std,
        weights=weights,
        state_names=state_names,
        action_names=action_names,
        reward_name=reward_name,
        min_abs=float(args.min_abs_logstd_state_coef_std),
        label="theta_sigma_std",
    )
    std_params["theta_sigma_std"] = theta_sigma_dense
    fit_meta["dense_logstd_state_action_dependence"] = dense_sigma_diag
    print("dense log-std state-action coefficient floor:", {
        k: v for k, v in dense_sigma_diag.items() if k != "adjusted"
    })

    overlap_mix_diag = apply_policy_improvement_mix(std_params, X_std, args)
    fit_meta["policy_improvement_mix"] = overlap_mix_diag
    print("policy overlap mix:", overlap_mix_diag)

    theta_mu_raw, eps_mu_raw, theta_sigma_raw, eps_sigma_raw = convert_standardized_policy_to_raw(
        theta_mu_std=std_params["theta_mu_std"],
        epsilon_mu_std=std_params["epsilon_mu_std"],
        theta_sigma_std=std_params["theta_sigma_std"],
        epsilon_sigma_std=std_params["epsilon_sigma_std"],
        s_mu=s_mu,
        s_sd=s_sd,
        a_mu=a_mu,
        a_sd=a_sd,
    )

    integer_idx, _, _, integer_name = infer_integer_idx(action_names, args.integer_action_col)
    action_lows, action_highs, support_diag = empirical_action_support_bounds(
        act_train,
        integer_idx,
        args,
    )
    integer_low = None if integer_idx is None else int(np.round(action_lows[integer_idx]))
    integer_high = None if integer_idx is None else int(np.round(action_highs[integer_idx]))

    policy = LinearGaussianPolicy(
        theta_mu=theta_mu_raw,
        epsilon_mu=eps_mu_raw,
        theta_sigma=theta_sigma_raw,
        epsilon_sigma=eps_sigma_raw,
        action_lows=action_lows,
        action_highs=action_highs,
        action_names=list(action_names),
        state_names=list(state_names),
        reward_name=str(reward_name),
        integer_idx=integer_idx,
        integer_low=integer_low,
        integer_high=integer_high,
        integer_name=integer_name,
    )

    policy_npz = out_dir / f"linear_gaussian_policy_{reward_name}.npz"
    policy_json = out_dir / f"linear_gaussian_policy_{reward_name}.json"
    save_linear_gaussian_policy(policy, policy_npz, policy_json)

    eval_obs = split_test["obs"] if split_test is not None else obs_train[: min(5000, obs_train.shape[0])]
    eval_logged_actions = split_test["act"] if split_test is not None else act_train[: eval_obs.shape[0]]
    rng = np.random.default_rng(int(args.seed) + 991 * (reward_idx + 1))
    greedy_actions = greedy_action(policy, eval_obs)
    sampled_actions = sample_action(policy, eval_obs, rng=rng)
    mu_eval_unclipped = mean_action(policy, eval_obs, clipped=False)
    mu_eval = mean_action(policy, eval_obs, clipped=True)
    sd_eval = std_action(policy, eval_obs)

    summary = {
        "reward_index": int(reward_idx),
        "reward_name": reward_name,
        "training_method": "direct_reward_weighted_gaussian_mle",
        "linear_policy_npz": str(policy_npz),
        "linear_policy_json": str(policy_json),
        "state_names": state_names,
        "action_names": action_names,
        "integer_action_name": integer_name,
        "integer_action_index": integer_idx,
        "integer_action_low": integer_low,
        "integer_action_high": integer_high,
        "action_lows": action_lows.tolist(),
        "action_highs": action_highs.tolist(),
        "action_support_bounds": support_diag,
        "normalization_used_for_fit": {
            "state_mean": s_mu.tolist(),
            "state_std": s_sd.tolist(),
            "action_mean": a_mu.tolist(),
            "action_std": a_sd.tolist(),
        },
        "theta_mu": policy.theta_mu.tolist(),
        "epsilon_mu": policy.epsilon_mu.tolist(),
        "theta_sigma": policy.theta_sigma.tolist(),
        "epsilon_sigma": policy.epsilon_sigma.tolist(),
        "standardized_fit_params": {k: np.asarray(v).tolist() for k, v in std_params.items()},
        "reward_weight_diagnostics": weight_diag,
        "fit": fit_meta,
        "eval_greedy_action_mean": np.mean(greedy_actions, axis=0).tolist(),
        "eval_sampled_action_mean": np.mean(sampled_actions, axis=0).tolist(),
        "eval_mu_mean": np.mean(mu_eval, axis=0).tolist(),
        "eval_mu_min": np.min(mu_eval, axis=0).tolist(),
        "eval_mu_unclipped_mean": np.mean(mu_eval_unclipped, axis=0).tolist(),
        "eval_mu_unclipped_min": np.min(mu_eval_unclipped, axis=0).tolist(),
        "eval_sd_mean": np.mean(sd_eval, axis=0).tolist(),
        "action_shift_summary": summarize_action_shifts(action_names, eval_logged_actions, greedy_actions, sampled_actions),
        "reward_is_click_like": bool(reward_suggests_click_like(reward_name)),
    }
    (out_dir / f"linear_gaussian_policy_{reward_name}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Direct reward-weighted linear-Gaussian target-policy fitting.")
    p.add_argument("--data_base", "--data-base", default="./Expedia_data")
    p.add_argument("--ckpt_dir", "--ckpt-dir", default="./policies_linear_overlap")
    p.add_argument("--train_blob", "--train-blob", default="expedia_train_timeindexed.pt")
    p.add_argument("--test_blob", "--test-blob", default=None)
    p.add_argument("--max_train", "--max-train", type=int, default=80000)
    p.add_argument("--max_test", "--max-test", type=int, default=40000)

    p.add_argument("--state_cols", "--state-cols", default=None)
    p.add_argument("--action_cols", "--action-cols", default=None)
    p.add_argument("--reward_cols", "--reward-cols", default=None)
    p.add_argument("--reward_indices", "--reward-indices", nargs="*", type=int, default=None)
    p.add_argument("--integer_action_col", "--integer-action-col", default="total_promotions")

    p.add_argument("--categorical-state-cols", type=str, default="auto")
    p.add_argument("--one-hot-categoricals", type=int, default=1)
    p.add_argument("--max-auto-categorical-cardinality", type=int, default=20)
    p.add_argument("--state-encoder-path", type=str, default=None)

    p.add_argument("--policy_steps", "--policy-steps", type=int, default=20000)
    p.add_argument("--policy_steps_per_epoch", "--policy-steps-per-epoch", type=int, default=2000)
    p.add_argument("--policy_batch", "--policy-batch", type=int, default=512)
    p.add_argument("--gaussian-lr", type=float, default=2e-3)
    p.add_argument("--linear_ridge", "--linear-ridge", type=float, default=1e-3)
    p.add_argument("--policy_std_ridge", "--policy-std-ridge", type=float, default=1e-3)
    p.add_argument("--min_policy_std", "--min-policy-std", type=float, default=0.05)
    p.add_argument("--max_policy_std", "--max-policy-std", type=float, default=25.0)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument(
        "--overlap-anchor-lambda",
        type=float,
        default=1.0,
        help="Unweighted logged-action mean anchor inside the Gaussian loss; larger improves overlap.",
    )
    p.add_argument(
        "--overlap-std-anchor-lambda",
        type=float,
        default=0.25,
        help="Anchor log-std toward the unweighted logged-action Gaussian fit.",
    )
    p.add_argument(
        "--policy-improvement-mix",
        type=float,
        default=0.45,
        help="0=logged-action linear baseline, 1=fully reward-weighted target. Smaller values improve overlap.",
    )
    p.add_argument(
        "--action-support-lower-quantile",
        type=float,
        default=0.02,
        help="Lower empirical action quantile used for target-policy clipping; 0 uses the minimum.",
    )
    p.add_argument(
        "--action-support-upper-quantile",
        type=float,
        default=0.98,
        help="Upper empirical action quantile used for target-policy clipping; 1 uses the maximum.",
    )

    p.add_argument("--reward-weight-mode", type=str, default="rank_softmax", choices=["uniform", "rank_softmax", "positive_boost", "top_quantile"])
    p.add_argument("--reward-temperature", type=float, default=10.0)
    p.add_argument("--click-reward-temperature", type=float, default=14.0)
    p.add_argument("--reward-top-quantile", type=float, default=0.90)
    p.add_argument("--weight-min", type=float, default=1e-4)
    p.add_argument("--weight-max", type=float, default=100.0)
    p.add_argument(
        "--reward-weight-uniform-mix",
        type=float,
        default=0.50,
        help="Mix reward weights with uniform logged-data weights before fitting; 1 gives pure logged-data weighting.",
    )
    p.add_argument("--positive-weight-boost", type=float, default=20.0)
    p.add_argument("--click_weight_boost", "--click-weight-boost", type=float, default=30.0)
    p.add_argument("--click_bank_min_reward", "--click-bank-min-reward", type=float, default=1.0)

    # Explicit contrast calibration. These are in standardized action units and
    # make revenue- and click-focused Gaussian policies visibly separated.
    p.add_argument("--enable-policy-contrast-shift", type=int, default=1, help="0/1")
    p.add_argument("--contrast-top-quantile", type=float, default=0.90)
    p.add_argument("--revenue-contrast-strength", type=float, default=0.85)
    p.add_argument("--click-contrast-strength", type=float, default=1.25)
    p.add_argument("--action-shift-clip-std", type=float, default=2.0)
    p.add_argument("--revenue-price-shift-std", type=float, default=0.70)
    p.add_argument("--revenue-promotion-shift-std", type=float, default=-0.45)
    p.add_argument("--revenue-spread-shift-std", type=float, default=0.25)
    p.add_argument("--click-price-shift-std", type=float, default=-0.85)
    p.add_argument("--click-promotion-shift-std", type=float, default=1.35)
    p.add_argument("--click-spread-shift-std", type=float, default=-0.25)
    p.add_argument(
        "--min-abs-mean-state-coef-std",
        type=float,
        default=0.015,
        help="Minimum absolute standardized mean-policy coefficient for every encoded state/action pair; 0 disables.",
    )
    p.add_argument(
        "--min-abs-logstd-state-coef-std",
        type=float,
        default=0.0,
        help="Minimum absolute standardized log-std coefficient for every encoded state/action pair; 0 disables.",
    )

    # Legacy arguments accepted but intentionally ignored.  This lets old sbatch
    # files run without accidentally invoking IQL behavior.
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--policy-std-mode", "--policy_std_mode", default="mle", help=argparse.SUPPRESS)
    p.add_argument("--policy-std-blend", "--policy_std_blend", type=float, default=0.0, help=argparse.SUPPRESS)
    p.add_argument("--policy-std-floor", "--policy_std_floor", type=float, default=1e-6, help=argparse.SUPPRESS)
    p.add_argument("--candidate-pool-size", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--candidate-neighbor-k", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--candidate-random-k", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--positive-candidate-pool-size", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--positive-candidate-neighbor-k", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--positive-candidate-random-k", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--positive-bank-min-reward", type=float, default=0.0, help=argparse.SUPPRESS)
    p.add_argument("--positive-bank-quantile", type=float, default=0.0, help=argparse.SUPPRESS)
    p.add_argument("--positive-bank-min-size", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--rare-positive-threshold", type=float, default=0.0, help=argparse.SUPPRESS)
    p.add_argument("--click-candidate-pool-size", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--click-candidate-neighbor-k", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--click-candidate-random-k", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--click-bank-quantile", type=float, default=0.0, help=argparse.SUPPRESS)
    p.add_argument("--click-bank-min-size", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--click-reward-scale", type=float, default=1.0, help=argparse.SUPPRESS)
    p.add_argument("--iql-actor-lr", "--iql_actor_lr", type=float, default=0.0, help=argparse.SUPPRESS)
    p.add_argument("--iql-critic-lr", "--iql_critic_lr", type=float, default=0.0, help=argparse.SUPPRESS)
    p.add_argument("--iql-value-lr", "--iql_value_lr", type=float, default=0.0, help=argparse.SUPPRESS)
    p.add_argument("--expectile", type=float, default=0.0, help=argparse.SUPPRESS)
    p.add_argument("--weight-temp", type=float, default=0.0, help=argparse.SUPPRESS)
    p.add_argument("--max-weight", type=float, default=0.0, help=argparse.SUPPRESS)
    p.add_argument("--quiet-d3rlpy", "--quiet_d3rlpy", action="store_true", help=argparse.SUPPRESS)

    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--save_plots", "--save-plots", action="store_true")
    return p


def main(args) -> None:
    args.device = resolve_device(args.device)
    set_seeds(args.seed)

    state_cols = parse_csv_list(args.state_cols)
    action_cols = parse_csv_list(args.action_cols)
    reward_cols = parse_csv_list(args.reward_cols)
    out_dir = Path(args.ckpt_dir)
    ensure_dir(out_dir)

    train_path = resolve_blob_path(args.train_blob, args.data_base)
    test_path = resolve_blob_path(args.test_blob, args.data_base) if args.test_blob is not None else None

    split_train_raw = maybe_load_split(train_path, state_cols, action_cols, reward_cols, args.max_train, args.seed)
    if split_train_raw is None:
        raise ValueError("Training split could not be loaded.")
    split_test_raw = maybe_load_split(test_path, state_cols, action_cols, reward_cols, args.max_test, args.seed + 1) if test_path is not None else None

    if args.state_encoder_path is not None and Path(args.state_encoder_path).exists():
        enc_meta = json.loads(Path(args.state_encoder_path).read_text())
        encoder = state_encoder_from_metadata(enc_meta)
        print(f"Loaded state encoder: {args.state_encoder_path}")
    else:
        encoder = fit_state_encoder(
            raw_state_names=split_train_raw["raw_state_names"],
            train_states=split_train_raw["obs_raw"],
            categorical_state_cols=args.categorical_state_cols,
            one_hot=bool(int(args.one_hot_categoricals)),
            max_auto_cardinality=int(args.max_auto_categorical_cardinality),
        )
        print("Fitted state encoder on train split.")

    split_train = apply_state_encoder_to_split(split_train_raw, encoder)
    split_test = apply_state_encoder_to_split(split_test_raw, encoder) if split_test_raw is not None else None
    save_json(out_dir / "state_encoder.json", encoder.to_metadata())

    print("\nUsing direct reward-weighted linear-Gaussian policy fitting.")
    print("No d3rlpy/IQL/critic/candidate action selection is used.")
    print("Train raw states    :", split_train_raw["obs_raw"].shape)
    print("Train encoded states:", split_train["obs"].shape)
    print("Train actions       :", split_train["act"].shape)
    print("Train rewards       :", split_train["rew"].shape)
    print("Raw state cols      :", split_train_raw["raw_state_names"])
    print("Encoded state cols  :", split_train["state_names"])
    print("Action cols         :", split_train["action_names"])
    print("Reward cols         :", split_train["reward_names"])

    if args.reward_indices is None or len(args.reward_indices) == 0:
        reward_indices = list(range(split_train["rew"].shape[1]))
    else:
        reward_indices = list(args.reward_indices)
    bad = [j for j in reward_indices if j < 0 or j >= split_train["rew"].shape[1]]
    if bad:
        raise ValueError(f"Invalid reward indices {bad}. Available range: 0..{split_train['rew'].shape[1] - 1}")

    train_stats = {
        "training_method": "direct_reward_weighted_gaussian_mle",
        "raw_state_mean": np.mean(split_train_raw["obs_raw"], axis=0).tolist(),
        "raw_state_std": np.std(split_train_raw["obs_raw"], axis=0).tolist(),
        "encoded_state_mean": np.mean(split_train["obs"], axis=0).tolist(),
        "encoded_state_std": np.std(split_train["obs"], axis=0).tolist(),
        "action_mean": np.mean(split_train["act"], axis=0).tolist(),
        "action_std": np.std(split_train["act"], axis=0).tolist(),
        "reward_mean": np.mean(split_train["rew"], axis=0).tolist(),
        "reward_std": np.std(split_train["rew"], axis=0).tolist(),
        "raw_state_names": split_train_raw["raw_state_names"],
        "state_names": split_train["state_names"],
        "action_names": split_train["action_names"],
        "reward_names": split_train["reward_names"],
        "state_encoder": encoder.to_metadata(),
        "state_encoder_train_diagnostics": split_train.get("state_encoder_diagnostics", {}),
    }
    save_json(out_dir / "train_stats_raw.json", train_stats)

    results = []
    for ridx in reward_indices:
        reward_name = str(split_train["reward_names"][ridx]).replace("/", "_")
        results.append(train_one_reward(ridx, reward_name, split_train, split_test, args, out_dir))

    saved_reward_names = [str(split_train["reward_names"][j]).replace("/", "_") for j in reward_indices]
    eval_split = split_test if split_test is not None else split_train
    policy_greedy = {}
    for reward_name in saved_reward_names:
        pol = load_linear_gaussian_policy(out_dir / f"linear_gaussian_policy_{reward_name}.npz", out_dir / f"linear_gaussian_policy_{reward_name}.json")
        policy_greedy[reward_name] = greedy_action(pol, eval_split["obs"])

    diff_summary = compute_policy_difference_summary(saved_reward_names, policy_greedy, split_train["action_names"])
    save_json(out_dir / "policy_difference_summary.json", diff_summary)

    plot_paths = {}
    if bool(args.save_plots):
        plots_dir = out_dir / "plots"
        ensure_dir(plots_dir)
        integer_idx, _, _, _ = infer_integer_idx(split_train["action_names"], args.integer_action_col)
        save_overlay_action_distribution_plots(
            action_names=split_train["action_names"],
            logged_actions=eval_split["act"],
            greedy_actions_by_policy=policy_greedy,
            integer_idx=integer_idx,
            out_path=plots_dir / "overlay_action_distributions_greedy.png",
        )
        plot_paths["overlay_action_distributions_greedy"] = str(plots_dir / "overlay_action_distributions_greedy.png")
        greedy_means = {name: np.mean(policy_greedy[name], axis=0) for name in saved_reward_names}
        save_overlay_action_mean_profile_plot(
            action_names=split_train["action_names"],
            train_action_mean=np.mean(split_train["act"], axis=0),
            train_action_std=np.std(split_train["act"], axis=0),
            logged_eval_action_mean=np.mean(eval_split["act"], axis=0),
            greedy_action_means_by_policy=greedy_means,
            out_path=plots_dir / "overlay_action_mean_profile_zscore.png",
        )
        plot_paths["overlay_action_mean_profile_zscore"] = str(plots_dir / "overlay_action_mean_profile_zscore.png")

    run_summary = {
        "training_method": "direct_reward_weighted_gaussian_mle",
        "seed": args.seed,
        "device": args.device,
        "overlap_controls": {
            "reward_weight_uniform_mix": float(args.reward_weight_uniform_mix),
            "overlap_anchor_lambda": float(args.overlap_anchor_lambda),
            "overlap_std_anchor_lambda": float(args.overlap_std_anchor_lambda),
            "policy_improvement_mix": float(args.policy_improvement_mix),
            "action_support_lower_quantile": float(args.action_support_lower_quantile),
            "action_support_upper_quantile": float(args.action_support_upper_quantile),
        },
        "train_blob": str(train_path),
        "test_blob": str(test_path) if test_path is not None else None,
        "raw_state_names": split_train_raw["raw_state_names"],
        "state_names": split_train["state_names"],
        "action_names": split_train["action_names"],
        "reward_names": split_train["reward_names"],
        "results": results,
        "policy_difference_summary_path": str(out_dir / "policy_difference_summary.json"),
        "plot_paths": plot_paths,
    }
    save_json(out_dir / "run_summary.json", run_summary)

    print("\nSaved artifact paths:")
    print(f"  {out_dir / 'state_encoder.json'}")
    print(f"  {out_dir / 'train_stats_raw.json'}")
    print(f"  {out_dir / 'run_summary.json'}")
    print(f"  {out_dir / 'policy_difference_summary.json'}")
    for reward_name in saved_reward_names:
        print(f"  {out_dir / f'linear_gaussian_policy_{reward_name}.npz'}")
        print(f"  {out_dir / f'linear_gaussian_policy_{reward_name}.json'}")
        print(f"  {out_dir / f'linear_gaussian_policy_{reward_name}_summary.json'}")
    print(f"\nFinished. Artifacts saved in: {out_dir}")


# public helpers

def get_linear_policy_greedy_actions(policy_npz: str, policy_json: str, states: np.ndarray) -> np.ndarray:
    return greedy_action(load_linear_gaussian_policy(policy_npz, policy_json), states)


def sample_linear_policy_actions(policy_npz: str, policy_json: str, states: np.ndarray, seed: int = 123) -> np.ndarray:
    return sample_action(load_linear_gaussian_policy(policy_npz, policy_json), states, rng=np.random.default_rng(seed))


def get_linear_policy_params(policy_npz: str, policy_json: str) -> Dict[str, np.ndarray]:
    policy = load_linear_gaussian_policy(policy_npz, policy_json)
    return {
        "theta_mu": policy.theta_mu,
        "epsilon_mu": policy.epsilon_mu,
        "theta_sigma": policy.theta_sigma,
        "epsilon_sigma": policy.epsilon_sigma,
        "action_lows": policy.action_lows,
        "action_highs": policy.action_highs,
        "integer_idx": None if policy.integer_idx is None else np.array([policy.integer_idx]),
        "integer_low": None if policy.integer_low is None else np.array([policy.integer_low]),
        "integer_high": None if policy.integer_high is None else np.array([policy.integer_high]),
    }


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
