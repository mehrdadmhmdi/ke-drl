#!/usr/bin/env python3
import argparse
import contextlib
import inspect
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import warnings

from expedia_preprocessing import fit_state_encoder, state_encoder_from_metadata

warnings.filterwarnings("ignore")
with open(os.devnull, "w") as fnull, contextlib.redirect_stderr(fnull):
    import d3rlpy

from ke_drl.api import (
    build_plot_config,
    compute_marginals_from_beta,
    estimate_embedding as _estimate_embedding_base,
    mean_embedding_all,
    plot_bellman_error,
    plot_densities,
    plot_operator_check_2d,
    plot_total_loss,
    recover_joint_beta,
)
from ke_drl.evaluation_metric import embedding_test_risk
from ke_drl.matern_kernel import matern_kernel


# ==================================== #
#               Defaults               #
# ==================================== #
LEGACY_FULL_STATE_COLS = [
    "srch_length_of_stay",
    "srch_room_count",
    "srch_saturday_night_bool",
    "prop_review_score",
    "prop_location_score2",
    "prop_location_score1",
    "prop_log_historical_price",
    "prop_starrating",
    "comp_rate",
    "comp_inv",
    "n_props",
    "mean_hist_price",
    "std_hist_price",
    "corr_pos_price",
    "corr_pos_review",
]

LEGACY_REDUCED_STATE_COLS = [
    "mean_hist_price",
    "std_hist_price",
    "prop_location_score1",
    "srch_room_count",
    "srch_length_of_stay",
    "prop_review_score",
    "comp_inv",
    "corr_pos_price",
]

DEFAULT_REWARD_COLS = ["gross_revenue_per_night", "total_clicks"]
SQRT_2 = math.sqrt(2.0)

import logging
import matplotlib.font_manager as fm


def _set_safe_matplotlib_fonts() -> None:
    """Use fonts that exist on most clusters and suppress Nimbus Roman fallback spam."""
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Liberation Serif", "Times New Roman"],
        "mathtext.fontset": "dejavuserif",
        "axes.unicode_minus": False,
    })


def _patch_findfont_nimbus() -> None:
    """Redirect any later request for Nimbus Roman to DejaVu Serif.

    Some plotting helpers may set font.family='Nimbus Roman' internally.  On many
    HPC systems that font is absent, so Matplotlib prints hundreds of
    `findfont` fallback messages.  This patch keeps plots visually serif while
    avoiding the warning noise.
    """
    if getattr(fm.findfont, "_ke_drl_safe_patch", False):
        return
    _orig_findfont = fm.findfont

    def _safe_findfont(prop, *args, **kwargs):
        try:
            fam = prop.get_family() if hasattr(prop, "get_family") else []
            fam_list = fam if isinstance(fam, (list, tuple)) else [fam]
            if any("Nimbus Roman" in str(x) for x in fam_list):
                prop = fm.FontProperties(family=["DejaVu Serif"])
        except Exception:
            pass
        return _orig_findfont(prop, *args, **kwargs)

    _safe_findfont._ke_drl_safe_patch = True
    fm.findfont = _safe_findfont


_set_safe_matplotlib_fonts()
_patch_findfont_nimbus()
# ==================================== #
#           Argument parsing           #
# ==================================== #
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate one or both reward-specific linear-Gaussian policies with KE-DRL."
    )

    p.add_argument("--cfg-index", type=int, default=None)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--run-test", type=int, default=1, help="0/1")
    p.add_argument("--do-plots", type=int, default=0, help="0/1")
    p.add_argument("--out-root", type=str, default="evaluation_results/gridsearch_runs")
    p.add_argument("--data-base", type=str, default="Expedia_data")

    p.add_argument("--train-blob", type=str, default="expedia_train_timeindexed.pt")
    p.add_argument("--val-blob", type=str, default="expedia_val_timeindexed.pt")
    p.add_argument("--test-blob", type=str, default="expedia_test_timeindexed.pt")

    p.add_argument("--max-train", type=int, default=10000)
    p.add_argument("--max-val", type=int, default=2000)
    p.add_argument("--max-test", type=int, default=3000)

    p.add_argument("--num-steps", type=int, default=5000)
    p.add_argument("--num-grid-points", type=int, default=300)
    p.add_argument(
        "--mean-embedding-basis-size",
        "--mean_embedding_basis_size",
        dest="mean_embedding_basis_size",
        type=int,
        default=0,
        help=(
            "State-action mean-embedding conditioning basis size L. "
            "Use 0 to let ke_drl use the full training state-action dictionary. "
            "When L>0 this is passed directly to ke_drl.api.estimate_embedding, "
            "whose native parameterization returns B_hat with shape L x m."
        ),
    )
    p.add_argument(
        "--mean-embedding-basis-ridge",
        "--mean_embedding_basis_ridge",
        dest="mean_embedding_basis_ridge",
        type=float,
        default=1e-6,
        help="Ridge used when projecting the full mean-embedding operator onto the L-point basis.",
    )
    p.add_argument(
        "--mean-embedding-basis-method",
        "--mean_embedding_basis_method",
        dest="mean_embedding_basis_method",
        type=str,
        default="kmeans",
        choices=["full", "all", "none", "random", "subsample", "uniform", "kmeans", "kmeans_landmarks", "landmark", "landmarks"],
        help="Native ke_drl conditioning-basis method for the L-row mean-embedding operator.",
    )
    p.add_argument(
        "--mean-embedding-basis-seed",
        "--mean_embedding_basis_seed",
        dest="mean_embedding_basis_seed",
        type=int,
        default=None,
        help="Seed for native ke_drl state-action basis selection. Defaults to --seed.",
    )
    p.add_argument(
        "--mean-embedding-basis-standardize",
        "--mean_embedding_basis_standardize",
        dest="mean_embedding_basis_standardize",
        type=int,
        default=1,
        help="0/1; standardize X=(S,A) before native kmeans/random basis selection diagnostics.",
    )
    p.add_argument(
        "--mean-embedding-basis-candidate-pool",
        "--mean_embedding_basis_candidate_pool",
        dest="mean_embedding_basis_candidate_pool",
        type=int,
        default=0,
        help="Candidate pool size for native ke_drl kmeans basis selection. Use 0 for package default.",
    )
    p.add_argument(
        "--mean-embedding-basis-max-iter",
        "--mean_embedding_basis_max_iter",
        dest="mean_embedding_basis_max_iter",
        type=int,
        default=20,
        help="Maximum native kmeans iterations for the mean-embedding basis.",
    )
    p.add_argument(
        "--mean-embedding-basis-batch-size",
        "--mean_embedding_basis_batch_size",
        dest="mean_embedding_basis_batch_size",
        type=int,
        default=8192,
        help="Batch size used by native ke_drl basis selection.",
    )
    p.add_argument(
        "--lambda-B",
        "--lambda_B",
        dest="lambda_B",
        type=float,
        default=0.0,
        help="Native ke_drl RKHS ridge on B: lambda_B tr(B.T K_U B).",
    )

    p.add_argument("--policy-objective",type=str,default="both",help="Reward name to evaluate, or 'both' to evaluate both reward-specific policies.")
    p.add_argument("--policy-ckpt-dir", type=str, default="checkpoints")

    p.add_argument("--state-cols",type=str,default=None,help="Optional comma-separated state variable names."    )
    p.add_argument("--action-cols",type=str,default=None,help="Optional comma-separated action variable names.",)
    p.add_argument("--reward-cols",type=str,default=None,help="Optional comma-separated reward variable names.")
    p.add_argument(
        "--categorical-state-cols",
        type=str,
        default="auto",
        help="Comma-separated state columns to one-hot encode, 'auto', or 'none'.",
    )
    p.add_argument("--one-hot-categoricals", type=int, default=1, help="0/1")
    p.add_argument("--max-auto-categorical-cardinality", type=int, default=20)
    p.add_argument("--state-encoder-path", type=str, default=None)

    p.add_argument("--full-state-cols-path", type=str, default=None)
    p.add_argument("--reduced-state-cols", type=str, default=",".join(LEGACY_REDUCED_STATE_COLS))
    p.add_argument("--reduced-idx", type=str, default=None)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--plot-n", type=int, default=3000)

    # KE-DRL / kernel parameters made explicit.
    p.add_argument("--nu-Z", type=float, default=3.5)
    p.add_argument("--ell-Z", type=float, default=0.8)
    p.add_argument("--lambda-reg", type=float, default=7e-3)
    p.add_argument("--sigma-Z", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--hull-expand-factor", type=float, default=1.0)
    p.add_argument("--embedding-lr", type=float, default=1e-3)
    p.add_argument("--fixed-point-constraint", type=int, default=1, help="0/1")
    p.add_argument("--fp-penalty-lambda", type=float, default=1e2)
    p.add_argument("--sum-one-W", type=int, default=1, help="0/1")
    p.add_argument("--nonneg-W", type=int, default=1, help="0/1")

    # Post-fit calibration: makes the learned conditional atom weights closer to
    # valid probability vectors on the training support.
    p.add_argument("--simplex-calibrate-B", type=int, default=1, help="0/1")
    p.add_argument("--simplex-calib-ridge", type=float, default=1e-3)
    p.add_argument("--simplex-calib-max-rows", type=int, default=5000)
    p.add_argument("--mass-anchor-lambda", type=float, default=10.0)
    p.add_argument("--target-mass", type=float, default=1.0)
    p.add_argument("--bandwidth", type=float, default=0.5)
    p.add_argument("--bandwidth-per-dim", type=str, default=None, help="Optional comma-separated bandwidths for reward marginals, one per reward dimension.")
    p.add_argument("--lambda-rec", type=float, default=1e-3)
    p.add_argument("--method", type=str, default="song")

    # Density recovery from embedding coefficients.  These control the
    # induced-embedding matching optimizer used after KE-DRL fitting.  This is
    # the same recovery logic used in plot_recovered_densities.py: do NOT treat
    # beta as a probability vector; find probability weights whose induced
    # embedding best matches beta.
    p.add_argument("--density-recovery-device", type=str, default="same", help="same/cpu/cuda; device for post-fit density recovery.")
    p.add_argument("--density-recovery-ridge", type=float, default=1e-4, help="Ridge in beta_tilde=(K+ridge I)^(-1)Aw.")
    p.add_argument("--density-recovery-lr", type=float, default=1e-2)
    p.add_argument("--density-recovery-steps", type=int, default=20000)
    p.add_argument("--density-recovery-tol", type=float, default=1e-10)
    p.add_argument("--density-recovery-init", type=str, default="positive_beta", choices=["uniform", "abs_beta", "positive_beta"])
    p.add_argument("--density-recovery-print-every", type=int, default=500)
    p.add_argument("--density-recovery-quad-points", type=int, default=31)
    p.add_argument("--density-recovery-batch-atoms", type=int, default=64)
    p.add_argument("--density-recovery-num-points", type=int, default=500)

    # Continuous 2D mean-embedding contour plot.  This is separate from density
    # recovery and always treats the two plotted reward coordinates as a
    # continuous evaluation canvas, even when one reward is count-valued.
    p.add_argument("--mean-embedding-contour-n1", type=int, default=160)
    p.add_argument("--mean-embedding-contour-n2", type=int, default=160)
    p.add_argument("--mean-embedding-contour-pad", type=float, default=0.05)
    p.add_argument("--mean-embedding-contour-levels", type=int, default=30)

    # Policy diagnostics.
    p.add_argument("--support-sample-n", type=int, default=1024)
    p.add_argument("--clip-sampled-actions", type=int, default=0, help="0/1")
    p.add_argument("--std-cap-for-diagnostics", type=float, default=None)
    p.add_argument("--require-exactly-two-rewards", type=int, default=1, help="0/1")
    p.add_argument("--integer-rounding-in-surrogate", type=int, default=0, help="0/1; kept for completeness, not used in KE-DRL surrogate")
    p.add_argument("--export-subsets", type=int, default=1, help="0/1")
    p.add_argument("--subsets-file", type=str, default=None)
    p.add_argument(
        "--discrete-reward-cols",
        type=str,
        default=None,
        help="Optional comma-separated reward column names to force as discrete, e.g. 'total_clicks'.",
    )

    return p.parse_args()


# ==================================== #
#               Utilities              #
# ==================================== #
def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_str: str) -> torch.device:
    if str(device_str).startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device("cpu")


def _parse_csv_list(x: Optional[str]) -> Optional[List[str]]:
    if x is None:
        return None
    x = x.strip()
    if x == "":
        return None
    return [c.strip() for c in x.split(",") if c.strip()]


def _resolve_discrete_reward_dims(
    discrete_reward_cols_arg: Optional[str],
    reward_cols: Sequence[str],
) -> List[int]:
    requested = _parse_csv_list(discrete_reward_cols_arg) or []
    if not requested:
        return []
    reward_cols = list(reward_cols)
    name_to_idx = {name: j for j, name in enumerate(reward_cols)}
    missing = [name for name in requested if name not in name_to_idx]
    if missing:
        raise ValueError(
            f"--discrete-reward-cols contains unknown reward columns {missing}. Available reward_cols={reward_cols}"
        )
    return [name_to_idx[name] for name in requested]


def _subsample_idx(n: int, max_n: Optional[int], seed: int) -> torch.Tensor:
    if max_n is None or max_n <= 0 or n <= max_n:
        return torch.arange(n)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return torch.randperm(n, generator=g)[:max_n]


def _zscore(x: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    return (x - mu) / (sd + 1e-12)


def _denorm(x: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    return x * sd + mu


def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _array_str(x, precision: int = 4) -> str:
    return np.array2string(np.asarray(x), precision=precision, suppress_small=False, max_line_width=160)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.size == 0 or y.size == 0:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def tic(msg: str) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.time()
    print(msg, flush=True)
    return t


def toc(t0: float, msg: str) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"{msg}: {time.time() - t0:.2f}s", flush=True)


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)



def _call_estimate_embedding_compatible(**kwargs):
    """Call ke_drl.api.estimate_embedding while remaining compatible with older APIs."""
    sig = inspect.signature(_estimate_embedding_base)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        print(
            "estimate_embedding does not accept these optional arguments; "
            f"they will be handled in policy_evaluation.py when possible: {dropped}",
            flush=True,
        )
    return _estimate_embedding_base(**filtered)


def _basis_size_from_args(args: argparse.Namespace, n_train: int) -> int:
    L = int(getattr(args, "mean_embedding_basis_size", 0) or 0)
    if L <= 0 or L >= int(n_train):
        return int(n_train)
    return max(1, L)


def _kmeanspp_basis_indices(
    x: torch.Tensor,
    L: int,
    seed: int,
) -> torch.Tensor:
    """Deterministic kmeans++-style row selection without running Lloyd iterations."""
    n = int(x.shape[0])
    L = min(max(1, int(L)), n)
    if L >= n:
        return torch.arange(n, device=x.device)

    x_work = x.detach()
    mu = x_work.mean(dim=0, keepdim=True)
    sd = x_work.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    xz = (x_work - mu) / sd

    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed))
    first = int(torch.randint(n, (1,), generator=gen, device=x.device).item())
    chosen = torch.empty(L, dtype=torch.long, device=x.device)
    chosen[0] = first
    min_dist = torch.sum((xz - xz[first:first + 1]) ** 2, dim=1).clamp_min(0.0)

    for t in range(1, L):
        probs = min_dist / min_dist.sum().clamp_min(1e-30)
        nxt = int(torch.multinomial(probs, 1, replacement=False, generator=gen).item())
        chosen[t] = nxt
        dist_new = torch.sum((xz - xz[nxt:nxt + 1]) ** 2, dim=1).clamp_min(0.0)
        min_dist = torch.minimum(min_dist, dist_new)
        min_dist[chosen[: t + 1]] = 0.0
    return chosen


def _select_mean_embedding_basis_indices(
    sa_train: torch.Tensor,
    L: int,
    seed: int,
    method: str,
) -> torch.Tensor:
    n = int(sa_train.shape[0])
    L = min(max(1, int(L)), n)
    if L >= n:
        return torch.arange(n, device=sa_train.device)

    method = str(method).strip().lower()
    if method == "first":
        return torch.arange(L, device=sa_train.device)
    if method == "kmeans++":
        return _kmeanspp_basis_indices(sa_train, L=L, seed=seed)

    gen = torch.Generator(device=sa_train.device)
    gen.manual_seed(int(seed))
    return torch.randperm(n, generator=gen, device=sa_train.device)[:L]


def _find_pre_basis_tensor(pre: dict, expected_rows: int, fallback_dim: int) -> Optional[torch.Tensor]:
    if not isinstance(pre, dict):
        return None
    candidate_keys = [
        "X_basis",                 # current ke_drl native key
        "sa_basis",                # compatibility aliases
        "SA_basis",
        "state_action_basis",
        "basis_sa",
        "basis_points",
        "basis",
        "dictionary_sa",
    ]
    for key in candidate_keys:
        val = pre.get(key, None)
        if isinstance(val, torch.Tensor) and val.ndim == 2 and int(val.shape[0]) == int(expected_rows):
            if int(val.shape[1]) == int(fallback_dim):
                return val
    return None


def _project_B_to_mean_embedding_basis(
    *,
    B_hat: torch.Tensor,
    pre: dict,
    sa_train: torch.Tensor,
    args: argparse.Namespace,
    nu: float,
    length_scale: float,
    sigma: float,
) -> Tuple[torch.Tensor, dict, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    r"""Resolve the native ke_drl L-point state-action conditioning basis.

    Current ke_drl implements the L-basis inside estimate_embedding.  It returns
    B_hat with shape (L, m), pre['X_basis'] with shape (L, d_s+d_a),
    pre['basis_indices'], and pre['mean_embedding_basis'] metadata.  This helper
    validates those objects and returns them for downstream validation/test
    kernels.  It keeps pre['k_sa'], pre['Phi'], and pre['K_sa'] in the package's
    native orientation so recover_joint_beta remains compatible.

    A post-fit projection fallback is kept only for older installed package
    versions that ignore the basis arguments and return a full N x m operator.
    """
    if not isinstance(pre, dict):
        pre = {}

    n_train = int(sa_train.shape[0])
    d_sa = int(sa_train.shape[1])
    requested_raw = int(getattr(args, "mean_embedding_basis_size", 0) or 0)
    requested_L = _basis_size_from_args(args, n_train)
    basis_method = str(getattr(args, "mean_embedding_basis_method", "kmeans"))
    basis_ridge = float(getattr(args, "mean_embedding_basis_ridge", 1e-6))

    if B_hat.ndim != 2:
        raise ValueError(f"B_hat must be 2D, got shape={tuple(B_hat.shape)}")

    B_rows = int(B_hat.shape[0])
    B_cols = int(B_hat.shape[1])

    # Native current ke_drl path: B rows are the selected conditioning basis size L.
    # The public package stores the basis as pre['X_basis'] and the indices as
    # pre['basis_indices'].
    native_basis = _find_pre_basis_tensor(pre, expected_rows=B_rows, fallback_dim=d_sa)
    native_meta = pre.get("mean_embedding_basis", {}) if isinstance(pre.get("mean_embedding_basis", {}), dict) else {}
    native_basis_idx = pre.get("basis_indices", pre.get("basis_idx", None))

    if native_basis is not None:
        basis = native_basis.to(device=sa_train.device, dtype=sa_train.dtype)
        if isinstance(native_basis_idx, torch.Tensor) and int(native_basis_idx.numel()) == B_rows:
            basis_idx = native_basis_idx.to(device=sa_train.device, dtype=torch.long)
        else:
            basis_idx = torch.arange(B_rows, device=sa_train.device, dtype=torch.long)

        K_basis_basis = pre.get("K_sa", pre.get("K_basis", None))
        if not isinstance(K_basis_basis, torch.Tensor) or tuple(K_basis_basis.shape) != (B_rows, B_rows):
            K_basis_basis = matern_kernel(basis, basis, nu=nu, length_scale=length_scale, sigma=sigma)
        else:
            K_basis_basis = K_basis_basis.to(device=B_hat.device, dtype=B_hat.dtype)

        pre_out = dict(pre)
        # Add aliases only.  Do not overwrite pre['k_sa']; in current ke_drl it is
        # k_X(X_basis, X_star), which recover_joint_beta expects.
        pre_out["X_basis"] = basis
        pre_out["sa_basis"] = basis
        pre_out["basis_indices"] = basis_idx
        pre_out["basis_idx"] = basis_idx
        pre_out["K_basis"] = K_basis_basis
        if "K_sa" not in pre_out or not isinstance(pre_out.get("K_sa"), torch.Tensor):
            pre_out["K_sa"] = K_basis_basis

        mode = "native_ke_drl_basis" if B_rows != n_train or requested_raw > 0 else "native_full_training_dictionary"
        info = {
            "mode": mode,
            "requested_basis_size": int(requested_raw),
            "effective_basis_size": int(B_rows),
            "train_dictionary_size": int(n_train),
            "B_shape_before": [int(B_rows), int(B_cols)],
            "B_shape_after": [int(B_rows), int(B_cols)],
            "basis_method": str(native_meta.get("method", basis_method)),
            "basis_ridge": float(basis_ridge),
            "ke_drl_native_basis_metadata": native_meta,
            "basis_indices_preview": [int(x) for x in basis_idx[: min(20, int(basis_idx.numel()))].detach().cpu().tolist()],
        }
        return B_hat, pre_out, basis, basis_idx, K_basis_basis, info

    # If no basis tensor was returned but the operator is full-size, treat the full
    # training dictionary as the basis.  This keeps compatibility with older full
    # dictionary fits and does not alter package-native pre['k_sa']/pre['Phi'].
    if B_rows == n_train and requested_L >= n_train:
        basis = sa_train
        basis_idx = torch.arange(n_train, device=sa_train.device, dtype=torch.long)
        K_basis_basis = pre.get("K_sa", pre.get("K_basis", None))
        if not isinstance(K_basis_basis, torch.Tensor) or tuple(K_basis_basis.shape) != (n_train, n_train):
            K_basis_basis = matern_kernel(basis, basis, nu=nu, length_scale=length_scale, sigma=sigma)
        else:
            K_basis_basis = K_basis_basis.to(device=B_hat.device, dtype=B_hat.dtype)
        pre_out = dict(pre)
        pre_out["X_basis"] = basis
        pre_out["sa_basis"] = basis
        pre_out["basis_indices"] = basis_idx
        pre_out["basis_idx"] = basis_idx
        pre_out["K_basis"] = K_basis_basis
        info = {
            "mode": "full_training_dictionary_no_native_basis_key",
            "requested_basis_size": int(requested_raw),
            "effective_basis_size": int(n_train),
            "train_dictionary_size": int(n_train),
            "B_shape_before": [int(B_rows), int(B_cols)],
            "B_shape_after": [int(B_rows), int(B_cols)],
            "basis_method": "full",
            "basis_ridge": float(basis_ridge),
        }
        return B_hat, pre_out, basis, basis_idx, K_basis_basis, info

    # Fallback only for older installed ke_drl versions that do not implement the
    # native L-basis and returned B with N rows even though L<N was requested.
    if B_rows != n_train:
        raise ValueError(
            "B_hat row count does not match the training dictionary, and no native "
            "basis tensor was found in pre. Current ke_drl should return pre['X_basis']."
        )

    basis_idx = _select_mean_embedding_basis_indices(
        sa_train=sa_train,
        L=requested_L,
        seed=int(getattr(args, "mean_embedding_basis_seed", None) or getattr(args, "seed", 0)) + 7919,
        method="random" if basis_method in {"kmeans", "kmeans_landmarks", "landmark", "landmarks"} else basis_method,
    )
    basis = sa_train.index_select(0, basis_idx)
    K_basis_basis = matern_kernel(basis, basis, nu=nu, length_scale=length_scale, sigma=sigma).to(dtype=B_hat.dtype)
    K_basis_train = matern_kernel(basis, sa_train, nu=nu, length_scale=length_scale, sigma=sigma).to(dtype=B_hat.dtype)
    eye = torch.eye(K_basis_basis.shape[0], device=K_basis_basis.device, dtype=K_basis_basis.dtype)
    B_basis = torch.linalg.solve(K_basis_basis + basis_ridge * eye, K_basis_train @ B_hat)

    # Recompute target-space k_sa and Phi for the fallback if enough package
    # internals were returned.  This preserves recover_joint_beta dimensions.
    pre_out = dict(pre)
    pre_out["X_basis"] = basis
    pre_out["sa_basis"] = basis
    pre_out["basis_indices"] = basis_idx
    pre_out["basis_idx"] = basis_idx
    pre_out["K_basis"] = K_basis_basis
    pre_out["K_sa"] = K_basis_basis
    if isinstance(pre.get("X_star", None), torch.Tensor):
        pre_out["k_sa"] = matern_kernel(basis, pre["X_star"].to(device=basis.device, dtype=basis.dtype), nu=nu, length_scale=length_scale, sigma=sigma).to(dtype=B_hat.dtype)
    if all(isinstance(pre.get(k, None), torch.Tensor) for k in ["X_successor", "Gamma", "eta_plus"]):
        K_basis_plus = matern_kernel(basis, pre["X_successor"].to(device=basis.device, dtype=basis.dtype), nu=nu, length_scale=length_scale, sigma=sigma).to(dtype=B_hat.dtype)
        pre_out["K_basis_plus"] = K_basis_plus
        pre_out["Phi"] = K_basis_plus @ (pre["Gamma"].to(device=K_basis_plus.device, dtype=K_basis_plus.dtype) * pre["eta_plus"].reshape(-1, 1).to(device=K_basis_plus.device, dtype=K_basis_plus.dtype))

    info = {
        "mode": "legacy_postfit_nystrom_projection_fallback",
        "requested_basis_size": int(requested_raw),
        "effective_basis_size": int(basis.shape[0]),
        "train_dictionary_size": int(n_train),
        "B_shape_before": [int(B_rows), int(B_cols)],
        "B_shape_after": [int(B_basis.shape[0]), int(B_basis.shape[1])],
        "basis_method": "postfit_random_fallback" if basis_method.startswith("kmeans") else basis_method,
        "basis_ridge": float(basis_ridge),
        "warning": "Native ke_drl L-basis was not used; update/install the current ke-drl package for the intended training-time L-basis.",
        "basis_indices_preview": [int(x) for x in basis_idx[: min(20, int(basis_idx.numel()))].detach().cpu().tolist()],
    }
    return B_basis, pre_out, basis, basis_idx, K_basis_basis, info

def _save_array_csv(path: Path, arr, col_names: Optional[Sequence[str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(arr, torch.Tensor):
        x = arr.detach().cpu().numpy()
    else:
        x = np.asarray(arr)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    header = "" if col_names is None else ",".join([str(c) for c in col_names])
    np.savetxt(path, x, delimiter=",", header=header, comments="")


def _save_atom_weight_table(
    path: Path,
    *,
    beta,
    density_weights,
    Z_norm,
    Z_raw,
    reward_cols: Sequence[str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    beta_np = np.asarray(beta.detach().cpu().numpy() if isinstance(beta, torch.Tensor) else beta, dtype=float).reshape(-1)
    w_np = np.asarray(
        density_weights.detach().cpu().numpy() if isinstance(density_weights, torch.Tensor) else density_weights,
        dtype=float,
    ).reshape(-1)
    Zn = np.asarray(Z_norm.detach().cpu().numpy() if isinstance(Z_norm, torch.Tensor) else Z_norm, dtype=float)
    Zr = np.asarray(Z_raw.detach().cpu().numpy() if isinstance(Z_raw, torch.Tensor) else Z_raw, dtype=float)
    m = beta_np.size
    if w_np.size != m or Zn.shape[0] != m or Zr.shape[0] != m:
        raise ValueError("Incompatible shapes for atom weight table export.")
    atom = np.arange(m, dtype=float).reshape(-1, 1)
    arr = np.column_stack([atom, beta_np, w_np, Zn, Zr])
    cols = ["atom_index", "beta_mean_embedding", "density_weight"]
    cols += [f"Z_norm_{c}" for c in reward_cols]
    cols += [f"Z_raw_{c}" for c in reward_cols]
    np.savetxt(path, arr, delimiter=",", header=",".join(cols), comments="")

def save_reproducibility_subsets(
    path: Path,
    *,
    cfg_index: int,
    seed: int,
    train_blob_path: str,
    val_blob_path: str,
    test_blob_path: str,
    state_cols: List[str],
    action_cols: List[str],
    reward_cols: List[str],
    idx_tr: torch.Tensor,
    idx_val: torch.Tensor,
    idx_test: torch.Tensor,
    s0_tr_raw: torch.Tensor,
    s1_tr_raw: Optional[torch.Tensor],
    a0_tr_raw: torch.Tensor,
    a1_tr_raw: Optional[torch.Tensor],
    r_tr_raw: torch.Tensor,
    s0_val_raw: torch.Tensor,
    a0_val_raw: torch.Tensor,
    r_val_raw: torch.Tensor,
    s0_test_raw: torch.Tensor,
    a0_test_raw: torch.Tensor,
    r_test_raw: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "cfg_index": int(cfg_index),
        "seed": int(seed),
        "train_blob_path": train_blob_path,
        "val_blob_path": val_blob_path,
        "test_blob_path": test_blob_path,
        "state_cols": list(state_cols),
        "action_cols": list(action_cols),
        "reward_cols": list(reward_cols),
        "idx_tr": idx_tr.clone().cpu(),
        "idx_val": idx_val.clone().cpu(),
        "idx_test": idx_test.clone().cpu(),
        "train": {
            "s0": s0_tr_raw.clone().cpu(),
            "s1": None if s1_tr_raw is None else s1_tr_raw.clone().cpu(),
            "a0": a0_tr_raw.clone().cpu(),
            "a1": None if a1_tr_raw is None else a1_tr_raw.clone().cpu(),
            "r": r_tr_raw.clone().cpu(),
        },
        "val": {
            "s0": s0_val_raw.clone().cpu(),
            "a0": a0_val_raw.clone().cpu(),
            "r": r_val_raw.clone().cpu(),
        },
        "test": {
            "s0": s0_test_raw.clone().cpu(),
            "a0": a0_test_raw.clone().cpu(),
            "r": r_test_raw.clone().cpu(),
        },
    }
    torch.save(payload, path)

    save_json(
        path.with_suffix(".json"),
        {
            "cfg_index": int(cfg_index),
            "seed": int(seed),
            "saved_pt": str(path),
            "train_n": int(s0_tr_raw.shape[0]),
            "val_n": int(s0_val_raw.shape[0]),
            "test_n": int(s0_test_raw.shape[0]),
            "state_cols": list(state_cols),
            "action_cols": list(action_cols),
            "reward_cols": list(reward_cols),
        },
    )

def _resolve_cfg(args: argparse.Namespace) -> dict:
    return {
        "nu_Z": float(args.nu_Z),
        "ell_Z": float(args.ell_Z),
        "lambda_reg": float(args.lambda_reg),
        "sigma_Z": float(args.sigma_Z),
    }

def _save_array_csv(path: Path, arr, col_names=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(arr, torch.Tensor):
        x = arr.detach().cpu().numpy()
    else:
        x = np.asarray(arr)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    header = "" if col_names is None else ",".join([str(c) for c in col_names])
    np.savetxt(path, x, delimiter=",", header=header, comments="")


# ==================================== #
#         Blob / metadata utils        #
# ==================================== #
def normalize_blob_payload(blob: dict) -> dict:
    if not isinstance(blob, dict):
        raise TypeError(f"Loaded blob must be a dict, got {type(blob)}")

    if any(k in blob for k in ["s0", "a0", "r0", "s1", "a1"]):
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
        f"Top-level keys: {list(blob.keys())}"
    )


def get_2d_tensor_from_blob(blob: dict, tensor_key: str) -> torch.Tensor:
    if tensor_key not in blob:
        raise KeyError(f"Blob missing key '{tensor_key}'.")
    x = blob[tensor_key]
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if x.ndim == 1:
        x = x.unsqueeze(1)
    if x.ndim != 2:
        raise ValueError(f"{tensor_key} must be 1D or 2D, got shape={tuple(x.shape)}")
    return x.float()


def load_full_state_cols(maybe_path: Optional[Path]) -> Optional[List[str]]:
    if maybe_path is None or (not maybe_path.exists()):
        return None
    if maybe_path.suffix.lower() == ".json":
        cols = json.loads(maybe_path.read_text())
        if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
            raise ValueError(f"{maybe_path} must be a JSON list of strings.")
        return cols
    cols = [ln.strip() for ln in maybe_path.read_text().splitlines() if ln.strip()]
    return cols or None


def default_names_for_blob_section(
    blob: dict,
    tensor_key: str,
    names_key: str,
    external_state_cols: Optional[List[str]] = None,
) -> List[str]:
    x = get_2d_tensor_from_blob(blob, tensor_key)

    names = blob.get(names_key, None)
    if names is not None:
        if len(names) != x.shape[1]:
            raise ValueError(
                f"{names_key} has length {len(names)} but {tensor_key} has {x.shape[1]} columns."
            )
        return list(names)

    if tensor_key == "s0" and external_state_cols is not None:
        if len(external_state_cols) != x.shape[1]:
            raise ValueError(
                f"external_state_cols length {len(external_state_cols)} != s0 dim {x.shape[1]}"
            )
        return list(external_state_cols)

    if tensor_key == "r0" and x.shape[1] == len(DEFAULT_REWARD_COLS):
        return list(DEFAULT_REWARD_COLS)

    return [f"{tensor_key}_{j}" for j in range(x.shape[1])]


def select_named_columns(
    blob: dict,
    wanted_cols: List[str],
    tensor_key: str,
    names_key: str,
    external_state_cols: Optional[List[str]] = None,
) -> torch.Tensor:
    x = get_2d_tensor_from_blob(blob, tensor_key)
    names = default_names_for_blob_section(
        blob=blob,
        tensor_key=tensor_key,
        names_key=names_key,
        external_state_cols=external_state_cols,
    )
    name_to_idx = {name: j for j, name in enumerate(names)}
    missing = [c for c in wanted_cols if c not in name_to_idx]
    if missing:
        raise ValueError(
            f"Requested columns not found in {tensor_key}: {missing}\n"
            f"Available: {sorted(name_to_idx.keys())}"
        )
    idx = [name_to_idx[c] for c in wanted_cols]
    return x[:, idx].float()


def select_named_pair(
    blob: dict,
    wanted_cols: List[str],
    current_key: str,
    next_key: str,
    names_key: str,
    external_state_cols: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    x0 = get_2d_tensor_from_blob(blob, current_key)
    x1 = get_2d_tensor_from_blob(blob, next_key) if next_key in blob else None

    names = default_names_for_blob_section(
        blob=blob,
        tensor_key=current_key,
        names_key=names_key,
        external_state_cols=external_state_cols,
    )
    if len(names) != x0.shape[1]:
        raise ValueError(
            f"{names_key} has length {len(names)} but {current_key} has {x0.shape[1]} columns."
        )
    if x1 is not None and x1.shape[1] != x0.shape[1]:
        raise ValueError(f"{current_key} and {next_key} must have same dimension.")

    name_to_idx = {name: j for j, name in enumerate(names)}
    missing = [c for c in wanted_cols if c not in name_to_idx]
    if missing:
        raise ValueError(
            f"Requested cols not found in {current_key}/{next_key}: {missing}\n"
            f"Available: {sorted(name_to_idx.keys())}"
        )

    idx = [name_to_idx[c] for c in wanted_cols]
    cur = x0[:, idx].float()
    nxt = x1[:, idx].float() if x1 is not None else None
    return cur, nxt


def resolve_reduced_indices(
    full_state_cols: Optional[List[str]],
    reduced_state_cols: List[str],
    reduced_idx: Optional[List[int]],
    full_state_dim: int,
) -> List[int]:
    if reduced_idx:
        return reduced_idx
    if full_state_cols is not None:
        state_idx = {name: i for i, name in enumerate(full_state_cols)}
        missing = [c for c in reduced_state_cols if c not in state_idx]
        if missing:
            raise ValueError(f"Reduced cols not found in full_state_cols: {missing}")
        return [state_idx[c] for c in reduced_state_cols]
    k = len(reduced_state_cols)
    if full_state_dim < k:
        raise ValueError(f"Full state dim {full_state_dim} < reduced dim {k}.")
    return list(range(k))


# ==================================== #
#        Diagnostics / summaries       #
# ==================================== #
def summarize_tensor_by_columns(
    x: torch.Tensor,
    col_names: List[str],
    title: str,
    quantiles: Tuple[float, ...] = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99),
) -> Dict[str, dict]:
    x_np = _np(x).astype(np.float64)
    out: Dict[str, dict] = {}

    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    for j, name in enumerate(col_names):
        v = x_np[:, j]
        qvals = np.quantile(v, quantiles)
        stats = {
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "quantiles": {str(q): float(val) for q, val in zip(quantiles, qvals)},
        }
        out[name] = stats
        qtxt = " | ".join([f"q{int(100*q):02d}={val:.4f}" for q, val in zip(quantiles, qvals)])
        print(
            f"{name:30s} "
            f"mean={stats['mean']:10.4f} std={stats['std']:10.4f} "
            f"min={stats['min']:10.4f} max={stats['max']:10.4f} | {qtxt}"
        )
    return out


def _finite_rows(x: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(x), axis=1)


def off_support_rate_against_train(
    a_candidate_raw: torch.Tensor,
    a_train_raw: torch.Tensor,
    action_cols: List[str],
    tol: float = 0.0,
    title: str = "OFF-SUPPORT CHECK",
) -> Dict[str, dict]:
    cand = _np(a_candidate_raw)
    train = _np(a_train_raw)

    finite_rows = _finite_rows(cand)
    cand_f = cand[finite_rows] if np.any(finite_rows) else np.zeros((0, cand.shape[1]), dtype=cand.dtype)

    train_min = train.min(axis=0)
    train_max = train.max(axis=0)

    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(f"finite_row_fraction={float(finite_rows.mean()) if finite_rows.size else float('nan'):.6f}")

    out: Dict[str, dict] = {}
    for j, name in enumerate(action_cols):
        if cand_f.shape[0] == 0:
            stats = {
                "train_min": float(train_min[j]),
                "train_max": float(train_max[j]),
                "candidate_mean": float("nan"),
                "candidate_min": float("nan"),
                "candidate_max": float("nan"),
                "frac_below_min": float("nan"),
                "frac_above_max": float("nan"),
                "frac_outside_range": float("nan"),
            }
        else:
            below = cand_f[:, j] < (train_min[j] - tol)
            above = cand_f[:, j] > (train_max[j] + tol)
            outside = below | above
            stats = {
                "train_min": float(train_min[j]),
                "train_max": float(train_max[j]),
                "candidate_mean": float(cand_f[:, j].mean()),
                "candidate_min": float(cand_f[:, j].min()),
                "candidate_max": float(cand_f[:, j].max()),
                "frac_below_min": float(np.mean(below)),
                "frac_above_max": float(np.mean(above)),
                "frac_outside_range": float(np.mean(outside)),
            }
        out[name] = stats
        print(
            f"{name:30s} "
            f"train[min,max]=[{stats['train_min']:10.4f}, {stats['train_max']:10.4f}] "
            f"cand[min,max,mean]=[{stats['candidate_min']:10.4f}, {stats['candidate_max']:10.4f}, {stats['candidate_mean']:10.4f}] "
            f"| below={stats['frac_below_min']:.4f} above={stats['frac_above_max']:.4f} outside={stats['frac_outside_range']:.4f}"
        )

    any_off_support = False
    if cand_f.shape[0] > 0:
        any_off_support = bool(np.any((cand_f < train_min) | (cand_f > train_max)))
    print(f"\nAny off-support action at all? {any_off_support}")
    out["finite_row_fraction"] = float(finite_rows.mean()) if finite_rows.size else float("nan")
    out["n_rows"] = int(cand.shape[0])
    out["n_finite_rows"] = int(finite_rows.sum())
    return out


def _robust_hist_range(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    z = np.concatenate([a.reshape(-1), b.reshape(-1)])
    z = z[np.isfinite(z)]
    if z.size == 0:
        return (0.0, 1.0)
    lo = float(np.quantile(z, 0.005))
    hi = float(np.quantile(z, 0.995))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(z))
        hi = float(np.max(z))
        if hi <= lo:
            hi = lo + 1.0
    return lo, hi


def save_action_hist_compare(
    a_data_raw: torch.Tensor,
    a_policy_raw: torch.Tensor,
    action_cols: List[str],
    out_dir: Path,
    prefix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data_np = _np(a_data_raw)
    pol_np = _np(a_policy_raw)

    for j, name in enumerate(action_cols):
        x = data_np[:, j]
        y = pol_np[:, j]
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        if x.size == 0 or y.size == 0:
            continue
        lo, hi = _robust_hist_range(x, y)
        fig = plt.figure()
        plt.hist(x, bins=60, range=(lo, hi), alpha=0.6, label="data")
        plt.hist(y, bins=60, range=(lo, hi), alpha=0.6, label="policy")
        plt.title(f"{prefix}: {name} (raw scale)")
        plt.legend()
        fig.savefig(out_dir / f"{prefix}_{name}_raw_hist.png", bbox_inches="tight")
        plt.close(fig)


def _to_cpu_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, dict):
        return {k: _to_cpu_serializable(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_to_cpu_serializable(v) for v in obj)
    if isinstance(obj, list):
        return [_to_cpu_serializable(v) for v in obj]
    return obj


def _collect_named_arrays(obj, prefix: str = "root") -> List[Tuple[str, np.ndarray]]:
    out: List[Tuple[str, np.ndarray]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_collect_named_arrays(v, f"{prefix}.{k}"))
        return out
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.extend(_collect_named_arrays(v, f"{prefix}[{i}]"))
        return out
    if isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().numpy()
    elif isinstance(obj, np.ndarray):
        arr = obj
    else:
        try:
            arr = np.asarray(obj)
        except Exception:
            return out
    if arr.dtype == object or arr.size == 0:
        return out
    out.append((prefix, arr))
    return out


def _normalize_nonnegative_density(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.maximum(y, 0.0)
    if y.shape[0] == x.shape[0] and x.shape[0] > 1:
        area = np.trapezoid(y, x)
        if area > 0:
            y = y / area
    return y


def _normalize_joint_surface_by_kind(
    z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    kind_dim0: str = "continuous",
    kind_dim1: str = "continuous",
) -> np.ndarray:
    """
    Normalize a 2D reward display on axes:
      rows -> reward dimension 0, grid x
      cols -> reward dimension 1, grid y

    Correct normalization depends on reward type:
      continuous/continuous:  int int f(x,y) dy dx = 1
      continuous/discrete  :  sum_y int p(y,x) dx = 1
      discrete/continuous  :  sum_x int p(x,y) dy = 1
      discrete/discrete    :  sum_x sum_y p(x,y) = 1
    """
    z = np.asarray(z, dtype=float)
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    z = np.maximum(z, 0.0)

    if z.ndim != 2 or z.shape != (x.size, y.size):
        return z

    k0 = str(kind_dim0).lower()
    k1 = str(kind_dim1).lower()

    area = 0.0
    if k0 == "continuous" and k1 == "continuous":
        if x.size > 1 and y.size > 1:
            area = float(np.trapezoid(np.trapezoid(z, y, axis=1), x))
    elif k0 == "continuous" and k1 == "discrete":
        if x.size > 1:
            area = float(np.trapezoid(z.sum(axis=1), x))
    elif k0 == "discrete" and k1 == "continuous":
        if y.size > 1:
            area = float(np.trapezoid(z.sum(axis=0), y))
    else:
        area = float(z.sum())

    if area > 0.0 and np.isfinite(area):
        z = z / area
    return z


def _normalize_surface(z: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Backward-compatible default: both dimensions continuous.
    return _normalize_joint_surface_by_kind(
        z=z, x=x, y=y, kind_dim0="continuous", kind_dim1="continuous"
    )


def _parse_optional_float_list(x: Optional[str]) -> Optional[List[float]]:
    if x is None:
        return None
    x = str(x).strip()
    if x == "":
        return None
    vals = [float(v.strip()) for v in x.split(",") if v.strip()]
    return vals or None


def _resolve_marginal_bandwidths(
    reward_dim: int,
    scalar_bandwidth: float,
    per_dim_bandwidth: Optional[Sequence[float]],
) -> List[float]:
    if per_dim_bandwidth is None:
        return [float(scalar_bandwidth)] * int(reward_dim)

    vals = [float(v) for v in per_dim_bandwidth]
    if len(vals) == 1:
        return vals * int(reward_dim)
    if len(vals) != int(reward_dim):
        raise ValueError(
            f"bandwidth_per_dim must have length 1 or reward_dim={reward_dim}, got {len(vals)}."
        )
    if any(v <= 0 for v in vals):
        raise ValueError(f"All marginal bandwidths must be > 0, got {vals}.")
    return vals


def _project_to_simplex(v: torch.Tensor) -> torch.Tensor:
    """Euclidean projection onto the probability simplex {w >= 0, sum w = 1}."""
    x = torch.as_tensor(v, dtype=torch.float64).reshape(-1)
    if x.numel() == 0:
        return x.to(dtype=torch.float32)
    if x.numel() == 1:
        return torch.ones_like(x, dtype=torch.float32)

    u, _ = torch.sort(x, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1.0
    ind = torch.arange(1, x.numel() + 1, device=x.device, dtype=x.dtype)
    cond = u - cssv / ind > 0
    if not bool(torch.any(cond)):
        return torch.full_like(x, 1.0 / x.numel(), dtype=torch.float64).to(dtype=torch.float32)
    rho = int(ind[cond][-1].item())
    theta = cssv[rho - 1] / float(rho)
    w = torch.clamp(x - theta, min=0.0)
    total = float(w.sum().item())
    if total <= 0.0 or not np.isfinite(total):
        return torch.full_like(x, 1.0 / x.numel(), dtype=torch.float64).to(dtype=torch.float32)
    return (w / total).to(dtype=torch.float32)

def _project_rows_to_simplex(W: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
    W = torch.as_tensor(W, dtype=torch.float32)
    rows = []
    for start in range(0, W.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), W.shape[0])
        rows.append(torch.stack([_project_to_simplex(row) for row in W[start:end]], dim=0))
    return torch.cat(rows, dim=0).to(dtype=W.dtype, device=W.device)


def _simplex_diagnostics(W: torch.Tensor, prefix: str) -> Dict[str, float]:
    W = torch.as_tensor(W)
    out = {
        f"{prefix}_min": float(W.min().detach().cpu()),
        f"{prefix}_max": float(W.max().detach().cpu()),
        f"{prefix}_row_sum_mean": float(W.sum(1).mean().detach().cpu()),
        f"{prefix}_row_sum_std": float(W.sum(1).std().detach().cpu()),
        f"{prefix}_negative_mass_mean": float(torch.clamp(-W, min=0).sum(1).mean().detach().cpu()),
        f"{prefix}_l1_mass_mean": float(W.abs().sum(1).mean().detach().cpu()),
    }
    return out


def _calibrate_B_hat_to_simplex(
    *,
    B_hat: torch.Tensor,
    sa_train: torch.Tensor,
    nu: float,
    length_scale: float,
    sigma: float,
    ridge: float,
    max_rows: int,
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Post-fit calibration.

    Step 1: compute training weights W_train = K_train @ B_hat.
    Step 2: project each row of W_train onto the probability simplex.
    Step 3: solve (K_train + ridge I) B_cal = W_projected.

    This preserves the RKHS operator form but makes the implied atom weights
    probability-like on the training support.
    """
    device = B_hat.device
    dtype = B_hat.dtype
    n = sa_train.shape[0]

    if max_rows is not None and int(max_rows) > 0 and n > int(max_rows):
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        idx = torch.randperm(n, generator=g)[: int(max_rows)]
        idx = idx.to(sa_train.device)
        sa_fit = sa_train.index_select(0, idx)
    else:
        sa_fit = sa_train

    K_fit = matern_kernel(
        sa_fit,
        sa_fit,
        nu=float(nu),
        length_scale=float(length_scale),
        sigma=float(sigma),
    ).to(device=device, dtype=dtype)

    W_raw = K_fit @ B_hat
    W_proj = _project_rows_to_simplex(W_raw)

    eye = torch.eye(K_fit.shape[0], device=device, dtype=dtype)
    B_cal = torch.linalg.solve(K_fit + float(ridge) * eye, W_proj)

    # If calibration used a subset, B_cal has subset rows. That is not compatible
    # with k_sa @ B_hat for the full training dictionary. Therefore, when max_rows
    # subsamples, return an operator tied to the subset and diagnostics only.
    # To keep the rest of the code unchanged, require full calibration.
    if sa_fit.shape[0] != sa_train.shape[0]:
        raise ValueError(
            "simplex calibration currently requires --simplex-calib-max-rows >= max_train "
            "or set --simplex-calib-max-rows 0 to use all rows."
        )

    diag = {}
    diag.update(_simplex_diagnostics(W_raw, "before_calibration"))
    diag.update(_simplex_diagnostics(K_fit @ B_cal, "after_calibration"))
    diag["simplex_calib_ridge"] = float(ridge)
    diag["simplex_calib_n_rows"] = int(sa_fit.shape[0])
    return B_cal.to(device=device, dtype=dtype), diag

def _safe_quadratic_value(w: torch.Tensor, beta: torch.Tensor, K: torch.Tensor) -> float:
    d = (w - beta).reshape(-1, 1)
    val = (d.T @ K @ d).squeeze()
    return float(val.detach().cpu().item())


def _rkhs_metric_simplex_projection(
    beta: torch.Tensor,
    K_Z: torch.Tensor,
    max_iter: int = 2000,
    tol: float = 1e-10,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    r"""
    Project raw embedding coefficients onto the simplex using RKHS geometry:

        min_{w in Delta_m} (w - beta)^T K_Z (w - beta),
        Delta_m = {w_i >= 0, sum_i w_i = 1}.

    This is the probability measure on the atoms that is closest to the raw
    coefficient embedding in the RKHS metric induced by the atom Gram matrix.
    """
    beta64 = torch.as_tensor(beta, dtype=torch.float64).reshape(-1).cpu()
    K64 = torch.as_tensor(K_Z, dtype=torch.float64).cpu()
    m = beta64.numel()
    if K64.shape != (m, m):
        raise ValueError(f"K_Z must have shape {(m, m)}, got {tuple(K64.shape)}")

    K64 = 0.5 * (K64 + K64.T)
    eye = torch.eye(m, dtype=torch.float64)

    # Make the quadratic numerically PSD.  Kernel matrices can have tiny negative
    # eigenvalues from float32 construction or support snapping.
    eigvals = torch.linalg.eigvalsh(K64)
    eig_min_before = float(eigvals.min().item())
    eig_max_before = float(eigvals.max().item())
    eig_min = eig_min_before
    eig_max = eig_max_before
    jitter = max(0.0, -eig_min_before + 1e-10)
    if jitter > 0.0:
        K64 = K64 + jitter * eye
        eigvals = torch.linalg.eigvalsh(K64)
        eig_min = float(eigvals.min().item())
        eig_max = float(eigvals.max().item())

    L = max(float(eigvals.max().item()), 1e-8)

    # FISTA with Euclidean simplex projection.  This solves the convex QP well
    # enough for visualization/export without adding a scipy/cvxpy dependency.
    w = _project_to_simplex(beta64).to(dtype=torch.float64)
    y = w.clone()
    t = 1.0
    obj_prev = _safe_quadratic_value(w, beta64, K64)
    n_iter_done = 0

    for it in range(int(max_iter)):
        grad = K64 @ (y - beta64)
        w_new = _project_to_simplex(y - grad / L).to(dtype=torch.float64)
        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        y = w_new + ((t - 1.0) / t_new) * (w_new - w)

        step = float(torch.linalg.norm(w_new - w).item())
        obj_new = _safe_quadratic_value(w_new, beta64, K64)
        w = w_new
        t = t_new
        n_iter_done = it + 1
        if abs(obj_prev - obj_new) <= tol * max(1.0, abs(obj_prev)) and step <= math.sqrt(tol):
            obj_prev = obj_new
            break
        obj_prev = obj_new

    w = _project_to_simplex(w).to(dtype=torch.float64)
    euclid_w = _project_to_simplex(beta64).to(dtype=torch.float64)

    diag = {
        "projection_method": "rkhs_metric_simplex_projected_gradient",
        "projection_objective": "min_(w in simplex) (w-beta)^T K_Z (w-beta)",
        "projection_iterations": int(n_iter_done),
        "projection_tol": float(tol),
        "projection_step_L": float(L),
        "K_eig_min_before_jitter": float(eig_min_before),
        "K_eig_max_before_jitter": float(eig_max_before),
        "K_eig_min_after_jitter": float(eig_min),
        "K_eig_max_after_jitter": float(eig_max),
        "K_jitter_added": float(jitter),
        "metric_distance_sq": _safe_quadratic_value(w, beta64, K64),
        "metric_distance_sq_euclidean_simplex_projection": _safe_quadratic_value(euclid_w, beta64, K64),
        "euclidean_l2_shift": float(torch.linalg.norm(w - beta64).item()),
        "projection_l1_shift": float(torch.abs(w - beta64).sum().item()),
        "simplex_sum": float(w.sum().item()),
        "simplex_min": float(w.min().item()),
        "simplex_max": float(w.max().item()),
        "simplex_n_positive": int((w > 1e-12).sum().item()),
    }
    return w.to(dtype=torch.float32), diag


def _density_projection_gram(
    Z_grid_kernel: Optional[torch.Tensor],
    *,
    nu: Optional[float],
    length_scale: Optional[float],
    sigma: Optional[float],
) -> Optional[torch.Tensor]:
    if Z_grid_kernel is None or nu is None or length_scale is None:
        return None
    Z = torch.as_tensor(Z_grid_kernel, dtype=torch.float32).detach().cpu()
    if Z.ndim != 2 or Z.shape[0] == 0:
        return None
    K = matern_kernel(
        Z,
        Z,
        nu=float(nu),
        length_scale=float(length_scale),
        sigma=float(1.0 if sigma is None else sigma),
    )
    return K.detach().cpu()


def _prepare_density_weights(
    beta_full: torch.Tensor,
    n_atoms: int,
    K_Z: Optional[torch.Tensor] = None,
    projection_max_iter: int = 2000,
    projection_tol: float = 1e-10,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    beta = torch.as_tensor(beta_full, dtype=torch.float32).reshape(-1)
    if beta.numel() != int(n_atoms):
        raise ValueError(f"beta_full has {beta.numel()} elements but Z_grid has {n_atoms} atoms.")

    beta = torch.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)

    base_diag = {
        "raw_sum": float(beta.sum().item()),
        "raw_min": float(beta.min().item()) if beta.numel() else float("nan"),
        "raw_max": float(beta.max().item()) if beta.numel() else float("nan"),
        "raw_l1_negative_mass": float(torch.clamp(-beta, min=0.0).sum().item()),
        "raw_l1_mass": float(torch.abs(beta).sum().item()),
        "n_atoms": int(beta.numel()),
    }

    if K_Z is None:
        w = _project_to_simplex(beta)
        diag = {
            **base_diag,
            "projection_method": "euclidean_simplex_fallback_no_K_Z",
            "projection_objective": "min_(w in simplex) ||w-beta||_2^2",
            "projection_l1_shift": float(torch.abs(w - beta).sum().item()),
            "euclidean_l2_shift": float(torch.linalg.norm(w - beta).item()),
            "simplex_sum": float(w.sum().item()),
            "simplex_min": float(w.min().item()),
            "simplex_max": float(w.max().item()),
            "simplex_n_positive": int((w > 1e-12).sum().item()),
        }
        return w, diag

    w, metric_diag = _rkhs_metric_simplex_projection(
        beta=beta,
        K_Z=K_Z,
        max_iter=int(projection_max_iter),
        tol=float(projection_tol),
    )
    return w, {**base_diag, **metric_diag}

def _finite_1d_np(x: Optional[np.ndarray]) -> np.ndarray:
    if x is None:
        return np.asarray([], dtype=float)
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def _name_suggests_discrete(name: str) -> bool:
    s = str(name).strip().lower()
    keys = ["click", "count", "booking", "impression", "visit", "n_book", "num_"]
    return any(k in s for k in keys)


def _name_suggests_nonnegative(name: str) -> bool:
    s = str(name).strip().lower()
    keys = ["revenue", "sales", "price", "click", "count", "booking", "cost", "amount"]
    return any(k in s for k in keys)


def _is_nearly_integer_valued(x: np.ndarray, tol: float = 0.15, frac_threshold: float = 0.98) -> bool:
    arr = _finite_1d_np(x)
    if arr.size == 0:
        return False
    rounded = np.round(arr)
    frac = np.mean(np.abs(arr - rounded) <= tol)
    return bool(frac >= frac_threshold)


def _infer_reward_kind(name: str, observed_vals: Optional[np.ndarray], atom_vals: np.ndarray) -> str:
    obs = _finite_1d_np(observed_vals)
    atoms = _finite_1d_np(atom_vals)
    pooled = np.concatenate([obs, atoms]) if obs.size else atoms
    if _name_suggests_discrete(name):
        return "discrete"
    if pooled.size == 0:
        return "continuous"
    if _is_nearly_integer_valued(pooled):
        uniq = np.unique(np.round(pooled).astype(int))
        if uniq.size <= 80:
            return "discrete"
    return "continuous"


def _infer_lower_bound(name: str, observed_vals: Optional[np.ndarray], atom_vals: np.ndarray) -> Optional[float]:
    obs = _finite_1d_np(observed_vals)
    atoms = _finite_1d_np(atom_vals)

    # If the observed raw data are nonnegative, respect that support regardless of recovered atoms.
    if obs.size > 0:
        if _name_suggests_nonnegative(name):
            if np.nanmin(obs) >= -1e-8:
                return 0.0
        if np.nanmin(obs) >= -1e-8 and np.nanquantile(obs, 0.01) >= -1e-6:
            return 0.0

    pooled = np.concatenate([obs, atoms]) if obs.size else atoms
    if pooled.size == 0:
        return None
    if _name_suggests_nonnegative(name) and np.nanmin(pooled) >= -1e-8:
        return 0.0
    if np.nanmin(pooled) >= -1e-8 and np.nanquantile(pooled, 0.01) >= -1e-6:
        return 0.0
    return None


def _build_discrete_support(observed_vals: Optional[np.ndarray], atom_vals: np.ndarray, max_support_size: int = 80) -> np.ndarray:
    obs = _finite_1d_np(observed_vals)
    atoms = _finite_1d_np(atom_vals)

    # For truly discrete observed variables, use the observed support only.
    if obs.size > 0 and _is_nearly_integer_valued(obs):
        obs_int = np.unique(np.round(obs).astype(int))
        if obs_int.size <= max_support_size:
            return obs_int.astype(float)
        lo = int(np.nanquantile(obs_int, 0.001))
        hi = int(np.nanquantile(obs_int, 0.999))
        if hi >= lo and (hi - lo + 1) <= max_support_size:
            return np.arange(lo, hi + 1, dtype=float)

    pooled = np.concatenate([obs, atoms]) if obs.size else atoms
    if pooled.size == 0:
        return np.asarray([0.0], dtype=float)

    if _is_nearly_integer_valued(pooled):
        pooled_int = np.round(pooled).astype(int)
        uniq = np.unique(pooled_int)
        if uniq.size <= max_support_size:
            return uniq.astype(float)
        lo = int(np.nanquantile(pooled_int, 0.001))
        hi = int(np.nanquantile(pooled_int, 0.999))
        if hi >= lo and (hi - lo + 1) <= max_support_size:
            return np.arange(lo, hi + 1, dtype=float)

    uniq = np.unique(np.round(pooled, 6))
    if uniq.size <= max_support_size:
        return uniq.astype(float)

    q = np.linspace(0.0, 1.0, max_support_size)
    return np.unique(np.quantile(pooled, q)).astype(float)


def _snap_to_support(values: np.ndarray, support: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float).reshape(-1)
    sup = np.asarray(support, dtype=float).reshape(-1)
    if vals.size == 0 or sup.size == 0:
        return vals.astype(float)
    idx = np.abs(vals[:, None] - sup[None, :]).argmin(axis=1)
    return sup[idx].astype(float)


def _enforce_reward_support_raw(
    Z_grid_raw: torch.Tensor,
    reward_cols: List[str],
    observed_rewards_raw: Optional[torch.Tensor] = None,
    force_discrete_dims: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, Dict[str, dict]]:
    """
    Enforce known raw-scale support constraints on reward atoms.

    - Discrete/count-like rewards are snapped to the observed integer support.
    - Nonnegative rewards are clipped below at 0.

    This is applied before evaluation plots and before the holdout-risk calculation
    that depends on reward atom locations.

    force_discrete_dims can be used to override the heuristic detection for
    selected reward dimensions.
    """
    Z_raw = torch.as_tensor(Z_grid_raw, dtype=torch.float32).detach().cpu().clone()
    if Z_raw.ndim != 2:
        raise ValueError(f"Z_grid_raw must be 2D, got shape={tuple(Z_raw.shape)}")

    obs_np = None
    if observed_rewards_raw is not None:
        obs_np = torch.as_tensor(observed_rewards_raw).detach().cpu().numpy().astype(float)
        if obs_np.ndim != 2 or obs_np.shape[1] != Z_raw.shape[1]:
            raise ValueError(
                f"observed_rewards_raw must have shape (n,{Z_raw.shape[1]}), got {tuple(obs_np.shape)}"
            )

    out = Z_raw.clone()
    info: Dict[str, dict] = {}
    forced_discrete_set = set(int(j) for j in (force_discrete_dims or []))

    for j in range(Z_raw.shape[1]):
        name = reward_cols[j] if j < len(reward_cols) else f"reward_{j}"
        z_j = Z_raw[:, j].numpy().astype(float)
        obs_j = None if obs_np is None else obs_np[:, j]
        kind = (
            "discrete"
            if j in forced_discrete_set
            else _infer_reward_kind(name=name, observed_vals=obs_j, atom_vals=z_j)
        )
        lower_bound = _infer_lower_bound(name=name, observed_vals=obs_j, atom_vals=z_j)

        support_used = None
        if kind == "discrete":
            support = _build_discrete_support(observed_vals=obs_j, atom_vals=np.asarray([], dtype=float))
            if support.size == 0:
                support = _build_discrete_support(observed_vals=obs_j, atom_vals=z_j)
            z_new = _snap_to_support(z_j, support)
            support_used = support.astype(float).tolist()
        else:
            z_new = z_j.copy()

        if lower_bound is not None:
            z_new = np.maximum(z_new, float(lower_bound))

        out[:, j] = torch.as_tensor(z_new, dtype=torch.float32)
        info[str(j)] = {
            "name": name,
            "kind": kind,
            "lower_bound": None if lower_bound is None else float(lower_bound),
            "support": support_used,
            "forced_discrete": bool(j in forced_discrete_set),
            "min_before": float(np.min(z_j)) if z_j.size else float("nan"),
            "max_before": float(np.max(z_j)) if z_j.size else float("nan"),
            "min_after": float(np.min(z_new)) if z_new.size else float("nan"),
            "max_after": float(np.max(z_new)) if z_new.size else float("nan"),
        }

    return out, info


def recover_marginal_densities_per_dim(
    beta_full: torch.Tensor,
    Z_grid: torch.Tensor,
    bandwidths: Sequence[float],
    reward_cols: List[str],
    observed_rewards_raw: Optional[torch.Tensor] = None,
    num_points: int = 400,
    grid_pad_std: float = 4.0,
    force_discrete_dims: Optional[Sequence[int]] = None,
    Z_grid_kernel: Optional[torch.Tensor] = None,
    kernel_nu: Optional[float] = None,
    kernel_length_scale: Optional[float] = None,
    kernel_sigma: Optional[float] = None,
) -> Dict[str, object]:
    """
    Recover per-dimension marginals in a way that respects reward type.

    - Continuous dimensions -> Gaussian-kernel density on a continuous grid.
    - Discrete dimensions   -> PMF on a discrete support grid.

    The recovered atom coefficients are projected onto the probability simplex
    using the RKHS metric induced by the atom Gram matrix whenever Z_grid_kernel
    and kernel parameters are provided. This gives the valid atom probability
    vector closest to the raw embedding coefficients in RKHS norm.
    """
    Z = torch.as_tensor(Z_grid, dtype=torch.float32)
    if Z.ndim != 2:
        raise ValueError(f"Z_grid must be 2D, got shape={tuple(Z.shape)}")

    m, d = Z.shape
    bw = _resolve_marginal_bandwidths(d, scalar_bandwidth=1.0, per_dim_bandwidth=bandwidths)
    K_proj = _density_projection_gram(
        Z_grid_kernel,
        nu=kernel_nu,
        length_scale=kernel_length_scale,
        sigma=kernel_sigma,
    )
    weights_t, proj_diag = _prepare_density_weights(
        beta_full=beta_full,
        n_atoms=m,
        K_Z=K_proj,
    )
    weights_np = weights_t.detach().cpu().numpy().astype(float)
    Z_np = Z.detach().cpu().numpy().astype(float)
    obs_np = None
    if observed_rewards_raw is not None:
        obs_np = torch.as_tensor(observed_rewards_raw).detach().cpu().numpy().astype(float)
        if obs_np.ndim != 2 or obs_np.shape[1] != d:
            raise ValueError(
                f"observed_rewards_raw must have shape (n,{d}), got {tuple(obs_np.shape)}"
            )

    marginals: Dict[str, dict] = {}
    normal_const = math.sqrt(2.0 * math.pi)
    forced_discrete_set = set(int(j) for j in (force_discrete_dims or []))

    for j in range(d):
        h = float(bw[j])
        if h <= 0:
            raise ValueError(f"Bandwidth for dimension {j} must be > 0, got {h}.")

        name = reward_cols[j] if j < len(reward_cols) else f"reward_{j}"
        z_j = Z_np[:, j].reshape(-1)
        obs_j = None if obs_np is None else obs_np[:, j].reshape(-1)
        kind = (
            "discrete"
            if j in forced_discrete_set
            else _infer_reward_kind(name=name, observed_vals=obs_j, atom_vals=z_j)
        )

        if kind == "discrete":
            support = _build_discrete_support(observed_vals=obs_j, atom_vals=z_j)
            # Discrete reward: allocate atom mass to the nearest observed support
            # point.  This avoids treating counts as a continuous density.
            alloc = _hard_discrete_allocation(z_j, support)
            pmf = alloc @ weights_np
            pmf = np.maximum(pmf, 0.0)
            s = float(pmf.sum())
            if s <= 0.0 or not np.isfinite(s):
                pmf = np.full_like(support, 1.0 / max(1, support.size), dtype=float)
            else:
                pmf = pmf / s
            marginals[str(j)] = {
                "kind": "discrete",
                "grid": support.astype(float),
                "density": pmf.astype(float),
                "bandwidth": None,
                "ylabel": "Probability",
                "support_lower": float(np.min(support)),
                "support_upper": float(np.max(support)),
                "forced_discrete": bool(j in forced_discrete_set),
                "allocation": "hard_nearest_support",
            }
            continue

        pooled = np.concatenate([_finite_1d_np(obs_j), _finite_1d_np(z_j)])
        if pooled.size == 0:
            pooled = np.asarray([0.0, 1.0], dtype=float)
        z_min = float(np.min(pooled))
        z_max = float(np.max(pooled))
        lo = z_min - grid_pad_std * h
        hi = z_max + grid_pad_std * h
        lower_bound = _infer_lower_bound(name=name, observed_vals=obs_j, atom_vals=z_j)
        if lower_bound is not None:
            lo = max(lo, lower_bound)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            center = float(np.mean(pooled)) if pooled.size else 0.0
            lo, hi = center - 1.0, center + 1.0
            if lower_bound is not None:
                lo = max(lo, lower_bound)
        grid = np.linspace(lo, hi, int(max(50, num_points)), dtype=float)
        u = (grid[:, None] - z_j[None, :]) / h
        K = np.exp(-0.5 * u * u) / (normal_const * h)
        dens = K @ weights_np
        dens = np.maximum(dens, 0.0)
        area = float(np.trapezoid(dens, grid)) if grid.size >= 2 else 0.0
        if area <= 0.0 or (not np.isfinite(area)):
            dens = np.full_like(grid, 1.0 / max(1, grid.size), dtype=float)
            area = float(np.trapezoid(dens, grid)) if grid.size >= 2 else 1.0
        dens = dens / max(area, 1e-12)
        marginals[str(j)] = {
            "kind": "continuous",
            "grid": grid.astype(float),
            "density": dens.astype(float),
            "bandwidth": float(h),
            "ylabel": "Density",
            "support_lower": None if lower_bound is None else float(lower_bound),
            "support_upper": float(np.max(pooled)),
            "forced_discrete": bool(j in forced_discrete_set),
        }

    return {
        "weights": weights_t.detach().cpu(),
        "projection": proj_diag,
        "marginals": marginals,
    }


# -----------------------------------------------------------------------------
# Induced-embedding density recovery and continuous 2D mean-embedding contours
# -----------------------------------------------------------------------------
def _density_recovery_device(args: argparse.Namespace, fallback_device: torch.device) -> str:
    requested = str(getattr(args, "density_recovery_device", "same")).strip().lower()
    if requested in {"same", "auto", ""}:
        return str(fallback_device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: --density-recovery-device requested CUDA but CUDA is unavailable; using cpu.")
        return "cpu"
    return requested


def _gaussian_quadrature_nodes(n: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    nodes_np, weights_np = np.polynomial.hermite.hermgauss(int(n))
    # For X ~ N(mu, h^2): X = mu + sqrt(2) h node, weight / sqrt(pi)
    nodes = torch.as_tensor(nodes_np, dtype=dtype, device=device)
    weights = torch.as_tensor(weights_np / math.sqrt(math.pi), dtype=dtype, device=device)
    return nodes, weights


def _build_induced_density_A_matrix(
    *,
    Z_norm_dict: torch.Tensor,
    Z_raw_atoms: torch.Tensor,
    r_mu: torch.Tensor,
    r_sd: torch.Tensor,
    bandwidths_raw: Sequence[float],
    discrete_dims: Sequence[int],
    nu: float,
    length_scale: float,
    sigma: float,
    quad_points: int,
    device: str,
    batch_atoms: int,
) -> torch.Tensor:
    r"""
    Construct A used by induced-embedding density recovery.

    A[ell,i] = E_{X_i}[ k(Z_norm_dict[ell], X_i_norm) ],
    where X_i is the smoothed atom distribution centered at raw atom Z_raw_atoms[i].

    Continuous reward coordinates use Gaussian smoothing with raw-scale bandwidth h_j.
    Discrete reward coordinates remain fixed at the atom value.
    """
    dev = torch.device(device)
    dtype = torch.float64
    Zdict = torch.as_tensor(Z_norm_dict, dtype=dtype, device=dev)
    Zraw = torch.as_tensor(Z_raw_atoms, dtype=dtype, device=dev)
    mu = torch.as_tensor(r_mu, dtype=dtype, device=dev).reshape(1, -1)
    sd = torch.as_tensor(r_sd, dtype=dtype, device=dev).reshape(1, -1)
    bw = torch.as_tensor(list(bandwidths_raw), dtype=dtype, device=dev).reshape(1, -1)

    if Zdict.ndim != 2 or Zraw.ndim != 2 or Zdict.shape != Zraw.shape:
        raise ValueError(f"Z_norm_dict and Z_raw_atoms must have the same 2D shape, got {tuple(Zdict.shape)} and {tuple(Zraw.shape)}")

    m, d = Zraw.shape
    discrete_set = set(int(j) for j in (discrete_dims or []))
    cont_dims = [j for j in range(d) if j not in discrete_set]
    nodes, qweights = _gaussian_quadrature_nodes(int(quad_points), dev, dtype)

    # Product Gauss-Hermite quadrature over continuous coordinates.
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
            offsets[:, j] = SQRT_2 * offsets_small[:, k]

    if offsets.shape[0] > 5000:
        raise ValueError(
            f"Density-recovery quadrature has {offsets.shape[0]} points. "
            "Reduce --density-recovery-quad-points or reduce the number of continuous reward dimensions."
        )

    A_cols = []
    batch_atoms = max(1, int(batch_atoms))
    for start in range(0, m, batch_atoms):
        end = min(m, start + batch_atoms)
        atoms = Zraw[start:end]  # b x d
        samples = atoms[:, None, :] + offsets[None, :, :] * bw[None, :, :]
        for j in discrete_set:
            samples[:, :, j] = atoms[:, None, j]
        samples_norm = (samples - mu) / (sd + 1e-12)
        flat = samples_norm.reshape(-1, d)
        K_flat = matern_kernel(
            Zdict.float(),
            flat.float(),
            nu=float(nu),
            length_scale=float(length_scale),
            sigma=float(sigma),
        ).to(dtype)
        K_flat = K_flat.reshape(m, end - start, offsets.shape[0])
        A_block = torch.sum(K_flat * weights.reshape(1, 1, -1), dim=2)
        A_cols.append(A_block.detach().cpu())
        del samples, samples_norm, flat, K_flat, A_block
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return torch.cat(A_cols, dim=1)


def _optimize_induced_probability_weights(
    *,
    beta_hat: torch.Tensor,
    Z_norm_dict: torch.Tensor,
    A: torch.Tensor,
    nu: float,
    length_scale: float,
    sigma: float,
    ridge: float,
    lr: float,
    steps: int,
    tol: float,
    init: str,
    device: str,
    print_every: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float], List[dict]]:
    r"""
    Find probability atom weights w by matching the induced embedding coefficients:

        beta_tilde(w) = (K + ridge I)^(-1) A w,
        min_{w in simplex} (beta_tilde(w)-beta_hat)^T K (beta_tilde(w)-beta_hat).

    The simplex constraint is enforced by w = softmax(theta), so w_i >= 0 and sum_i w_i = 1.
    """
    dev = torch.device(device)
    dtype = torch.float64
    beta = torch.as_tensor(beta_hat, dtype=dtype, device=dev).reshape(-1)
    Zdict = torch.as_tensor(Z_norm_dict, dtype=dtype, device=dev)
    A = torch.as_tensor(A, dtype=dtype, device=dev)
    m = beta.numel()
    if A.shape != (m, m):
        raise ValueError(f"A must have shape {(m, m)}, got {tuple(A.shape)}")

    K = matern_kernel(
        Zdict.float(),
        Zdict.float(),
        nu=float(nu),
        length_scale=float(length_scale),
        sigma=float(sigma),
    ).to(dtype)
    K = 0.5 * (K + K.T)
    K_reg = K + float(ridge) * torch.eye(m, dtype=dtype, device=dev)
    M = torch.linalg.solve(K_reg, A)

    init = str(init)
    if init == "abs_beta":
        init_w = torch.abs(beta) + 1e-8
        init_w = init_w / init_w.sum().clamp_min(1e-30)
        theta = torch.log(init_w).clone().detach().requires_grad_(True)
    elif init == "positive_beta":
        init_w = torch.clamp(beta, min=0.0) + 1e-8
        init_w = init_w / init_w.sum().clamp_min(1e-30)
        theta = torch.log(init_w).clone().detach().requires_grad_(True)
    else:
        theta = torch.zeros(m, dtype=dtype, device=dev, requires_grad=True)

    opt = torch.optim.Adam([theta], lr=float(lr))
    history: List[dict] = []
    prev_obj = None
    prev_w = None
    converged = False

    print_every = max(1, int(print_every))
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
            rec = {
                "iter": int(t + 1),
                "objective": obj_val,
                "relative_objective_change": float(rel_change),
                "w_l1_change": float(l1_w_change),
            }
            history.append(rec)
            if (t == 0) or ((t + 1) % print_every == 0) or (t + 1 == int(steps)):
                print(
                    f"density-recovery iter={t+1:6d} obj={obj_val:.8e} "
                    f"rel_change={rel_change:.3e} w_l1_change={l1_w_change:.3e} "
                    f"w_min={float(w_now.min()):.3e} w_max={float(w_now.max()):.3e}",
                    flush=True,
                )
            if prev_obj is not None and rel_change < float(tol) and l1_w_change < math.sqrt(float(tol)):
                converged = True
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
            "density_recovery_method": "induced_embedding_matching_softmax_simplex",
            "density_recovery_converged": bool(converged),
            "density_recovery_iterations": int(len(history)),
            "density_recovery_steps_requested": int(steps),
            "density_recovery_lr": float(lr),
            "density_recovery_tol": float(tol),
            "density_recovery_ridge": float(ridge),
            "density_recovery_init": str(init),
            "objective_rkhs_sq": float(obj.cpu().item()),
            "objective_rkhs": float(torch.sqrt(torch.clamp(obj, min=0)).cpu().item()),
            "target_rkhs_norm_sq": float(target_norm_sq.cpu().item()),
            "target_rkhs_norm": float(torch.sqrt(torch.clamp(target_norm_sq, min=0)).cpu().item()),
            "induced_rkhs_norm_sq": float(induced_norm_sq.cpu().item()),
            "relative_rkhs_error": float(torch.sqrt(torch.clamp(obj, min=0) / torch.clamp(target_norm_sq, min=1e-30)).cpu().item()),
            "beta_rmse": float(np.sqrt(np.mean((beta_tilde_np - beta_np) ** 2))),
            "beta_l2": float(np.linalg.norm(beta_tilde_np - beta_np)),
            "beta_max_abs": float(np.max(np.abs(beta_tilde_np - beta_np))),
            "embedding_value_rmse_on_atoms": float(np.sqrt(np.mean(values_diff.detach().cpu().numpy() ** 2))),
            "embedding_value_corr_on_atoms": corr,
            "w_sum": float(w_np.sum()),
            "w_min": float(w_np.min()),
            "w_max": float(w_np.max()),
            "w_entropy": entropy,
            "w_effective_n_exp_entropy": float(np.exp(entropy)),
            "w_effective_n_inverse_hhi": float(1.0 / np.sum(w_np ** 2)),
            "beta_hat_sum": float(beta_np.sum()),
            "beta_hat_min": float(beta_np.min()),
            "beta_hat_max": float(beta_np.max()),
            "beta_tilde_sum": float(beta_tilde_np.sum()),
            "beta_tilde_min": float(beta_tilde_np.min()),
            "beta_tilde_max": float(beta_tilde_np.max()),
        }
    return w.detach().cpu().float(), beta_tilde.detach().cpu().float(), diagnostics, history


def recover_marginal_densities_per_dim_induced(
    *,
    beta_full: torch.Tensor,
    Z_grid_raw: torch.Tensor,
    Z_grid_norm_dict: torch.Tensor,
    r_mu: torch.Tensor,
    r_sd: torch.Tensor,
    bandwidths: Sequence[float],
    reward_cols: List[str],
    observed_rewards_raw: Optional[torch.Tensor],
    force_discrete_dims: Optional[Sequence[int]],
    kernel_nu: float,
    kernel_length_scale: float,
    kernel_sigma: float,
    args: argparse.Namespace,
    fallback_device: torch.device,
) -> Dict[str, object]:
    """Recovered densities using induced-embedding matching, not beta simplex projection."""
    Z_raw = torch.as_tensor(Z_grid_raw, dtype=torch.float32).detach().cpu()
    Z_norm = torch.as_tensor(Z_grid_norm_dict, dtype=torch.float32).detach().cpu()
    if Z_raw.ndim != 2 or Z_norm.ndim != 2 or Z_raw.shape != Z_norm.shape:
        raise ValueError(f"Z_grid_raw and Z_grid_norm_dict must have the same 2D shape, got {tuple(Z_raw.shape)} and {tuple(Z_norm.shape)}")

    m, d = Z_raw.shape
    bw = _resolve_marginal_bandwidths(d, scalar_bandwidth=1.0, per_dim_bandwidth=bandwidths)
    rec_device = _density_recovery_device(args, fallback_device)

    A = _build_induced_density_A_matrix(
        Z_norm_dict=Z_norm,
        Z_raw_atoms=Z_raw,
        r_mu=torch.as_tensor(r_mu).detach().cpu(),
        r_sd=torch.as_tensor(r_sd).detach().cpu(),
        bandwidths_raw=bw,
        discrete_dims=force_discrete_dims or [],
        nu=float(kernel_nu),
        length_scale=float(kernel_length_scale),
        sigma=float(kernel_sigma),
        quad_points=int(args.density_recovery_quad_points),
        device=rec_device,
        batch_atoms=int(args.density_recovery_batch_atoms),
    )

    weights_t, beta_tilde_t, diag, history = _optimize_induced_probability_weights(
        beta_hat=torch.as_tensor(beta_full).detach().cpu(),
        Z_norm_dict=Z_norm,
        A=A,
        nu=float(kernel_nu),
        length_scale=float(kernel_length_scale),
        sigma=float(kernel_sigma),
        ridge=float(args.density_recovery_ridge),
        lr=float(args.density_recovery_lr),
        steps=int(args.density_recovery_steps),
        tol=float(args.density_recovery_tol),
        init=str(args.density_recovery_init),
        device=rec_device,
        print_every=int(args.density_recovery_print_every),
    )

    weights_np = weights_t.detach().cpu().numpy().astype(float).reshape(-1)
    Z_np = Z_raw.detach().cpu().numpy().astype(float)
    obs_np = None
    if observed_rewards_raw is not None:
        obs_np = torch.as_tensor(observed_rewards_raw).detach().cpu().numpy().astype(float)
        if obs_np.ndim != 2 or obs_np.shape[1] != d:
            raise ValueError(f"observed_rewards_raw must have shape (n,{d}), got {tuple(obs_np.shape)}")

    marginals: Dict[str, dict] = {}
    normal_const = math.sqrt(2.0 * math.pi)
    forced_discrete_set = set(int(j) for j in (force_discrete_dims or []))
    num_points = max(50, int(getattr(args, "density_recovery_num_points", 500)))

    for j in range(d):
        h = float(bw[j])
        if h <= 0:
            raise ValueError(f"Bandwidth for dimension {j} must be > 0, got {h}.")

        name = reward_cols[j] if j < len(reward_cols) else f"reward_{j}"
        z_j = Z_np[:, j].reshape(-1)
        obs_j = None if obs_np is None else obs_np[:, j].reshape(-1)
        kind = "discrete" if j in forced_discrete_set else _infer_reward_kind(name=name, observed_vals=obs_j, atom_vals=z_j)

        if kind == "discrete":
            support = _build_discrete_support(observed_vals=obs_j, atom_vals=z_j)
            alloc = _hard_discrete_allocation(z_j, support)
            pmf = alloc @ weights_np
            pmf = np.maximum(pmf, 0.0)
            s = float(pmf.sum())
            pmf = pmf / s if s > 0 and np.isfinite(s) else np.full_like(support, 1.0 / max(1, support.size), dtype=float)
            marginals[str(j)] = {
                "kind": "discrete",
                "grid": support.astype(float),
                "density": pmf.astype(float),
                "bandwidth": None,
                "ylabel": "Probability",
                "support_lower": float(np.min(support)),
                "support_upper": float(np.max(support)),
                "forced_discrete": bool(j in forced_discrete_set),
                "allocation": "hard_nearest_support",
            }
            continue

        pooled = np.concatenate([_finite_1d_np(obs_j), _finite_1d_np(z_j)])
        if pooled.size == 0:
            pooled = np.asarray([0.0, 1.0], dtype=float)
        lo = float(np.min(pooled)) - 4.0 * h
        hi = float(np.max(pooled)) + 4.0 * h
        lower_bound = _infer_lower_bound(name=name, observed_vals=obs_j, atom_vals=z_j)
        if lower_bound is not None:
            lo = max(lo, float(lower_bound))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            center = float(np.mean(pooled)) if pooled.size else 0.0
            lo, hi = center - 1.0, center + 1.0
            if lower_bound is not None:
                lo = max(lo, float(lower_bound))
        grid = np.linspace(lo, hi, num_points, dtype=float)
        u = (grid[:, None] - z_j[None, :]) / h
        Kd = np.exp(-0.5 * u * u) / (normal_const * h)
        dens = Kd @ weights_np
        dens = _normalize_nonnegative_density(dens, grid)
        marginals[str(j)] = {
            "kind": "continuous",
            "grid": grid.astype(float),
            "density": dens.astype(float),
            "bandwidth": float(h),
            "ylabel": "Density",
            "support_lower": None if lower_bound is None else float(lower_bound),
            "support_upper": float(np.max(pooled)),
            "forced_discrete": bool(j in forced_discrete_set),
        }

    return {
        "weights": weights_t.detach().cpu(),
        "beta_tilde": beta_tilde_t.detach().cpu(),
        "projection": diag,
        "density_recovery": diag,
        "optimization_history": history,
        "marginals": marginals,
    }


def plot_mean_embedding_2d_continuous_contour(
    *,
    beta_full: torch.Tensor,
    Z_grid_norm: torch.Tensor,
    Z_grid_raw_for_limits: torch.Tensor,
    r_mu: torch.Tensor,
    r_sd: torch.Tensor,
    reward_cols: Sequence[str],
    outdir: Path,
    policy_name: str,
    nu: float,
    length_scale: float,
    sigma: float,
    n1: int = 160,
    n2: int = 160,
    pad_frac: float = 0.05,
    levels: int = 30,
) -> Optional[dict]:
    """
    Continuous contour map of the 2D mean embedding surface.

    This is a visualization of z -> <mu_hat, k(z,.)> on a continuous raw-scale
    reward canvas. It is intentionally independent of the density recovery step.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if len(reward_cols) < 2:
        return None

    beta = torch.as_tensor(beta_full, dtype=torch.float32).detach().cpu().reshape(-1)
    Z_norm = torch.as_tensor(Z_grid_norm, dtype=torch.float32).detach().cpu()
    Z_raw = torch.as_tensor(Z_grid_raw_for_limits, dtype=torch.float32).detach().cpu()
    r_mu_cpu = torch.as_tensor(r_mu, dtype=torch.float32).detach().cpu().reshape(-1)
    r_sd_cpu = torch.as_tensor(r_sd, dtype=torch.float32).detach().cpu().reshape(-1)
    if Z_norm.ndim != 2 or Z_norm.shape[1] < 2 or beta.numel() != Z_norm.shape[0]:
        return None

    x_raw = Z_raw[:, 0].numpy().astype(float)
    y_raw = Z_raw[:, 1].numpy().astype(float)
    x_min, x_max = float(np.nanmin(x_raw)), float(np.nanmax(x_raw))
    y_min, y_max = float(np.nanmin(y_raw)), float(np.nanmax(y_raw))
    x_pad = max(1e-8, float(pad_frac) * max(1e-8, x_max - x_min))
    y_pad = max(1e-8, float(pad_frac) * max(1e-8, y_max - y_min))
    x = np.linspace(x_min - x_pad, x_max + x_pad, int(max(20, n1)))
    y = np.linspace(y_min - y_pad, y_max + y_pad, int(max(20, n2)))
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Fill unplotted reward dimensions with the atom-grid raw median, then z-score.
    d = Z_norm.shape[1]
    raw_query = np.tile(np.nanmedian(Z_raw.numpy(), axis=0).reshape(1, -1), (X.size, 1))
    raw_query[:, 0] = X.reshape(-1)
    raw_query[:, 1] = Y.reshape(-1)
    q_norm = (raw_query - r_mu_cpu.numpy().reshape(1, -1)) / (r_sd_cpu.numpy().reshape(1, -1) + 1e-12)
    Q = torch.as_tensor(q_norm, dtype=torch.float32)

    vals = []
    batch = 20000
    for start in range(0, Q.shape[0], batch):
        end = min(Q.shape[0], start + batch)
        Kq = matern_kernel(Q[start:end], Z_norm, nu=float(nu), length_scale=float(length_scale), sigma=float(sigma))
        vals.append((Kq @ beta).detach().cpu())
    V = torch.cat(vals, dim=0).numpy().reshape(X.shape)

    # Save CSV as rows: x_raw, y_raw, mean_embedding_value
    csv_arr = np.column_stack([X.reshape(-1), Y.reshape(-1), V.reshape(-1)])
    csv_path = outdir / f"{policy_name}_mean_embedding_2d_continuous_values.csv"
    np.savetxt(csv_path, csv_arr, delimiter=",", header=f"{reward_cols[0]},{reward_cols[1]},mean_embedding_value", comments="")

    fig = plt.figure(figsize=(7.5, 5.8))
    cf = plt.contourf(Y, X, V, levels=int(max(5, levels)))
    plt.colorbar(cf, label="Mean embedding value")
    plt.xlabel(_pretty_label(reward_cols[1]))
    plt.ylabel(_pretty_label(reward_cols[0]))
    plt.title(f"{policy_name}: continuous 2D mean embedding contour")
    contour_path = outdir / f"{policy_name}_mean_embedding_2d_continuous_contour.png"
    fig.savefig(contour_path, bbox_inches="tight", dpi=700)
    plt.close(fig)

    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(Y, X, V, alpha=0.75, linewidth=0, antialiased=True)
        ax.set_xlabel(_pretty_label(reward_cols[1]))
        ax.set_ylabel(_pretty_label(reward_cols[0]))
        ax.set_zlabel("Mean embedding value", labelpad=14)
        fig.subplots_adjust(left=0.03, right=0.88, bottom=0.05, top=0.92)
        surface_path = outdir / f"{policy_name}_mean_embedding_2d_continuous_surface3d.png"
        fig.savefig(surface_path, bbox_inches="tight", pad_inches=0.4, dpi=700)
        plt.close(fig)
    except Exception:
        surface_path = None

    return {
        "x": x,
        "y": y,
        "z": V,
        "csv": str(csv_path),
        "contour_png": str(contour_path),
        "surface3d_png": None if surface_path is None else str(surface_path),
    }


def plot_densities_per_dim(
    marginal_payload: Dict[str, object],
    reward_cols: List[str],
    outdir: str,
) -> None:
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    marginals = marginal_payload.get("marginals", {}) if isinstance(marginal_payload, dict) else {}
    for key, item in marginals.items():
        try:
            j = int(key)
        except Exception:
            continue
        if not isinstance(item, dict):
            continue
        label = reward_cols[j] if j < len(reward_cols) else f"reward_{j}"
        grid = np.asarray(item.get("grid", []), dtype=float).reshape(-1)
        dens = np.asarray(item.get("density", []), dtype=float).reshape(-1)
        kind = str(item.get("kind", "continuous"))
        ylabel = str(item.get("ylabel", "Density"))
        bw = item.get("bandwidth", None)
        if grid.size == 0 or dens.size != grid.size:
            continue

        fig = plt.figure()
        if kind == "discrete":
            markerline, stemlines, baseline = plt.stem(grid, dens)
            plt.setp(markerline, markersize=5)
            plt.setp(stemlines, linewidth=1.8)
            plt.setp(baseline, linewidth=0.8)
            title = f"Recovered PMF: {label}"
        else:
            dens = _normalize_nonnegative_density(dens, grid)
            plt.plot(grid, dens, linewidth=2.5)
            title = f"Recovered Density: {label}"
        if bw is not None:
            title += f" (bandwidth={float(bw):.4g})"
        plt.xlabel(label)
        plt.ylabel(ylabel)
        plt.title(title)
        fig.savefig(out_path / f"density_recovered_dim{j}_{label}.png", bbox_inches="tight")
        plt.close(fig)


def _get_from_dict_by_candidates(d: dict, candidates):
    if not isinstance(d, dict):
        return None
    for k in candidates:
        if k in d:
            return d[k]
    return None


def _candidate_1d_arrays(obj, prefix="root"):
    out = []
    for name, arr in _collect_named_arrays(obj, prefix):
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1 and arr.size >= 2 and np.isfinite(arr).any():
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            out.append((name, arr.reshape(-1)))
    return out

def _candidate_2d_arrays(obj, prefix="root"):
    out = []
    for name, arr in _collect_named_arrays(obj, prefix):
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 2 and min(arr.shape) >= 5 and np.isfinite(arr).any():
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            out.append((name, arr))
    return out


def _resample_surface_regular_grid(
    zsurf: np.ndarray,
    old_x: np.ndarray,
    old_y: np.ndarray,
    new_x: np.ndarray,
    new_y: np.ndarray,
) -> np.ndarray:
    """
    Resample a surface zsurf defined on the regular grid (old_x, old_y)
    onto the regular/mixed grid (new_x, new_y) using sequential 1D interpolation.

    Here:
      - rows correspond to old_x / new_x   (reward dim 0)
      - cols correspond to old_y / new_y   (reward dim 1)
    """
    zsurf = np.asarray(zsurf, dtype=float)
    old_x = np.asarray(old_x, dtype=float).reshape(-1)
    old_y = np.asarray(old_y, dtype=float).reshape(-1)
    new_x = np.asarray(new_x, dtype=float).reshape(-1)
    new_y = np.asarray(new_y, dtype=float).reshape(-1)

    # interpolate along columns (dim 1)
    tmp = np.empty((zsurf.shape[0], new_y.size), dtype=float)
    for i in range(zsurf.shape[0]):
        tmp[i, :] = np.interp(new_y, old_y, zsurf[i, :], left=0.0, right=0.0)

    # interpolate along rows (dim 0)
    out = np.empty((new_x.size, new_y.size), dtype=float)
    for j in range(tmp.shape[1]):
        out[:, j] = np.interp(new_x, old_x, tmp[:, j], left=0.0, right=0.0)

    return out


def _hard_discrete_allocation(atom_vals: np.ndarray, support: np.ndarray) -> np.ndarray:
    """
    Allocate each atom to exactly one discrete support point. This implements the
    mixed discrete-continuous law with indicators 1{z_j = c}, rather than a
    continuous interpolation over the count dimension.
    """
    atom_vals = np.asarray(atom_vals, dtype=float).reshape(-1)
    support = np.asarray(support, dtype=float).reshape(-1)
    alloc = np.zeros((support.size, atom_vals.size), dtype=float)
    if support.size == 0 or atom_vals.size == 0:
        return alloc
    nearest = np.abs(atom_vals[:, None] - support[None, :]).argmin(axis=1)
    alloc[nearest, np.arange(atom_vals.size)] = 1.0
    return alloc


def _continuous_kernel_matrix(grid: np.ndarray, atom_vals: np.ndarray, bandwidth: float) -> np.ndarray:
    grid = np.asarray(grid, dtype=float).reshape(-1)
    atom_vals = np.asarray(atom_vals, dtype=float).reshape(-1)
    h = max(float(bandwidth), 1e-8)
    u = (grid[:, None] - atom_vals[None, :]) / h
    return np.exp(-0.5 * u * u) / (math.sqrt(2.0 * math.pi) * h)




def _safe_float(value, default: float = 1.0) -> float:
    """Return default when value is None/NaN/non-numeric."""
    try:
        if value is None:
            return float(default)
        out = float(value)
        if not np.isfinite(out):
            return float(default)
        return out
    except Exception:
        return float(default)

def _build_joint_surface_from_atoms(
    *,
    beta_full,
    Zg_raw,
    overlay_one_d: Dict[str, dict],
    reward_cols: List[str],
    density_weights=None,
) -> Optional[dict]:
    """
    Build the two-dimensional display directly from atom masses and atom locations.

    This is the corrected object for mixed rewards. If one dimension is discrete
    and the other is continuous, the result satisfies

        sum_c int p(c,r) dr = 1,

    not int int p(c,r) dc dr = 1.
    """
    if beta_full is None:
        return None

    Z = np.asarray(Zg_raw.detach().cpu().numpy() if isinstance(Zg_raw, torch.Tensor) else Zg_raw, dtype=float)
    if Z.ndim != 2 or Z.shape[1] < 2 or Z.shape[0] == 0:
        return None

    if density_weights is not None:
        weights = np.asarray(
            density_weights.detach().cpu().numpy() if isinstance(density_weights, torch.Tensor) else density_weights,
            dtype=float,
        ).reshape(-1)
    else:
        try:
            weights_t, _ = _prepare_density_weights(torch.as_tensor(beta_full), n_atoms=Z.shape[0])
        except Exception:
            return None
        weights = weights_t.detach().cpu().numpy().astype(float).reshape(-1)
    if weights.size != Z.shape[0]:
        return None
    weights = np.maximum(np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    total_w = float(weights.sum())
    if total_w <= 0.0 or not np.isfinite(total_w):
        return None
    weights = weights / total_w

    dim_payload = []
    for d in [0, 1]:
        item = overlay_one_d.get(str(d), {})
        kind = str(item.get("kind", "continuous")).lower()
        grid = np.asarray(item.get("grid", []), dtype=float).reshape(-1)
        bw = _safe_float(item.get("bandwidth", None), default=1.0)

        if grid.size == 0:
            z_d = Z[:, d]
            if kind == "discrete":
                grid = np.unique(np.round(z_d).astype(int)).astype(float)
            else:
                lo = float(np.nanmin(z_d)) - 4.0 * bw
                hi = float(np.nanmax(z_d)) + 4.0 * bw
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    lo, hi = -1.0, 1.0
                grid = np.linspace(lo, hi, 120)

        if kind == "discrete":
            A = _hard_discrete_allocation(Z[:, d], grid)
        else:
            A = _continuous_kernel_matrix(grid, Z[:, d], bandwidth=bw)

        dim_payload.append((kind, grid, A, bw))

    kind0, x, A0, bw0 = dim_payload[0]
    kind1, y, A1, bw1 = dim_payload[1]

    # A0: (len(x), m), A1: (len(y), m)
    z = (A0 * weights.reshape(1, -1)) @ A1.T
    z = _normalize_joint_surface_by_kind(z, x, y, kind0, kind1)

    return {
        "x": x,   # reward dim 0 axis; rows of z
        "y": y,   # reward dim 1 axis; columns of z
        "z": z,
        "kind_dim0": kind0,
        "kind_dim1": kind1,
        "bandwidth_dim0": float(bw0),
        "bandwidth_dim1": float(bw1),
        "construction": "atom_mixture_mixed_discrete_continuous",
    }


def _build_overlay_data_explicit(
    marginal_payload_raw,
    Zg_raw,
    reward_cols: List[str],
    beta_full=None,
    joint_surface_raw=None,
) -> dict:
    if isinstance(Zg_raw, torch.Tensor):
        Zg_raw_np = Zg_raw.detach().cpu().numpy()
    else:
        Zg_raw_np = np.asarray(Zg_raw, dtype=float)

    overlay_one_d = {}
    marginals = {}

    if isinstance(marginal_payload_raw, dict):
        marginals = marginal_payload_raw.get("marginals", {}) or {}

        for d in range(min(2, len(reward_cols))):
            item = marginals.get(str(d), None)
            if not isinstance(item, dict):
                continue

            grid = np.asarray(item.get("grid", []), dtype=float).reshape(-1)
            dens = np.asarray(item.get("density", []), dtype=float).reshape(-1)
            if grid.size == 0 or dens.size != grid.size:
                continue

            kind = str(item.get("kind", "continuous")).lower()
            if kind == "continuous":
                dens = _normalize_nonnegative_density(dens, grid)
            else:
                dens = np.maximum(dens, 0.0)
                s = dens.sum()
                dens = dens / s if s > 0 else dens

            overlay_one_d[str(d)] = {
                "kind": kind,
                "grid": grid,
                "density": dens,
                "ylabel": str(item.get("ylabel", "Density")),
                "bandwidth": _safe_float(item.get("bandwidth", None), default=1.0),
            }

    density_weights = None
    if isinstance(marginal_payload_raw, dict) and "weights" in marginal_payload_raw:
        density_weights = marginal_payload_raw.get("weights")

    # Correct construction: use atom masses directly. This avoids interpolating a
    # continuous 2D surface onto an integer/count axis.
    overlay_joint = _build_joint_surface_from_atoms(
        beta_full=beta_full,
        Zg_raw=Zg_raw_np,
        overlay_one_d=overlay_one_d,
        reward_cols=reward_cols,
        density_weights=density_weights,
    )

    # Backward-compatible fallback only if beta is unavailable.
    if overlay_joint is None and joint_surface_raw is not None:
        zsurf = np.asarray(joint_surface_raw, dtype=float)
        if zsurf.ndim == 2 and min(zsurf.shape) >= 5:
            nx, ny = zsurf.shape

            old_x = np.linspace(float(np.nanmin(Zg_raw_np[:, 0])), float(np.nanmax(Zg_raw_np[:, 0])), nx)
            old_y = np.linspace(float(np.nanmin(Zg_raw_np[:, 1])), float(np.nanmax(Zg_raw_np[:, 1])), ny)

            if overlay_one_d.get("0", {}).get("kind", "continuous") == "discrete":
                new_x = np.asarray(overlay_one_d["0"]["grid"], dtype=float)
            else:
                new_x = old_x

            if overlay_one_d.get("1", {}).get("kind", "continuous") == "discrete":
                new_y = np.asarray(overlay_one_d["1"]["grid"], dtype=float)
            else:
                new_y = old_y

            zsurf_resampled = _resample_surface_regular_grid(
                zsurf=zsurf,
                old_x=old_x,
                old_y=old_y,
                new_x=new_x,
                new_y=new_y,
            )

            kind0 = overlay_one_d.get("0", {}).get("kind", "continuous")
            kind1 = overlay_one_d.get("1", {}).get("kind", "continuous")
            overlay_joint = {
                "x": new_x,
                "y": new_y,
                "z": _normalize_joint_surface_by_kind(zsurf_resampled, new_x, new_y, kind0, kind1),
                "kind_dim0": kind0,
                "kind_dim1": kind1,
                "construction": "fallback_resampled_continuous_surface",
            }

    return {
        "one_d": overlay_one_d,
        "joint": overlay_joint,
    }


def _extract_joint_surface_from_payload(payload: dict) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if not isinstance(payload, dict):
        return None
    overlay_data = payload.get("overlay_data", {})
    joint = overlay_data.get("joint", None)
    if isinstance(joint, dict) and all(k in joint for k in ["x", "y", "z"]):
        x = np.asarray(joint["x"], dtype=float)
        y = np.asarray(joint["y"], dtype=float)
        z = np.asarray(joint["z"], dtype=float)
        kind0 = str(joint.get("kind_dim0", "continuous"))
        kind1 = str(joint.get("kind_dim1", "continuous"))
        if x.size >= 1 and y.size >= 1 and z.ndim == 2:
            return x, y, _normalize_joint_surface_by_kind(z, x, y, kind0, kind1)
    return None


def _extract_1d_density_from_payload(payload: dict, dim_idx: int) -> Optional[Dict[str, np.ndarray]]:
    if not isinstance(payload, dict):
        return None
    overlay_data = payload.get("overlay_data", {})
    one_d = overlay_data.get("one_d", {})
    item = one_d.get(str(dim_idx), None)
    if not isinstance(item, dict):
        return None
    grid = np.asarray(item.get("grid", []), dtype=float)
    dens = np.asarray(item.get("density", []), dtype=float)
    kind = str(item.get("kind", "continuous"))
    ylabel = str(item.get("ylabel", "Density"))
    if grid.size >= 1 and dens.size == grid.size:
        if kind == "continuous":
            dens = _normalize_nonnegative_density(dens, grid)
        else:
            dens = np.maximum(dens, 0.0)
            s = dens.sum()
            dens = dens / s if s > 0 else dens
        return {"kind": kind, "grid": grid, "density": dens, "ylabel": ylabel}
    return None


def _pretty_label(name: str) -> str:
    mapping = {
        "gross_revenue_per_night": "Gross Revenue per Night",
        "total_clicks": "Total Clicks",
        "avg_price_per_night": "Average Price per Night",
        "total_promotions": "Total Promotions",
        "std_price_usd": "Price Standard Deviation (USD)",
    }
    return mapping.get(name, name.replace("_", " ").title())


def _policy_overlay_label(name: str) -> str:
    mapping = {
        "gross_revenue_per_night": r"Policy $\pi^{\mathrm{rev}}$: Revenue-focused",
        "total_clicks": r"Policy $\pi^{\mathrm{clk}}$: Click-focused",
    }
    return mapping.get(name, _pretty_label(name))


def _mask_zero_surface(z: np.ndarray, rel_tol: float = 1e-6) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if z.size == 0:
        return z
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    zmax = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 0.0
    if zmax <= 0.0:
        return np.full_like(z, np.nan)
    thresh = max(rel_tol * zmax, 1e-12)
    z_masked = z.copy()
    z_masked[np.abs(z_masked) <= thresh] = np.nan
    return z_masked


def _plot_two_policy_reward_overlays(payload_A: dict, payload_B: dict, reward_cols: List[str], label_A: str, label_B: str, out_dir: Path) -> Dict[str, str]:
    """Create two-policy plots using RKHS mean-embedding surfaces, not joint recovered densities.

    Top panels use payload['continuous_mean_embedding_2d'], which contains
        z -> <mu_hat, k(z, .)>
    evaluated on a continuous raw-scale reward canvas.  The recovered
    density/PMF is used only for the marginal panels.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    color_A = "#13294B"  # Navy
    color_B = "#FF5F05"  # Orange
    ls_A = "-"
    ls_B = "--"

    pretty_A = _policy_overlay_label(label_A)
    pretty_B = _policy_overlay_label(label_B)
    pretty_rewards = [_pretty_label(r) for r in reward_cols[:2]]

    def _extract_mean_embedding_surface(payload: dict) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if not isinstance(payload, dict):
            return None
        item = payload.get("continuous_mean_embedding_2d", None)
        if not isinstance(item, dict):
            return None
        x = np.asarray(item.get("x", []), dtype=float).reshape(-1)  # reward dim 0
        y = np.asarray(item.get("y", []), dtype=float).reshape(-1)  # reward dim 1
        z = np.asarray(item.get("z", []), dtype=float)
        if x.size < 2 or y.size < 2 or z.ndim != 2:
            return None
        if z.shape == (x.size, y.size):
            pass
        elif z.shape == (y.size, x.size):
            z = z.T
        else:
            return None
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        return x, y, z

    def _surface_grid(surface_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        x, y, z = surface_tuple
        X, Y = np.meshgrid(y, x)  # X: reward dim 1, Y: reward dim 0
        return X, Y, z

    def _draw_contour(ax, surface_tuple, color: str, linestyle: str, label: str) -> None:
        X, Y, z = _surface_grid(surface_tuple)

        # Do not draw the near-zero boundary contour. That boundary is what makes
        # the mean-embedding plot look like it spills outside the useful support.
        z_plot = _mask_zero_surface(z, rel_tol=1e-4)
        finite = z_plot[np.isfinite(z_plot)]
        if finite.size == 0:
            return
        zmin, zmax = float(np.min(finite)), float(np.max(finite))
        if not np.isfinite(zmin) or not np.isfinite(zmax) or abs(zmax - zmin) < 1e-12:
            ax.plot([], [], color=color, linestyle=linestyle, linewidth=2.5, label=label)
            return
        levels = np.linspace(zmin, zmax, 12)
        ax.contour(X, Y, z_plot, levels=levels, colors=[color], linewidths=2.2, linestyles=linestyle, alpha=0.95)

    def _set_discrete_ticks_if_needed(ax, surface_tuple) -> None:
        x, y, _ = surface_tuple

        # x-axis in the plots = reward_cols[1], here Total Clicks.
        # Never show negative counts, and cap the display at 6 clicks.
        if len(reward_cols) > 1 and _name_suggests_discrete(reward_cols[1]):
            ax.set_xlim(0, 6)
            ax.set_xticks(np.arange(0, 7))

        # y-axis = reward_cols[0], only needed if first reward is discrete.
        if len(reward_cols) > 0 and _name_suggests_discrete(reward_cols[0]):
            ax.set_ylim(bottom=0)
            lo, hi = 0, min(6, int(np.floor(np.nanmax(x))))
            if hi >= lo:
                ax.set_yticks(np.arange(lo, hi + 1))

    def _pmf_plot_arrays(gA, fA, gB, fB, reward_name: str, zero_tol: float = 1e-12):
        """Align two PMFs and remove empty zero-probability tail support."""
        all_support = np.unique(np.concatenate([np.asarray(gA, dtype=float), np.asarray(gB, dtype=float)]))
        all_support = all_support[np.isfinite(all_support)]
        if all_support.size == 0:
            return all_support, all_support, all_support, None, None

        vals_A = np.zeros_like(all_support, dtype=float)
        vals_B = np.zeros_like(all_support, dtype=float)
        idx_map = {float(x): i for i, x in enumerate(all_support)}
        for gx, px in zip(gA, fA):
            key = float(gx)
            if key in idx_map:
                vals_A[idx_map[key]] = float(px)
        for gx, px in zip(gB, fB):
            key = float(gx)
            if key in idx_map:
                vals_B[idx_map[key]] = float(px)

        if _name_suggests_discrete(reward_name):
            nonzero = (vals_A > zero_tol) | (vals_B > zero_tol)
            if np.any(nonzero):
                hi = int(np.ceil(np.max(all_support[nonzero])))
            else:
                hi = 0
            # For total_clicks/count-like rewards, do not display the empty tail.
            # Also never allow the axis to run beyond 6.
            hi = min(max(hi, 1), 6)
            support = np.arange(0, hi + 1, dtype=float)
            old_A = {float(x): float(v) for x, v in zip(all_support, vals_A)}
            old_B = {float(x): float(v) for x, v in zip(all_support, vals_B)}
            vals_A = np.asarray([old_A.get(float(x), 0.0) for x in support], dtype=float)
            vals_B = np.asarray([old_B.get(float(x), 0.0) for x in support], dtype=float)
            return support, vals_A, vals_B, (-0.5, hi + 0.5), np.arange(0, hi + 1)

        return all_support, vals_A, vals_B, None, None

    # -------------------------
    # One-dimensional recovered marginal overlays
    # -------------------------
    one_d = {}
    for d, rname in enumerate(reward_cols[:2]):
        A = _extract_1d_density_from_payload(payload_A, d)
        B = _extract_1d_density_from_payload(payload_B, d)
        if A is None or B is None:
            continue

        gA, fA, kindA = A["grid"], A["density"], A["kind"]
        gB, fB, kindB = B["grid"], B["density"], B["kind"]
        kind = kindA if kindA == kindB else "continuous"

        fig = plt.figure(figsize=(7, 5))
        if kind == "discrete":
            all_support, vals_A, vals_B, xlim, xticks = _pmf_plot_arrays(gA, fA, gB, fB, rname)
            width = 0.35
            plt.bar(all_support - width / 2.0, vals_A, width=width, alpha=0.75, color=color_A, label=pretty_A)
            plt.bar(all_support + width / 2.0, vals_B, width=width, alpha=0.55, color=color_B, label=pretty_B)
            if xlim is not None:
                plt.xlim(*xlim)
                plt.xticks(xticks)
            ylabel = "Probability"
            title = f"Recovered Marginal PMF: {pretty_rewards[d]}"
        else:
            plt.plot(gA, fA, color=color_A, linestyle=ls_A, linewidth=2.5, alpha=0.95, label=pretty_A)
            plt.plot(gB, fB, color=color_B, linestyle=ls_B, linewidth=2.5, alpha=0.95, label=pretty_B)
            ylabel = "Density"
            title = f"Recovered Marginal Density: {pretty_rewards[d]}"
        plt.xlabel(pretty_rewards[d])
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(frameon=True)
        p = out_dir / f"overlay_1d_{rname}.png"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.15, dpi=700)
        plt.close(fig)

        paths[f"overlay_1d_{rname}"] = str(p)
        one_d[d] = {"kind": kind, "gA": gA, "fA": fA, "gB": gB, "fB": fB}

    mean_A = _extract_mean_embedding_surface(payload_A)
    mean_B = _extract_mean_embedding_surface(payload_B)

    if mean_A is not None and mean_B is not None:
        contour_handles = [
            Line2D([0], [0], color=color_A, linestyle=ls_A, linewidth=2.5, label=pretty_A),
            Line2D([0], [0], color=color_B, linestyle=ls_B, linewidth=2.5, label=pretty_B),
        ]
        surface_handles = [
            Patch(facecolor=color_A, edgecolor=color_A, alpha=0.45, label=pretty_A),
            Patch(facecolor=color_B, edgecolor=color_B, alpha=0.45, label=pretty_B),
        ]

        # -------------------------
        # Overlay contour of the RKHS mean embedding, not density.
        # -------------------------
        fig = plt.figure(figsize=(7.5, 5.8))
        ax = fig.add_subplot(111)
        _draw_contour(ax, mean_A, color_A, ls_A, pretty_A)
        _draw_contour(ax, mean_B, color_B, ls_B, pretty_B)
        _set_discrete_ticks_if_needed(ax, mean_A)
        ax.set_xlabel(pretty_rewards[1])
        ax.set_ylabel(pretty_rewards[0])
        ax.legend(handles=contour_handles, frameon=True)
        p = out_dir / "overlay_2d_mean_embedding_contour.png"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.20, dpi=700)
        plt.close(fig)
        paths["overlay_2d_mean_embedding_contour"] = str(p)

        # -------------------------
        # Overlay 3D RKHS mean-embedding surface, not recovered joint density.
        # -------------------------
        XA, YA, zA = _surface_grid(mean_A)
        XB, YB, zB = _surface_grid(mean_B)
        zA_plot = _mask_zero_surface(zA, rel_tol=1e-4)
        zB_plot = _mask_zero_surface(zB, rel_tol=1e-4)

        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(11.2, 8.2), constrained_layout=False)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(XA, YA, zA_plot, color=color_A, alpha=0.45, linewidth=0, antialiased=True)
        ax.plot_surface(XB, YB, zB_plot, color=color_B, alpha=0.45, linewidth=0, antialiased=True)
        ax.set_xlabel(pretty_rewards[1], labelpad=10)
        ax.set_ylabel(pretty_rewards[0], labelpad=10)
        ax.set_zlabel("Mean embedding value", labelpad=10)
        ax.set_box_aspect((1.25, 1.05, 0.70))
        ax.view_init(elev=25, azim=-55)
        _set_discrete_ticks_if_needed(ax, mean_A)
        ax.legend(handles=surface_handles, loc="upper right")
        fig.subplots_adjust(left=0.04, right=0.94, bottom=0.08, top=0.92)
        p = out_dir / "overlay_3d_mean_embedding_surface.png"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.25, dpi=700)
        plt.close(fig)
        paths["overlay_3d_mean_embedding_surface"] = str(p)

        # -------------------------
        # Combined panel: mean embedding top row + recovered marginals bottom row.
        # No density surface and no difference heatmap.
        # -------------------------
        fig = plt.figure(figsize=(15.8, 11.2), constrained_layout=False)

        ax1 = fig.add_subplot(2, 2, 1)
        _draw_contour(ax1, mean_A, color_A, ls_A, pretty_A)
        _draw_contour(ax1, mean_B, color_B, ls_B, pretty_B)
        _set_discrete_ticks_if_needed(ax1, mean_A)
        ax1.set_xlabel(pretty_rewards[1])
        ax1.set_ylabel(pretty_rewards[0])
        ax1.set_title("Mean-embedding Contours")
        ax1.legend(handles=contour_handles, frameon=True)

        ax2 = fig.add_subplot(2, 2, 2, projection="3d")
        ax2.plot_surface(XA, YA, zA_plot, color=color_A, alpha=0.45, linewidth=0, antialiased=True)
        ax2.plot_surface(XB, YB, zB_plot, color=color_B, alpha=0.45, linewidth=0, antialiased=True)
        ax2.set_xlabel(pretty_rewards[1], labelpad=8)
        ax2.set_ylabel(pretty_rewards[0], labelpad=8)
        ax2.set_zlabel("Mean embedding value", labelpad=8)
        ax2.set_box_aspect((1.25, 1.05, 0.70))
        ax2.view_init(elev=25, azim=-55)
        _set_discrete_ticks_if_needed(ax2, mean_A)
        ax2.set_title("Mean-embedding Surface")
        ax2.legend(handles=surface_handles, loc="upper right")

        ax3 = fig.add_subplot(2, 2, 3)
        if 0 in one_d:
            item = one_d[0]
            if item["kind"] == "discrete":
                all_support, vals_A, vals_B, xlim, xticks = _pmf_plot_arrays(
                    item["gA"], item["fA"], item["gB"], item["fB"], reward_cols[0]
                )
                width = 0.35
                ax3.bar(all_support - width / 2.0, vals_A, width=width, alpha=0.75, color=color_A, label=pretty_A)
                ax3.bar(all_support + width / 2.0, vals_B, width=width, alpha=0.55, color=color_B, label=pretty_B)
                if xlim is not None:
                    ax3.set_xlim(*xlim)
                    ax3.set_xticks(xticks)
                ax3.set_ylabel("Probability")
                ax3.set_title(f"Recovered PMF: {pretty_rewards[0]}")
            else:
                ax3.plot(item["gA"], item["fA"], color=color_A, linestyle=ls_A, linewidth=2.5, alpha=0.95, label=pretty_A)
                ax3.plot(item["gB"], item["fB"], color=color_B, linestyle=ls_B, linewidth=2.5, alpha=0.95, label=pretty_B)
                ax3.set_ylabel("Density")
                ax3.set_title(f"Recovered Density: {pretty_rewards[0]}")
            ax3.set_xlabel(pretty_rewards[0])
            ax3.legend(frameon=True)

        ax4 = fig.add_subplot(2, 2, 4)
        if 1 in one_d:
            item = one_d[1]
            if item["kind"] == "discrete":
                all_support, vals_A, vals_B, xlim, xticks = _pmf_plot_arrays(
                    item["gA"], item["fA"], item["gB"], item["fB"], reward_cols[1]
                )
                width = 0.35
                ax4.bar(all_support - width / 2.0, vals_A, width=width, alpha=0.75, color=color_A, label=pretty_A)
                ax4.bar(all_support + width / 2.0, vals_B, width=width, alpha=0.55, color=color_B, label=pretty_B)
                if xlim is not None:
                    ax4.set_xlim(*xlim)
                    ax4.set_xticks(xticks)
                ax4.set_ylabel("Probability")
                ax4.set_title(f"Recovered PMF: {pretty_rewards[1]}")
            else:
                ax4.plot(item["gA"], item["fA"], color=color_A, linestyle=ls_A, linewidth=2.5, alpha=0.95, label=pretty_A)
                ax4.plot(item["gB"], item["fB"], color=color_B, linestyle=ls_B, linewidth=2.5, alpha=0.95, label=pretty_B)
                ax4.set_ylabel("Density")
                ax4.set_title(f"Recovered Density: {pretty_rewards[1]}")
            ax4.set_xlabel(pretty_rewards[1])
            ax4.legend(frameon=True)

        fig.suptitle(f"Mean Embedding and Recovered Marginal Comparison: {pretty_A} vs {pretty_B}", y=0.965)
        fig.subplots_adjust(left=0.07, right=0.96, bottom=0.06, top=0.90, wspace=0.28, hspace=0.36)
        p = out_dir / "overlay_all_mean_embedding_marginal_comparison.png"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.25, dpi=700)
        plt.close(fig)
        paths["overlay_all_mean_embedding_marginal_comparison"] = str(p)
    else:
        print("Warning: mean-embedding overlay was not produced because continuous_mean_embedding_2d was missing or malformed.\n")

    if not paths:
        print("Warning: no two-policy overlay plots were produced.\n")
    else:
        print(f"Saved overlay plots to: {out_dir}\n")

    return paths

def summarize_policy_outputs(mu_norm: torch.Tensor,log_std_norm: torch.Tensor,a_mu: torch.Tensor,a_sd: torch.Tensor,action_cols: List[str],title: str) -> Dict[str, dict]:
    mu_raw = _denorm(mu_norm, a_mu, a_sd)
    std_norm = log_std_norm.exp()
    std_raw = std_norm * a_sd

    out: Dict[str, dict] = {}
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    for j, name in enumerate(action_cols):
        stats = {
            "mu_mean_raw": float(mu_raw[:, j].mean().item()),
            "mu_std_raw": float(mu_raw[:, j].std(unbiased=False).item()),
            "mu_min_raw": float(mu_raw[:, j].min().item()),
            "mu_max_raw": float(mu_raw[:, j].max().item()),
            "avg_std_raw": float(std_raw[:, j].mean().item()),
            "min_std_raw": float(std_raw[:, j].min().item()),
            "max_std_raw": float(std_raw[:, j].max().item()),
            "avg_std_norm": float(std_norm[:, j].mean().item()),
            "min_log_std_norm": float(log_std_norm[:, j].min().item()),
            "max_log_std_norm": float(log_std_norm[:, j].max().item()),
        }
        out[name] = stats
        print(
            f"{name:30s} "
            f"mu_raw[mean,std,min,max]=[{stats['mu_mean_raw']:.4f}, {stats['mu_std_raw']:.4f}, {stats['mu_min_raw']:.4f}, {stats['mu_max_raw']:.4f}] "
            f"| std_raw[mean,min,max]=[{stats['avg_std_raw']:.4f}, {stats['min_std_raw']:.4f}, {stats['max_std_raw']:.4f}] "
            f"| logstd_norm[min,max]=[{stats['min_log_std_norm']:.4f}, {stats['max_log_std_norm']:.4f}]"
        )
    return out


# ==================================== #
#    policy artifact compatibility     #
# ==================================== #
@dataclass
class ActionSpec:
    action_names: List[str]
    lows: List[float]
    highs: List[float]
    integer_idx: Optional[int]
    integer_low: Optional[int]
    integer_high: Optional[int]
    integer_name: Optional[str]


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        layers: List[nn.Module] = []
        d = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            d = hidden_dim
        self.net = nn.Sequential(*layers)
        self.output_dim = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianRoundedPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        depth: int = 2,
        log_std_min: float = -4.0,
        log_std_max: float = 1.0,
        action_lows: Optional[Sequence[float]] = None,
        action_highs: Optional[Sequence[float]] = None,
        integer_idx: Optional[int] = None,
        integer_low: Optional[int] = None,
        integer_high: Optional[int] = None,
    ):
        super().__init__()
        self.backbone = MLP(state_dim, hidden_dim, depth)
        h = self.backbone.output_dim
        self.mu_head = nn.Linear(h, action_dim)
        self.log_std_head = nn.Linear(h, action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        if action_lows is None:
            action_lows = np.full(action_dim, -np.inf, dtype=np.float32)
        if action_highs is None:
            action_highs = np.full(action_dim, np.inf, dtype=np.float32)

        self.register_buffer("action_lows", torch.as_tensor(action_lows, dtype=torch.float32))
        self.register_buffer("action_highs", torch.as_tensor(action_highs, dtype=torch.float32))

        self.integer_idx = -1 if integer_idx is None else int(integer_idx)
        self.integer_low = None if integer_low is None else int(integer_low)
        self.integer_high = None if integer_high is None else int(integer_high)

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.backbone(states)
        mu = self.mu_head(z)
        log_std = torch.clamp(self.log_std_head(z), self.log_std_min, self.log_std_max)
        return mu, log_std

    def gaussian_params(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(states)
        return mu, torch.exp(log_std)

    def _clamp_to_support(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.min(x, self.action_highs), self.action_lows)

    def _round_integer_dim(self, x: torch.Tensor) -> torch.Tensor:
        if self.integer_idx < 0:
            return x
        y = x.clone()
        y[:, self.integer_idx] = torch.round(y[:, self.integer_idx])
        if self.integer_low is not None:
            y[:, self.integer_idx] = torch.clamp(y[:, self.integer_idx], min=float(self.integer_low))
        if self.integer_high is not None:
            y[:, self.integer_idx] = torch.clamp(y[:, self.integer_idx], max=float(self.integer_high))
        return y

    def greedy_actions(self, states: torch.Tensor) -> torch.Tensor:
        mu, _ = self.forward(states)
        a = self._clamp_to_support(mu)
        a = self._round_integer_dim(a)
        return a

    def sample_actions(self, states: torch.Tensor) -> torch.Tensor:
        mu, log_std = self.forward(states)
        std = torch.exp(log_std)
        a = mu + std * torch.randn_like(std)
        a = self._clamp_to_support(a)
        a = self._round_integer_dim(a)
        return a


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


def _sanitize_reward_tag(x: str) -> str:
    return str(x).replace('/', '_').replace(' ', '_')


def resolve_gaussian_policy_paths(ckpt_dir: Path, objective_name: str) -> Tuple[Path, Path]:
    tags = [objective_name, _sanitize_reward_tag(objective_name)]
    tried = []
    for tag in tags:
        npz = ckpt_dir / f'linear_gaussian_policy_{tag}.npz'
        js = ckpt_dir / f'linear_gaussian_policy_{tag}.json'
        tried.append((npz, js))
        if npz.exists() and js.exists():
            return npz, js
    for tag in tags:
        pt = ckpt_dir / f'gaussian_policy_{tag}.pt'
        js = ckpt_dir / f'gaussian_policy_{tag}.json'
        tried.append((pt, js))
        if pt.exists() and js.exists():
            return pt, js
    raise FileNotFoundError(f'Could not find policy artifacts for {objective_name}. Tried: {tried}')


def resolve_value_model_path(ckpt_dir: Path, objective_name: str) -> Optional[Path]:
    tags = [objective_name, _sanitize_reward_tag(objective_name)]
    for tag in tags:
        p = ckpt_dir / f'iql_value_{tag}.d3'
        if p.exists():
            return p
    return None


def _maybe_none_int(v) -> Optional[int]:
    if v is None:
        return None
    iv = int(v)
    return None if iv < 0 else iv


def _load_linear_policy(path_npz: Path, path_json: Path):
    meta = json.loads(path_json.read_text())
    arr = np.load(path_npz)
    policy = LinearGaussianPolicy(
        theta_mu=np.asarray(arr['theta_mu'], dtype=np.float64),
        epsilon_mu=np.asarray(arr['epsilon_mu'], dtype=np.float64),
        theta_sigma=np.asarray(arr['theta_sigma'], dtype=np.float64),
        epsilon_sigma=np.asarray(arr['epsilon_sigma'], dtype=np.float64),
        action_lows=np.asarray(arr['action_lows'], dtype=np.float64),
        action_highs=np.asarray(arr['action_highs'], dtype=np.float64),
        action_names=list(meta['action_names']),
        state_names=list(meta['state_names']),
        reward_name=str(meta.get('reward_name', path_npz.stem)),
        integer_idx=_maybe_none_int(meta.get('integer_action_index')),
        integer_low=_maybe_none_int(meta.get('integer_action_low')),
        integer_high=_maybe_none_int(meta.get('integer_action_high')),
        integer_name=meta.get('integer_action_name', None),
    )
    spec = ActionSpec(
        action_names=list(meta['action_names']),
        lows=[float(v) for v in policy.action_lows.tolist()],
        highs=[float(v) for v in policy.action_highs.tolist()],
        integer_idx=policy.integer_idx,
        integer_low=policy.integer_low,
        integer_high=policy.integer_high,
        integer_name=policy.integer_name,
    )
    return policy, meta, spec


def load_gaussian_policy(path_pt: Path, path_json: Path, device: torch.device):
    if str(path_pt).endswith('.npz'):
        return _load_linear_policy(path_pt, path_json)

    meta = json.loads(path_json.read_text())
    model = GaussianRoundedPolicy(
        state_dim=int(meta['state_dim']),
        action_dim=int(meta['action_dim']),
        hidden_dim=int(meta['hidden_dim']),
        depth=int(meta['depth']),
        log_std_min=float(meta['log_std_min']),
        log_std_max=float(meta['log_std_max']),
        action_lows=meta['action_lows'],
        action_highs=meta['action_highs'],
        integer_idx=meta.get('integer_action_index', None),
        integer_low=meta.get('integer_action_low', None),
        integer_high=meta.get('integer_action_high', None),
    ).to(device)
    state = torch.load(path_pt, map_location='cpu')
    model.load_state_dict(state, strict=True)
    model.eval()
    spec = ActionSpec(
        action_names=list(meta['action_names']),
        lows=[float(v) for v in meta['action_lows']],
        highs=[float(v) for v in meta['action_highs']],
        integer_idx=meta.get('integer_action_index', None),
        integer_low=meta.get('integer_action_low', None),
        integer_high=meta.get('integer_action_high', None),
        integer_name=meta.get('integer_action_name', None),
    )
    return model, meta, spec


def _linear_policy_clamp_and_round(policy: LinearGaussianPolicy, x: torch.Tensor) -> torch.Tensor:
    y = torch.max(torch.min(x, torch.as_tensor(policy.action_highs, dtype=x.dtype, device=x.device)),
                  torch.as_tensor(policy.action_lows, dtype=x.dtype, device=x.device))
    if policy.integer_idx is not None:
        j = int(policy.integer_idx)
        y = y.clone()
        y[:, j] = torch.round(y[:, j])
        if policy.integer_low is not None:
            y[:, j] = torch.clamp(y[:, j], min=float(policy.integer_low))
        if policy.integer_high is not None:
            y[:, j] = torch.clamp(y[:, j], max=float(policy.integer_high))
    return y


def gaussian_policy_stats_raw(model, states_raw: torch.Tensor) -> Dict[str, torch.Tensor]:
    if isinstance(model, LinearGaussianPolicy):
        with torch.no_grad():
            theta_mu = torch.as_tensor(model.theta_mu, dtype=states_raw.dtype, device=states_raw.device)
            eps_mu = torch.as_tensor(model.epsilon_mu, dtype=states_raw.dtype, device=states_raw.device)
            theta_sigma = torch.as_tensor(model.theta_sigma, dtype=states_raw.dtype, device=states_raw.device)
            eps_sigma = torch.as_tensor(model.epsilon_sigma, dtype=states_raw.dtype, device=states_raw.device)
            mu_raw_unclipped = states_raw @ theta_mu + eps_mu
            log_std_raw = states_raw @ theta_sigma + eps_sigma
            std_raw = torch.exp(log_std_raw)
            mu_raw = _linear_policy_clamp_and_round(model, mu_raw_unclipped)
            greedy_raw = mu_raw.clone()
        return {'mu_raw': mu_raw, 'std_raw': std_raw, 'greedy_raw': greedy_raw}

    with torch.no_grad():
        mu_raw, std_raw = model.gaussian_params(states_raw)
        mu_raw = model._clamp_to_support(mu_raw)
        greedy_raw = model.greedy_actions(states_raw)
    return {
        'mu_raw': mu_raw,
        'std_raw': std_raw,
        'greedy_raw': greedy_raw,
    }


def sample_gaussian_actions_raw(model, states_raw: torch.Tensor, n_samples: int) -> torch.Tensor:
    if isinstance(model, LinearGaussianPolicy):
        with torch.no_grad():
            s_rep = states_raw.repeat_interleave(n_samples, dim=0)
            theta_mu = torch.as_tensor(model.theta_mu, dtype=s_rep.dtype, device=s_rep.device)
            eps_mu = torch.as_tensor(model.epsilon_mu, dtype=s_rep.dtype, device=s_rep.device)
            theta_sigma = torch.as_tensor(model.theta_sigma, dtype=s_rep.dtype, device=s_rep.device)
            eps_sigma = torch.as_tensor(model.epsilon_sigma, dtype=s_rep.dtype, device=s_rep.device)
            mu_raw = s_rep @ theta_mu + eps_mu
            std_raw = torch.exp(s_rep @ theta_sigma + eps_sigma)
            a = mu_raw + std_raw * torch.randn_like(std_raw)
            return _linear_policy_clamp_and_round(model, a)

    with torch.no_grad():
        s_rep = states_raw.repeat_interleave(n_samples, dim=0)
        return model.sample_actions(s_rep)


def distill_gaussian_policy_to_linear_surrogate(
    model,
    s_raw: torch.Tensor,
    s_norm: torch.Tensor,
    a_mu: torch.Tensor,
    a_sd: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float, torch.Tensor, torch.Tensor]:
    stats = gaussian_policy_stats_raw(model, s_raw)
    mu_raw = stats['mu_raw']
    std_raw = stats['std_raw'].clamp_min(1e-6)

    mu_norm = (mu_raw - a_mu) / (a_sd + 1e-12)
    std_norm = std_raw / (a_sd + 1e-12)
    log_std_norm = torch.log(std_norm.clamp_min(1e-6))

    n, d_s = s_norm.shape
    S = torch.cat([s_norm, torch.ones(n, 1, device=s_norm.device, dtype=s_norm.dtype)], dim=1)
    beta_mu = torch.linalg.lstsq(S, mu_norm).solution
    beta_ls = torch.linalg.lstsq(S, log_std_norm).solution

    theta_mean_vec = beta_mu[:d_s, :]
    epsilon_mean = beta_mu[d_s, :]
    theta_std_vec = beta_ls[:d_s, :]
    epsilon_std = beta_ls[d_s, :]

    with torch.no_grad():
        mu_fit = S @ beta_mu
        ls_fit = S @ beta_ls
        mu_mse = torch.mean((mu_norm - mu_fit) ** 2).item()
        ls_mse = torch.mean((log_std_norm - ls_fit) ** 2).item()

    return (
        theta_mean_vec,
        theta_std_vec,
        epsilon_mean,
        epsilon_std,
        float(mu_mse),
        float(ls_mse),
        mu_norm,
        log_std_norm,
    )


def print_raw_policy_difference(ckpt_dir: Path, reward_a: str, reward_b: str, device: torch.device) -> None:
    pA, jA = resolve_gaussian_policy_paths(ckpt_dir, reward_a)
    pB, jB = resolve_gaussian_policy_paths(ckpt_dir, reward_b)
    modelA, _, _ = load_gaussian_policy(pA, jA, device)
    modelB, _, _ = load_gaussian_policy(pB, jB, device)
    print('\n=== RAW POLICY PARAMETER DIFF ===')
    print(reward_a, ':', pA)
    print(reward_b, ':', pB)

    if isinstance(modelA, LinearGaussianPolicy) and isinstance(modelB, LinearGaussianPolicy):
        pairs = {
            'theta_mu': (modelA.theta_mu, modelB.theta_mu),
            'epsilon_mu': (modelA.epsilon_mu, modelB.epsilon_mu),
            'theta_sigma': (modelA.theta_sigma, modelB.theta_sigma),
            'epsilon_sigma': (modelA.epsilon_sigma, modelB.epsilon_sigma),
        }
        for k, (a, b) in pairs.items():
            d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
            print(f'{k:40s} L2={np.linalg.norm(d):.6g} max|Δ|={np.max(np.abs(d)):.6g}')
    else:
        sdA = modelA.state_dict() if hasattr(modelA, 'state_dict') else {}
        sdB = modelB.state_dict() if hasattr(modelB, 'state_dict') else {}
        for k in sorted(set(sdA.keys()).intersection(sdB.keys())):
            d = sdA[k].float() - sdB[k].float()
            print(f'{k:40s} L2={d.norm().item():.6g} max|Δ|={d.abs().max().item():.6g}')
    print('==============================\n')


def print_policy_spec(
    name: str,
    theta_mean_vec: torch.Tensor,
    theta_std_vec: torch.Tensor,
    epsilon_mean: torch.Tensor,
    epsilon_std: torch.Tensor,
    state_cols: List[str],
    action_cols: List[str],
) -> None:
    print('\n' + '=' * 70)
    print(f'POLICY SPECIFICATION: {name}')
    print('=' * 70)
    print('state_cols       :', state_cols)
    print('action_cols      :', action_cols)
    print('\ntheta_mean_vec:')
    print(theta_mean_vec.detach().cpu())
    print('\nepsilon_mean:')
    print(epsilon_mean.detach().cpu())
    print('\ntheta_std_vec:')
    print(theta_std_vec.detach().cpu())
    print('\nepsilon_std:')
    print(epsilon_std.detach().cpu())
    print('\nNorms:')
    print('||theta_mean||_F =', float(torch.norm(theta_mean_vec).item()))
    print('||theta_std||_F  =', float(torch.norm(theta_std_vec).item()))
    print('=' * 70 + '\n')


def _normal_cdf(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z / SQRT_2))


def gaussian_offsupport_probability_diag(
    mu_raw: torch.Tensor,
    std_raw: torch.Tensor,
    train_min_raw: torch.Tensor,
    train_max_raw: torch.Tensor,
    action_cols: List[str],
    eps: float = 1e-8,
) -> Dict[str, dict]:
    std = std_raw.clamp_min(eps)
    z_lo = (train_min_raw - mu_raw) / std
    z_hi = (train_max_raw - mu_raw) / std
    inside = (_normal_cdf(z_hi) - _normal_cdf(z_lo)).clamp(0.0, 1.0)
    outside = 1.0 - inside

    out: Dict[str, dict] = {}
    print("\n" + "=" * 100)
    print("ANALYTIC GAUSSIAN OFF-SUPPORT PROBABILITY")
    print("=" * 100)
    for j, name in enumerate(action_cols):
        stats = {
            "train_min": float(train_min_raw[j].item()),
            "train_max": float(train_max_raw[j].item()),
            "avg_outside_prob": float(outside[:, j].mean().item()),
            "median_outside_prob": float(outside[:, j].median().item()),
            "max_outside_prob": float(outside[:, j].max().item()),
        }
        out[name] = stats
        print(
            f"{name:30s} train[min,max]=[{stats['train_min']:.4f}, {stats['train_max']:.4f}] "
            f"| outside_prob[avg,med,max]=[{stats['avg_outside_prob']:.6f}, {stats['median_outside_prob']:.6f}, {stats['max_outside_prob']:.6f}]"
        )
    return out


def _subsample_rows_tensor(x: torch.Tensor, max_rows: int, seed: int) -> torch.Tensor:
    if x.shape[0] <= max_rows:
        return x
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=g)[:max_rows]
    return x[idx.to(x.device)]


# ==================================== #
#           d3rlpy utilities           #
# ==================================== #
def load_d3(path: str, device: Optional[str] = None):
    kwargs = {}
    if device is not None:
        kwargs['device'] = device
    if hasattr(d3rlpy, 'load_learnable'):
        try:
            return d3rlpy.load_learnable(path, **kwargs)
        except TypeError:
            return d3rlpy.load_learnable(path)
    if hasattr(d3rlpy, 'load_learner'):
        try:
            return d3rlpy.load_learner(path, **kwargs)
        except TypeError:
            return d3rlpy.load_learner(path)
    if hasattr(d3rlpy, 'load'):
        return d3rlpy.load(path)
    raise AttributeError('No compatible d3rlpy load function found.')


def predict_value_any(learner, s_np: np.ndarray, a_np: np.ndarray) -> np.ndarray:
    if hasattr(learner, 'predict_value'):
        q = learner.predict_value(s_np, a_np)
    elif hasattr(learner, 'predict_q'):
        q = learner.predict_q(s_np, a_np)
    elif hasattr(learner, 'impl') and hasattr(learner.impl, 'predict_value'):
        q = learner.impl.predict_value(s_np, a_np)
    else:
        raise AttributeError('Cannot find predict_value(obs, act) on the learner.')
    q = np.asarray(q)
    if q.ndim == 2 and q.shape[1] == 1:
        q = q[:, 0]
    return q


# ==================================== #
#         Single-policy pipeline       #
# ==================================== #
def run_one_policy(
    policy_name: str,
    out_dir: Path,
    do_plots: bool,
    run_test: bool,
    cfg: dict,
    args: argparse.Namespace,
    m_Z: int,
    device: torch.device,
    s0_tr_raw: torch.Tensor,
    s1_tr_raw: Optional[torch.Tensor],
    a0_tr_raw: torch.Tensor,
    a1_tr_raw: Optional[torch.Tensor],
    r_tr_raw: torch.Tensor,
    s0_val_raw: torch.Tensor,
    a0_val_raw: torch.Tensor,
    r_val_raw: torch.Tensor,
    s0_test_raw: torch.Tensor,
    a0_test_raw: torch.Tensor,
    r_test_raw: torch.Tensor,
    s0_tr: torch.Tensor,
    s1_tr: torch.Tensor,
    a0_tr: torch.Tensor,
    a1_tr: torch.Tensor,
    r_tr: torch.Tensor,
    s0_val: torch.Tensor,
    a0_val: torch.Tensor,
    r_val: torch.Tensor,
    s0_test: torch.Tensor,
    a0_test: torch.Tensor,
    r_test: torch.Tensor,
    s_star: torch.Tensor,
    a_star: torch.Tensor,
    s_mu: torch.Tensor,
    s_sd: torch.Tensor,
    a_mu: torch.Tensor,
    a_sd: torch.Tensor,
    r_mu: torch.Tensor,
    r_sd: torch.Tensor,
    state_cols: List[str],
    action_cols: List[str],
    reward_cols: List[str],
    ckpt_dir: Path,
    plot_n: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    d_s = s0_tr.shape[1]
    d_a = a0_tr.shape[1]
    d_r = r_tr.shape[1]
    discrete_reward_dims = _resolve_discrete_reward_dims(args.discrete_reward_cols, reward_cols)
    discrete_reward_cols = [reward_cols[j] for j in discrete_reward_dims]
    if discrete_reward_cols:
        print(
            f"Forcing reward columns {discrete_reward_cols} (dims={discrete_reward_dims}) to be treated as discrete."
        )

    actor_pt, actor_json = resolve_gaussian_policy_paths(ckpt_dir, policy_name)
    model, meta, spec = load_gaussian_policy(actor_pt, actor_json, device)

    actor_state_names = list(meta.get('state_names', []))
    actor_action_names = list(meta.get('action_names', []))
    if actor_state_names and actor_state_names != list(state_cols):
        raise ValueError(
            f'State columns mismatch for policy {policy_name}. '
            f'Actor expects {actor_state_names}, evaluator uses {state_cols}.'
        )
    if actor_action_names and actor_action_names != list(action_cols):
        raise ValueError(
            f'Action columns mismatch for policy {policy_name}. '
            f'Actor expects {actor_action_names}, evaluator uses {action_cols}.'
        )

    (
        theta_mean_vec,
        theta_std_vec,
        epsilon_mean,
        epsilon_std,
        mu_mse,
        ls_mse,
        _mu_tr_norm_sur,
        _logstd_tr_norm_sur,
    ) = distill_gaussian_policy_to_linear_surrogate(
        model=model,
        s_raw=s0_tr_raw,
        s_norm=s0_tr,
        a_mu=a_mu,
        a_sd=a_sd,
    )

    print(f'\nLoaded gaussian-rounded policy: {actor_pt}')
    print(f'Gaussian-surrogate distill MSE: mu={mu_mse:.6e} | log_std={ls_mse:.6e}')
    print_policy_spec(
        name=policy_name,
        theta_mean_vec=theta_mean_vec,
        theta_std_vec=theta_std_vec,
        epsilon_mean=epsilon_mean,
        epsilon_std=epsilon_std,
        state_cols=state_cols,
        action_cols=action_cols,
    )

    stats_tr = gaussian_policy_stats_raw(model, s0_tr_raw)
    stats_val = gaussian_policy_stats_raw(model, s0_val_raw)
    stats_test = gaussian_policy_stats_raw(model, s0_test_raw)

    def _raw_to_norm_stats(stats: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_norm = (stats['mu_raw'] - a_mu) / (a_sd + 1e-12)
        logstd_norm = torch.log((stats['std_raw'] / (a_sd + 1e-12)).clamp_min(1e-6))
        return mu_norm, logstd_norm

    mu_tr_norm, logstd_tr_norm = _raw_to_norm_stats(stats_tr)
    mu_val_norm, logstd_val_norm = _raw_to_norm_stats(stats_val)
    mu_test_norm, logstd_test_norm = _raw_to_norm_stats(stats_test)

    mu_tr_raw = stats_tr['mu_raw']
    mu_val_raw = stats_val['mu_raw']
    mu_test_raw = stats_test['mu_raw']
    std_tr_raw = stats_tr['std_raw']
    std_val_raw = stats_val['std_raw']
    std_test_raw = stats_test['std_raw']
    greedy_tr_raw = stats_tr['greedy_raw']
    greedy_val_raw = stats_val['greedy_raw']
    greedy_test_raw = stats_test['greedy_raw']

    policy_output_summary = {
        'train': summarize_policy_outputs(mu_tr_norm, logstd_tr_norm, a_mu, a_sd, action_cols, f'POLICY OUTPUT SUMMARY ON TRAIN: {policy_name}'),
        'val': summarize_policy_outputs(mu_val_norm, logstd_val_norm, a_mu, a_sd, action_cols, f'POLICY OUTPUT SUMMARY ON VAL: {policy_name}'),
        'test': summarize_policy_outputs(mu_test_norm, logstd_test_norm, a_mu, a_sd, action_cols, f'POLICY OUTPUT SUMMARY ON TEST: {policy_name}'),
    }

    train_min_raw = a0_tr_raw.min(dim=0).values
    train_max_raw = a0_tr_raw.max(dim=0).values

    off_support_greedy_train = off_support_rate_against_train(
        a_candidate_raw=greedy_tr_raw,
        a_train_raw=a0_tr_raw,
        action_cols=action_cols,
        title=f'OFF-SUPPORT CHECK ON TRAIN STATES (GREEDY ACTION): {policy_name}',
    )
    off_support_greedy_test = off_support_rate_against_train(
        a_candidate_raw=greedy_test_raw,
        a_train_raw=a0_tr_raw,
        action_cols=action_cols,
        title=f'OFF-SUPPORT CHECK ON TEST STATES (GREEDY ACTION): {policy_name}',
    )

    std_for_diag = std_test_raw
    if args.std_cap_for_diagnostics is not None:
        std_for_diag = torch.clamp(std_test_raw, max=float(args.std_cap_for_diagnostics))
    analytic_offsupport_test = gaussian_offsupport_probability_diag(
        mu_raw=mu_test_raw,
        std_raw=std_for_diag,
        train_min_raw=train_min_raw,
        train_max_raw=train_max_raw,
        action_cols=action_cols,
    )

    plot_n_eff = min(int(plot_n), s0_test_raw.shape[0])
    sample_n = max(1, int(args.support_sample_n))
    sampled_test_raw = sample_gaussian_actions_raw(
        model=model,
        states_raw=s0_test_raw[:plot_n_eff],
        n_samples=sample_n,
    )
    if bool(int(args.clip_sampled_actions)):
        sampled_test_raw = torch.max(sampled_test_raw, train_min_raw.view(1, -1))
        sampled_test_raw = torch.min(sampled_test_raw, train_max_raw.view(1, -1))
    off_support_sampled_test = off_support_rate_against_train(
        a_candidate_raw=sampled_test_raw,
        a_train_raw=a0_tr_raw,
        action_cols=action_cols,
        title=f'OFF-SUPPORT CHECK ON TEST STATES (SAMPLED ACTIONS): {policy_name}',
    )

    save_action_hist_compare(
        a_data_raw=a0_test_raw[:plot_n_eff],
        a_policy_raw=greedy_test_raw[:plot_n_eff],
        action_cols=action_cols,
        out_dir=plots_dir,
        prefix=f'{policy_name}_greedy_vs_testdata',
    )
    sampled_for_plot = _subsample_rows_tensor(
        sampled_test_raw,
        max_rows=plot_n_eff,
        seed=int(args.seed) + 10007,
    )
    save_action_hist_compare(
        a_data_raw=a0_test_raw[:plot_n_eff],
        a_policy_raw=sampled_for_plot,
        action_cols=action_cols,
        out_dir=plots_dir,
        prefix=f'{policy_name}_sampled_vs_testdata',
    )

    value_metrics = {}
    for reward_name in reward_cols:
        value_path = resolve_value_model_path(ckpt_dir, reward_name)
        if value_path is None:
            continue
        learner = load_d3(str(value_path), device='cpu')
        value_metrics[f'Q_{reward_name}_on_policy_greedy_mean'] = float(
            predict_value_any(learner, _np(s0_test_raw), _np(greedy_test_raw)).mean()
        )
        value_metrics[f'Q_{reward_name}_on_policy_meanaction_mean'] = float(
            predict_value_any(learner, _np(s0_test_raw), _np(mu_test_raw)).mean()
        )
        value_metrics[f'Q_{reward_name}_on_data_mean'] = float(
            predict_value_any(learner, _np(s0_test_raw), _np(a0_test_raw)).mean()
        )

    target_p_choice = 'gaussian'
    target_p_params = {
        'gaussian': {
            'theta_mean': theta_mean_vec,
            'theta_std': theta_std_vec,
            'epsilon_mean': epsilon_mean,
            'epsilon_std': epsilon_std,
        }
    }

    nu_Z = float(cfg['nu_Z'])
    ell_Z = float(cfg['ell_Z'])
    lambda_reg = float(cfg['lambda_reg'])
    sigma_Z = float(cfg['sigma_Z'])

    gamma_val = float(args.gamma)
    hull_expand_factor = float(args.hull_expand_factor)
    lr = float(args.embedding_lr)

    fixed_point_constraint = bool(int(args.fixed_point_constraint))
    FP_penalty_lambda = float(args.fp_penalty_lambda)
    Sum_one_W = bool(int(args.sum_one_W))
    NonNeg_W = bool(int(args.nonneg_W))
    mass_anchor_lambda = float(args.mass_anchor_lambda)
    target_mass = float(args.target_mass)

    bandwidth = float(args.bandwidth)
    bandwidth_per_dim = _parse_optional_float_list(args.bandwidth_per_dim)
    marginal_bandwidths = _resolve_marginal_bandwidths(d_r, bandwidth, bandwidth_per_dim)
    lambda_rec = float(args.lambda_rec)
    method = str(args.method)

    sa_tr = torch.cat([s0_tr, a0_tr], dim=1)

    t_train_start = time.time()
    t0 = tic(f'START estimate_embedding [{policy_name}]')
    embedding_kwargs = dict(
        s0=s0_tr,
        s1=s1_tr,
        a0=a0_tr,
        a1=a1_tr,
        s_star=s_star,
        a_star=a_star,
        r=r_tr,
        discrete_dims=discrete_reward_dims,
        target_p_choice=target_p_choice,
        target_p_params=target_p_params,
        nu=nu_Z,
        length_scale=ell_Z,
        sigma=sigma_Z,
        gamma_val=gamma_val,
        lambda_reg=lambda_reg,
        lambda_B=float(getattr(args, "lambda_B", 0.0)),
        num_grid_points=m_Z,
        hull_expand_factor=hull_expand_factor,
        lr=lr,
        num_steps=int(args.num_steps),
        FP_penalty_lambda=FP_penalty_lambda,
        fixed_point_constraint=fixed_point_constraint,
        Sum_one_W=Sum_one_W,
        NonNeg_W=NonNeg_W,
        mass_anchor_lambda=mass_anchor_lambda,
        target_mass=target_mass,
        mean_embedding_basis_size=(None if int(getattr(args, "mean_embedding_basis_size", 0) or 0) <= 0 else int(getattr(args, "mean_embedding_basis_size", 0))),
        mean_embedding_basis_method=str(getattr(args, "mean_embedding_basis_method", "kmeans")),
        mean_embedding_basis_seed=(int(getattr(args, "mean_embedding_basis_seed", getattr(args, "seed", 0))) if getattr(args, "mean_embedding_basis_seed", None) is not None else int(getattr(args, "seed", 0))),
        mean_embedding_basis_standardize=bool(int(getattr(args, "mean_embedding_basis_standardize", 1))),
        mean_embedding_basis_candidate_pool=(None if int(getattr(args, "mean_embedding_basis_candidate_pool", 0) or 0) <= 0 else int(getattr(args, "mean_embedding_basis_candidate_pool", 0))),
        mean_embedding_basis_max_iter=int(getattr(args, "mean_embedding_basis_max_iter", 20)),
        mean_embedding_basis_batch_size=int(getattr(args, "mean_embedding_basis_batch_size", 8192)),
        device=str(device),
        dtype=torch.float32,
    )
    B_hat, hist_obj, hist_be, pre = _call_estimate_embedding_compatible(**embedding_kwargs)
    toc(t0, f'DONE estimate_embedding [{policy_name}]')

    t0 = tic(f'START mean-embedding state-action basis resolution [{policy_name}]')
    B_hat, pre_basis, sa_basis, sa_basis_idx, K_sa_basis, mean_embedding_basis_info = _project_B_to_mean_embedding_basis(
        B_hat=B_hat,
        pre=pre,
        sa_train=sa_tr,
        args=args,
        nu=nu_Z,
        length_scale=ell_Z,
        sigma=sigma_Z,
    )
    pre = pre_basis
    toc(t0, f'DONE mean-embedding state-action basis resolution [{policy_name}]')
    print(
        f"Mean-embedding operator basis: mode={mean_embedding_basis_info['mode']} | "
        f"B_shape={tuple(B_hat.shape)} | L={mean_embedding_basis_info['effective_basis_size']} | "
        f"m={B_hat.shape[1]}",
        flush=True,
    )
    train_time = time.time() - t_train_start

    # IMPORTANT:
    # B_hat was optimized/projected against pre['Z_grid'].  Do not replace that grid after
    # training for risk computation.  If support corrections are needed for
    # visualization, apply them only to plotting atoms below.
    Z_grid = pre['Z_grid']

    observed_rewards_all_raw = torch.cat([r_tr_raw, r_val_raw, r_test_raw], dim=0)
    Z_grid_raw_est = _denorm(Z_grid, r_mu, r_sd)
    Z_grid_raw_supported, reward_support_info = _enforce_reward_support_raw(
        Z_grid_raw=Z_grid_raw_est,
        reward_cols=reward_cols,
        observed_rewards_raw=observed_rewards_all_raw,
        force_discrete_dims=discrete_reward_dims,
    )

    # Export optimized KE-DRL Z-grid
    _save_array_csv(
        out_dir / "Z_grid_normalized_optimized.csv",
        Z_grid.detach().cpu(),
        reward_cols,
    )

    _save_array_csv(
        out_dir / "Z_grid_raw_est.csv",
        Z_grid_raw_est.detach().cpu(),
        reward_cols,
    )

    _save_array_csv(
        out_dir / "Z_grid_raw_supported_for_plots.csv",
        Z_grid_raw_supported.detach().cpu(),
        reward_cols,
    )

    # Export KE-DRL learned embedding operator weights and its state-action basis.
    _save_array_csv(
        out_dir / "B_hat_mean_embedding_operator.csv",
        B_hat.detach().cpu(),
    )
    _save_array_csv(
        out_dir / "mean_embedding_state_action_basis.csv",
        sa_basis.detach().cpu(),
    )
    _save_array_csv(
        out_dir / "mean_embedding_state_action_basis_indices.csv",
        sa_basis_idx.detach().cpu(),
        ["basis_row_index"],
    )
    save_json(out_dir / "mean_embedding_basis_diagnostics.json", mean_embedding_basis_info)
    Z_grid_raw_support_shift_max = float(
        torch.max(torch.abs(Z_grid_raw_supported.to(Z_grid_raw_est.device) - Z_grid_raw_est)).item()
    )
    if Z_grid_raw_support_shift_max > 1e-6:
        print(
            f"Warning: support projection would move Z_grid atoms by max raw-scale shift "
            f"{Z_grid_raw_support_shift_max:.6g}. Risk is computed on the original optimized grid; "
            f"support-projected atoms are used only for plots."
        )
        simplex_calibration_diagnostics = {"simplex_calibrate_B": bool(int(args.simplex_calibrate_B))}
        if bool(int(args.simplex_calibrate_B)):
            print("\nSTART simplex calibration of B_hat")
            B_hat, simplex_calibration_diagnostics = _calibrate_B_hat_to_simplex(
                B_hat=B_hat,
                sa_train=sa_basis,
                nu=nu_Z,
                length_scale=ell_Z,
                sigma=sigma_Z,
                ridge=float(args.simplex_calib_ridge),
                max_rows=int(args.simplex_calib_max_rows),
                seed=int(args.seed),
            )
            print("DONE simplex calibration of B_hat")
            for k, v in simplex_calibration_diagnostics.items():
                print(f"  {k}: {v}")

        save_json(out_dir / "simplex_calibration_diagnostics.json", simplex_calibration_diagnostics)

    # Export the learned reward atom grid in both normalized and raw scales.
    _save_array_csv(out_dir / "Z_grid_normalized_optimized.csv", Z_grid.detach().cpu(), reward_cols)
    _save_array_csv(out_dir / "Z_grid_raw_est.csv", Z_grid_raw_est.detach().cpu(), reward_cols)
    _save_array_csv(out_dir / "Z_grid_raw_supported_for_plots.csv", Z_grid_raw_supported.detach().cpu(), reward_cols)
    _save_array_csv(out_dir / "B_hat_mean_embedding_operator.csv", B_hat.detach().cpu())
    _save_array_csv(out_dir / "mean_embedding_state_action_basis.csv", sa_basis.detach().cpu())
    _save_array_csv(out_dir / "mean_embedding_state_action_basis_indices.csv", sa_basis_idx.detach().cpu(), ["basis_row_index"])
    save_json(out_dir / "mean_embedding_basis_diagnostics.json", mean_embedding_basis_info)

    t_val_start = time.time()
    t0 = tic(f'START val kernel [{policy_name}]')
    sa_val = torch.cat([s0_val, a0_val], dim=1)
    k_sa_val = matern_kernel(sa_val, sa_basis, nu=nu_Z, length_scale=ell_Z, sigma=sigma_Z)
    toc(t0, f'DONE val kernel [{policy_name}]')

    def diagnose_W(k_sa, B_hat, name):
        W = k_sa @ B_hat
        print(f"\n{name} W diagnostics")
        print("shape:", tuple(W.shape))
        print("min:", float(W.min()))
        print("max:", float(W.max()))
        print("row-sum mean:", float(W.sum(1).mean()))
        print("row-sum std:", float(W.sum(1).std()))
        print("negative mass mean:", float(torch.clamp(-W, min=0).sum(1).mean()))
        print("L1 mass mean:", float(W.abs().sum(1).mean()))
        print("")
    diagnose_W(k_sa_val, B_hat, "VAL")

    risk_val = embedding_test_risk(
        Z_test=r_val,
        k_sa_test=k_sa_val,
        B_hat_torch=B_hat,
        Z_grid=Z_grid,
        nu=nu_Z,
        length_scale=ell_Z,
        sigma=sigma_Z,
    )
    val_time = time.time() - t_val_start
    risk_val = float(risk_val)

    risk_test = None
    test_time = None
    if run_test:
        t_test_start = time.time()
        t0 = tic(f'START test kernel [{policy_name}]')
        sa_test = torch.cat([s0_test, a0_test], dim=1)
        k_sa_test = matern_kernel(sa_test, sa_basis, nu=nu_Z, length_scale=ell_Z, sigma=sigma_Z)
        toc(t0, f'DONE test kernel [{policy_name}]')
        diagnose_W(k_sa_test, B_hat, "TEST")
        risk_test = embedding_test_risk(
            Z_test=r_test,
            k_sa_test=k_sa_test,
            B_hat_torch=B_hat,
            Z_grid=Z_grid,
            nu=nu_Z,
            length_scale=ell_Z,
            sigma=sigma_Z,
        )
        test_time = time.time() - t_test_start
        risk_test = float(risk_test)

    print(
        f'[{policy_name}] val_risk={risk_val:.6f} | train={train_time/3600:.2f}h | val={val_time/60:.2f}m'
        + (f' | test_risk={risk_test:.6f} | test={test_time/60:.2f}m' if run_test else '')
    )

    metrics = {
        'policy_name': policy_name,
        'gaussian_policy_path': str(actor_pt),
        'gaussian_policy_meta_path': str(actor_json),
        'cfg': cfg,
        'embedding_config': {
            'gamma': gamma_val,
            'hull_expand_factor': hull_expand_factor,
            'embedding_lr': lr,
            'fixed_point_constraint': fixed_point_constraint,
            'FP_penalty_lambda': FP_penalty_lambda,
            'Sum_one_W': Sum_one_W,
            'NonNeg_W': NonNeg_W,
            'mass_anchor_lambda': mass_anchor_lambda,
            'target_mass': target_mass,
            'bandwidth': bandwidth,
            'bandwidth_per_dim': marginal_bandwidths,
            'lambda_rec': lambda_rec,
            'method': method,
            'num_steps': int(args.num_steps),
            'num_grid_points': int(m_Z),
            'mean_embedding_basis_size_requested': int(getattr(args, "mean_embedding_basis_size", 0) or 0),
            'mean_embedding_basis_size_effective': int(mean_embedding_basis_info['effective_basis_size']),
            'mean_embedding_basis_method': str(mean_embedding_basis_info['basis_method']),
            'mean_embedding_basis_mode': str(mean_embedding_basis_info['mode']),
            'mean_embedding_basis_ridge': float(mean_embedding_basis_info['basis_ridge']),
            'B_hat_shape': [int(B_hat.shape[0]), int(B_hat.shape[1])],
            'discrete_reward_cols': discrete_reward_cols,
            'discrete_reward_dims': discrete_reward_dims,
        },
        'train_time_sec': float(train_time),
        'val_time_sec': float(val_time),
        'test_time_sec': None if test_time is None else float(test_time),
        'val_risk': float(risk_val),
        'test_risk': None if risk_test is None else float(risk_test),
        'surrogate_distill_mu_mse': float(mu_mse),
        'surrogate_distill_logstd_mse': float(ls_mse),
        'policy_output_summary': policy_output_summary,
        'train_policy_action_mean_raw': _np(greedy_tr_raw.mean(0)).tolist(),
        'val_policy_action_mean_raw': _np(greedy_val_raw.mean(0)).tolist(),
        'test_policy_action_mean_raw': _np(greedy_test_raw.mean(0)).tolist(),
        'test_policy_gaussian_mean_raw': _np(mu_test_raw.mean(0)).tolist(),
        'test_policy_action_std_raw_mean': _np(std_test_raw.mean(0)).tolist(),
        'test_data_action_mean_raw': _np(a0_test_raw.mean(0)).tolist(),
        'off_support_greedy_train': off_support_greedy_train,
        'off_support_greedy_test': off_support_greedy_test,
        'off_support_sampled_test': off_support_sampled_test,
        'analytic_gaussian_offsupport_test': analytic_offsupport_test,
        'value_metrics': value_metrics,
        'discrete_reward_cols': discrete_reward_cols,
        'discrete_reward_dims': discrete_reward_dims,
        'reward_support_info': reward_support_info,
        'Z_grid_raw_support_shift_max': float(Z_grid_raw_support_shift_max),
        'mean_embedding_basis_info': mean_embedding_basis_info,
    }
    save_json(out_dir / 'metrics.json', metrics)

    torch.save(
        {
            'B_hat': B_hat.detach().cpu(),
            'B_hat_shape': tuple(B_hat.shape),
            'mean_embedding_basis_info': mean_embedding_basis_info,
            'mean_embedding_state_action_basis': sa_basis.detach().cpu(),
            'mean_embedding_state_action_basis_indices': sa_basis_idx.detach().cpu(),
            'K_mean_embedding_basis': K_sa_basis.detach().cpu(),
            'Z_grid': Z_grid.detach().cpu(),
            'Z_grid_normalized_optimized': Z_grid.detach().cpu(),
            's_star': s_star.detach().cpu(),
            'a_star': a_star.detach().cpu(),
            'normalization': {
                's_mu': s_mu.detach().cpu(),
                's_sd': s_sd.detach().cpu(),
                'a_mu': a_mu.detach().cpu(),
                'a_sd': a_sd.detach().cpu(),
                'r_mu': r_mu.detach().cpu(),
                'r_sd': r_sd.detach().cpu(),
            },
            'target_policy_linear_params': {
                'theta_mean': theta_mean_vec.detach().cpu(),
                'theta_std': theta_std_vec.detach().cpu(),
                'epsilon_mean': epsilon_mean.detach().cpu(),
                'epsilon_std': epsilon_std.detach().cpu(),
            },
            'raw_action_means': {
                'train_policy_action_mean_raw': greedy_tr_raw.mean(0).detach().cpu(),
                'val_policy_action_mean_raw': greedy_val_raw.mean(0).detach().cpu(),
                'test_policy_action_mean_raw': greedy_test_raw.mean(0).detach().cpu(),
                'test_policy_gaussian_mean_raw': mu_test_raw.mean(0).detach().cpu(),
                'test_policy_action_std_raw_mean': std_test_raw.mean(0).detach().cpu(),
                'test_data_action_mean_raw': a0_test_raw.mean(0).detach().cpu(),
            },
            'discrete_reward_cols': discrete_reward_cols,
            'discrete_reward_dims': discrete_reward_dims,
            'reward_support_info': reward_support_info,
            'Z_grid_raw_est': Z_grid_raw_est.detach().cpu(),
            'Z_grid_raw_supported_for_plots': Z_grid_raw_supported.detach().cpu(),
            'Z_grid_normalized_supported_for_plots': _zscore(
                Z_grid_raw_supported,
                r_mu.detach().cpu(),
                r_sd.detach().cpu()
            ).detach().cpu(),            'Z_grid_raw_support_shift_max': float(Z_grid_raw_support_shift_max),
            "Z_grid_normalized_optimized": Z_grid.detach().cpu(),
            "Z_grid_raw_est": Z_grid_raw_est.detach().cpu(),
            "Z_grid_raw_supported_for_plots": Z_grid_raw_supported.detach().cpu(),
            "B_hat_mean_embedding_operator": B_hat.detach().cpu(),
            "mean_embedding_state_action_basis": sa_basis.detach().cpu(),
            "mean_embedding_state_action_basis_indices": sa_basis_idx.detach().cpu(),
            "K_mean_embedding_basis": K_sa_basis.detach().cpu(),
        },

        out_dir / 'artifacts.pt',
    )

    try:
        save_json(
            out_dir / 'history.json',
            {
                'total_loss': [float(x) for x in hist_obj],
                'bellman_error': [float(x) for x in hist_be],
            },
        )
    except Exception:
        torch.save({'hist_obj': hist_obj, 'hist_be': hist_be}, out_dir / 'history.pt')

    if do_plots:
        plot_cfg = build_plot_config(
            lr=lr,
            fixed_point_constraint=fixed_point_constraint,
            FP_penalty_lambda=FP_penalty_lambda,
            Sum_one_W=Sum_one_W,
            NonNeg_W=NonNeg_W,
            mass_anchor_lambda=mass_anchor_lambda,
            target_mass=target_mass,
            num_steps=int(args.num_steps),
            nu=nu_Z,
            length_scale=ell_Z,
            sigma_k=sigma_Z,
            gamma_val=gamma_val,
            num_grid_points=m_Z,
            hull_expand_factor=hull_expand_factor,
            lambda_reg=lambda_reg,
            bandwidth=bandwidth,
            lambda_rec=lambda_rec,
            method=method,
            state_dim=d_s,
            reward_dim=d_r,
            action_dim=d_a,
            s_star=s_star,
            a_star=a_star,
            target_policy=target_p_choice,
        )

        _set_safe_matplotlib_fonts()
        plot_bellman_error(hist_be, config=plot_cfg, outdir=str(plots_dir))
        _set_safe_matplotlib_fonts()
        plot_total_loss(hist_obj, config=plot_cfg, outdir=str(plots_dir))
        _set_safe_matplotlib_fonts()

        beta, Zg = recover_joint_beta(
            B=B_hat,
            k_sa=pre['k_sa'],
            Z_grid=Z_grid,
            Phi=pre['Phi'],
            K_sa=pre['K_sa'],
            config=plot_cfg,
        )
        _save_array_csv(
            out_dir / "mean_embedding_coefficients_beta.csv",
            beta.detach().cpu(),
            ["beta_mean_embedding"],
        )
        Zg_norm_optimized = Zg.detach().cpu().clone()
        Zg_raw_est = _denorm(Zg_norm_optimized.clone(), r_mu.detach().cpu(), r_sd.detach().cpu())
        Zg_raw, recovered_support_info = _enforce_reward_support_raw(
            Z_grid_raw=Zg_raw_est,
            reward_cols=reward_cols,
            observed_rewards_raw=observed_rewards_all_raw,
            force_discrete_dims=discrete_reward_dims,
        )
        Zg_norm_for_density = _zscore(Zg_raw, r_mu.detach().cpu(), r_sd.detach().cpu())

        # Export the specific atom grid used for recovered densities.  This may
        # differ from the optimized grid only by raw-scale support corrections
        # used for plotting, e.g. snapping total_clicks to integer support.
        _save_array_csv(out_dir / "density_atoms_Z_grid_normalized_optimized.csv", Zg_norm_optimized, reward_cols)
        _save_array_csv(out_dir / "density_atoms_Z_grid_raw_est.csv", Zg_raw_est, reward_cols)
        _save_array_csv(out_dir / "density_atoms_Z_grid_raw_supported.csv", Zg_raw, reward_cols)
        _save_array_csv(out_dir / "density_atoms_Z_grid_normalized_supported.csv", Zg_norm_for_density, reward_cols)
        _save_array_csv(out_dir / "mean_embedding_coefficients_beta.csv", beta.detach().cpu(), ["beta_mean_embedding"])

        # Recovered density/PMF uses induced-embedding matching, not direct simplex
        # projection of beta.  This mirrors plot_recovered_densities.py and is
        # controlled by --density-recovery-* CLI/sbatch parameters.
        marginal_payload = recover_marginal_densities_per_dim_induced(
            beta_full=beta,
            Z_grid_raw=Zg_raw,
            Z_grid_norm_dict=Zg_norm_optimized,
            r_mu=r_mu.detach().cpu(),
            r_sd=r_sd.detach().cpu(),
            bandwidths=marginal_bandwidths,
            reward_cols=reward_cols,
            observed_rewards_raw=observed_rewards_all_raw,
            force_discrete_dims=discrete_reward_dims,
            kernel_nu=nu_Z,
            kernel_length_scale=ell_Z,
            kernel_sigma=sigma_Z,
            args=args,
            fallback_device=device,
        )

        density_weights = marginal_payload.get("weights", None)
        beta_tilde_induced = marginal_payload.get("beta_tilde", None)
        if density_weights is not None:
            _save_array_csv(out_dir / "density_weights_induced_embedding_match.csv", density_weights, ["density_weight"])
            # Backward-compatible filename for downstream scripts that already read this file.
            _save_array_csv(out_dir / "density_weights_rkhs_projected.csv", density_weights, ["density_weight"])
            _save_atom_weight_table(
                out_dir / "density_atom_weights_table.csv",
                beta=beta.detach().cpu(),
                density_weights=density_weights,
                Z_norm=Zg_norm_for_density,
                Z_raw=Zg_raw,
                reward_cols=reward_cols,
            )
        if beta_tilde_induced is not None:
            _save_array_csv(out_dir / "beta_tilde_induced_density_recovery.csv", beta_tilde_induced, ["beta_tilde_induced"])
        try:
            save_json(out_dir / "density_recovery_diagnostics.json", marginal_payload.get("density_recovery", {}))
            save_json(out_dir / "density_recovery_optimization_history.json", {"history": marginal_payload.get("optimization_history", [])})
            # Backward-compatible filename.
            save_json(out_dir / "density_weight_projection_diagnostics.json", marginal_payload.get("density_recovery", {}))
        except Exception:
            pass

        _set_safe_matplotlib_fonts()
        plot_densities_per_dim(
            marginal_payload=marginal_payload,
            reward_cols=reward_cols,
            outdir=str(plots_dir),
        )

        joint_dims = (0, 1)

        joint_surface_raw = None
        cache_raw = None

        # Continuous 2D mean-embedding contour map.  This is always produced,
        # including mixed continuous/discrete rewards, because it evaluates the
        # RKHS mean embedding on a continuous 2D raw-scale canvas.
        _set_safe_matplotlib_fonts()
        continuous_mean_embedding_2d = plot_mean_embedding_2d_continuous_contour(
            beta_full=beta,
            Z_grid_norm=Zg_norm_optimized,
            Z_grid_raw_for_limits=Zg_raw,
            r_mu=r_mu.detach().cpu(),
            r_sd=r_sd.detach().cpu(),
            reward_cols=reward_cols,
            outdir=plots_dir,
            policy_name=policy_name,
            nu=nu_Z,
            length_scale=ell_Z,
            sigma=sigma_Z,
            n1=int(args.mean_embedding_contour_n1),
            n2=int(args.mean_embedding_contour_n2),
            pad_frac=float(args.mean_embedding_contour_pad),
            levels=int(args.mean_embedding_contour_levels),
        )
        if continuous_mean_embedding_2d is not None:
            joint_surface_raw = continuous_mean_embedding_2d.get("z", None)

        # Preserve the package's legacy mean_embedding_all/operator-check plots only
        # for fully continuous rewards. For mixed rewards, the new continuous map
        # above replaces the skipped old contour while density displays are still
        # built from atom mixtures.
        if len(discrete_reward_dims) == 0:
            _set_safe_matplotlib_fonts()
            try:
                cache_raw, _ = mean_embedding_all(
                    beta_full=beta,
                    Z_grid=Zg_norm_optimized.to(beta.device),
                    config=plot_cfg,
                    do_joint_dims=joint_dims,
                    n1=120,
                    n2=120,
                    outdir=str(plots_dir),
                )

                for joint_csv in [plots_dir / "mu2D_hat_dims01.csv", Path("./mu/mu2D_hat_dims01.csv")]:
                    if joint_csv.exists():
                        try:
                            joint_surface_raw = np.loadtxt(joint_csv, delimiter=",")
                            break
                        except Exception:
                            pass

                try:
                    plot_operator_check_2d(
                        cache_raw,
                        r_obs=r_test,
                        gamma=plot_cfg['gamma_val'],
                        config=plot_cfg,
                        outdir=str(plots_dir),
                    )
                except Exception as e:
                    print(f'Warning: normalized-scale operator check plot failed for {policy_name}: {e}')
            except Exception as e:
                print(f"Warning: legacy mean_embedding_all failed for {policy_name}: {e}")
        else:
            print(
                f"Continuous 2D mean-embedding contour was produced for {policy_name}; "
                f"legacy mean_embedding_all/operator-check skipped because discrete reward dims={discrete_reward_dims}."
            )

        if r_test_raw.shape[1] >= 2:
            fig = plt.figure()
            pts = min(plot_n_eff, r_test_raw.shape[0])
            plt.scatter(_np(r_test_raw[:pts, 0]), _np(r_test_raw[:pts, 1]), s=5, alpha=0.4)
            plt.xlabel(reward_cols[0])
            plt.ylabel(reward_cols[1])
            plt.title(f'{policy_name}: test rewards (raw scale)')
            fig.savefig(plots_dir / f'{policy_name}_test_rewards_raw_scatter.png', bbox_inches='tight')
            plt.close(fig)

        # Explicit overlay-ready arrays.
        raw_plot_payload = {
            'marginal_bandwidths': _to_cpu_serializable(torch.as_tensor(marginal_bandwidths, dtype=torch.float32)),
            'Zg_norm_optimized': _to_cpu_serializable(Zg_norm_optimized),
            'Zg_raw_est': _to_cpu_serializable(Zg_raw_est),
            'Zg_raw': _to_cpu_serializable(Zg_raw),
            'Zg_norm_for_density': _to_cpu_serializable(Zg_norm_for_density),
            'beta': _to_cpu_serializable(beta),
            'density_weights': _to_cpu_serializable(density_weights),
            'density_weight_projection': _to_cpu_serializable(marginal_payload.get('projection', {})),
            'marginal_payload': _to_cpu_serializable(marginal_payload),
            'reward_support_info': reward_support_info,
            'recovered_support_info': recovered_support_info,
            'cache_raw': _to_cpu_serializable(cache_raw),
            'joint_surface_raw': _to_cpu_serializable(joint_surface_raw),
            'continuous_mean_embedding_2d': _to_cpu_serializable(locals().get('continuous_mean_embedding_2d', None)),
        }

        overlay_data = _build_overlay_data_explicit(
            marginal_payload_raw=raw_plot_payload['marginal_payload'],
            Zg_raw=raw_plot_payload['Zg_raw'],
            reward_cols=reward_cols,
            beta_full=raw_plot_payload['beta'],
            joint_surface_raw=raw_plot_payload['joint_surface_raw'],  # fallback only
        )

        torch.save(
            {
                'policy_name': policy_name,
                'reward_cols': reward_cols,
                'plot_cfg': _to_cpu_serializable(plot_cfg),
                **raw_plot_payload,
                'overlay_data': overlay_data,
            },
            out_dir / 'plot_payload.pt',
        )

    return metrics


# ==================================== #
#                Main                  #
# ==================================== #
def main() -> None:
    args = _parse_args()
    _set_seeds(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    slurm_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
    torch.set_num_threads(max(1, slurm_cpus))

    cfg_index = 0 if args.cfg_index is None else int(args.cfg_index)
    cfg = _resolve_cfg(args)

    run_test = bool(int(args.run_test))
    do_plots = bool(int(args.do_plots))
    device = _resolve_device(args.device)

    base = Path(args.data_base)
    ckpt_dir = Path(args.policy_ckpt_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f'\n=== KE-DRL evaluation on {device} ===')
    print('Chosen cfg:', cfg)

    full_cols_path = Path(args.full_state_cols_path) if args.full_state_cols_path else None
    if full_cols_path is None:
        for cand in [ckpt_dir / 'full_state_cols.json', base / 'full_state_cols.json']:
            if cand.exists():
                full_cols_path = cand
                break
    external_state_cols = load_full_state_cols(full_cols_path) if full_cols_path else LEGACY_FULL_STATE_COLS

    train_blob = normalize_blob_payload(torch.load(base / args.train_blob, map_location='cpu'))
    val_blob = normalize_blob_payload(torch.load(base / args.val_blob, map_location='cpu'))
    test_blob = normalize_blob_payload(torch.load(base / args.test_blob, map_location='cpu'))

    if args.state_cols is not None:
        state_cols = _parse_csv_list(args.state_cols)
    elif 'state_cols' in train_blob:
        state_cols = list(train_blob['state_cols'])
    else:
        s0_full = get_2d_tensor_from_blob(train_blob, 's0')
        reduced_idx = resolve_reduced_indices(
            full_state_cols=external_state_cols,
            reduced_state_cols=[c.strip() for c in args.reduced_state_cols.split(',') if c.strip()],
            reduced_idx=[int(x) for x in args.reduced_idx.split(',')] if args.reduced_idx else None,
            full_state_dim=s0_full.shape[1],
        )
        state_cols = [external_state_cols[i] for i in reduced_idx]

    if args.action_cols is not None:
        action_cols = _parse_csv_list(args.action_cols)
    elif 'action_cols' in train_blob:
        action_cols = list(train_blob['action_cols'])
    else:
        action_cols = default_names_for_blob_section(train_blob, 'a0', 'action_cols', external_state_cols)

    reward_tensor_key = 'r0' if 'r0' in train_blob else 'r'
    if args.reward_cols is not None:
        reward_cols = _parse_csv_list(args.reward_cols)
    elif 'reward_cols' in train_blob:
        reward_cols = list(train_blob['reward_cols'])
    else:
        reward_cols = default_names_for_blob_section(train_blob, reward_tensor_key, 'reward_cols', external_state_cols)

    if bool(int(args.require_exactly_two_rewards)) and len(reward_cols) != 2:
        raise ValueError(f'This script expects exactly 2 reward columns, got {reward_cols}')

    print('\nSelected metadata:')
    print('  state_cols :', state_cols)
    print('  action_cols:', action_cols)
    print('  reward_cols:', reward_cols)
    print('  discrete_reward_cols:', _parse_csv_list(args.discrete_reward_cols) or [])

    s0_tr_raw, s1_tr_raw = select_named_pair(train_blob, state_cols, 's0', 's1', 'state_cols', external_state_cols)
    a0_tr_raw, a1_tr_raw = select_named_pair(train_blob, action_cols, 'a0', 'a1', 'action_cols', external_state_cols)
    r_tr_raw = select_named_columns(train_blob, reward_cols, reward_tensor_key, 'reward_cols', external_state_cols)

    reward_tensor_key_val = 'r0' if 'r0' in val_blob else 'r'
    reward_tensor_key_test = 'r0' if 'r0' in test_blob else 'r'

    s0_val_raw, _ = select_named_pair(val_blob, state_cols, 's0', 's1', 'state_cols', external_state_cols)
    a0_val_raw, _ = select_named_pair(val_blob, action_cols, 'a0', 'a1', 'action_cols', external_state_cols)
    r_val_raw = select_named_columns(val_blob, reward_cols, reward_tensor_key_val, 'reward_cols', external_state_cols)

    s0_test_raw, _ = select_named_pair(test_blob, state_cols, 's0', 's1', 'state_cols', external_state_cols)
    a0_test_raw, _ = select_named_pair(test_blob, action_cols, 'a0', 'a1', 'action_cols', external_state_cols)
    r_test_raw = select_named_columns(test_blob, reward_cols, reward_tensor_key_test, 'reward_cols', external_state_cols)

    idx_tr = _subsample_idx(s0_tr_raw.shape[0], args.max_train, args.seed)
    idx_val = _subsample_idx(s0_val_raw.shape[0], args.max_val, args.seed + 1)
    idx_test = _subsample_idx(s0_test_raw.shape[0], args.max_test, args.seed + 2)

    s0_tr_raw = s0_tr_raw[idx_tr]
    s1_tr_raw = s1_tr_raw[idx_tr] if s1_tr_raw is not None else None
    a0_tr_raw = a0_tr_raw[idx_tr]
    a1_tr_raw = a1_tr_raw[idx_tr] if a1_tr_raw is not None else None
    r_tr_raw = r_tr_raw[idx_tr]

    s0_val_raw = s0_val_raw[idx_val]
    a0_val_raw = a0_val_raw[idx_val]
    r_val_raw = r_val_raw[idx_val]

    s0_test_raw = s0_test_raw[idx_test]
    a0_test_raw = a0_test_raw[idx_test]
    r_test_raw = r_test_raw[idx_test]

    if bool(int(getattr(args, "export_subsets", 1))):
        subsets_path = (
            Path(args.subsets_file)
            if args.subsets_file is not None
            else (out_root / f"selected_subsets_cfg{cfg_index:03d}.pt")
        )
        save_reproducibility_subsets(
            path=subsets_path,
            cfg_index=int(cfg_index),
            seed=int(args.seed),
            train_blob_path=str(base / args.train_blob),
            val_blob_path=str(base / args.val_blob),
            test_blob_path=str(base / args.test_blob),
            state_cols=state_cols,
            action_cols=action_cols,
            reward_cols=reward_cols,
            idx_tr=idx_tr,
            idx_val=idx_val,
            idx_test=idx_test,
            s0_tr_raw=s0_tr_raw,
            s1_tr_raw=s1_tr_raw,
            a0_tr_raw=a0_tr_raw,
            a1_tr_raw=a1_tr_raw,
            r_tr_raw=r_tr_raw,
            s0_val_raw=s0_val_raw,
            a0_val_raw=a0_val_raw,
            r_val_raw=r_val_raw,
            s0_test_raw=s0_test_raw,
            a0_test_raw=a0_test_raw,
            r_test_raw=r_test_raw,
        )
        print(f"Saved reproducibility subsets: {subsets_path}")

    raw_state_cols = list(state_cols)
    s0_tr_raw_unencoded = s0_tr_raw.clone()
    s1_tr_raw_unencoded = s1_tr_raw.clone() if s1_tr_raw is not None else None
    s0_val_raw_unencoded = s0_val_raw.clone()
    s0_test_raw_unencoded = s0_test_raw.clone()

    encoder_path = Path(args.state_encoder_path) if args.state_encoder_path else (ckpt_dir / "state_encoder.json")
    if encoder_path.exists():
        state_encoder = state_encoder_from_metadata(json.loads(encoder_path.read_text()))
        encoder_source = str(encoder_path)
        if list(state_encoder.raw_state_names) != raw_state_cols:
            raise ValueError(
                "State encoder raw columns do not match evaluator state columns. "
                f"encoder={state_encoder.raw_state_names}, evaluator={raw_state_cols}"
            )
    else:
        state_encoder = fit_state_encoder(
            raw_state_names=raw_state_cols,
            train_states=_np(s0_tr_raw),
            categorical_state_cols=args.categorical_state_cols,
            one_hot=bool(int(args.one_hot_categoricals)),
            max_auto_cardinality=int(args.max_auto_categorical_cardinality),
        )
        encoder_source = "fit_from_evaluation_train_subset"

    state_encoder_meta = state_encoder.to_metadata()
    state_encoding_diagnostics = {
        "encoder_source": encoder_source,
        "train": state_encoder.diagnostics(_np(s0_tr_raw_unencoded)),
        "val": state_encoder.diagnostics(_np(s0_val_raw_unencoded)),
        "test": state_encoder.diagnostics(_np(s0_test_raw_unencoded)),
    }

    def _encode_state_tensor(x: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(
            state_encoder.transform(_np(x)),
            dtype=torch.float32,
        )

    s0_tr_raw = _encode_state_tensor(s0_tr_raw_unencoded)
    s1_tr_raw = _encode_state_tensor(s1_tr_raw_unencoded) if s1_tr_raw_unencoded is not None else None
    s0_val_raw = _encode_state_tensor(s0_val_raw_unencoded)
    s0_test_raw = _encode_state_tensor(s0_test_raw_unencoded)
    state_cols = list(state_encoder.encoded_state_names)

    print("\nState encoding:")
    print("  raw_state_cols        :", raw_state_cols)
    print("  categorical_state_cols:", state_encoder_meta["categorical_state_names"])
    print("  encoder_source        :", encoder_source)
    print("  encoded_state_dim     :", len(state_cols))

    print('\nShapes after named selection + subsample:')
    print('  train:', tuple(s0_tr_raw.shape), tuple(a0_tr_raw.shape), tuple(r_tr_raw.shape))
    print('  val  :', tuple(s0_val_raw.shape), tuple(a0_val_raw.shape), tuple(r_val_raw.shape))
    print('  test :', tuple(s0_test_raw.shape), tuple(a0_test_raw.shape), tuple(r_test_raw.shape))

    data_summary = {
        'train_raw_state_summary': summarize_tensor_by_columns(s0_tr_raw_unencoded, raw_state_cols, 'TRAIN RAW STATE SUMMARY (selected columns)'),
        'train_encoded_state_summary': summarize_tensor_by_columns(s0_tr_raw, state_cols, 'TRAIN ENCODED STATE SUMMARY (model basis)'),
        'train_action_summary': summarize_tensor_by_columns(a0_tr_raw, action_cols, 'TRAIN ACTION SUMMARY (raw scale)'),
        'train_reward_summary': summarize_tensor_by_columns(r_tr_raw, reward_cols, 'TRAIN REWARD SUMMARY (raw scale)'),
        'val_action_summary': summarize_tensor_by_columns(a0_val_raw, action_cols, 'VAL ACTION SUMMARY (raw scale)'),
        'test_action_summary': summarize_tensor_by_columns(a0_test_raw, action_cols, 'TEST ACTION SUMMARY (raw scale)'),
        'state_encoder': state_encoder_meta,
        'state_encoding_diagnostics': state_encoding_diagnostics,
    }

    if s1_tr_raw is None or a1_tr_raw is None:
        raise ValueError('This evaluation script requires s1 and a1 in the training blob.')

    s0_tr_raw = s0_tr_raw.to(device)
    s1_tr_raw = s1_tr_raw.to(device)
    a0_tr_raw = a0_tr_raw.to(device)
    a1_tr_raw = a1_tr_raw.to(device)
    r_tr_raw = r_tr_raw.to(device)

    s0_val_raw = s0_val_raw.to(device)
    a0_val_raw = a0_val_raw.to(device)
    r_val_raw = r_val_raw.to(device)

    s0_test_raw = s0_test_raw.to(device)
    a0_test_raw = a0_test_raw.to(device)
    r_test_raw = r_test_raw.to(device)

    s_mu = s0_tr_raw.mean(0)
    s_sd = s0_tr_raw.std(0, unbiased=False).clamp_min(1e-6)
    a_mu = a0_tr_raw.mean(0)
    a_sd = a0_tr_raw.std(0, unbiased=False).clamp_min(1e-6)
    r_mu = r_tr_raw.mean(0)
    r_sd = r_tr_raw.std(0, unbiased=False).clamp_min(1e-6)

    s0_tr = _zscore(s0_tr_raw, s_mu, s_sd)
    s1_tr = _zscore(s1_tr_raw, s_mu, s_sd)
    a0_tr = _zscore(a0_tr_raw, a_mu, a_sd)
    a1_tr = _zscore(a1_tr_raw, a_mu, a_sd)
    r_tr = _zscore(r_tr_raw, r_mu, r_sd)

    s0_val = _zscore(s0_val_raw, s_mu, s_sd)
    a0_val = _zscore(a0_val_raw, a_mu, a_sd)
    r_val = _zscore(r_val_raw, r_mu, r_sd)

    s0_test = _zscore(s0_test_raw, s_mu, s_sd)
    a0_test = _zscore(a0_test_raw, a_mu, a_sd)
    r_test = _zscore(r_test_raw, r_mu, r_sd)

    sa_tr = torch.cat([s0_tr, a0_tr], dim=1)
    mu_sa = sa_tr.mean(0)
    sd_sa = sa_tr.std(0, unbiased=False).clamp_min(1e-6)
    z = ((sa_tr - mu_sa) / sd_sa).pow(2).sum(1)
    idx_star = torch.argmin(z).view(1)
    s_star = s0_tr[idx_star].clone()
    a_star = a0_tr[idx_star].clone()

    if args.policy_objective == 'both':
        policies_to_run = list(reward_cols)
    else:
        if args.policy_objective not in reward_cols:
            raise ValueError(f'--policy-objective={args.policy_objective} not in reward_cols={reward_cols}')
        policies_to_run = [args.policy_objective]

    try:
        if len(reward_cols) >= 2:
            print_raw_policy_difference(ckpt_dir, reward_cols[0], reward_cols[1], device)
    except Exception as e:
        print(f'Could not print raw checkpoint difference: {e}')

    compare_results = {}
    for policy_name in policies_to_run:
        tag = f"{_sanitize_reward_tag(policy_name)}_cfg{cfg_index:03d}_nu{cfg['nu_Z']}_ell{cfg['ell_Z']}_lam{cfg['lambda_reg']:g}_sigmaZ{cfg['sigma_Z']}"
        policy_out_dir = out_root / tag
        result = run_one_policy(
            policy_name=policy_name,
            out_dir=policy_out_dir,
            do_plots=do_plots,
            run_test=run_test,
            cfg=cfg,
            args=args,
            m_Z=int(args.num_grid_points),
            device=device,
            s0_tr_raw=s0_tr_raw,
            s1_tr_raw=s1_tr_raw,
            a0_tr_raw=a0_tr_raw,
            a1_tr_raw=a1_tr_raw,
            r_tr_raw=r_tr_raw,
            s0_val_raw=s0_val_raw,
            a0_val_raw=a0_val_raw,
            r_val_raw=r_val_raw,
            s0_test_raw=s0_test_raw,
            a0_test_raw=a0_test_raw,
            r_test_raw=r_test_raw,
            s0_tr=s0_tr,
            s1_tr=s1_tr,
            a0_tr=a0_tr,
            a1_tr=a1_tr,
            r_tr=r_tr,
            s0_val=s0_val,
            a0_val=a0_val,
            r_val=r_val,
            s0_test=s0_test,
            a0_test=a0_test,
            r_test=r_test,
            s_star=s_star,
            a_star=a_star,
            s_mu=s_mu,
            s_sd=s_sd,
            a_mu=a_mu,
            a_sd=a_sd,
            r_mu=r_mu,
            r_sd=r_sd,
            state_cols=state_cols,
            action_cols=action_cols,
            reward_cols=reward_cols,
            ckpt_dir=ckpt_dir,
            plot_n=int(args.plot_n),
        )
        compare_results[policy_name] = result

    summary = {
        'cfg_index': int(cfg_index),
        'cfg': cfg,
        'seed': int(args.seed),
        'raw_state_cols': raw_state_cols,
        'state_cols': state_cols,
        'action_cols': action_cols,
        'reward_cols': reward_cols,
        'discrete_reward_cols': _parse_csv_list(args.discrete_reward_cols) or [],
        'state_encoder': state_encoder_meta,
        'state_encoding_diagnostics': state_encoding_diagnostics,
        'normalization': {
            's_mu': _np(s_mu).tolist(),
            's_sd': _np(s_sd).tolist(),
            'a_mu': _np(a_mu).tolist(),
            'a_sd': _np(a_sd).tolist(),
            'r_mu': _np(r_mu).tolist(),
            'r_sd': _np(r_sd).tolist(),
        },
        'data_summary': data_summary,
        'results': compare_results,
    }

    if len(compare_results) == 2:
        A, B = reward_cols[0], reward_cols[1]
        if A in compare_results and B in compare_results:
            pA_pt, pA_json = resolve_gaussian_policy_paths(ckpt_dir, A)
            pB_pt, pB_json = resolve_gaussian_policy_paths(ckpt_dir, B)
            netA, _, _ = load_gaussian_policy(pA_pt, pA_json, device)
            netB, _, _ = load_gaussian_policy(pB_pt, pB_json, device)

            aA_raw = gaussian_policy_stats_raw(netA, s0_test_raw)['greedy_raw']
            aB_raw = gaussian_policy_stats_raw(netB, s0_test_raw)['greedy_raw']
            aA_norm = _zscore(aA_raw, a_mu, a_sd)
            aB_norm = _zscore(aB_raw, a_mu, a_sd)

            delta_norm = _np(aA_norm - aB_norm)
            delta_raw = _np(aA_raw - aB_raw)

            per_dim_abs_delta_norm = np.mean(np.abs(delta_norm), axis=0)
            per_dim_abs_delta_raw = np.mean(np.abs(delta_raw), axis=0)
            per_dim_corr = [_safe_corr(_np(aA_raw[:, j]), _np(aB_raw[:, j])) for j in range(aA_raw.shape[1])]

            print('\n' + '=' * 80)
            print('FINAL TWO-POLICY COMPARISON')
            print('=' * 80)
            print(f'Compared policies: π_{A} vs π_{B}')
            print('Action columns:', action_cols)
            print('State columns :', state_cols)
            print('\nNormalized-action comparison:')
            print(f'  Mean ||a_{A} - a_{B}||_2    :', float(np.linalg.norm(delta_norm, axis=1).mean()))
            print(f'  Median ||a_{A} - a_{B}||_2  :', float(np.median(np.linalg.norm(delta_norm, axis=1))))
            print(f'  Disagree-rate max|Δ|>1e-3   :', float((np.max(np.abs(delta_norm), axis=1) > 1e-3).mean()))
            print('  Per-dim mean |Δ| normalized :', _array_str(per_dim_abs_delta_norm))
            print('  Per-dim corr (raw actions)  :', _array_str(np.asarray(per_dim_corr)))

            print('\nRaw-scale action comparison:')
            print(f'  Test data action mean       : {_array_str(_np(a0_test_raw.mean(0)))}')
            print(f'  {A} policy action mean      : {_array_str(_np(aA_raw.mean(0)))}')
            print(f'  {B} policy action mean      : {_array_str(_np(aB_raw.mean(0)))}')
            print('  Per-dim mean |Δ| raw        :', _array_str(per_dim_abs_delta_raw))
            print('\nKE-DRL risks:')
            print(f"  {A}: val={compare_results[A]['val_risk']:.6f} test={compare_results[A]['test_risk'] if compare_results[A]['test_risk'] is not None else 'NA'}")
            print(f"  {B}: val={compare_results[B]['val_risk']:.6f} test={compare_results[B]['test_risk'] if compare_results[B]['test_risk'] is not None else 'NA'}")
            print('\nValue-model cross-metrics:')
            print(f"  {A} policy metrics:", compare_results[A].get('value_metrics', {}))
            print(f"  {B} policy metrics:", compare_results[B].get('value_metrics', {}))
            print('=' * 80)

            summary['two_policy_comparison'] = {
                'policy_A': A,
                'policy_B': B,
                'mean_l2_normalized': float(np.linalg.norm(delta_norm, axis=1).mean()),
                'median_l2_normalized': float(np.median(np.linalg.norm(delta_norm, axis=1))),
                'disagree_rate_maxabs_gt_1e3': float((np.max(np.abs(delta_norm), axis=1) > 1e-3).mean()),
                'per_dim_mean_abs_delta_normalized': per_dim_abs_delta_norm.tolist(),
                'per_dim_mean_abs_delta_raw': per_dim_abs_delta_raw.tolist(),
                'per_dim_corr_raw': per_dim_corr,
                'policy_A_action_mean_raw': _np(aA_raw.mean(0)).tolist(),
                'policy_B_action_mean_raw': _np(aB_raw.mean(0)).tolist(),
                'test_data_action_mean_raw': _np(a0_test_raw.mean(0)).tolist(),
                'val_risk_A': compare_results[A]['val_risk'],
                'val_risk_B': compare_results[B]['val_risk'],
                'test_risk_A': compare_results[A]['test_risk'],
                'test_risk_B': compare_results[B]['test_risk'],
            }

            if do_plots:
                payload_A = torch.load(
                    out_root / f"{_sanitize_reward_tag(A)}_cfg{cfg_index:03d}_nu{cfg['nu_Z']}_ell{cfg['ell_Z']}_lam{cfg['lambda_reg']:g}_sigmaZ{cfg['sigma_Z']}" / "plot_payload.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                payload_B = torch.load(
                    out_root / f"{_sanitize_reward_tag(B)}_cfg{cfg_index:03d}_nu{cfg['nu_Z']}_ell{cfg['ell_Z']}_lam{cfg['lambda_reg']:g}_sigmaZ{cfg['sigma_Z']}" / "plot_payload.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                overlay_dir = out_root / "two_policy_reward_overlay_plots"
                overlay_paths = _plot_two_policy_reward_overlays(
                    payload_A=payload_A,
                    payload_B=payload_B,
                    reward_cols=reward_cols,
                    label_A=A,
                    label_B=B,
                    out_dir=overlay_dir,
                )
                summary.setdefault("two_policy_comparison", {})["overlay_plot_paths"] = overlay_paths

    save_json(out_root / f"summary_cfg{cfg_index:03d}.json", summary)
    print(f"\nSaved summary: {out_root / f'summary_cfg{cfg_index:03d}.json'}")


if __name__ == '__main__':
    main()
