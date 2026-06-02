from __future__ import annotations

import math
from typing import Iterable

import torch

from .matern_kernel import matern_kernel


def _to_2d_tensor(x, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    out = torch.as_tensor(x).to(dtype=dtype, device=device)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    if out.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape {tuple(out.shape)}.")
    return out.contiguous()


def _select_truth_points(Z: torch.Tensor, max_points: int | None) -> torch.Tensor:
    if max_points is None or int(max_points) <= 0 or Z.shape[0] <= int(max_points):
        return Z.contiguous()
    n = int(max_points)
    idx = torch.linspace(0, Z.shape[0] - 1, n, device=Z.device).round().long()
    return Z.index_select(0, idx).contiguous()


def empirical_embedding_inner_product(
    Z_left: torch.Tensor,
    Z_right: torch.Tensor,
    *,
    nu: float,
    length_scale: float,
    sigma: float,
    batch_size: int = 1000,
) -> torch.Tensor:
    """Empirical RKHS inner product between two Monte Carlo return embeddings."""
    if Z_left.ndim != 2 or Z_right.ndim != 2 or Z_left.shape[1] != Z_right.shape[1]:
        raise ValueError("Z_left and Z_right must be 2D tensors with the same feature dimension.")
    if Z_left.shape[0] == 0 or Z_right.shape[0] == 0:
        raise ValueError("Monte Carlo truth samples must be nonempty.")

    total = torch.zeros((), dtype=Z_left.dtype, device=Z_left.device)
    batch = max(1, int(batch_size))
    for left_start in range(0, Z_left.shape[0], batch):
        left = Z_left[left_start : left_start + batch]
        for right_start in range(0, Z_right.shape[0], batch):
            right = Z_right[right_start : right_start + batch]
            total = total + matern_kernel(left, right, nu=nu, length_scale=length_scale, sigma=sigma).sum()
    return total / float(Z_left.shape[0] * Z_right.shape[0])


def empirical_embedding_mmd2_components(
    beta: torch.Tensor,
    Z_grid: torch.Tensor,
    Z_true: torch.Tensor,
    *,
    nu: float,
    length_scale: float,
    sigma: float,
    K_grid: torch.Tensor | None = None,
    batch_size: int = 1000,
    max_truth_points: int | None = None,
) -> dict[str, torch.Tensor | int]:
    """Squared RKHS distance between a grid-weighted embedding and MC truth."""
    beta = torch.as_tensor(beta).reshape(-1)
    Z_grid = torch.as_tensor(Z_grid).to(dtype=beta.dtype, device=beta.device)
    if Z_grid.ndim != 2:
        raise ValueError(f"Z_grid must be 2D, got shape {tuple(Z_grid.shape)}.")
    if beta.numel() != Z_grid.shape[0]:
        raise ValueError(f"beta has length {beta.numel()} but Z_grid has {Z_grid.shape[0]} rows.")
    Z_true = _to_2d_tensor(Z_true, dtype=beta.dtype, device=beta.device)
    Z_true = _select_truth_points(Z_true, max_truth_points)

    if K_grid is None:
        K_grid = matern_kernel(Z_grid, Z_grid, nu=nu, length_scale=length_scale, sigma=sigma)
    else:
        K_grid = torch.as_tensor(K_grid).to(dtype=beta.dtype, device=beta.device)
    if K_grid.shape != (Z_grid.shape[0], Z_grid.shape[0]):
        raise ValueError("K_grid must have shape (len(Z_grid), len(Z_grid)).")

    quad = beta @ (K_grid @ beta)
    cross_sum = torch.zeros((), dtype=beta.dtype, device=beta.device)
    batch = max(1, int(batch_size))
    for start in range(0, Z_true.shape[0], batch):
        chunk = Z_true[start : start + batch]
        cross_sum = cross_sum + (matern_kernel(chunk, Z_grid, nu=nu, length_scale=length_scale, sigma=sigma) @ beta).sum()
    cross = cross_sum / float(Z_true.shape[0])
    truth_self = empirical_embedding_inner_product(
        Z_true,
        Z_true,
        nu=nu,
        length_scale=length_scale,
        sigma=sigma,
        batch_size=batch,
    )
    mmd2_raw = quad - 2.0 * cross + truth_self
    mmd2 = torch.clamp(mmd2_raw, min=0.0)
    return {
        "mmd2": mmd2,
        "mmd2_raw": mmd2_raw,
        "quad": quad,
        "cross": cross,
        "truth_self": truth_self,
        "truth_points_used": int(Z_true.shape[0]),
    }


def empirical_embedding_mmd2(
    beta: torch.Tensor,
    Z_grid: torch.Tensor,
    Z_true: torch.Tensor,
    *,
    nu: float,
    length_scale: float,
    sigma: float,
    batch_size: int = 1000,
    max_truth_points: int | None = None,
) -> torch.Tensor:
    """Return only the squared RKHS distance to the MC truth embedding."""
    return empirical_embedding_mmd2_components(
        beta,
        Z_grid,
        Z_true,
        nu=nu,
        length_scale=length_scale,
        sigma=sigma,
        batch_size=batch_size,
        max_truth_points=max_truth_points,
    )["mmd2"]


def _float(x: torch.Tensor | float) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu())
    return float(x)


def _maybe_float(x: torch.Tensor) -> float:
    val = _float(x)
    return val if math.isfinite(val) else float("nan")


def embedding_explained_signal_from_true_samples(
    betas: Iterable[torch.Tensor],
    Z_grid: torch.Tensor,
    Z_true_list: Iterable[torch.Tensor],
    *,
    nu: float,
    length_scale: float,
    sigma: float,
    batch_size: int = 1000,
    max_truth_points: int | None = 512,
    eps: float = 1e-12,
) -> dict[str, object]:
    """Signal-normalized embedding fit diagnostic for simulation settings.

    For each target this computes
    ``1 - ||mu_hat_i - mu_i^MC||_H^2 / (||mu_i^MC||_H^2 + eps)``.
    Unlike the legacy RKHS R^2 helper below, this does not center the MC truth
    embeddings and does not build a target-by-target truth Gram matrix.
    """
    beta_list = list(betas)
    raw_truth_list = list(Z_true_list)
    if len(beta_list) != len(raw_truth_list):
        raise ValueError(f"Expected the same number of betas and truth samples, got {len(beta_list)} and {len(raw_truth_list)}.")
    if not beta_list:
        raise ValueError("At least one evaluation embedding is required.")

    first_beta = torch.as_tensor(beta_list[0]).reshape(-1)
    device = first_beta.device
    dtype = first_beta.dtype
    Z_grid_t = _to_2d_tensor(Z_grid, dtype=dtype, device=device)
    K_grid = matern_kernel(Z_grid_t, Z_grid_t, nu=nu, length_scale=length_scale, sigma=sigma)

    truth_sets = [
        _select_truth_points(_to_2d_tensor(Z, dtype=dtype, device=device), max_truth_points)
        for Z in raw_truth_list
    ]
    beta_tensors = [torch.as_tensor(beta).to(dtype=dtype, device=device).reshape(-1) for beta in beta_list]

    components = [
        empirical_embedding_mmd2_components(
            beta,
            Z_grid_t,
            truth,
            nu=nu,
            length_scale=length_scale,
            sigma=sigma,
            K_grid=K_grid,
            batch_size=batch_size,
            max_truth_points=None,
        )
        for beta, truth in zip(beta_tensors, truth_sets)
    ]

    errors = torch.stack([comp["mmd2"] for comp in components])  # type: ignore[arg-type]
    signals = torch.stack([torch.clamp(comp["truth_self"], min=0.0) for comp in components])  # type: ignore[arg-type]
    hat_norms = torch.stack([torch.clamp(comp["quad"], min=0.0) for comp in components])  # type: ignore[arg-type]

    relative_error = errors / (signals + float(eps))
    explained_signal = 1.0 - relative_error
    error_total = errors.sum()
    signal_total = signals.sum()
    relative_global = error_total / (signal_total + float(eps))
    explained_global = 1.0 - relative_global
    truth_points = [int(comp["truth_points_used"]) for comp in components]

    return {
        "embedding_error_mmd2_total": _float(error_total),
        "embedding_truth_signal_total": _float(signal_total),
        "relative_embedding_error_global": _maybe_float(relative_global),
        "explained_embedding_signal_global": _maybe_float(explained_global),
        "embedding_error_mmd2_mean": _float(errors.mean()),
        "embedding_truth_signal_mean": _float(signals.mean()),
        "relative_embedding_error_mean": _float(relative_error.mean()),
        "explained_embedding_signal_mean": _float(explained_signal.mean()),
        "embedding_error_mmd2": [_float(x) for x in errors],
        "embedding_truth_signal": [_float(x) for x in signals],
        "relative_embedding_error": [_float(x) for x in relative_error],
        "explained_embedding_signal": [_float(x) for x in explained_signal],
        "embedding_hat_norm2": [_float(x) for x in hat_norms],
        "embedding_truth_points_used": truth_points,
        "embedding_truth_points_used_min": int(min(truth_points)),
        "embedding_truth_points_used_max": int(max(truth_points)),
        "embedding_truth_points_used_mean": float(sum(truth_points) / len(truth_points)),
    }


def normalized_bellman_error(
    betas: Iterable[torch.Tensor],
    Z_grid: torch.Tensor,
    bellman_residuals: Iterable[torch.Tensor | float],
    *,
    nu: float,
    length_scale: float,
    sigma: float,
    eps: float = 1e-12,
) -> dict[str, object]:
    """Normalize held-out Bellman residuals by the estimated embedding signal."""
    beta_list = list(betas)
    residual_list = list(bellman_residuals)
    if len(beta_list) != len(residual_list):
        raise ValueError(f"Expected the same number of betas and Bellman residuals, got {len(beta_list)} and {len(residual_list)}.")
    if not beta_list:
        raise ValueError("At least one evaluation embedding is required.")

    first_beta = torch.as_tensor(beta_list[0]).reshape(-1)
    device = first_beta.device
    dtype = first_beta.dtype
    Z_grid_t = _to_2d_tensor(Z_grid, dtype=dtype, device=device)
    K_grid = matern_kernel(Z_grid_t, Z_grid_t, nu=nu, length_scale=length_scale, sigma=sigma)

    beta_tensors = [torch.as_tensor(beta).to(dtype=dtype, device=device).reshape(-1) for beta in beta_list]
    hat_norms = []
    for beta in beta_tensors:
        if beta.numel() != Z_grid_t.shape[0]:
            raise ValueError(f"beta has length {beta.numel()} but Z_grid has {Z_grid_t.shape[0]} rows.")
        hat_norms.append(torch.clamp(beta @ (K_grid @ beta), min=0.0))
    hat_norm2 = torch.stack(hat_norms)
    residuals = torch.stack([
        torch.as_tensor(residual, dtype=dtype, device=device).reshape(())
        for residual in residual_list
    ])
    residuals = torch.clamp(residuals, min=0.0)

    nbe = residuals / (hat_norm2 + float(eps))
    fit = 1.0 - nbe
    residual_total = residuals.sum()
    norm_total = hat_norm2.sum()
    nbe_global = residual_total / (norm_total + float(eps))
    fit_global = 1.0 - nbe_global

    return {
        "normalized_bellman_error_global": _maybe_float(nbe_global),
        "bellman_fit_global": _maybe_float(fit_global),
        "normalized_bellman_error_mean": _float(nbe.mean()),
        "bellman_fit_mean": _float(fit.mean()),
        "bellman_residual_total": _float(residual_total),
        "embedding_hat_norm2_total": _float(norm_total),
        "normalized_bellman_error": [_float(x) for x in nbe],
        "bellman_fit": [_float(x) for x in fit],
        "bellman_residual": [_float(x) for x in residuals],
        "embedding_hat_norm2": [_float(x) for x in hat_norm2],
    }


def embedding_r2_from_true_samples(
    betas: Iterable[torch.Tensor],
    Z_grid: torch.Tensor,
    Z_true_list: Iterable[torch.Tensor],
    *,
    nu: float,
    length_scale: float,
    sigma: float,
    batch_size: int = 1000,
    max_truth_points: int | None = 512,
    eps: float = 1e-12,
) -> dict[str, object]:
    """RKHS analogue of R^2 using only Monte Carlo true return samples.

    The numerator is the sum of empirical MMD^2 errors between estimated
    embeddings and MC truth embeddings. The denominator is the RKHS variance
    of the MC truth embeddings around their empirical mean embedding.
    """
    beta_list = list(betas)
    raw_truth_list = list(Z_true_list)
    if len(beta_list) != len(raw_truth_list):
        raise ValueError(f"Expected the same number of betas and truth samples, got {len(beta_list)} and {len(raw_truth_list)}.")
    if not beta_list:
        raise ValueError("At least one evaluation embedding is required.")

    first_beta = torch.as_tensor(beta_list[0]).reshape(-1)
    device = first_beta.device
    dtype = first_beta.dtype
    Z_grid_t = _to_2d_tensor(Z_grid, dtype=dtype, device=device)
    K_grid = matern_kernel(Z_grid_t, Z_grid_t, nu=nu, length_scale=length_scale, sigma=sigma)

    truth_sets = [
        _select_truth_points(_to_2d_tensor(Z, dtype=dtype, device=device), max_truth_points)
        for Z in raw_truth_list
    ]
    beta_tensors = [torch.as_tensor(beta).to(dtype=dtype, device=device).reshape(-1) for beta in beta_list]

    components = [
        empirical_embedding_mmd2_components(
            beta,
            Z_grid_t,
            truth,
            nu=nu,
            length_scale=length_scale,
            sigma=sigma,
            K_grid=K_grid,
            batch_size=batch_size,
            max_truth_points=None,
        )
        for beta, truth in zip(beta_tensors, truth_sets)
    ]

    numerators = torch.stack([comp["mmd2"] for comp in components])  # type: ignore[arg-type]
    numerators_raw = torch.stack([comp["mmd2_raw"] for comp in components])  # type: ignore[arg-type]

    q = len(truth_sets)
    true_gram = torch.empty((q, q), dtype=dtype, device=device)
    for i in range(q):
        for j in range(i, q):
            val = empirical_embedding_inner_product(
                truth_sets[i],
                truth_sets[j],
                nu=nu,
                length_scale=length_scale,
                sigma=sigma,
                batch_size=batch_size,
            )
            true_gram[i, j] = val
            true_gram[j, i] = val

    row_mean = true_gram.mean(dim=1)
    grand_mean = true_gram.mean()
    baseline_raw = true_gram.diag() - 2.0 * row_mean + grand_mean
    baseline = torch.clamp(baseline_raw, min=0.0)
    numerator_total = numerators.sum()
    denominator_total = baseline.sum()
    if _float(denominator_total) > float(eps):
        r2_global = 1.0 - numerator_total / denominator_total
    else:
        r2_global = torch.full((), float("nan"), dtype=dtype, device=device)

    nan = torch.full_like(baseline, float("nan"))
    pointwise_r2 = torch.where(baseline > float(eps), 1.0 - numerators / baseline, nan)
    truth_points = [int(comp["truth_points_used"]) for comp in components]

    return {
        "embedding_r2_global": _maybe_float(r2_global),
        "embedding_mmd2_total": _float(numerator_total),
        "embedding_baseline_mmd2_total": _float(denominator_total),
        "embedding_mmd2_mean": _float(numerators.mean()),
        "embedding_baseline_mmd2_mean": _float(baseline.mean()),
        "embedding_mmd2_to_true": [_float(x) for x in numerators],
        "embedding_mmd2_raw_to_true": [_float(x) for x in numerators_raw],
        "embedding_baseline_mmd2": [_float(x) for x in baseline],
        "embedding_baseline_mmd2_raw": [_float(x) for x in baseline_raw],
        "embedding_r2_pointwise": [_maybe_float(x) for x in pointwise_r2],
        "embedding_truth_points_used": truth_points,
        "embedding_truth_points_used_min": int(min(truth_points)),
        "embedding_truth_points_used_max": int(max(truth_points)),
        "embedding_truth_points_used_mean": float(sum(truth_points) / len(truth_points)),
        "embedding_truth_gram": true_gram.detach().cpu().tolist(),
    }
