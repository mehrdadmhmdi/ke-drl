from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch


def _format_threshold(thr: float) -> str:
    if thr > 0:
        exponent = round(math.log10(float(thr)))
        if math.isclose(float(thr), 10.0**exponent, rel_tol=1e-12, abs_tol=0.0):
            return f"1em{abs(exponent)}" if exponent < 0 else f"1e{exponent}"
    text = f"{float(thr):.0e}".replace("+", "")
    text = text.replace("-", "m").replace(".", "p")
    text = text.replace("m0", "m")
    return text


@torch.no_grad()
def matrix_rank_diagnostics(
    B: Any,
    *,
    atol: float | None = None,
    rtol: float | None = None,
    relative_thresholds: Iterable[float] = (1e-2, 1e-3, 1e-4),
    prefix: str = "",
    return_singular_values: bool = False,
) -> dict[str, float | int | torch.Tensor]:
    """Compute numerical and effective-rank diagnostics for a 2-D matrix.

    The default numerical rank uses the same scale as ``torch.linalg.matrix_rank``:
    ``tol = max(m, n) * eps * s_max`` unless ``atol`` or ``rtol`` is supplied.
    All computations stay on the tensor's existing device, so a CUDA ``B`` uses
    the GPU SVD implementation.
    """
    x = torch.as_tensor(B)
    if x.ndim != 2:
        raise ValueError(f"B must be a 2-D matrix, got shape {tuple(x.shape)}.")
    if not (x.is_floating_point() or torch.is_complex(x)):
        x = x.to(torch.float32)
    if x.numel() == 0:
        raise ValueError("B must be nonempty.")

    s = torch.linalg.svdvals(x)
    if torch.is_complex(s):
        s = s.real
    s = torch.sort(s, descending=True).values
    min_dim = int(min(x.shape))
    max_dim = int(max(x.shape))
    s_max = float(s[0].detach().cpu())
    eps = torch.finfo(s.dtype).eps
    tol = float(0.0 if atol is None else atol)
    tol += float(max_dim * eps if rtol is None else rtol) * s_max

    rank = int((s > tol).sum().detach().cpu())
    fro_sq_t = torch.sum(s * s)
    nuclear_t = torch.sum(s)
    fro_norm = float(torch.sqrt(fro_sq_t).detach().cpu())
    nuclear_norm = float(nuclear_t.detach().cpu())
    stable_rank = float((fro_sq_t / (s[0] * s[0])).detach().cpu()) if s_max > 0 else 0.0

    if nuclear_norm > 0:
        p = s / nuclear_t
        entropy_t = -(p * torch.log(p.clamp_min(torch.finfo(p.dtype).tiny))).sum()
        entropy = float(entropy_t.detach().cpu())
        effective_rank = float(math.exp(entropy))
    else:
        entropy = 0.0
        effective_rank = 0.0

    s_min = float(s[-1].detach().cpu())
    condition_number = float(s_max / s_min) if s_min > 0 else float("inf")
    smallest_kept = float(s[rank - 1].detach().cpu()) if rank > 0 else 0.0
    condition_number_tol = float(s_max / smallest_kept) if smallest_kept > 0 else float("inf")

    out: dict[str, float | int | torch.Tensor] = {
        f"{prefix}num_rows": int(x.shape[0]),
        f"{prefix}num_cols": int(x.shape[1]),
        f"{prefix}min_dim": min_dim,
        f"{prefix}numerical_rank": rank,
        f"{prefix}rank_fraction": float(rank / min_dim) if min_dim else 0.0,
        f"{prefix}rank_tolerance": tol,
        f"{prefix}stable_rank": stable_rank,
        f"{prefix}effective_rank": effective_rank,
        f"{prefix}entropy_rank": entropy,
        f"{prefix}spectral_norm": s_max,
        f"{prefix}frobenius_norm": fro_norm,
        f"{prefix}nuclear_norm": nuclear_norm,
        f"{prefix}condition_number": condition_number,
        f"{prefix}condition_number_tol": condition_number_tol,
        f"{prefix}smallest_singular_value": s_min,
    }
    for thr in relative_thresholds:
        thr = float(thr)
        key = _format_threshold(thr)
        cutoff = thr * s_max
        out[f"{prefix}rank_rel_{key}"] = int((s > cutoff).sum().detach().cpu())
    if return_singular_values:
        out[f"{prefix}singular_values"] = s.detach().cpu()
    return out


__all__ = ["matrix_rank_diagnostics"]
