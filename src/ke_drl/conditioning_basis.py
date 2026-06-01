from __future__ import annotations

from typing import Any, Optional

import torch


def _randperm_cpu(n: int, seed: Optional[int]) -> torch.Tensor:
    if seed is None:
        return torch.randperm(int(n))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return torch.randperm(int(n), generator=gen)


@torch.no_grad()
def _nearest_rows_to_centers(
    X: torch.Tensor,
    centers: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    best_dist = torch.full(
        (centers.shape[0],), float("inf"), device=X.device, dtype=X.dtype
    )
    best_idx = torch.full((centers.shape[0],), -1, device=X.device, dtype=torch.long)
    for start in range(0, X.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), X.shape[0])
        dist = torch.cdist(X[start:stop], centers)
        vals, pos = dist.min(dim=0)
        mask = vals < best_dist
        best_dist[mask] = vals[mask]
        best_idx[mask] = pos[mask] + start
    return best_idx[best_idx >= 0].detach().cpu()


@torch.no_grad()
def select_conditioning_basis(
    X: torch.Tensor,
    *,
    n_basis: Optional[int] = None,
    method: str = "full",
    seed: Optional[int] = None,
    standardize: bool = True,
    candidate_pool: Optional[int] = None,
    max_iter: int = 20,
    batch_size: int = 8192,
    device: torch.device | str | None = None,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Choose an L-point conditioning basis in current X=(S,A) space.

    The selected basis controls the number of rows in the mean-embedding
    coefficient matrix B. It does not discard transitions from the Bellman
    target/operator construction.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D tensor.")
    n = int(X.shape[0])
    if n < 1:
        raise ValueError("X must contain at least one row.")

    if n_basis is None:
        requested = n
    else:
        requested = int(n_basis)
        if requested < 1:
            raise ValueError("n_basis must be at least 1.")
    requested = min(requested, n)

    method_l = str(method).strip().lower()
    if requested >= n or method_l in {"full", "all", "none"}:
        idx = torch.arange(n, dtype=torch.long)
        meta = {
            "method": "full",
            "requested_basis": requested,
            "basis_size": n,
            "original_rows": n,
            "standardize": False,
        }
        return X, idx, meta

    work_device = torch.device(device) if device is not None else X.device
    X_work = X.to(device=work_device, dtype=torch.float32)
    if bool(standardize):
        mean = X_work.mean(dim=0, keepdim=True)
        sd = X_work.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        X_work = (X_work - mean) / sd

    perm = _randperm_cpu(n, seed)
    if method_l in {"random", "subsample", "uniform"}:
        idx = perm[:requested]
    elif method_l in {"kmeans", "kmeans_landmarks", "landmark", "landmarks"}:
        pool = candidate_pool
        if pool is None:
            pool = max(20000, 20 * requested)
        pool = min(max(int(pool), requested), n)
        pool_idx = perm[:pool].to(work_device)
        candidates = X_work.index_select(0, pool_idx)
        init_idx = _randperm_cpu(pool, None if seed is None else int(seed) + 17)[:requested].to(work_device)
        centers = candidates.index_select(0, init_idx).clone()

        for it in range(max(1, int(max_iter))):
            labels = torch.cdist(candidates, centers).argmin(dim=1)
            sums = torch.zeros_like(centers)
            sums.index_add_(0, labels, candidates)
            counts = torch.bincount(labels, minlength=requested).to(
                device=work_device, dtype=X_work.dtype
            )
            empty = counts <= 0
            new_centers = sums / counts.clamp_min(1.0).unsqueeze(1)
            new_centers[empty] = centers[empty]
            shift = (new_centers - centers).pow(2).sum(dim=1).sqrt().mean()
            centers = new_centers
            if verbose and (it == 0 or it + 1 == int(max_iter) or (it + 1) % 5 == 0):
                print(
                    f"[mean_embedding_basis] kmeans iter {it + 1}/{int(max_iter)}, "
                    f"mean center shift={float(shift):.4e}",
                    flush=True,
                )
            if float(shift) < 1e-5:
                break
        idx = _nearest_rows_to_centers(X_work, centers, batch_size=int(batch_size))
    else:
        raise ValueError(
            f"Unknown mean-embedding basis method={method!r}. Use full, kmeans, or random."
        )

    idx = torch.unique(idx.to(dtype=torch.long), sorted=True)
    if idx.numel() < requested:
        mask = ~torch.isin(perm, idx)
        idx = torch.cat([idx, perm[mask][: requested - idx.numel()]])
        idx = torch.unique(idx.to(dtype=torch.long), sorted=True)
    if idx.numel() > requested:
        idx = idx[:requested]

    X_basis = X.index_select(0, idx.to(X.device))
    meta = {
        "method": method_l,
        "requested_basis": requested,
        "basis_size": int(X_basis.shape[0]),
        "original_rows": n,
        "standardize": bool(standardize),
        "candidate_pool": None if candidate_pool is None else int(candidate_pool),
        "max_iter": int(max_iter),
    }
    return X_basis, idx.detach().cpu(), meta
