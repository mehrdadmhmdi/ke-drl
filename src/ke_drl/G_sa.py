"""Construction of the G operator for the KE-DRL Bellman residual.

Given a 1D weight vector Gamma in R^N or a stack of L weight vectors
Gamma in R^{N x L}, this module returns

    G[i, j]     = sum_{u,v} Gamma[u]    Gamma[v]    k_Z(gamma z_i + r_u, gamma z_j + r_v)
    G[l, i, j]  = sum_{u,v} Gamma[u,l]  Gamma[v,l]  k_Z(gamma z_i + r_u, gamma z_j + r_v)

following equation (g_ij) in draft2.tex.

The stacked path is vectorized over the L target points: the (n, n) kernel
block k_Z(gamma z_i + R, gamma z_j + R) is computed exactly once per (i, j)
and then turned into the L bilinear forms via a single batched matmul. This
removes the L-fold Python loop and roughly an L-times kernel cost present
in the previous per-l implementation.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Optional

from .matern_kernel import matern_kernel


def compute_transformed_grid_pytorch(
    Z_grid: torch.Tensor, reward_set: torch.Tensor, gamma_val: float
) -> torch.Tensor:
    """Build (m, n, d) tensor with entry [i, u, :] = gamma * Z[i] + R[u]."""
    return float(gamma_val) * Z_grid.unsqueeze(1) + reward_set.unsqueeze(0)


def _bilinear_block_stacked(K_block: torch.Tensor, Gamma_2d: torch.Tensor) -> torch.Tensor:
    """For a kernel block K of shape (bi, n, bjm, n) and Gamma (n, L), return
    a (L, bi, bjm) tensor with entry [l, i, j] = sum_{u,v} K[i,u,j,v] Gamma[u,l] Gamma[v,l].

    The computation is done in two matmul-style reductions to keep the peak
    intermediate at O(L * bi * bjm * n) instead of materializing an L-fold
    expansion of K. With the default block_i=1 and block_j=n used by the
    estimator this stays at L * 1 * 1 * n = O(L n) per (i, j).
    """
    bi, n, bjm, n2 = K_block.shape
    assert n == n2, "K_block must be (bi, n, bjm, n)."

    # Step 1: weight over u via a single (n x L) matmul reused across (i, bjm, v).
    # K_block.reshape(bi*n, bjm*n).T -> (bjm*n, bi*n); but we instead reduce
    # u by viewing K_block as (bi, n, bjm*n) and contracting (n, L) on the u-axis.
    Kflat = K_block.reshape(bi, n, bjm * n)                   # (bi, n, bjm*n)
    # tmp[l, i, w] = sum_u Gamma[u, l] * Kflat[i, u, w]  with w = (bjm, v)
    tmp = torch.einsum("ul,iuw->liw", Gamma_2d, Kflat)        # (L, bi, bjm*n)
    tmp = tmp.reshape(-1, bi, bjm, n)                         # (L, bi, bjm, n)
    # Step 2: reduce over v with Gamma[v, l]
    out = torch.einsum("lijv,vl->lij", tmp, Gamma_2d)          # (L, bi, bjm)
    return out


def compute_G_pytorch_batched(
    transformed: torch.Tensor,
    Gamma_sa: torch.Tensor,
    nu: float,
    length_scale: float,
    sigma: float = 1.0,
    block_i: int = 1,
    block_j: Optional[int] = None,
    check_props: bool = False,
) -> torch.Tensor:
    """Batched G-matrix computation, vectorized over the L target points.

    transformed has shape (m, n, d) where transformed[i, u, :] = gamma z_i + r_u.
    Gamma_sa has shape (n,) or (n, L).

    Returns:
        - (m, m) if Gamma_sa is 1D,
        - (L, m, m) if Gamma_sa is 2D.

    The kernel block between {gamma z_i + R} and {gamma z_j + R} only depends
    on Z, R, gamma, and the kernel parameters. It is built once per (i, j)
    sub-block and reused across every l in the stacked path, instead of
    being rebuilt L times.
    """
    m, n, d = transformed.shape
    device, dtype = transformed.device, transformed.dtype

    if block_j is None:
        block_j = n
    if block_j % n != 0:
        raise ValueError("block_j must be a multiple of n.")
    bjm_per_block = block_j // n

    is_stack = (Gamma_sa.ndim == 2 and Gamma_sa.shape[1] > 1)
    if Gamma_sa.ndim == 2 and Gamma_sa.shape[0] != n:
        raise ValueError(f"Gamma_sa rows {Gamma_sa.shape[0]} must equal transformed.shape[1] {n}.")
    if Gamma_sa.ndim == 1 and Gamma_sa.numel() != n:
        raise ValueError(f"Gamma_sa length {Gamma_sa.numel()} must equal transformed.shape[1] {n}.")

    if is_stack:
        Gamma_2d = Gamma_sa.to(device=device, dtype=dtype)            # (n, L)
        L = Gamma_2d.shape[1]
        G = torch.zeros((L, m, m), device=device, dtype=dtype)
    else:
        Gamma_1d = Gamma_sa.reshape(-1).to(device=device, dtype=dtype)
        Gamma_2d = Gamma_1d.unsqueeze(1)                              # (n, 1)
        G = torch.zeros((m, m), device=device, dtype=dtype)

    for i0 in range(0, m, block_i):
        i1 = min(m, i0 + block_i)
        bi = i1 - i0
        blk_rows = transformed[i0:i1].reshape(bi * n, d)              # (bi*n, d)

        for j0 in range(0, m * n, block_j):
            j1 = min(m * n, j0 + block_j)
            bj = j1 - j0
            bjm = bj // n
            j_i0 = j0 // n
            j_i1 = j_i0 + bjm

            blk_cols = transformed[j_i0:j_i1].reshape(bj, d)          # (bj, d)

            # Kernel block K[(i, u), (j, v)] — independent of Gamma.
            Kblk = matern_kernel(blk_rows, blk_cols, nu, length_scale, sigma)
            K4 = Kblk.view(bi, n, bjm, n)

            # Vectorized over L (or L=1 for the single-Gamma path)
            G_block = _bilinear_block_stacked(K4, Gamma_2d)            # (L, bi, bjm)

            if is_stack:
                G[:, i0:i1, j_i0:j_i1] += G_block
            else:
                G[i0:i1, j_i0:j_i1] += G_block.squeeze(0)

            del Kblk, K4, G_block
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if check_props:
        if is_stack:
            print(f"G stack finite: {bool(torch.isfinite(G).all())}")
        else:
            _check_G_properties(G.detach().cpu().numpy())

    return G


def compute_G_pytorch_semivectorized(
    transformed: torch.Tensor, Gamma_sa: torch.Tensor, nu: float, length_scale: float, sigma: float = 1.0
) -> torch.Tensor:
    """Reference O(m^2) double-loop implementation for verification only."""
    m, n, _ = transformed.shape
    if Gamma_sa.ndim != 1:
        raise ValueError("Reference implementation expects a 1D Gamma.")
    G = torch.empty((m, m), device=transformed.device, dtype=transformed.dtype)
    for i in range(m):
        for j in range(i, m):
            Kij = matern_kernel(transformed[i], transformed[j], nu, length_scale, sigma)
            tmp = torch.mv(Kij, Gamma_sa)
            val = torch.dot(Gamma_sa, tmp)
            G[i, j] = G[j, i] = val
    return G


def _check_G_properties(G) -> None:
    """Optional symmetry / PSD diagnostics for the single-Gamma path."""
    G = np.asarray(G)
    print("G is symmetric:", np.allclose(G, G.T, atol=1e-8))
    try:
        eigs = np.linalg.eigvalsh(0.5 * (G + G.T))
        print("G min eigenvalue:", float(eigs.min()))
    except np.linalg.LinAlgError:
        print("G eigen decomposition failed.")
