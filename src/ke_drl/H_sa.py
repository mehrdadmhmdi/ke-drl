"""Construction of the H operator for the KE-DRL Bellman residual.

Given a 1D weight vector Gamma in R^N or a stack of L weight vectors
Gamma in R^{N x L}, this module returns

    H[i, j]      = sum_p Gamma[p]     * k_Z(R[p], Z[i] - gamma Z[j])     (single)
    H[l, i, j]   = sum_p Gamma[p, l]  * k_Z(R[p], Z[i] - gamma Z[j])     (stacked)

following equation (h_ij) in draft2.tex. The stacked path is the hot path used
by the global KE-DRL estimator and is fully vectorized across the L target
points: the (N, b*m) kernel block is computed once per row-batch and reused
for every l via a single matmul, instead of being rebuilt L times in a Python
loop.
"""

from __future__ import annotations

import torch

from .matern_kernel import matern_kernel


def compute_H_rows(i_batch, Gamma_sa, gamma, R, Z, kernel):
    """Compute H[i_batch, :] for one row-batch.

    Returns a chunk of shape (b, m) if Gamma_sa is 1D or (n, 1), and a chunk
    of shape (L, b, m) if Gamma_sa has shape (n, L). The kernel block
    K(R, shifts_flat) is constructed once and reused across all L stacked
    weight vectors.
    """
    device, dtype = Z.device, Z.dtype

    if not isinstance(i_batch, torch.Tensor):
        i_batch_t = torch.as_tensor(i_batch, device=device, dtype=torch.long)
    else:
        i_batch_t = i_batch.to(device=device, dtype=torch.long)

    R = R.to(device=device, dtype=dtype)
    Z = Z.to(device=device, dtype=dtype)
    gamma_t = torch.as_tensor(gamma, device=device, dtype=dtype)

    b = int(i_batch_t.numel())
    nR = R.shape[0]
    nZ = Z.shape[0]
    d = R.shape[1]

    Zi = Z.index_select(dim=0, index=i_batch_t)                # (b, d)
    shifts = Zi.unsqueeze(1) - gamma_t * Z.unsqueeze(0)        # (b, m, d)
    shifts_flat = shifts.reshape(b * nZ, d)                    # (b*m, d)

    # K(R, shifts_flat): (n, b*m) — computed exactly once per i_batch.
    K_full = kernel(R, shifts_flat)

    g = Gamma_sa.to(device=device, dtype=dtype)
    if g.ndim == 2 and g.shape[1] > 1:
        if g.shape[0] != nR:
            raise ValueError("Gamma rows {} must equal R rows {}.".format(g.shape[0], nR))
        L = g.shape[1]
        out = (g.transpose(0, 1) @ K_full).reshape(L, b, nZ)   # (L, b, m)
    else:
        if g.ndim == 2:
            g = g.reshape(-1)
        if g.numel() != nR:
            raise ValueError("Gamma length {} must equal R rows {}.".format(g.numel(), nR))
        out = (g.unsqueeze(0) @ K_full).reshape(b, nZ)         # (b, m)

    return i_batch_t, out


@torch.no_grad()
def H_sa(Gamma_sa, gamma, R, Z, nu, length_scale, sigma=1.0, batch_size=10, check_props=False):
    """Construct H for either a single Gamma or a stack of L Gammas.

    R has shape (n, d), Z has shape (m, d), Gamma_sa has shape (n,) or (n, L).
    The (R, shifts) kernel block for each row-batch is constructed once and
    reused for all L stacked target points, cutting kernel work by a factor
    of L over the previous per-l Python loop.

    ``batch_size`` controls how many Z-rows are processed at once.  For
    m > 50 the effective batch size is automatically raised to at least
    m // 5 (capped at 100) for better GPU utilization, unless the caller
    explicitly passes a larger value.
    """
    R = torch.as_tensor(R)
    Z = torch.as_tensor(Z)
    Gamma_sa = torch.as_tensor(Gamma_sa)

    device, dtype = Z.device, Z.dtype
    nR = R.shape[0]
    mZ = Z.shape[0]

    # Auto-tune: each batch computes a (n, b*m) kernel block.
    # Larger batches amortize Python overhead and improve GPU occupancy.
    _bs = max(int(batch_size), mZ // 5) if mZ > 50 and batch_size <= 10 else int(batch_size)

    kernel = lambda x1, x2: matern_kernel(x1, x2, length_scale=length_scale, nu=nu, sigma=sigma)
    row_indices = torch.arange(mZ, device=device)

    if Gamma_sa.ndim == 2 and Gamma_sa.shape[1] > 1:
        if Gamma_sa.shape[0] != nR:
            raise ValueError("Gamma_sa rows {} must equal R rows {}.".format(Gamma_sa.shape[0], nR))
        L = Gamma_sa.shape[1]
        H_stack = torch.zeros((L, mZ, mZ), device=device, dtype=dtype)
        for i0 in range(0, mZ, _bs):
            i_batch = row_indices[i0 : i0 + _bs]
            _, chunk = compute_H_rows(i_batch, Gamma_sa, gamma, R, Z, kernel)
            H_stack[:, i_batch.to(torch.long), :] = chunk
        if check_props:
            print("H stacked finite:", bool(torch.isfinite(H_stack).all()))
        return H_stack

    g = Gamma_sa.reshape(-1) if Gamma_sa.ndim == 2 else Gamma_sa
    if g.numel() != nR:
        raise ValueError("Gamma_sa length {} must equal R rows {}.".format(g.numel(), nR))

    H = torch.zeros((mZ, mZ), device=device, dtype=dtype)
    for i0 in range(0, mZ, _bs):
        i_batch = row_indices[i0 : i0 + _bs]
        _, chunk = compute_H_rows(i_batch, g, gamma, R, Z, kernel)
        H.index_copy_(0, i_batch.to(dtype=torch.long), chunk)

    if check_props:
        print("H finite:", bool(torch.isfinite(H).all()))
    return H
