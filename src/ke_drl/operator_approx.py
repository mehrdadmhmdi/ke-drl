from __future__ import annotations

import math
from typing import Optional

import torch


def _check_half_integer_nu(nu: float) -> int:
    df = 2.0 * float(nu)
    df_int = int(round(df))
    if abs(df - df_int) > 1e-8 or df_int < 1:
        raise ValueError("RFF Matern approximation expects half-integer nu so 2 * nu is a positive integer.")
    return df_int


def sample_matern_rff(
    *,
    num_features: int,
    input_dim: int,
    nu: float,
    length_scale: float,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
    seed: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Sample random Fourier features for the Matern kernel.

    The Matern spectral measure is a multivariate Student-t distribution with
    df=2*nu and scale 1/length_scale.  The code samples it as z/sqrt(chi2/df).
    """
    if num_features < 1:
        raise ValueError("num_features must be positive.")
    if input_dim < 1:
        raise ValueError("input_dim must be positive.")
    if length_scale <= 0 or sigma <= 0:
        raise ValueError("length_scale and sigma must be positive.")

    df = _check_half_integer_nu(nu)
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))

    z = torch.randn((num_features, input_dim), generator=gen, device=device, dtype=dtype)
    chi = torch.randn((num_features, df), generator=gen, device=device, dtype=dtype).pow_(2).sum(dim=1, keepdim=True)
    omega = (z / torch.sqrt(chi / float(df))) / float(length_scale)
    phase = 2.0 * math.pi * torch.rand((num_features,), generator=gen, device=device, dtype=dtype)
    scale = float(sigma) * math.sqrt(2.0 / float(num_features))
    return omega, phase, scale


def rff_features(x: torch.Tensor, omega: torch.Tensor, phase: torch.Tensor, scale: float) -> torch.Tensor:
    return float(scale) * torch.cos(x @ omega.transpose(0, 1) + phase)


def compute_H_rff(
    Gamma_sa: torch.Tensor,
    gamma: float,
    R: torch.Tensor,
    Z: torch.Tensor,
    *,
    omega: torch.Tensor,
    phase: torch.Tensor,
    scale: float,
    batch_size: int = 16,
) -> torch.Tensor:
    """Approximate H_l[i,j] = sum_p Gamma_l[p] k(R_p, Z_i - gamma Z_j)."""
    if Gamma_sa.ndim == 1:
        Gamma_sa = Gamma_sa.unsqueeze(1)
    if Gamma_sa.ndim != 2:
        raise ValueError("Gamma_sa must have shape (N,) or (N,L).")
    if Gamma_sa.shape[0] != R.shape[0]:
        raise ValueError("Gamma_sa rows must match R rows.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    device, dtype = Z.device, Z.dtype
    Gamma_sa = Gamma_sa.to(device=device, dtype=dtype)
    R = R.to(device=device, dtype=dtype)
    Z = Z.to(device=device, dtype=dtype)
    omega = omega.to(device=device, dtype=dtype)
    phase = phase.to(device=device, dtype=dtype)

    n_targets = Gamma_sa.shape[1]
    m, d = Z.shape
    feat_R = rff_features(R, omega, phase, scale)
    weighted_reward_features = Gamma_sa.transpose(0, 1) @ feat_R

    H = torch.empty((n_targets, m, m), device=device, dtype=dtype)
    gamma_t = torch.as_tensor(float(gamma), device=device, dtype=dtype)
    for i0 in range(0, m, int(batch_size)):
        i1 = min(m, i0 + int(batch_size))
        shifts = Z[i0:i1].unsqueeze(1) - gamma_t * Z.unsqueeze(0)
        feat_shift = rff_features(shifts.reshape(-1, d), omega, phase, scale)
        chunk = weighted_reward_features @ feat_shift.transpose(0, 1)
        H[:, i0:i1, :] = chunk.reshape(n_targets, i1 - i0, m)
        del shifts, feat_shift, chunk
    return H


def compute_G_rff(
    Gamma_sa: torch.Tensor,
    gamma: float,
    R: torch.Tensor,
    Z: torch.Tensor,
    *,
    omega: torch.Tensor,
    phase: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Approximate G_l from Matern random Fourier features.

    G_l[i,j] = <sum_u Gamma_l[u] phi(gamma Z_i + R_u),
                sum_v Gamma_l[v] phi(gamma Z_j + R_v)>.

    The addition identity avoids the direct L*m*N*q feature construction.
    """
    if Gamma_sa.ndim == 1:
        Gamma_sa = Gamma_sa.unsqueeze(1)
    if Gamma_sa.ndim != 2:
        raise ValueError("Gamma_sa must have shape (N,) or (N,L).")
    if Gamma_sa.shape[0] != R.shape[0]:
        raise ValueError("Gamma_sa rows must match R rows.")

    device, dtype = Z.device, Z.dtype
    Gamma_sa = Gamma_sa.to(device=device, dtype=dtype)
    R = R.to(device=device, dtype=dtype)
    Z = Z.to(device=device, dtype=dtype)
    omega = omega.to(device=device, dtype=dtype)
    phase = phase.to(device=device, dtype=dtype)

    reward_arg = R @ omega.transpose(0, 1) + phase
    c_reward = Gamma_sa.transpose(0, 1) @ torch.cos(reward_arg)
    s_reward = Gamma_sa.transpose(0, 1) @ torch.sin(reward_arg)

    z_arg = (float(gamma) * Z) @ omega.transpose(0, 1)
    c_z = torch.cos(z_arg)
    s_z = torch.sin(z_arg)

    feature_sums = float(scale) * (
        c_reward.unsqueeze(1) * c_z.unsqueeze(0)
        - s_reward.unsqueeze(1) * s_z.unsqueeze(0)
    )
    return torch.bmm(feature_sums, feature_sums.transpose(1, 2))
