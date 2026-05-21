from __future__ import annotations

import math
import os
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


def _rff_features_core(x: torch.Tensor, omega: torch.Tensor, phase: torch.Tensor, scale: float) -> torch.Tensor:
    """Inner RFF computation: scale * cos(x @ omega^T + phase)."""
    return scale * torch.cos(x @ omega.transpose(0, 1) + phase)


# Opt-in torch.compile for the RFF feature map (shares the env var with matern).
_COMPILE_RFF = os.environ.get("KEDRL_COMPILE_MATERN", "0").strip().lower() in {
    "1", "true", "yes", "on",
}
try:
    _rff_features_compiled = torch.compile(_rff_features_core) if _COMPILE_RFF else None
except Exception:
    _rff_features_compiled = None


def rff_features(x: torch.Tensor, omega: torch.Tensor, phase: torch.Tensor, scale: float) -> torch.Tensor:
    if _rff_features_compiled is not None:
        try:
            return _rff_features_compiled(x, omega, phase, float(scale))
        except Exception:
            pass
    return _rff_features_core(x, omega, phase, float(scale))


@torch.no_grad()
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
    """Approximate H with the same finite feature map used by G.

    For a stationary kernel, the exact expression

        sum_p Gamma_l[p] k(R_p, Z_i - gamma Z_j)

    equals <phi(Z_i), sum_p Gamma_l[p] phi(gamma Z_j + R_p)> in the RKHS.
    In the finite-RFF objective we must use this second form so that K_Z, H,
    and G are all built from one shared feature map. Otherwise the Bellman
    quadratic is no longer guaranteed to be nonnegative and the optimizer can
    exploit numerical negative directions.
    """
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
    m = Z.shape[0]

    feat_Z = rff_features(Z, omega, phase, scale)

    reward_arg = R @ omega.transpose(0, 1) + phase
    c_reward = Gamma_sa.transpose(0, 1) @ torch.cos(reward_arg)
    s_reward = Gamma_sa.transpose(0, 1) @ torch.sin(reward_arg)

    z_arg = (float(gamma) * Z) @ omega.transpose(0, 1)
    c_z = torch.cos(z_arg)
    s_z = torch.sin(z_arg)

    successor_feature_sums = float(scale) * (
        c_reward.unsqueeze(1) * c_z.unsqueeze(0)
        - s_reward.unsqueeze(1) * s_z.unsqueeze(0)
    )

    # Auto-tune batch_size: the einsum materializes (batch, L, m) per chunk.
    # Use at least m//5 rows per batch for GPU utilization.
    _bs = max(int(batch_size), m // 5) if m > 50 else int(batch_size)

    H = torch.empty((n_targets, m, m), device=device, dtype=dtype)
    for i0 in range(0, m, _bs):
        i1 = min(m, i0 + _bs)
        H[:, i0:i1, :] = torch.einsum("iq,ljq->lij", feat_Z[i0:i1], successor_feature_sums)
    return H


@torch.no_grad()
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
    )                                                                # (L, m, q)

    # For very large L, chunk the bmm to cap peak GPU allocation.
    _L = feature_sums.shape[0]
    _chunk = max(1, min(_L, 512))
    if _L <= _chunk:
        return torch.bmm(feature_sums, feature_sums.transpose(1, 2))
    G = torch.empty((_L, feature_sums.shape[1], feature_sums.shape[1]),
                     device=device, dtype=dtype)
    for _g0 in range(0, _L, _chunk):
        _g1 = min(_L, _g0 + _chunk)
        fs = feature_sums[_g0:_g1]
        G[_g0:_g1] = torch.bmm(fs, fs.transpose(1, 2))
    return G
