from functools import lru_cache
import torch
import math
import os
# import torch_bessel



# def matern_kernel(x:np.ndarray,
#                   y:np.ndarray,
#                   nu=1.5, length_scale=1.0):
#
#     kernel = Matern(length_scale=length_scale, nu=nu)
#     return kernel(x, y)

def matern_kernel_np(X1, X2, nu, length_scale, sigma=1.0):
    """
    Compute the Matérn kernel matrix between two sets of vectors.
    Parameters:
        - x (np.ndarray): Array of shape (n_samples_x, n_features) representing input points.
        - y (np.ndarray): Array of shape (n_samples_y, n_features) representing input points.
        - nu (float, optional): Smoothness parameter of the Matern kernel.
        - length_scale (float, optional): Length scale parameter of the Matern kernel.
        - sigma (float, optional):  parameter of the Matern kernel. default is 1.0.

    Returns:
        - kernel_matrix (np.ndarray): The computed Matern kernel matrix of shape (n_samples_x, n_samples_y).
    """
    from scipy.spatial.distance import cdist
    from scipy.special import gamma, kv
    import numpy as np

    dist = cdist(X1, X2, metric='euclidean')
    scaled_dist = np.sqrt(2 * nu) * dist / length_scale
    scaled_safe = np.maximum(scaled_dist, np.finfo(float).eps)

    coeff  = sigma ** 2 * (2 ** (1 - nu)) / gamma(nu)
    kernel = coeff * (scaled_safe ** nu) * kv(nu, scaled_safe)
    kernel[dist == 0] = sigma ** 2  # variance on the diagonal

    return kernel


@lru_cache(maxsize=None)
def _matern_half_integer_coefficients(p: int) -> tuple[tuple[float, ...], float]:
    coeffs = tuple(
        float(math.factorial(2 * p - m) // (math.factorial(p - m) * math.factorial(m)))
        for m in range(p + 1)
    )
    prefac = math.factorial(p) / math.factorial(2 * p)
    return coeffs, float(prefac)


def _matern_elementwise_p1(dists: torch.Tensor, sqrt2nu: float, prefac: float, a0: float, a1: float) -> torch.Tensor:
    """Fused elementwise chain for p=1 (nu=1.5): exp(-z) * prefac * (a0 + a1*2z)."""
    z = dists * sqrt2nu
    t = 2.0 * z
    poly = a0 + a1 * t
    return prefac * torch.exp(-z) * poly


def _matern_elementwise_p2(dists: torch.Tensor, sqrt2nu: float, prefac: float, a0: float, a1: float, a2: float) -> torch.Tensor:
    """Fused elementwise chain for p=2 (nu=2.5): exp(-z) * prefac * (a0 + t*(a1 + a2*t))."""
    z = dists * sqrt2nu
    t = 2.0 * z
    poly = a0 + t * (a1 + a2 * t)
    return prefac * torch.exp(-z) * poly


def _matern_elementwise_generic(dists: torch.Tensor, sqrt2nu: float, prefac: float, coeffs: tuple[float, ...]) -> torch.Tensor:
    """Fused elementwise chain for arbitrary p: exp(-z) * prefac * horner(2z, coeffs)."""
    z = dists * sqrt2nu
    t = 2.0 * z
    result = torch.full_like(dists, coeffs[-1])
    for m in range(len(coeffs) - 2, -1, -1):
        result = result * t + coeffs[m]
    return prefac * torch.exp(-z) * result


_COMPILE_MATERN = os.environ.get("KEDRL_COMPILE_MATERN", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# torch.compile can fail at first execution when a host compiler is missing.
# Keep it opt-in and fall back permanently if the compiled path raises.
try:
    if _COMPILE_MATERN:
        _matern_ewise_p1_compiled = torch.compile(_matern_elementwise_p1)
        _matern_ewise_p2_compiled = torch.compile(_matern_elementwise_p2)
        _matern_ewise_gen_compiled = torch.compile(_matern_elementwise_generic)
        _HAS_COMPILE = True
    else:
        _HAS_COMPILE = False
except Exception:
    _HAS_COMPILE = False


def matern_kernel(X1: torch.Tensor, X2: torch.Tensor, nu: float, length_scale: float, sigma: float = 1.0) -> torch.Tensor:
    """
    Matérn kernel for ν = p + 0.5 (p ∈ ℕ₀) using the closed-form
    exponential × polynomial representation in pure PyTorch.

    The elementwise chain (exp, polynomial, multiply) can be fused via
    torch.compile by setting KEDRL_COMPILE_MATERN=1. It is disabled by
    default because some cluster/Windows environments lack the compiler
    stack needed by PyTorch Inductor.

    Args:
        X1: Tensor of shape (N, D)
        X2: Tensor of shape (M, D)
        nu: Smoothness parameter, must satisfy nu = p + 0.5
        length_scale: Length-scale ℓ > 0
        sigma: Signal variance σ (default 1.0)

    Returns:
        Kernel matrix of shape (N, M)
    """
    if X1.ndim != 2 or X2.ndim != 2 or X1.size(1) != X2.size(1):
        raise ValueError("X1, X2 must be 2D with same feature dim.")
    if length_scale <= 0 or sigma <= 0:
        raise ValueError("length_scale and sigma must be > 0.")
    p = int(nu - 0.5)
    if abs(nu - (p + 0.5)) > 1e-8:
        raise ValueError(f"nu={nu} must be half-integer (p + 0.5)")

    X1 = X1.contiguous()
    X2 = X2.to(device=X1.device, dtype=X1.dtype).contiguous()

    # Pairwise distances scaled by 1/ℓ  (cdist is already a single fused kernel)
    dists = torch.cdist(X1, X2, p=2.0)
    dists.div_(float(length_scale))

    # Precompute scalar constants
    sqrt2nu = math.sqrt(2.0 * nu)
    a, prefac_base = _matern_half_integer_coefficients(p)
    prefac = float(sigma ** 2) * prefac_base

    # Dispatch to the compiled elementwise kernel
    global _HAS_COMPILE
    if _HAS_COMPILE:
        try:
            if p == 1:
                return _matern_ewise_p1_compiled(dists, sqrt2nu, prefac, a[0], a[1])
            if p == 2:
                return _matern_ewise_p2_compiled(dists, sqrt2nu, prefac, a[0], a[1], a[2])
            return _matern_ewise_gen_compiled(dists, sqrt2nu, prefac, a)
        except Exception:
            _HAS_COMPILE = False

    # Fallback: in-place ops (original path, no compilation)
    z = dists.mul_(sqrt2nu)
    exp_term = torch.exp(-z)
    z = z.mul_(2.0)
    result = z.new_full(z.shape, float(a[-1]))
    for m in range(p - 1, -1, -1):
        result.mul_(z).add_(float(a[m]))
    exp_term.mul_(prefac)
    exp_term.mul_(result)
    del dists, z, result, a
    return exp_term


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##### Usage Example
# input points (2D points or more)
# x = np.array([1.0, 2.0]).reshape(-1,1).T
