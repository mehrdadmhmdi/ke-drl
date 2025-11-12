import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import gamma, kv
import torch
import time
import math
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
    dist = cdist(X1, X2, metric='euclidean')
    scaled_dist = np.sqrt(2 * nu) * dist / length_scale
    scaled_safe = np.maximum(scaled_dist, np.finfo(float).eps)

    coeff  = sigma ** 2 * (2 ** (1 - nu)) / gamma(nu)
    kernel = coeff * (scaled_safe ** nu) * kv(nu, scaled_safe)
    kernel[dist == 0] = sigma ** 2  # variance on the diagonal

    return kernel


import math
import torch

def matern_kernel(
    X1: torch.Tensor,
    X2: torch.Tensor,
    nu: float,
    length_scale: float,
    sigma: float = 1.0,
    batch_size: int = None
) -> torch.Tensor:
    """
    Matérn kernel for ν = p + 0.5 (p ∈ ℕ₀) using the closed-form
    exponential × polynomial representation in pure PyTorch.
    Supports batching over X1 to reduce memory usage for large datasets.

    Args:
        X1: Tensor of shape (N, D)
        X2: Tensor of shape (M, D)
        nu: Smoothness parameter, must satisfy nu = p + 0.5
        length_scale: Length-scale ℓ > 0
        sigma: Signal variance σ (default 1.0)
        batch_size: Optional batch size for computing kernel in chunks over X1 (default None, no batching)

    Returns:
        Kernel matrix of shape (N, M)
    """
    if X1.ndim != 2 or X2.ndim != 2 or X1.size(1) != X2.size(1):
        raise ValueError("X1, X2 must be 2D with same feature dim.")
    if length_scale <= 0 or sigma <= 0:
        raise ValueError("length_scale and sigma must be > 0.")
    # Determine p = ν - 0.5 and verify half-integer
    p = int(nu - 0.5)
    if abs(nu - (p + 0.5)) > 1e-8:
        raise ValueError(f"nu={nu} must be half-integer (p + 0.5)")

    X1 = X1.contiguous()
    X2 = X2.to(device=X1.device, dtype=X1.dtype).contiguous()

    prefac = (sigma ** 2) * (math.factorial(p) / math.factorial(2 * p))
    sqrt_2nu = math.sqrt(2.0 * nu)

    # Precompute coefficients for the polynomial
    a = [math.factorial(2 * p - m) // (math.factorial(p - m) * math.factorial(m)) for m in range(p + 1)]

    if batch_size is None or batch_size >= X1.size(0):
        # Non-batched computation
        dists = torch.cdist(X1, X2, p=2.0)
        dists.div_(length_scale)
        z = dists.mul_(sqrt_2nu)
        exp_term = torch.exp(-z)
        z.mul_(2.0)  # t = 2z
        result = z.new_full(z.shape, float(a[-1]))
        for m in range(p - 1, -1, -1):
            result.mul_(z).add_(float(a[m]))
        exp_term.mul_(prefac).mul_(result)
        del dists, z, result, a
        return exp_term
    else:
        # Batched computation over X1
        kernel_parts = []
        for i in range(0, X1.size(0), batch_size):
            end = min(i + batch_size, X1.size(0))
            X1_batch = X1[i:end]
            dists = torch.cdist(X1_batch, X2, p=2.0)
            dists.div_(length_scale)
            z = dists.mul_(sqrt_2nu)
            exp_term = torch.exp(-z)
            z.mul_(2.0)  # t = 2z
            result = z.new_full(z.shape, float(a[-1]))
            for m in range(p - 1, -1, -1):
                result.mul_(z).add_(float(a[m]))
            exp_term.mul_(prefac).mul_(result)
            kernel_parts.append(exp_term)
            del dists, z, result, exp_term  # Explicitly free memory
            torch.cuda.empty_cache() if X1.device.type == 'cuda' else None
        del a
        return torch.cat(kernel_parts, dim=0)