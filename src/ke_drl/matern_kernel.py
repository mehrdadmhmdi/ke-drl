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


def matern_kernel( X1: torch.Tensor,X2: torch.Tensor, nu: float,length_scale: float,sigma: float = 1.0)-> torch.Tensor:
    """
    Matérn kernel for ν = p + 0.5 (p ∈ ℕ₀) using the closed-form
    exponential × polynomial representation in pure PyTorch.

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
    # Determine p = ν - 0.5 and verify half-integer
    p = int(nu - 0.5)
    if abs(nu - (p + 0.5)) > 1e-8:
        raise ValueError(f"nu={nu} must be half-integer (p + 0.5)")

    X1 = X1.contiguous()
    X2 = X2.to(device=X1.device, dtype=X1.dtype).contiguous()

    #  Compute scaled pairwise distances d / ℓ
    dists = torch.cdist(X1, X2, p=2.0)
    dists.div_(float(length_scale))
    # Build argument z = sqrt(2ν) * d / ℓ
    z        = dists.mul_(math.sqrt(2.0 * nu))
    exp_term = torch.exp(-z)

    # Horner evaluation of the polynomial in t = 2z (no .pow, no extra big temps)
    z = z.mul_(2.0)

    # Construct the polynomial sum ∑_{k=0}^p ( (p+k)! / (k!(p-k)!) ) · (2z)^{p-k}
    # coefficients for powers t^m, m=0..p: a_m = (2p - m)! / ((p - m)! * m!)
    # compute on CPU as Python ints, then cast
    a = [math.factorial(2 * p - m) // (math.factorial(p - m) * math.factorial(m)) for m in range(p + 1)]
    # Horner: result = a_p; for m=p-1..0: result = result * t + a_m
    result = z.new_full(z.shape, float(a[-1]))  # one output buffer
    for m in range(p - 1, -1, -1):
        result.mul_(z).add_(float(a[m]))                             # in-place

    # Prefactor and final multiply (reuse exp_term as K to avoid another N×M)
    prefac = (sigma ** 2) * (math.factorial(p) / math.factorial(2 * p))
    exp_term.mul_(prefac)                                            # exp_term := prefac * exp(-z)
    exp_term.mul_(result)                                            # exp_term := K
    del dists, z, result, a
    return exp_term


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##### Usage Example
# input points (2D points or more)
# x = np.array([1.0, 2.0]).reshape(-1,1).T   # Example: each data points with 2 dimensional x
# y = np.array([5.0, 6.0]).reshape(-1,1).T

#####  Matern kernel Params
#  nu = 1.5
#  length_scale = 1.0
#  kernel = Matern(length_scale=length_scale, nu=nu)

#### Compute kernel matrix
#  kernel_matrix = kernel(x, y)
#  print("Kernel matrix:\n", kernel_matrix)
