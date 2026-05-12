import torch

def Phi_sa(K_sa_prime: torch.Tensor,
           Gamma_sa  : torch.Tensor,
           eta_plus  : torch.Tensor) -> torch.Tensor:
    """
    Compute the off-policy continuation feature

        Phi(x) = K_+ D_eta Gamma(x).

    Here K_+ has rows anchored at the training inputs and columns anchored at
    logged successor inputs, D_eta is diagonal with successor density-ratio
    estimates eta_j^+, and Gamma may contain one or many target points.

    Args:
        K_sa_prime : (n, n) tensor, the cross Gram K_+ = k_X(x_r, x_j^+)
        Gamma_sa   : (n,), (n,1), or (n,L) tensor
        eta_plus   : (n,) or (n,1) tensor of eta_j^+

    Returns:
        Phi        : (n,), (n,1), or (n,L) tensor matching Gamma's target columns
    """
    gamma_was_vec = Gamma_sa.ndim == 1
    if gamma_was_vec:
        Gamma_sa = Gamma_sa.unsqueeze(1)
    if eta_plus.ndim == 1:
        eta_plus = eta_plus.unsqueeze(1)

    if Gamma_sa.ndim != 2:
        raise ValueError("Gamma_sa must be a vector or a 2D matrix.")
    if eta_plus.shape[0] != Gamma_sa.shape[0]:
        raise ValueError("eta_plus and Gamma_sa must have the same first dimension.")
    if K_sa_prime.shape[0] != Gamma_sa.shape[0] or K_sa_prime.shape[1] != Gamma_sa.shape[0]:
        raise ValueError("K_sa_prime must be square with size matching Gamma_sa rows.")

    weighted = Gamma_sa * eta_plus
    Phi = K_sa_prime @ weighted
    return Phi.squeeze(1) if gamma_was_vec else Phi
