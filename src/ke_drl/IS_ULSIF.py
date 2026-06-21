"""Unconstrained Least-Squares Importance Fitting (uLSIF) estimator.

Implements the kernel-based direct ratio estimator used to recover the
target/logged-data density ratio entering Phi(x) = K_+ D_eta Gamma(x). The
denominator is represented by the observed samples; no behavior-policy density
model is fit. The Bellman IS weight is the ordinary density ratio, so callers
that build ``D_eta`` should use ``alpha_mix=0.0``. Nonzero ``alpha_mix`` fits an
alpha-relative RuLSIF ratio, which is a different stabilization diagnostic and
not the Bellman continuation weight. Two basis choices are supported:

- ``basis_source="denominator"`` with ``n_basis=None``: every logged sample is
  a basis function (kernel-ridge-flavored uLSIF, the original repository
  behavior). Cost O(N^3) per fit; cleaner for small N.
- ``basis_source="numerator"`` (or ``"denominator"``) with ``n_basis=b`` for b < N:
  classical uLSIF with a randomly sampled subset of b centers (Kanamori et al.
  2009). Cost O(b^2 N + b^3) per fit; the standard speed-up for large N.

The real-data KE-DRL path deliberately does not fit a behavior policy model.
The denominator distribution is represented only by the observed samples
``(S, A)``, while the numerator distribution is represented by resampling
actions from the user-specified target policy at those same states.  Optional
nonnegativity and empirical mean calibration enforce basic density-ratio
identities without estimating either density separately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .Probability_Densities import Probability_Densities
from .matern_kernel import matern_kernel


class ULSIFEstimator:
    def __init__(
        self,
        kernel_func=matern_kernel,
        lambda_reg: float = 1e-3,
        nu: float = 1.5,
        length_scale: float = 1.0,
        sigma: float = 1.0,
        alpha_mix: Optional[float] = None,
    ):
        import os as _os
        self.kernel_func = kernel_func
        self.lambda_reg = float(lambda_reg)
        self.kernel_kwargs = {"nu": float(nu), "length_scale": float(length_scale), "sigma": float(sigma)}
        # RuLSIF alpha-relative mixing in [0,1). 0 == plain uLSIF.
        # If no explicit value is passed, keep the cluster env-var fallback for
        # backward compatibility with older sbatch files.
        if alpha_mix is None:
            alpha_mix = float(_os.environ.get("KEDRL_RULSIF_ALPHA", "0.0"))
        self.alpha_mix = float(alpha_mix)
        if not (0.0 <= self.alpha_mix < 1.0):
            raise ValueError("alpha_mix must be in [0, 1).")
        self.alpha: Optional[torch.Tensor] = None      # (n_basis, 1)
        self._X_basis: Optional[torch.Tensor] = None   # (n_basis, d)
        self.fit_diagnostics: dict[str, object] = {}

    # ------------------------------------------------------------------
    def _sample_target(
        self,
        action_dim: int,
        s: torch.Tensor,
        target_p_choice: str,
        target_p_params: dict,
    ) -> torch.Tensor:
        if not target_p_params:
            raise ValueError("target_p_params must be provided for Torch sampling.")
        prob_density = Probability_Densities(**target_p_params)

        if s.ndim != 2:
            raise ValueError("s must be (n, d_s).")
        n = s.shape[0]

        sample = prob_density.sample_pdf(target_p_choice, s)
        if sample is None:
            raise RuntimeError("sample_pdf returned None; check target_p_choice/params.")
        sample = torch.as_tensor(sample, device=s.device, dtype=s.dtype).reshape(n, -1)

        k = sample.shape[1]
        if k == action_dim:
            return sample
        if k == 1:
            return sample.repeat(1, action_dim)
        reps = (action_dim + k - 1) // k
        return sample.repeat(1, reps)[:, :action_dim]

    # ------------------------------------------------------------------
    def fit(
        self,
        S: torch.Tensor,
        A: torch.Tensor,
        target_p_choice: str,
        target_p_params: dict,
        *,
        n_basis: Optional[int] = None,
        basis_source: str = "numerator",
        basis_seed: Optional[int] = None,
        target_sample_multiplier: int = 1,
        nonnegative_alpha: bool = True,
        calibrate_mean: bool = True,
        plot: bool = False,
    ) -> torch.Tensor:
        """Fit the uLSIF coefficients.

        Args:
            S, A: behavior samples, both 2D tensors of shape (N, d_s) and (N, d_a).
            target_p_choice, target_p_params: configuration for sampling the
                target-policy action used to form the numerator points.
            n_basis: number of basis centers. ``None`` keeps the original
                behavior of using every denominator sample as a basis (cost
                grows as O(N^3)).
            basis_source: ``"numerator"`` (default) or ``"denominator"``. The
                default matches the canonical direct-uLSIF construction.
            basis_seed: optional integer seed for the basis subsample.
            target_sample_multiplier: number of target-policy action draws per
                observed state used to form numerator samples. This still
                estimates the direct target/data ratio; it does not estimate a
                behavior policy.
            nonnegative_alpha: clamp negative kernel coefficients to zero after
                the linear solve, the standard uLSIF post-processing that keeps
                the fitted ratio nonnegative for positive kernels.
            calibrate_mean: for ordinary uLSIF (``alpha_mix=0``), rescale the
                fitted ratio so its empirical denominator mean is one. The true
                density ratio obeys this identity.
            plot: optional diagnostic plots; off by default to keep cluster runs
                lightweight.
        """
        if S.ndim != 2 or A.ndim != 2:
            raise ValueError("S and A must be 2D tensors.")
        device = S.device
        dtype = S.dtype
        n, d_a = A.shape

        X_beta = torch.cat([S, A], dim=1).to(device=device, dtype=dtype)             # denominator

        target_sample_multiplier = max(1, int(target_sample_multiplier or 1))
        S_nu = S.to(device, dtype)
        if target_sample_multiplier > 1:
            S_nu = S_nu.repeat((target_sample_multiplier, 1))
        a_pi = self._sample_target(d_a, S_nu, target_p_choice, target_p_params)
        X_pi = torch.cat([S_nu, a_pi], dim=1)                                        # numerator

        basis_source_l = str(basis_source).lower()
        if basis_source_l not in {"denominator", "numerator"}:
            raise ValueError("basis_source must be 'denominator' or 'numerator'.")
        source_pool = X_beta if basis_source_l == "denominator" else X_pi

        if n_basis is None or int(n_basis) <= 0 or int(n_basis) >= source_pool.shape[0]:
            X_basis = source_pool
        else:
            gen = torch.Generator(device="cpu")
            if basis_seed is not None:
                gen.manual_seed(int(basis_seed))
            idx = torch.randperm(source_pool.shape[0], generator=gen)[: int(n_basis)]
            X_basis = source_pool.index_select(0, idx.to(device))

        self._X_basis = X_basis
        b = X_basis.shape[0]
        n_de = X_beta.shape[0]
        n_nu = X_pi.shape[0]

        K_basis_de = self.kernel_func(X_basis, X_beta, **self.kernel_kwargs)          # (b, n_de)
        K_basis_nu = self.kernel_func(X_basis, X_pi, **self.kernel_kwargs)            # (b, n_nu)

        # RuLSIF: estimate the alpha-relative ratio
        #   r_alpha = p_pi / (alpha p_pi + (1-alpha) p_beta),  bounded above by 1/alpha.
        # alpha_mix == 0 recovers plain uLSIF (the original, unbounded estimator).
        am = float(getattr(self, "alpha_mix", 0.0))
        H_de = (K_basis_de @ K_basis_de.transpose(0, 1)) / float(n_de)                # (b, b)
        if am > 0.0:
            H_nu = (K_basis_nu @ K_basis_nu.transpose(0, 1)) / float(n_nu)            # (b, b)
            H = am * H_nu + (1.0 - am) * H_de
        else:
            H = H_de
        h = K_basis_nu.mean(dim=1)                                                    # (b,)

        I_b = torch.eye(b, device=device, dtype=dtype)
        A_mat = H + self.lambda_reg * I_b
        jitter = 1e-8 * torch.trace(A_mat).clamp_min(1.0) / float(b)
        try:
            L = torch.linalg.cholesky(A_mat + jitter * I_b)
            alpha = torch.cholesky_solve(h.unsqueeze(1), L).squeeze(1)
        except RuntimeError:
            alpha = torch.linalg.solve(A_mat + jitter * I_b, h)

        alpha_unconstrained = alpha
        alpha_unconstrained_neg_frac = float((alpha_unconstrained < 0).double().mean().detach().cpu().item())
        alpha_unconstrained_min = float(alpha_unconstrained.min().detach().cpu().item())
        alpha_unconstrained_max = float(alpha_unconstrained.max().detach().cpu().item())

        if bool(nonnegative_alpha):
            alpha = torch.clamp(alpha, min=0.0)

        eta_pre_calibration = (K_basis_de.transpose(0, 1) @ alpha.reshape(-1, 1)).squeeze(1)
        eta_pre_mean = eta_pre_calibration.mean()
        mean_calibration_scale = 1.0
        mean_calibration_applied = False
        mean_calibration_skipped = None
        if bool(calibrate_mean):
            if am != 0.0:
                mean_calibration_skipped = "alpha_relative_ratio_has_no_unit_mean_identity"
            elif torch.isfinite(eta_pre_mean) and eta_pre_mean > torch.finfo(dtype).eps:
                mean_calibration_scale = float((1.0 / eta_pre_mean).detach().cpu().item())
                alpha = alpha * mean_calibration_scale
                mean_calibration_applied = True
            else:
                mean_calibration_skipped = "nonpositive_or_nonfinite_denominator_mean"

        self.alpha = alpha.reshape(-1, 1)

        with torch.no_grad():
            eta_hat = (K_basis_de.transpose(0, 1) @ self.alpha).squeeze(1)
            neg = (eta_hat < 0).float().mean().item()
            sw = torch.clamp(eta_hat, min=0.0).sum()
            ess = 0.0
            if float(sw.detach().cpu()) > 0.0:
                ess = float(((sw * sw) / (torch.clamp(eta_hat, min=0.0).pow(2).sum() + 1e-12)).detach().cpu().item())
            self.fit_diagnostics = {
                "estimator": "direct_uLSIF",
                "ratio_type": "ordinary_target_to_logged_data_direct" if am == 0.0 else "alpha_relative_target_to_logged_data_direct",
                "fits_behavior_policy_model": False,
                "n_denominator": int(n_de),
                "n_numerator": int(n_nu),
                "target_sample_multiplier": int(target_sample_multiplier),
                "basis_size": int(b),
                "basis_source": str(basis_source_l),
                "basis_seed": None if basis_seed is None else int(basis_seed),
                "alpha_mix": float(am),
                "lambda_reg": float(self.lambda_reg),
                "nonnegative_alpha": bool(nonnegative_alpha),
                "alpha_unconstrained_neg_frac": alpha_unconstrained_neg_frac,
                "alpha_unconstrained_min": alpha_unconstrained_min,
                "alpha_unconstrained_max": alpha_unconstrained_max,
                "alpha_min": float(self.alpha.min().detach().cpu().item()),
                "alpha_max": float(self.alpha.max().detach().cpu().item()),
                "calibrate_mean": bool(calibrate_mean),
                "mean_calibration_applied": bool(mean_calibration_applied),
                "mean_calibration_scale": float(mean_calibration_scale),
                "mean_calibration_skipped": mean_calibration_skipped,
                "eta_pre_calibration_mean": float(eta_pre_calibration.mean().detach().cpu().item()),
                "eta_pre_calibration_min": float(eta_pre_calibration.min().detach().cpu().item()),
                "eta_pre_calibration_max": float(eta_pre_calibration.max().detach().cpu().item()),
                "eta_hat_mean": float(eta_hat.mean().detach().cpu().item()),
                "eta_hat_min": float(eta_hat.min().detach().cpu().item()),
                "eta_hat_max": float(eta_hat.max().detach().cpu().item()),
                "eta_hat_neg_frac": float(neg),
                "eta_hat_ess": float(ess),
            }
            print(
                "[uLSIF alpha_mix={:.2f}] basis={}/{} ({}); numerator={} target_draws/state={}; "
                "nonneg_alpha={}; mean_calib={} scale={:.3g}; eta_hat mean={:.3e} min={:.3e} max={:.3e} neg%={:.2f}".format(
                    am,
                    b,
                    source_pool.shape[0],
                    basis_source_l,
                    n_nu,
                    target_sample_multiplier,
                    bool(nonnegative_alpha),
                    bool(mean_calibration_applied),
                    mean_calibration_scale,
                    eta_hat.mean().item(),
                    eta_hat.min().item(),
                    eta_hat.max().item(),
                    100.0 * neg,
                )
            )

        if plot:
            import matplotlib.pyplot as plt
            import seaborn as sns

            Path("plots").mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                eta_train = (K_basis_de.transpose(0, 1) @ self.alpha).squeeze(1)
            plt.figure()
            sns.histplot(eta_train.detach().cpu().numpy(), kde=True, bins=30)
            plt.title(f"uLSIF eta on denominator (target={target_p_choice})")
            plt.savefig(f"./plots/eta_uLSIF_{S.shape[1]}_{A.shape[1]}.png")
            plt.close()

            plt.figure()
            sns.histplot(self.alpha.squeeze(1).detach().cpu().numpy(), kde=True, bins=30)
            plt.title(f"uLSIF alpha (target={target_p_choice})")
            plt.savefig(f"./plots/alpha_uLSIF_{S.shape[1]}_{A.shape[1]}.png")
            plt.close()

        return self.alpha

    # ------------------------------------------------------------------
    def predict(self, S_new: torch.Tensor, A_new: torch.Tensor) -> torch.Tensor:
        if self.alpha is None or self._X_basis is None:
            raise RuntimeError("Call fit() first.")
        if S_new.ndim != 2 or A_new.ndim != 2:
            raise ValueError("S_new and A_new must be 2D tensors.")
        X_new = torch.cat([S_new, A_new], dim=1).to(device=self._X_basis.device, dtype=self._X_basis.dtype)
        K_basis_new = self.kernel_func(self._X_basis, X_new, **self.kernel_kwargs)    # (b, n_new)
        return (K_basis_new.transpose(0, 1) @ self.alpha).reshape(-1, 1)              # (n_new, 1)

    # ------------------------------------------------------------------
    def compute_ess(self, S: torch.Tensor, A: torch.Tensor) -> float:
        """Effective sample size for the predicted (nonnegative) weights."""
        eta = self.predict(S, A).reshape(-1)
        w = torch.clamp(eta, min=0.0)
        sw = w.sum()
        if sw <= 0:
            return 0.0
        return float(((sw * sw) / (w.pow(2).sum() + 1e-12)).item())
