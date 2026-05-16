import math
from typing import Optional

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_


class RKDRL_Optimizer:
    """Optimizer for the global KE-DRL coefficient matrix.

    The revised estimator fits one policy-specific matrix B over a conditioning
    set X_star by minimizing the average Bellman embedding residual

        u_l^T K_Z u_l - 2 u_l^T H_l v_l + v_l^T G_l v_l,

    where u_l = B^T k_l and v_l = B^T Phi_l, plus the state-action RKHS
    ridge lambda_B tr(B^T K_X B).
    """

    def __init__(self, device=None, dtype=torch.float64):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.last_diagnostics: dict[str, list[float]] = {}

    @staticmethod
    def _ensure_target_matrix(name: str, value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 1:
            return value.unsqueeze(1)
        if value.ndim != 2:
            raise ValueError(f"{name} must have shape (N,), (N,1), or (N,L).")
        return value

    @staticmethod
    def _ensure_operator_stack(name: str, value: torch.Tensor, L: int) -> torch.Tensor:
        if value.ndim == 2:
            value = value.unsqueeze(0)
        if value.ndim != 3:
            raise ValueError(f"{name} must have shape (m,m) or (L,m,m).")
        if value.shape[0] == 1 and L != 1:
            value = value.expand(L, -1, -1)
        if value.shape[0] != L:
            raise ValueError(f"{name} has {value.shape[0]} target slices but expected {L}.")
        return value

    def initial_B(self, n: int, m: int, scale: float = 1e-3, seed: Optional[int] = None) -> torch.Tensor:
        if seed is not None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed)
            B = torch.randn((n, m), generator=gen, dtype=self.dtype)
            return (scale * B).to(self.dev)
        return scale * torch.randn((n, m), device=self.dev, dtype=self.dtype)

    def closed_form_B0(self, k_sa, Phi, K_Zpi, H_mat, G_mat):
        """Compatibility initializer for older callers.

        The revised global objective is fitted by gradient descent, so this no
        longer solves a closed-form system. It returns a small nonzero matrix
        with the correct shape.
        """
        k_mat = self._ensure_target_matrix("k_sa", torch.as_tensor(k_sa))
        m = torch.as_tensor(K_Zpi).shape[0]
        return self.initial_B(k_mat.shape[0], m)

    @staticmethod
    def _batch_residuals(B, k_batch, phi_batch, K_Z, H_batch, G_batch):
        # k_batch, phi_batch: (N,b); H_batch/G_batch: (b,m,m)
        u = k_batch.transpose(0, 1) @ B
        v = phi_batch.transpose(0, 1) @ B
        term1 = torch.einsum("bi,ij,bj->b", u, K_Z, u)
        term2 = -2.0 * torch.einsum("bi,bij,bj->b", u, H_batch, v)
        term3 = torch.einsum("bi,bij,bj->b", v, G_batch, v)
        return term1 + term2 + term3

    @staticmethod
    def _weighted_average(values: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        if weights is None:
            return values.mean()
        return torch.sum(values * weights)

    @staticmethod
    def _normal_target_weights(
        target_weights: Optional[torch.Tensor],
        L: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if target_weights is None:
            return None
        w = target_weights.to(device=device, dtype=dtype).reshape(-1)
        if w.numel() != L:
            raise ValueError(f"target_weights must have length {L}, got {w.numel()}.")
        if not torch.isfinite(w).all():
            raise ValueError("target_weights must be finite.")
        if torch.any(w < 0):
            raise ValueError("target_weights must be nonnegative.")
        total = w.sum()
        if total <= 0:
            raise ValueError("target_weights must have positive sum.")
        return w / total

    @staticmethod
    def _select_weights(weights: Optional[torch.Tensor], idx: torch.Tensor) -> Optional[torch.Tensor]:
        if weights is None:
            return None
        w = weights.index_select(0, idx)
        total = w.sum()
        if total <= 0:
            return torch.full_like(w, 1.0 / float(w.numel()))
        return w / total

    @staticmethod
    def _coefficient_vectors(B, k_batch) -> torch.Tensor:
        return k_batch.transpose(0, 1) @ B

    @classmethod
    def _mass_anchor_penalty(
        cls,
        B,
        k_batch,
        target_mass: float,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Penalize target-point coefficient masses away from a probability mass."""
        u = cls._coefficient_vectors(B, k_batch)
        mass = u.sum(dim=1)
        return cls._weighted_average((mass - float(target_mass)) ** 2, weights)

    @classmethod
    def _negativity_penalty(
        cls,
        B,
        k_batch,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Optional penalty for negative finite-grid coefficients."""
        u = cls._coefficient_vectors(B, k_batch)
        neg_sq = torch.clamp(-u, min=0.0).pow(2).sum(dim=1)
        return cls._weighted_average(neg_sq, weights)

    @staticmethod
    def _rkhs_ridge(B, K_X) -> torch.Tensor:
        return torch.sum(B * (K_X @ B))

    @staticmethod
    def _project_frobenius_ball_(B: torch.Tensor, max_norm: Optional[float]) -> None:
        if max_norm is None or max_norm <= 0:
            return
        norm = torch.linalg.vector_norm(B)
        if torch.isfinite(norm) and norm > float(max_norm):
            B.mul_(float(max_norm) / (norm + torch.finfo(B.dtype).eps))

    def optimize(
            self,
            k_sa: torch.Tensor,
            K_Zpi: torch.Tensor,
            H_mat: torch.Tensor,
            Phi: torch.Tensor,
            G_mat: torch.Tensor,
            *,
            K_X: Optional[torch.Tensor] = None,
            lambda_B: float = 0.0,
            target_batch_size: Optional[int] = None,
            initial_B: Optional[torch.Tensor] = None,
            lr: float = 1e-3,
            weight_decay: float = 0.0,
            num_steps: int = 2000,
            random_seed: Optional[int] = None,
            initial_scale: float = 1e-3,
            target_weights: Optional[torch.Tensor] = None,
            # Legacy keyword arguments kept for package/API compatibility.
            FP_penalty_lambda: float = 0.0,
            use_low_rank: bool = False,
            rank: Optional[int] = None,
            ortho_lambda: float = 0.0,
            B_positive: bool = False,
            fixed_point_constraint: bool = False,
            exact_projection: bool = False,
            B_conv: bool = False,
            Sum_one_W: bool = False,
            NonNeg_W: bool = False,
            mass_anchor_lambda: float = 0.0,
            target_mass: float = 1.0,
            negativity_penalty_lambda: float = 0.0,
            max_B_norm: Optional[float] = None,
            B_ridge_penalty: bool = False,
            verbose: bool = True,
    ) -> tuple[torch.Tensor, list[float], list[float]]:

        dev, dtype = self.dev, self.dtype
        k_mat = self._ensure_target_matrix("k_sa", k_sa.to(dev, dtype))
        phi_mat = self._ensure_target_matrix("Phi", Phi.to(dev, dtype))
        if k_mat.shape != phi_mat.shape:
            raise ValueError(f"k_sa and Phi must have the same shape; got {k_mat.shape} and {phi_mat.shape}.")

        N, L = k_mat.shape
        K_Z = K_Zpi.to(dev, dtype)
        H_stack = self._ensure_operator_stack("H_mat", H_mat.to(dev, dtype), L)
        G_stack = self._ensure_operator_stack("G_mat", G_mat.to(dev, dtype), L)
        m = K_Z.shape[0]
        if K_Z.shape != (m, m):
            raise ValueError("K_Zpi must be square.")
        if H_stack.shape[1:] != (m, m) or G_stack.shape[1:] != (m, m):
            raise ValueError("H_mat and G_mat slices must match K_Zpi shape.")

        if K_X is None:
            K_X = torch.eye(N, device=dev, dtype=dtype)
        else:
            K_X = K_X.to(dev, dtype)
        if K_X.shape != (N, N):
            raise ValueError(f"K_X must have shape {(N, N)}, got {tuple(K_X.shape)}.")
        weights_full = self._normal_target_weights(target_weights, L, device=dev, dtype=dtype)

        if initial_B is None:
            B0 = self.initial_B(N, m, scale=initial_scale, seed=random_seed)
        else:
            B0 = initial_B.to(dev, dtype)
        if B0.shape != (N, m):
            raise ValueError(f"initial_B must have shape {(N, m)}, got {tuple(B0.shape)}.")

        legacy_active = any([
            fixed_point_constraint, exact_projection, B_conv, Sum_one_W, NonNeg_W,
            B_positive, B_ridge_penalty, ortho_lambda > 0.0, use_low_rank,
        ])
        if verbose and legacy_active:
            print(
                "Ignoring legacy estimator constraints/penalties in the revised global objective: "
                "fixed_point, projection, simplex/nonnegativity, low-rank, and ad hoc ridge. "
                "The mass anchor is implemented separately when mass_anchor_lambda > 0."
            )
        if verbose and negativity_penalty_lambda > 0.0:
            print(f"Using finite-grid negativity penalty: lambda_neg={float(negativity_penalty_lambda):.3g}")
        if verbose and max_B_norm is not None and max_B_norm > 0:
            print(f"Projecting B onto Frobenius ball with radius {float(max_B_norm):.3g}")

        if target_batch_size is None or target_batch_size <= 0 or target_batch_size > L:
            target_batch_size = L

        B = torch.nn.Parameter(B0.clone())
        with torch.no_grad():
            self._project_frobenius_ball_(B, max_B_norm)
        opt = optim.AdamW([B], lr=lr, weight_decay=weight_decay)
        sched = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=20, threshold=1e-4, min_lr=1e-8
        )

        history_obj: list[float] = []
        history_be: list[float] = []
        history_components: dict[str, list[float]] = {
            "objective": [],
            "bellman": [],
            "rkhs_ridge": [],
            "mass": [],
            "negativity": [],
            "B_norm": [],
        }
        eps = torch.finfo(dtype).eps

        for step in range(1, int(num_steps) + 1):
            opt.zero_grad()
            if target_batch_size == L:
                idx = torch.arange(L, device=dev)
            else:
                idx = torch.randperm(L, device=dev)[:target_batch_size]
            weights_batch = self._select_weights(weights_full, idx)

            residuals = self._batch_residuals(
                B,
                k_mat[:, idx],
                phi_mat[:, idx],
                K_Z,
                H_stack.index_select(0, idx),
                G_stack.index_select(0, idx),
            )
            bellman_loss = self._weighted_average(residuals, weights_batch)
            ridge = float(lambda_B) * self._rkhs_ridge(B, K_X)
            mass_penalty = (
                float(mass_anchor_lambda)
                * self._mass_anchor_penalty(B, k_mat[:, idx], target_mass, weights_batch)
                if mass_anchor_lambda > 0.0
                else torch.zeros((), device=dev, dtype=dtype)
            )
            neg_penalty = (
                float(negativity_penalty_lambda)
                * self._negativity_penalty(B, k_mat[:, idx], weights_batch)
                if negativity_penalty_lambda > 0.0
                else torch.zeros((), device=dev, dtype=dtype)
            )
            loss = bellman_loss + ridge + mass_penalty + neg_penalty

            loss.backward()
            grad_norm = clip_grad_norm_([B], max_norm=1e2).item()
            opt.step()
            with torch.no_grad():
                self._project_frobenius_ball_(B, max_B_norm)
            sched.step(float(loss.detach().cpu()))

            with torch.no_grad():
                full_residuals = self._batch_residuals(B, k_mat, phi_mat, K_Z, H_stack, G_stack)
                full_bellman_raw = self._weighted_average(full_residuals, weights_full)
                full_bellman = full_bellman_raw.clamp_min(eps)
                full_ridge = float(lambda_B) * self._rkhs_ridge(B, K_X)
                full_mass = (
                    float(mass_anchor_lambda)
                    * self._mass_anchor_penalty(B, k_mat, target_mass, weights_full)
                    if mass_anchor_lambda > 0.0
                    else torch.zeros((), device=dev, dtype=dtype)
                )
                full_neg = (
                    float(negativity_penalty_lambda)
                    * self._negativity_penalty(B, k_mat, weights_full)
                    if negativity_penalty_lambda > 0.0
                    else torch.zeros((), device=dev, dtype=dtype)
                )
                full_loss_raw = full_bellman_raw + full_ridge + full_mass + full_neg
                full_loss = full_loss_raw.clamp_min(eps)
                history_obj.append(math.log(float(full_loss.detach().cpu())))
                history_be.append(math.log(float(torch.sqrt(full_bellman).detach().cpu())))
                history_components["objective"].append(float(full_loss_raw.detach().cpu()))
                history_components["bellman"].append(float(full_bellman_raw.detach().cpu()))
                history_components["rkhs_ridge"].append(float(full_ridge.detach().cpu()))
                history_components["mass"].append(float(full_mass.detach().cpu()))
                history_components["negativity"].append(float(full_neg.detach().cpu()))
                history_components["B_norm"].append(float(torch.linalg.vector_norm(B).detach().cpu()))

            if verbose and (step == 1 or step % max(1, int(num_steps) // 10) == 0):
                print(
                    f"Iter {step}/{num_steps} | log_obj={history_obj[-1]:.3e} "
                    f"| log_Bellman_root={history_be[-1]:.3e} | "
                    f"mass={history_components['mass'][-1]:.2e} "
                    f"| neg={history_components['negativity'][-1]:.2e} "
                    f"| grad={grad_norm:.2e}"
                )
            if grad_norm < 1e-8:
                if verbose:
                    print(f"Converged at step {step}: grad={grad_norm:.2e}")
                break

        self.last_diagnostics = history_components
        return B.detach().cpu(), history_obj, history_be
