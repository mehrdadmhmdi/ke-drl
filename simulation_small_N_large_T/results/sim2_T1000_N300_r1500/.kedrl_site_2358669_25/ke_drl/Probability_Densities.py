import torch
from torch import Tensor
from typing import Optional, Dict, Any


class Probability_Densities:
    """
    Torch-based conditional probability density/sampler class.

    Supported choices
    -----------------
    - beta     : parameters alpha(s), beta(s) > 0
    - gaussian : latent Normal(mean(s), std(s))
    - uniform  : lower(s), upper(s)
    - logistic : latent Logistic(loc(s), scale(s))

    Important
    ---------
    For 'gaussian' and 'logistic', `apply_sigmoid=True` means:
      1) sample a latent variable Z on R,
      2) map to X = sigmoid(Z) in (0,1).

    In that case, `pdf(...)` returns the *correct transformed density*
    using the change-of-variables formula, not sigmoid(raw_pdf).

    For 'uniform', `apply_sigmoid=True` means the interval endpoints are
    squashed first, i.e. lower = sigmoid(lower_latent),
    upper = sigmoid(upper_latent), and then a Uniform(lower, upper)
    distribution is used on the action scale.

    For 'beta', the support is already (0,1), so `apply_sigmoid` is ignored.
    """

    def __init__(self, **kwargs: Dict[str, Any]):
        self.distributions: Dict[str, Dict[str, Any]] = {}

        supported_pdfs = ["beta", "gaussian", "uniform", "logistic"]
        required_params = {
            "beta": ["theta_alpha", "theta_beta"],
            "gaussian": ["theta_mean", "theta_std"],
            "uniform": ["theta_lower", "theta_upper"],
            "logistic": ["theta_loc", "theta_scale"],
        }

        for pdf_choice in supported_pdfs:
            if pdf_choice not in kwargs:
                continue

            cfg = dict(kwargs[pdf_choice])
            required = required_params[pdf_choice]
            missing = [p for p in required if p not in cfg]
            if missing:
                continue

            epsilon_params = {}
            for theta_name in required:
                suffix = theta_name.replace("theta_", "")
                epsilon_params[suffix] = cfg.get(f"epsilon_{suffix}", 0.0)

            cleaned_params = {
                k: v
                for k, v in cfg.items()
                if not k.startswith("epsilon_")
                and k not in {"apply_sigmoid", "log_scale_min", "log_scale_max"}
            }

            self.distributions[pdf_choice] = {
                "params": cleaned_params,
                "epsilon": epsilon_params,
                "apply_sigmoid": bool(cfg.get("apply_sigmoid", False)),
                "log_scale_min": float(cfg.get("log_scale_min", -12.0)),
                "log_scale_max": float(cfg.get("log_scale_max", 8.0)),
            }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_tensor(x, like: Optional[Tensor] = None) -> Tensor:
        if isinstance(x, Tensor):
            if like is not None and (x.device != like.device or x.dtype != like.dtype):
                return x.to(device=like.device, dtype=like.dtype)
            return x
        if like is not None:
            return torch.as_tensor(x, dtype=like.dtype, device=like.device)
        return torch.as_tensor(x, dtype=torch.float64)

    @staticmethod
    def _lin(s: Tensor, theta: Tensor) -> Tensor:
        # (d,) @ (d,) -> ()
        # (n,d) @ (d,) -> (n,)
        # (n,d) @ (d,k) -> (n,k)
        return s.matmul(theta)

    @staticmethod
    def _normal_logpdf(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        var = std * std
        return -0.5 * (torch.log(2.0 * torch.pi * var) + (x - mean) ** 2 / var)

    @staticmethod
    def _normal_pdf(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        return torch.exp(Probability_Densities._normal_logpdf(x, mean, std))

    @staticmethod
    def _uniform_pdf(x: Tensor, low: Tensor, high: Tensor) -> Tensor:
        width = high - low
        in_support = (x >= low) & (x <= high)
        return torch.where(in_support & (width > 0), 1.0 / width, torch.zeros_like(x))

    @staticmethod
    def _logistic_logpdf(x: Tensor, loc: Tensor, scale: Tensor) -> Tensor:
        z = (x - loc) / scale
        return -z - torch.log(scale) - 2.0 * torch.nn.functional.softplus(-z)

    @staticmethod
    def _logistic_pdf(x: Tensor, loc: Tensor, scale: Tensor) -> Tensor:
        return torch.exp(Probability_Densities._logistic_logpdf(x, loc, scale))

    @staticmethod
    def _clamp_log_scale(log_scale: Tensor, lo: float, hi: float) -> Tensor:
        return torch.clamp(log_scale, min=lo, max=hi)

    @staticmethod
    def _safe_exp(log_x: Tensor, min_value: float = 1e-8) -> Tensor:
        return torch.exp(log_x).clamp(min=min_value)

    @staticmethod
    def _safe_unit_interval(x: Tensor) -> Tensor:
        eps = torch.finfo(x.dtype).eps
        return torch.clamp(x, eps, 1.0 - eps)

    @staticmethod
    def _logit(x: Tensor) -> Tensor:
        x = Probability_Densities._safe_unit_interval(x)
        return torch.log(x) - torch.log1p(-x)

    @staticmethod
    def _broadcast3(a: Tensor, b: Tensor, c: Tensor):
        return torch.broadcast_tensors(a, b, c)

    @staticmethod
    def _broadcast2(a: Tensor, b: Tensor):
        return torch.broadcast_tensors(a, b)

    @staticmethod
    def _zero_like_input(x):
        if isinstance(x, Tensor):
            return torch.zeros_like(x)
        return 0.0

    def _resolve_apply_sigmoid(self, dist_info: Dict[str, Any], apply_sigmoid: Optional[bool]) -> bool:
        return dist_info["apply_sigmoid"] if apply_sigmoid is None else bool(apply_sigmoid)

    def _get_scale_bounds(self, dist_info: Dict[str, Any]):
        return dist_info["log_scale_min"], dist_info["log_scale_max"]

    def _param_tensor(self, params: Dict[str, Any], name: str, like: Tensor) -> Tensor:
        return self._to_tensor(params[name], like=like)

    def _compute_positive_param(
        self,
        s: Tensor,
        theta: Tensor,
        epsilon,
        log_min: float,
        log_max: float,
        min_value: float = 1e-8,
    ) -> Tensor:
        log_param = self._lin(s, theta) + self._to_tensor(epsilon, like=s)
        log_param = self._clamp_log_scale(log_param, log_min, log_max)
        return self._safe_exp(log_param, min_value=min_value)

    def _transformed_density_via_sigmoid(self, latent_pdf: Tensor, x: Tensor) -> Tensor:
        x_safe = self._safe_unit_interval(x)
        jac = x_safe * (1.0 - x_safe)
        dens = latent_pdf / jac
        inside = (x > 0.0) & (x < 1.0)
        return torch.where(inside, dens, torch.zeros_like(dens))

    # ------------------------------------------------------------------
    # pdf
    # ------------------------------------------------------------------
    def pdf(self, pdf_choice, x, s, apply_sigmoid: Optional[bool] = None):
        if pdf_choice not in self.distributions:
            return self._zero_like_input(x)

        dist_info = self.distributions[pdf_choice]
        params = dist_info["params"]
        epsilon = dist_info["epsilon"]
        use_sigmoid = self._resolve_apply_sigmoid(dist_info, apply_sigmoid)
        log_scale_min, log_scale_max = self._get_scale_bounds(dist_info)

        try:
            s = self._to_tensor(s)
            x = self._to_tensor(x, like=s)

            def P(name: str) -> Tensor:
                return self._param_tensor(params, name, like=s)

            if pdf_choice == "beta":
                # Beta is already on (0,1); apply_sigmoid is intentionally ignored.
                alpha = self._compute_positive_param(
                    s, P("theta_alpha"), epsilon["alpha"], log_scale_min, log_scale_max, min_value=1e-6
                )
                beta_val = self._compute_positive_param(
                    s, P("theta_beta"), epsilon["beta"], log_scale_min, log_scale_max, min_value=1e-6
                )

                x_safe, alpha_b, beta_b = self._broadcast3(self._safe_unit_interval(x), alpha, beta_val)
                log_pdf = (
                    (alpha_b - 1.0) * torch.log(x_safe)
                    + (beta_b - 1.0) * torch.log1p(-x_safe)
                    - (torch.lgamma(alpha_b) + torch.lgamma(beta_b) - torch.lgamma(alpha_b + beta_b))
                )
                dens = torch.exp(log_pdf)
                inside = (x > 0.0) & (x < 1.0)
                return torch.where(inside, dens, torch.zeros_like(dens))

            if pdf_choice == "gaussian":
                mean = self._lin(s, P("theta_mean")) + self._to_tensor(epsilon["mean"], like=s)
                std = self._compute_positive_param(
                    s, P("theta_std"), epsilon["std"], log_scale_min, log_scale_max, min_value=1e-8
                )

                if use_sigmoid:
                    z = self._logit(x)
                    z_b, mean_b, std_b = self._broadcast3(z, mean, std)
                    latent_pdf = self._normal_pdf(z_b, mean_b, std_b)
                    return self._transformed_density_via_sigmoid(latent_pdf, x=self._to_tensor(x, like=z_b))

                x_b, mean_b, std_b = self._broadcast3(x, mean, std)
                return self._normal_pdf(x_b, mean_b, std_b)

            if pdf_choice == "uniform":
                lower = self._lin(s, P("theta_lower")) + self._to_tensor(epsilon["lower"], like=s)
                upper = self._lin(s, P("theta_upper")) + self._to_tensor(epsilon["upper"], like=s)
                upper = torch.where(upper <= lower, lower + 1.0, upper)

                if use_sigmoid:
                    lower = torch.sigmoid(lower)
                    upper = torch.sigmoid(upper)

                x_b, lower_b, upper_b = self._broadcast3(x, lower, upper)
                return self._uniform_pdf(x_b, lower_b, upper_b)

            if pdf_choice == "logistic":
                loc = self._lin(s, P("theta_loc")) + self._to_tensor(epsilon["loc"], like=s)
                loc = torch.clamp(loc, -50.0, 50.0)
                scale = self._compute_positive_param(
                    s, P("theta_scale"), epsilon["scale"], log_scale_min, log_scale_max, min_value=1e-8
                )

                if use_sigmoid:
                    z = self._logit(x)
                    z_b, loc_b, scale_b = self._broadcast3(z, loc, scale)
                    latent_pdf = self._logistic_pdf(z_b, loc_b, scale_b)
                    return self._transformed_density_via_sigmoid(latent_pdf, x=self._to_tensor(x, like=z_b))

                x_b, loc_b, scale_b = self._broadcast3(x, loc, scale)
                return self._logistic_pdf(x_b, loc_b, scale_b)

            return self._to_tensor(0.0, like=s)

        except Exception as e:
            print("[pdf error]", pdf_choice, "state:", s, "x:", x, "::", repr(e))
            return self._zero_like_input(x)

    # ------------------------------------------------------------------
    # sample
    # ------------------------------------------------------------------
    def sample_pdf(self, pdf_choice, s, apply_sigmoid: Optional[bool] = None):
        if pdf_choice not in self.distributions:
            return None

        dist_info = self.distributions[pdf_choice]
        params = dist_info["params"]
        epsilon = dist_info["epsilon"]
        use_sigmoid = self._resolve_apply_sigmoid(dist_info, apply_sigmoid)
        log_scale_min, log_scale_max = self._get_scale_bounds(dist_info)

        try:
            s = self._to_tensor(s)

            def P(name: str) -> Tensor:
                return self._param_tensor(params, name, like=s)

            if pdf_choice == "beta":
                # Beta is already on (0,1); apply_sigmoid is intentionally ignored.
                alpha = self._compute_positive_param(
                    s, P("theta_alpha"), epsilon["alpha"], log_scale_min, log_scale_max, min_value=1e-6
                )
                beta_val = self._compute_positive_param(
                    s, P("theta_beta"), epsilon["beta"], log_scale_min, log_scale_max, min_value=1e-6
                )
                dist = torch.distributions.Beta(alpha, beta_val)
                return dist.sample()

            if pdf_choice == "gaussian":
                mean = self._lin(s, P("theta_mean")) + self._to_tensor(epsilon["mean"], like=s)
                std = self._compute_positive_param(
                    s, P("theta_std"), epsilon["std"], log_scale_min, log_scale_max, min_value=1e-8
                )
                sample = torch.distributions.Normal(mean, std).sample()
                return torch.sigmoid(sample) if use_sigmoid else sample

            if pdf_choice == "uniform":
                lower = self._lin(s, P("theta_lower")) + self._to_tensor(epsilon["lower"], like=s)
                upper = self._lin(s, P("theta_upper")) + self._to_tensor(epsilon["upper"], like=s)
                upper = torch.where(upper <= lower, lower + 1.0, upper)
                if use_sigmoid:
                    lower = torch.sigmoid(lower)
                    upper = torch.sigmoid(upper)
                u = torch.rand_like(lower)
                return lower + (upper - lower) * u

            if pdf_choice == "logistic":
                loc = self._lin(s, P("theta_loc")) + self._to_tensor(epsilon["loc"], like=s)
                loc = torch.clamp(loc, -50.0, 50.0)
                scale = self._compute_positive_param(
                    s, P("theta_scale"), epsilon["scale"], log_scale_min, log_scale_max, min_value=1e-8
                )
                u = torch.clamp(torch.rand_like(loc), 1e-6, 1.0 - 1e-6)
                logistic_sample = loc + scale * torch.log(u / (1.0 - u))
                return torch.sigmoid(logistic_sample) if use_sigmoid else logistic_sample

            return None

        except Exception as e:
            print("[sample_pdf error]", pdf_choice, "state:", s, "::", repr(e))
            return None
