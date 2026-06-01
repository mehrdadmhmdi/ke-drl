# ke-drl

Offline multi-dimensional distributional reinforcement learning via RKHS
kernel mean embeddings.

## Installation

Requires Python 3.10 or newer.

```bash
python -m pip install "git+https://github.com/mehrdadmhmdi/ke-drl.git"
```

For local development:

```bash
git clone https://github.com/mehrdadmhmdi/ke-drl.git
cd ke-drl
python -m pip install -e .
```

## Estimator

The package fits one policy-specific global coefficient matrix `B` for a fixed
target policy. For historical state-action inputs `X_train`, a chosen
conditioning basis `U={u_1,...,u_L}`, and a return grid `Z_grid`, the fitted
conditional return embedding at a query input `x` is

```text
mu_hat(x) = sum_j omega_j(x; B) k_Z(z_j, .),
omega(x; B) = B.T psi_L(x),     psi_L(x) = k_X(U, x),
B in R^{L x m_grid}.
```

The target-point set `X_star` is used only to choose where the Bellman residual
is enforced during training. It is not a separate optimization for each
scientific evaluation point.

The implemented global objective follows `rz_new_version.tex`:

```text
mean_l [
  u_l.T K_Z u_l - 2 u_l.T H_l v_l + v_l.T G_l v_l
]
+ lambda_B tr(B.T K_U B)
+ lambda_mass mean_l (1.T u_l - target_mass)^2
+ lambda_neg mean_l ||negative_part(u_l)||_2^2
```

where `u_l = B.T psi_L(x_l)` and
`v_l = B.T sum_i Gamma_i(x_l) eta_i psi_L(x_i^+)`. The mass anchor is important:
without it, the Bellman residual and RKHS ridge are homogeneous in `B`, so
`B = 0` can become an artificial minimizer. The negativity penalty is optional.

## Main Entry Point

```python
from ke_drl import estimate_embedding

B_hat, history_obj, history_be, pre = estimate_embedding(
    s0=s0,
    s1=s1,
    a0=a0,
    a1=a1,
    s_star=s_star,
    a_star=a_star,
    r=r0,
    target_p_choice="logistic",
    target_p_params={"logistic": {...}},
    nu=3.5,
    length_scale=1.0,
    sigma=0.7,
    gamma_val=0.8,
    lambda_reg=1e-3,
    lambda_B=1e-3,
    mean_embedding_basis_size=1500,
    mean_embedding_basis_method="kmeans",
    mass_anchor_lambda=1.0,
    negativity_penalty_lambda=0.0,
)
```

`pre` contains the fitted grid, state-action Gram matrices, stacked
target-point operators, stabilized continuation density ratios, and optimizer
diagnostics.
