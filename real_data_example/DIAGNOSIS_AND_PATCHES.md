# Mean-embedding density recovery: diagnosis and patches

Two symptoms reported: (1) the recovered click distribution does **not** put more mass on
large click counts for the click-focused policy vs the revenue-focused policy, and
(2) the recovery is **unstable** across runs/specs.

## What the evidence shows

From the saved runs in `results/6.11.2026/...`:

- **Recovered separation is wrong, not just absent.** Across 18 paired configs,
  `E[clicks]_clk − E[clicks]_rev` is **negative in 17/18** (median −0.14). The
  click policy is recovered as having *fewer* clicks.
- **The embedding β barely carries the policy signal.** The two policies' β are
  nearly uncorrelated (median corr ≈ **0.08**, relative L2 diff ≈ **1.4**), i.e. they
  differ as noise, not signal.
- **The importance ratio η is pathological** (from logs): uLSIF returns negative
  ratios (min −83 … −382), ESS is **846–2455 / 10000 (8–25%)**, and after clipping
  negatives to 0 there is no renormalisation, so the used mean ranges 0.95–12
  (a density ratio should average ≈ 1).
- **The recovery is ill-posed (unidentified).** `relative_rkhs_error` median ≈ **0.48**
  (up to 0.96); effective atom count (1/HHI) swings from **1 to 366**; `w_max` from
  0.0075 to 0.9999 on neighbouring hyperparameters; click recovery converged in only
  **13/21** runs.
- **Decisive localisation test** (`downstream_recovery_diagnostic.py`): re-recovering the
  click density from the *same saved β* with three different methods gives
  `frac_correct(sep>0)` = 0.38 (current/uniform-anchored), **0.25** (well-posed RKHS
  projection), 0.75 (w ∝ max(β,0)) — and the sign **flips with the method** on the same β
  (cfg3, cfg5, cfg7). No downstream method reliably fixes separation.

**Conclusion / ordering:** the dominant blocker is **upstream** — the density ratio η is
unstable and noise-dominated, so β does not robustly encode the policy's click behaviour.
The downstream recovery is *also* ill-posed and amplifies that noise. Fix η first, then
the recovery. Anchoring the recovery to `uniform` additionally biases both policies toward
the same density (manufacturing non-separation).

---

## Patch 1 — Stabilise η (upstream, minimal, do this first)

`policy_evaluation.py`, in `embedding_kwargs` (~line 4148). These kwargs already exist in
`estimate_embedding`/`KE_DRL` but are not being passed, so they sit at unsafe defaults
(`normalize_eta=False`, `eta_clip_max=None`).

```python
    embedding_kwargs = dict(
        ...
        gamma_val=gamma_val,
        lambda_reg=lambda_reg,
        # --- ADD: self-normalised, variance-bounded importance ratio ---
        eta_clip_min=0.0,                 # ratio must be >= 0
        eta_clip_max=float(getattr(args, "eta_clip_max", 20.0)),  # cap the tail
        normalize_eta=True,               # rescale so mean(eta) ~= 1 (self-normalised IS)
        ratio_lambda_reg=float(getattr(args, "ratio_lambda_reg", 1e-2)),  # stronger ridge on the ratio fit
        ...
    )
```

Self-normalised IS keeps π/β consistent in expectation while bounding variance; the upper
clip + stronger `ratio_lambda_reg` lift ESS. **Gate on ESS**: the log already prints
`ESS for eta_plus`. Treat any embedding with ESS below ~1000/10000 as untrustworthy and do
not plot it.

## Patch 2 — Robust ratio estimator: RuLSIF (upstream, recommended)

uLSIF is unconstrained and heavy-tailed here. RuLSIF (Yamada et al. 2011) estimates the
**α-relative** ratio r_α = p_π / (α p_π + (1−α) p_β), which is bounded by 1/α and far more
stable. It is a 3-line change in `src/ke_drl/IS_ULSIF.py`, in `fit()` after the two kernel
matrices are formed (~line 130):

```python
        K_basis_de = self.kernel_func(X_basis, X_beta, **self.kernel_kwargs)   # (b, n_de)
        K_basis_nu = self.kernel_func(X_basis, X_pi,   **self.kernel_kwargs)   # (b, n_nu)

        # --- RuLSIF: alpha-relative ratio (alpha in [0,1); 0 == plain uLSIF) ---
        alpha_mix = float(getattr(self, "alpha_mix", 0.1))
        H_de = (K_basis_de @ K_basis_de.transpose(0, 1)) / float(n_de)         # (b, b)
        H_nu = (K_basis_nu @ K_basis_nu.transpose(0, 1)) / float(n_nu)         # (b, b)
        H = alpha_mix * H_nu + (1.0 - alpha_mix) * H_de                        # (b, b)
        h = K_basis_nu.mean(dim=1)                                             # (b,)
```

Set `alpha_mix` in `__init__` (add `self.alpha_mix = 0.1`) or expose it as a CLI flag.
The predicted r_α is bounded above by 1/α (=10 for α=0.1), which removes the ±hundreds
blow-ups and raises ESS. Note this changes the estimand from π/β to the α-relative ratio;
it is monotone in π/β and is a standard robustification for off-policy contrasts. Combine
with `normalize_eta=True` from Patch 1.

## Patch 3 — Make the recovery well-posed and data-anchored (downstream)

In `Job_ev.sbatch`:

```bash
  --density-recovery-anchor positive_beta \   # was: uniform  (uniform pulls both policies together)
  --density-recovery-init   positive_beta \   # was: uniform
  --density-recovery-l2-lambda 1e-2 \         # anchor more firmly to a data-informed prior
  --density-recovery-kl-lambda 0.0 \
  --num-grid-points 300 \                     # was: 1200; fewer atoms -> better-determined inverse
```

In `policy_evaluation.py`, `_optimize_induced_probability_weights` (~line 2166), set the
ridge from the spectrum of `K` instead of a fixed `1e-5` (which means different things at
each length scale):

```python
    K = 0.5 * (K + K.T)
    # scale ridge to the kernel spectrum (well-conditioned inverse)
    lam_max = float(torch.linalg.eigvalsh(K)[-1])
    ridge_eff = max(float(ridge), 1e-6 * lam_max)
    K_reg = K + ridge_eff * torch.eye(m, dtype=dtype, device=dev)
    M = torch.linalg.solve(K_reg, A)
```

Also **log a separation metric** so it's checked automatically. In the two-policy section
(~line 5174, where `payload_A`/`payload_B` are loaded), compute and save
`E_clk[clicks] − E_rev[clicks]` and its sign from the recovered click marginals, and the
β–β correlation between the two policies. If β-corr is ≈0 or ESS is low, separation is not
recoverable regardless of the recovery settings.

---

## Recommended run order

1. Apply Patch 1, re-run a few configs, and read `ESS for eta_plus` and the new β-β
   correlation. Expect ESS to rise and β-corr to become clearly positive if η is fixed.
2. If ESS is still low / β-corr ≈ 0, apply Patch 2 (RuLSIF) — this is the real fix for the
   heavy-tailed ratio.
3. Only then apply Patch 3 so the (now-informative) β is turned into a stable, separable
   density. Verify with `downstream_recovery_diagnostic.py` that `frac_correct(sep>0)`
   moves well above 0.5 and stops flipping with the recovery method.

## Notes / deeper considerations

- **Return vs single-step support.** The Z-grid atoms are k-means centres of *single-step*
  rewards, and `total_clicks` is heavily zero-inflated, so the recovered return density is
  confined to single-step click support and piles on 0. If you want the distribution of the
  discounted *return*, build the grid on the return scale (e.g. bootstrap returns, or scale
  the support by 1/(1−γ)); otherwise E[clicks] is structurally capped.
- **Bandwidth.** The revenue bandwidth sweep (30–90) oversmooths revenue, which is why the
  two revenue densities are visually identical; reduce it once η is fixed.
