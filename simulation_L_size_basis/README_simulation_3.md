# Simulation 3: Large-N / Small-T L-by-m Mean-Embedding Basis

This folder is a separate simulation branch copied from `simulation/`.

The architecture is the same as the original tuning workflow:

1. `Job_tune_prepare.sbatch` generates the shared offline data and Monte Carlo truth.
2. `Job_tune_global.sbatch` runs one tuning config and one offline replicate per array task.
3. `mu_plot.py`, `summarize_tuning.py`, and the existing report scripts consume the same output layout.

The difference is in `main_est.py`: KE-DRL keeps the return grid size

```text
m = num_grid_points
```

but parameterizes the conditional mean-embedding map with an explicit current
state-action basis `U_1,...,U_L`. Therefore the fitted matrix is
`B_hat.shape = (L, m)`, where `L = mean_embedding_basis.n_basis`, instead of
having one row per raw transition.

Default in `params_tune.yaml`:

```yaml
n_ids: 6000
n_timepoints: 3
mean_embedding_basis:
  method: kmeans
  n_basis: 5000
```

So the raw transition count is `6000 * (3 - 1) = 12000`, while the coefficient
matrix has `5000 x m` parameters. The optional `transition_reduction` block is
now only a computational data-bank reduction for the Bellman operator; it no
longer defines the row dimension of `B_hat`.

Run the usual tuning workflow from this folder:

```bash
jid_prep=$(sbatch --parsable Job_tune_prepare.sbatch)
sbatch --dependency=afterok:$jid_prep Job_tune_global.sbatch
```

If you change `n_ids`, `n_timepoints`, benchmark points, or basis settings, rerun
`Job_tune_prepare.sbatch` so the shared data and benchmark bundle match the config.

Important outputs per replicate:

- `metrics/risk_metrics_<rep>.csv`, including `mean_embedding_basis_size`
- `data/fit_<rep>.pt`, including the mean-embedding basis metadata and `B_hat`
