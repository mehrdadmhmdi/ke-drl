# KE-DRL Simulation Workflow

This folder runs the global-`B` simulation for the revised KE-DRL objective.
The important point is that the Monte Carlo truth is used only as a benchmark,
while the global coefficient matrix is fit from an empirical Bellman loss
summed over multiple target state-action points.

## Architecture

The main consistency experiment repeats the following pipeline 50 times.

1. Generate one offline dataset under the behavior policy.
2. Choose one benchmark state-action point from that offline dataset.
3. Generate one Monte Carlo true return sample `Z_true_i.pt` for that benchmark.
4. Choose 300 separate target state-action points from the same offline dataset.
5. Fit one global coefficient matrix `B_hat` by averaging the Bellman loss over
   those 300 target points.
6. Evaluate that fitted global `B_hat` at the one benchmark point and compare
   the estimated mean embedding against the Monte Carlo truth.

The replicate index is `i = 0,...,49`. Each replicate has its own offline data,
benchmark true-Z file, target-point set, fitted `B_hat`, and benchmark metrics.

```
Job_data.sbatch --array=0-49
   |
   +--> data/offline_data_i.pt

Job_Z.sbatch --array=0-49
   |
   +--> data/Z_true_i.pt
   +--> data/benchmark_point_i.csv

Job_E_P_sa.sbatch
   |
   +--> runs/main_<master_job_id>/
          |
          +--> Job_est.sbatch --array=0-49
          |      |
          |      +--> data/target_points_i.csv
          |      +--> data/fit_i.pt
          |      +--> data/Zgrid_i.pt
          |      +--> data/Zeval_i.pt
          |      +--> mu/mu_hat_i.csv
          |      +--> mu/mu_true_i.csv
          |      +--> metrics/run_metrics_i.csv
          |
          +--> Job_plot.sbatch
                 |
                 +--> metrics/per_run_metrics.csv
                 +--> metrics/aggregate_metrics.csv
                 +--> metrics/calibration_deming.csv
                 +--> plots/mu_summary_UG.png
```

## Parameter Files

The production settings are in `params.yaml`.

```yaml
experiment:
  num_replicates: 50

benchmark:
  num_points: 1

target_set:
  mode: train_subset
  num_points: 300
  exclude_benchmark: True
```

`benchmark.num_points` is intentionally fixed at 1. The target points in
`target_set.num_points` are not Monte Carlo benchmarks; they are the points
over which the empirical global Bellman loss is averaged.

Other important tuning parameters are:

- `n_ids`, `n_timepoints`: offline data size.
- `Z_sim.n_ids`, `Z_sim.n_timepoints`: Monte Carlo benchmark size.
- `num_grid_points`, `hull_expand_factor`: return-grid resolution and support.
- `lambda_reg`: Gamma/ridge regularization for conditional embedding weights.
- `lambda_B`: ridge regularization on the global coefficient matrix.
- `optimization.mass_anchor_lambda`: penalty enforcing learned target-point
  coefficient masses near `optimization.target_mass`. Keep this positive; with
  zero mass anchoring, the Bellman-only quadratic objective can collapse toward
  a near-zero embedding.
- `kernel.length_scale`, `kernel.sigma`, `kernel.nu`: Matern kernel settings.
- `optimization.lr`, `optimization.weight_decay`, `optimization.num_steps`.
- `optimization.target_batch_size`: number of target points used per optimizer
  step. With 300 target points, `300` means full target-batch optimization.

## Policy Alignment

The current behavior policy is uniform and the target policy is logistic.
The target logistic parameters were set to keep target actions inside the
behavior support.

```yaml
policy:
  evaluation_Target_policy: logistic
  Behvaioral_policy: uniform

  logistic:
    theta_loc: [0.0, -0.2, -0.2, -0.8, -0.6]
    theta_scale: [0.0, 0.0, 0.0, 0.0, 0.0]
    epsilon_loc: [0.1]
    epsilon_scale: [-3.0]
```

`validate_sim_config.py` checks policy dimensions and, when given an offline
data file, checks target-policy overlap with the behavior-policy support.

## Smoke Test

Use this before any large run:

```bash
sbatch Job_smoke.sbatch
```

The smoke test uses `params_smoke.yaml`:

- 3 offline replicates.
- 1 benchmark true-Z per replicate.
- 60 target points per replicate for the global loss.
- 50 return-grid points and 80 optimizer steps.

Expected output is under:

```text
runs/smoke_<job_id>/
```

The key success files are:

```text
runs/smoke_<job_id>/data/fit_0.pt
runs/smoke_<job_id>/metrics/per_run_metrics.csv
runs/smoke_<job_id>/plots/mu_summary_UG.png
```

## Tuning Run

Before the full 50-replicate run, use the small tuning sweep:

```bash
jid_tune=$(sbatch --parsable Job_tune_global.sbatch)
jid_sum=$(sbatch --parsable --dependency=afterok:${jid_tune} Job_tune_summary.sbatch)
```

Each tuning array task uses `params_tune.yaml`, applies one override from
`tuning_grid.yaml`, and runs the same architecture on 5 smaller replicates with
100 global-loss target points. The summary job writes:

```text
runs/tuning_summary.csv
logs/tune_summary.<job_id>.log
```

The tuning score combines RMSE, MAE, sup-norm error, and calibration slope. Use
the best stable setting to update `params.yaml` before the production run.

Each tuning run now records two benchmark families:

- `score_true_z`: external accuracy against the Monte Carlo benchmark true-Z.
- `score_risk`: internal empirical risk, using the final log regularized
  Bellman objective from the optimizer history.

`runs/tuning_summary.csv` also reports `true_z_rank`, `risk_rank`, and
`combined_rank`. Prefer settings that are good on both ranks. The true-Z score
is the main external validation target; the risk score checks whether the
empirical objective from `rz_new_version.tex` is actually being minimized.
Because kernel and grid changes can rescale the raw risk, inspect the two ranks
together rather than trusting a raw risk value alone.

Good first knobs:

- Increase `lambda_B` if `B_hat` looks unstable across replicates.
- Increase `lambda_reg` if Gamma/importance-weight behavior is noisy.
- Compare `optimization.mass_anchor_lambda` values around `0.3`, `1.0`, and
  `3.0` if the estimated embedding mass is too small or too rigid.
- Compare `kernel.length_scale` values around `0.7`, `1.0`, and `1.5`.
- Increase `num_grid_points` only after the small run is stable.
- Increase `n_ids` and `Z_sim.n_ids` when the estimator is stable but noisy.

## Production Run

Run from the `simulation/` directory.

If the `data/` folder contains old files from the previous workflow, clear the
generated simulation artifacts first:

```bash
rm -f data/offline_data_*.pt data/Z_true_*.pt data/benchmark_point_*.csv
rm -f data/target_points_*.csv data/fit_*.pt data/Zgrid_*.pt data/Zeval_*.pt
rm -f mu/mu_hat_*.csv mu/mu_true_*.csv mu/weights_*.csv
rm -f metrics/run_metrics_*.csv metrics/global_eval_metrics_*.csv
```

```bash
jid_data=$(sbatch --parsable Job_data.sbatch)
jid_z=$(sbatch --parsable --dependency=afterok:${jid_data} Job_Z.sbatch)
jid_run=$(sbatch --parsable --dependency=afterok:${jid_z} Job_E_P_sa.sbatch)
```

`Job_E_P_sa.sbatch` checks that all `offline_data_i.pt` and `Z_true_i.pt` files
exist before launching the estimator array.

Production outputs go under:

```text
runs/main_<master_job_id>/
```

The final figure is:

```text
runs/main_<master_job_id>/plots/mu_summary_UG.png
```

## Visualization Meaning

`mu_summary_UG.png` aggregates across offline replicates. It is appropriate for
checking consistency of the estimated global `B` because each curve is produced
from a fresh offline dataset, a fresh benchmark point, and a fresh global fit.

The four panels show:

- Mean estimated and Monte Carlo benchmark embeddings across replicates.
- Quantile calibration of estimated versus true benchmark mean embeddings.
- Per-replicate error summaries.
- ECDF of absolute mean-embedding error.

This is different from aggregating across the target points. The target
points are used inside the global loss; they are not the evaluation objects in
the final consistency plot.

## File Map

```text
params.yaml              production simulation settings
params_smoke.yaml        small smoke-test settings
params_tune.yaml         small tuning baseline
tuning_grid.yaml         tuning overrides

main_offlinedata.py      generates offline_data_i.pt
main_MonteCarloZ.py      generates benchmark Z_true_i.pt
main_est.py             fits one global B for replicate i and evaluates it
mu_plot.py              aggregates replicate metrics and plots
sim_utils.py            data generation, policy sampling, target-set selection
sim_eval.py             mean-embedding metrics and plots
validate_sim_config.py  policy and run-shape validation

Job_data.sbatch          offline data array, 0-49
Job_Z.sbatch             benchmark true-Z array, 0-49
Job_E_P_sa.sbatch        master launcher for estimator and plotting
Job_est.sbatch           global-B estimator array, 0-49
Job_plot.sbatch          aggregation and plotting job
Job_smoke.sbatch         end-to-end small smoke test
Job_tune_global.sbatch   tuning array over combo ids
Job_tune_summary.sbatch  tuning-result aggregation
```

The older `Job_tuning*.sbatch` scripts are legacy scripts from the previous
pointwise/evaluation-point workflow. Use `Job_tune_global.sbatch` for the
current global-`B` architecture.
