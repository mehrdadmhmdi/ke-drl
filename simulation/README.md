# KE-DRL Simulation Workflow

This folder runs the global-`B` simulation for the revised KE-DRL objective.
The important point is that the Monte Carlo truth is used only as a benchmark,
while the global coefficient matrix is fit from an empirical Bellman loss
summed over multiple target state-action points.

## Architecture

The main consistency experiment repeats the estimation step over 100 independent
offline datasets, but uses one fixed benchmark point for all of them.

1. Generate 100 offline datasets under the behavior policy:
   `D_i`, `i = 1,...,100`.
2. Fix one benchmark state-action point `(s*, a*)` independently of every
   `D_i`. By default it is specified directly in `params.yaml` and stored in
   `data/benchmark_point.csv`.
3. Generate one high-precision Monte Carlo benchmark sample `data/Z_true.pt`
   from `(s*, a*)` under the target policy, using `Z_sim.n_ids` independent
   trajectories and `Z_sim.n_timepoints` time points.
4. For each offline dataset `D_i`, choose 150 training target points
   `(\tilde s_j, \tilde a_j)` from that same `D_i`.
5. Fit one global coefficient matrix `B_i` by minimizing the empirical Bellman
   loss averaged over those 150 training target points.
6. Evaluate `B_i` only at the fixed benchmark point `(s*, a*)`, producing
   `\hat\mu_i((s*,a*);B_i)`, and compare it against the shared Monte Carlo truth.

The replicate index in the code is zero-based: `offline_data_0.pt` corresponds
to mathematical `D_1`.

```
Job_data.sbatch --array=0-99
   |
   +--> data/offline_data_i.pt

Job_Z.sbatch
   |
   +--> data/Z_true.pt
   +--> data/benchmark_point.csv

Job_E_P_sa.sbatch
   |
   +--> runs/main_<master_job_id>/
          |
          +--> Job_est.sbatch --array=0-99
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
  num_replicates: 100

benchmark:
  num_points: 1
  source: fixed_config
  s_star: [0.6, -0.8, 0.7, -0.5, 0.9]
  a_star: [0.75]
  output: Z_true.pt

target_set:
  mode: train_subset
  num_points: 150
  exclude_benchmark: True
```

`benchmark.num_points` is intentionally fixed at 1. `Job_Z.sbatch` generates
only one benchmark truth file, and this file does not depend on any `D_i`. The
target points in `target_set.num_points` are not Monte Carlo benchmarks; they
are the points over which the empirical global Bellman loss is averaged for
each offline dataset.

Other important tuning parameters are:

- `n_ids`, `n_timepoints`: offline data size.
- `Z_sim.n_ids`, `Z_sim.n_timepoints`: Monte Carlo benchmark size.
- `num_grid_points`, `hull_expand_factor`: return-grid resolution and support.
- `operator_approximation`: construction method for the return-space `H` and
  `G` operators. Use `method: rff` for non-smoke runs; exact `G` scales as
  `target_points * num_grid_points^2 * N^2` and is only appropriate for tiny
  tests.
- `lambda_reg`: Gamma/ridge regularization for conditional embedding weights.
- `lambda_B`: ridge regularization on the global coefficient matrix.
- `optimization.ridge_mode`: use `rkhs` for the draft's
  `tr(B^T K_X B)` penalty. `frobenius` is available only as a cheaper
  diagnostic variant and changes the estimator.
- `optimization.mass_anchor_lambda`: penalty enforcing learned target-point
  coefficient masses near `optimization.target_mass`. Keep this positive; with
  zero mass anchoring, the Bellman-only quadratic objective can collapse toward
  a near-zero embedding. Estimation logs and `metrics/risk_metrics_*.csv`
  report `target_mass_*` diagnostics; these should be close to 1 after fitting.
- `optimization.negativity_penalty_lambda`: optional penalty on negative
  finite-grid coefficients. Leave at `0.0` for a signed RKHS expansion; tune it
  upward only if the grid coefficients need a more probability-vector-like
  interpretation.
- `optimization.eta_clip_*` and `optimization.normalize_eta`: stabilize the
  uLSIF continuation density ratios used in `Phi`.
- `kernel.length_scale`, `kernel.sigma`, `kernel.nu`: Matern kernel settings.
- `optimization.lr`, `optimization.weight_decay`, `optimization.num_steps`.
- `optimization.target_batch_size`: number of target points used per optimizer
  step. With 150 target points, `150` means full target-batch optimization.

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
- 1 fixed benchmark true-Z shared by all smoke replicates.
- 10 target points per replicate for the global loss.
- 50 return-grid points and 80 optimizer steps.

Every Slurm script prints `KEDRL_SRC` and every Python stage prints the
resolved `ke_drl import source`. These should point to the cloned repository's
`src/ke_drl` when the full clone is present. If only the `simulation/` folder is
on the cluster, the scripts fall back to the installed `ke_drl` package and
print its import path.

To force a reinstall from GitHub inside a Slurm job, submit with:

```bash
sbatch --export=ALL,KEDRL_REINSTALL_FROM_GIT=1 Job_smoke.sbatch
```

The reinstall command used by the scripts is:

```bash
python -m pip install "git+https://github.com/mehrdadmhmdi/ke-drl.git"
```

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

Before the full 100-replicate run, use the smaller tuning sweep:

```bash
jid_tune=$(sbatch --parsable Job_tune_global.sbatch)
jid_sum=$(sbatch --parsable --dependency=afterok:${jid_tune} Job_tune_summary.sbatch)
```

Each tuning array task uses `params_tune.yaml`, applies one one-factor-at-a-time
override from `tuning_grid.yaml`, and runs the same fixed-benchmark architecture
on 50 tuning replicates. The current base tuning design is:

```yaml
n_ids: 3000
n_timepoints: 3
Z_sim: {n_ids: 10000, n_timepoints: 300}
gamma_val: 0.7
target_set.num_points: 100
num_grid_points: 400
optimization.num_steps: 400
optimization.target_batch_size: 30
lambda_reg: 0.001
lambda_B: 0.001
optimization.mass_anchor_lambda: 1.0
kernel: {nu: 3.5, length_scale: 0.7, sigma: 0.7}
operator_approximation.num_features: 512
```

The tuning grid currently has 19 array tasks: the base setting plus single-factor
changes over `lambda_reg`, `lambda_B`, kernel length scale, kernel sigma, kernel
smoothness `nu`, hull expansion, mass anchor, negativity penalty, and combined
tail-stability checks around the length-scale-0.7 baseline.
The summary job writes:

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
- `score_mass`: mass-constraint RMSE for the global-loss target points.

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
- Compare a mild `optimization.negativity_penalty_lambda` such as `0.1` if the
  fitted benchmark weights are too negative.
- Compare `kernel.length_scale` values around `0.5`, `0.7`, and `1.0`.
- Keep `num_grid_points: 400` for reported runs; use `200` only for quick smoke
  checks.

## Production Run

Run from the `simulation/` directory.

If the `data/` folder contains old files from the previous workflow, clear the
generated simulation artifacts first:

```bash
rm -f data/offline_data_*.pt data/Z_true.pt data/Z_true_*.pt data/benchmark_point.csv data/benchmark_point_*.csv
rm -f data/target_points_*.csv data/fit_*.pt data/Zgrid_*.pt data/Zeval_*.pt
rm -f mu/mu_hat_*.csv mu/mu_true_*.csv mu/weights_*.csv
rm -f metrics/run_metrics_*.csv metrics/global_eval_metrics_*.csv
```

```bash
jid_data=$(sbatch --parsable Job_data.sbatch)
jid_z=$(sbatch --parsable Job_Z.sbatch)
jid_run=$(sbatch --parsable --dependency=afterok:${jid_data}:${jid_z} Job_E_P_sa.sbatch)
```

`Job_E_P_sa.sbatch` checks that all `offline_data_i.pt` files and the shared
`Z_true.pt` file exist before launching the estimator array.

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
from a fresh offline dataset and a fresh global fit, but evaluated at the same
fixed benchmark point.

The four panels show:

- Mean estimated embeddings across `D_i` against the fixed Monte Carlo truth.
- Quantile calibration lines, one gray line per `D_i`, plus their mean line.
- Per-replicate error summaries, including the benchmark embedding test risk
  when available.
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
main_MonteCarloZ.py      generates the shared benchmark Z_true.pt
main_est.py             fits one global B for replicate i and evaluates it
mu_plot.py              aggregates replicate metrics and plots
sim_utils.py            data generation, policy sampling, target-set selection
sim_eval.py             mean-embedding metrics and plots
validate_sim_config.py  policy and run-shape validation

Job_data.sbatch          offline data array, 0-99
Job_Z.sbatch             shared benchmark true-Z job
Job_E_P_sa.sbatch        master launcher for estimator and plotting
Job_est.sbatch           global-B estimator array, 0-99
Job_plot.sbatch          aggregation and plotting job
Job_smoke.sbatch         end-to-end small smoke test
Job_tune_global.sbatch   tuning array over combo ids
Job_tune_summary.sbatch  tuning-result aggregation
```

The older `Job_tuning*.sbatch` scripts are legacy scripts from the previous
pointwise/evaluation-point workflow. Use `Job_tune_global.sbatch` for the
current global-`B` architecture.
