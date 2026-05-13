# KE-DRL Simulation Pipeline

This folder contains the cluster workflow for the current global-`B`
simulation. The design follows the formulation in `rz_new_version.tex`: one
policy-specific coefficient matrix `B^pi` is fitted globally, and the Bellman
loss is averaged over a set of evaluation/target points.

Most scientific settings are controlled by `params.yaml`. The `.sbatch` files
control Slurm logistics: account, partition, memory, time limits, dependency
submission, and the Delta Python/PyTorch environment.

## Current Architecture

The intended simulation has one offline dataset, 30 evaluation points, 30 Monte
Carlo truth files, and one global estimator fit.

```text
params.yaml
   |
   v
Job_data.sbatch
   |
   +--> data/offline_data_0.pt
   |
   v
Job_Z.sbatch --array=0-29
   |
   +--> data/Z_true_0.pt
   +--> ...
   +--> data/Z_true_29.pt
   |
   v
Job_E_P_sa.sbatch
   |
   +--> runs/global_eval_<master_job_id>/
          |
          +--> Job_est.sbatch
          |      |
          |      +--> main_est.py
          |             load offline_data_0.pt
          |             load all 30 Z_true_j.pt files
          |             fit one global B over the 30 target points
          |             evaluate the fitted map at each point
          |             save mu_hat_j, mu_true_j, metrics, B, and grids
          |
          +--> Job_plot.sbatch
                 |
                 +--> mu_plot.py
                        aggregate the 30 evaluation-point outputs
                        draw plots/mu_summary_UG.png
```

## Parameters

The global simulation controls are:

```yaml
evaluation:
  offline_data_id: 0
  num_points: 30

target_set:
  mode: evaluation_points
  num_points: 30
```

`evaluation.offline_data_id` tells every stage to use `offline_data_0.pt`.
`evaluation.num_points` sets the number of benchmark state-action points and
therefore the number of true-Z files required before estimation can start.

The main scientific settings are also in `params.yaml`:

- `n_ids`, `n_timepoints`: offline data size
- `Z_sim.n_ids`, `Z_sim.n_timepoints`: Monte Carlo benchmark size
- `state_dim`, `reward_dim`, `action_dim`: problem dimensions
- `MDP`: state-transition and reward matrices
- `policy`: behavior and target policy parameters
- `gamma_val`: discount factor
- `lambda_reg`: conditional embedding regularization
- `lambda_B`: global coefficient-matrix regularization
- `kernel`: Matern kernel parameters
- `optimization`: optimizer settings

## Stage 1: Offline Data

`main_offlinedata.py` generates the single offline dataset under the behavior
policy specified in `params.yaml`.

```text
data/offline_data_0.pt
```

The file contains pooled transition tensors:

```text
s0, a0, s1, a1, r0, r1, r, metadata
```

Run this stage with:

```bash
sbatch Job_data.sbatch
```

`Job_data.sbatch` is intentionally `--array=0-0`.

## Stage 2: Monte Carlo True Z

`main_MonteCarloZ.py` always loads `data/offline_data_0.pt`. The Slurm array
index is the evaluation-point id, not an offline-data replicate id.

For each `j = 0,...,29`, it selects a deterministic row from the single offline
dataset and simulates many target-policy discounted returns:

```text
data/Z_true_j.pt
data/sa_star_j.csv
```

Each `Z_true_j.pt` stores the Monte Carlo samples and the metadata for that
evaluation point, including `s_star`, `a_star`, `offline_row`, and `eval_id`.

Run this stage with:

```bash
sbatch Job_Z.sbatch
```

`Job_Z.sbatch` is intentionally `--array=0-29`.

## Stage 3: One Global B

`main_est.py` now requires all 30 true-Z files. If any are missing, it stops
with a clear error before fitting.

It loads:

```text
data/offline_data_0.pt
data/Z_true_0.pt
...
data/Z_true_29.pt
```

Then it constructs:

```text
s_star = [s_star_0; ...; s_star_29]
a_star = [a_star_0; ...; a_star_29]
```

and calls `KE_DRL(...)` once. The package optimizer already averages the
Bellman embedding residual over the rows of this target set, so this produces
one global `B_hat`, not 30 separate pointwise fits.

Important outputs from the estimator are:

```text
data/fit_global.pt
data/Zgrid_global.pt
data/Zeval_global.pt
data/evaluation_points.csv
mu/mu_hat_0.csv, ..., mu/mu_hat_29.csv
mu/mu_true_0.csv, ..., mu/mu_true_29.csv
metrics/global_eval_metrics.csv
```

## Aggregation and Visualization

`Job_E_P_sa.sbatch` does not draw the final figure directly. It submits
`Job_est.sbatch`, then submits `Job_plot.sbatch` after successful estimation.
`Job_plot.sbatch` runs:

```bash
python3 mu_plot.py
```

`mu_plot.py` aggregates the paired mean-embedding curves:

```text
mu/mu_hat_j.csv
mu/mu_true_j.csv
```

where `j` indexes the 30 evaluation points. These curves are evaluated on
`data/Zeval_global.pt`, a common deterministic grid drawn from the combined
Monte Carlo truth samples. This makes the figure a comparison over the same
return-space locations for every evaluation point.

The main outputs are:

```text
plots/mu_summary_UG.png
plots/mu_summary.png
plots/mu_calibration.png
metrics/per_point_metrics.csv
metrics/aggregate_metrics.csv
metrics/calibration_deming.csv
```

`plots/mu_summary_UG.png` summarizes whether the one fitted global `B_hat`
produces consistent embeddings across the 30 evaluation points. The four panels
show:

- average estimated and Monte Carlo mean embeddings across evaluation points
- quantile calibration of estimated vs. true embedding values
- per-point `|Bias|`, `MAE`, and `RMSE`
- empirical CDF of pointwise absolute embedding error

This is the correct visualization for the current global-`B` design: the
variation is across evaluation points after fitting one shared coefficient
matrix, not across separately fitted offline-data replicates.

`mu_plot.py` now fails explicitly if `plots/mu_summary_UG.png` is not created,
so a missing `matplotlib` installation will show up as a plotting-job error
instead of a silent success.

## Cluster Commands

Run from the simulation folder on Delta:

```bash
cd /path/to/ke-drl/simulation
mkdir -p logs
```

For a small smoke test, run one self-contained job:

```bash
sbatch Job_smoke.sbatch
```

This uses `params_smoke.yaml`, not the production `params.yaml`. It creates a
fresh folder:

```text
runs/smoke_<job_id>/
```

The smoke test uses 1 offline dataset, 3 evaluation points, 20 return-grid
points, 20 optimizer steps, and small Monte Carlo truth samples. It should only
be used to check that files, paths, imports, global-`B` fitting, metrics, and
plotting work end-to-end.

Submit the full dependency chain:

```bash
jid_data=$(sbatch --parsable Job_data.sbatch)
jid_z=$(sbatch --parsable --dependency=afterok:${jid_data} Job_Z.sbatch)
jid_run=$(sbatch --parsable --dependency=afterok:${jid_z} Job_E_P_sa.sbatch)
echo "data=$jid_data z=$jid_z run=$jid_run"
```

After it finishes, inspect the master log to find the run folder:

```bash
ls -ltr logs/master-launch.*.log
```

The estimator and plot logs will be inside:

```text
runs/global_eval_<master_job_id>/logs/
```

## Main Files

```text
params.yaml              simulation, MDP, policy, and estimator settings
sim_utils.py             synthetic MDP, policy sampling, and MC simulation
sim_eval.py              embedding metrics and aggregation helpers
main_offlinedata.py      generate offline_data_0.pt
main_MonteCarloZ.py      generate Z_true_0.pt through Z_true_29.pt
main_est.py              fit one global B and evaluate all points
mu_plot.py               aggregate metrics and plots
params_smoke.yaml        tiny smoke-test settings
Job_data.sbatch          offline data job, array 0-0
Job_Z.sbatch             true-Z job, array 0-29
Job_smoke.sbatch         one-job end-to-end smoke test, 3 points
Job_E_P_sa.sbatch        master global-fit launcher
Job_est.sbatch           estimator job, no array
Job_plot.sbatch          aggregation/plotting job, no array
```

Generated data, logs, runs, plots, and metrics are ignored by Git and should
remain local to the cluster run.
