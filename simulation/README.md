# KE-DRL Simulation Pipeline

This folder contains the simulation workflow for generating offline data,
generating Monte Carlo benchmark returns, and running the global-X KE-DRL
estimator.

Most scientific settings are controlled by `params.yaml`. The `.sbatch` files
only control cluster logistics such as account, partition, memory, time limits,
array ranges, and the Python environment.

## Architecture

The simulation has three main stages.

```text
params.yaml
   |
   v
Job_data.sbatch -> data/offline_data_i.pt
   |
   v
Job_Z.sbatch -> data/Z_true_j.pt
   |
   v
Job_E_P_sa.sbatch
   |
   +--> runs/run_j/
          |
          +--> Job_est.sbatch over i = 0,...,100
                 |
                 +--> main_est.py
                        fit global B
                        evaluate at fixed (s*, a*)
                        save weights, metrics, and plots
```

## Stage 1: Offline Data Generation

`main_offlinedata.py` generates offline trajectories under the behavior policy
specified in `params.yaml`.

It saves one file per Slurm array index:

```text
data/offline_data_<array_id>.pt
```

Each file contains:

```text
s0, a0, s1, a1, r0, r1, r, metadata
```

These tensors represent observed transitions:

```text
(s_t, a_t, r_t, s_{t+1}, a_{t+1})
```

Run by:

```text
Job_data.sbatch
```

Current array range:

```text
0-100
```

## Stage 2: Monte Carlo True Z Generation

`main_MonteCarloZ.py` loads one offline dataset, chooses one benchmark point
`(s*, a*)`, and simulates many future discounted returns under the target
policy.

It saves:

```text
data/Z_true_<array_id>.pt
```

Each `Z_true` file contains Monte Carlo samples of:

```text
Z = sum_t gamma^t R_t
```

starting from the selected benchmark point `(s*, a*)`.

Run by:

```text
Job_Z.sbatch
```

Current array range:

```text
0-30
```

This creates 31 Monte Carlo benchmark points. If one benchmark is needed for
every offline replicate, change the array range in `Job_Z.sbatch` from `0-30`
to `0-100`.

## Stage 3: Global KE-DRL Estimation

`main_est.py` loads:

- one offline dataset, `offline_data_i.pt`
- one Monte Carlo truth file, either `Z_true.pt` or `Z_true_i.pt`

It fits the global KE-DRL coefficient matrix:

```text
B^pi
```

The global target set is controlled by `params.yaml`:

```yaml
target_set:
  mode: train_subset
  num_points: 128
  seed_offset: 7919
```

After fitting the global map, `main_est.py` evaluates the fitted embedding at
the fixed Monte Carlo benchmark point `(s*, a*)` and compares the estimated
embedding against the Monte Carlo truth.

Run by:

```text
Job_est.sbatch
```

Usually this is launched through:

```text
Job_E_P_sa.sbatch
```

The master script creates:

```text
runs/run_0/
runs/run_1/
...
runs/run_30/
```

Each `run_j` fixes one Monte Carlo benchmark file `Z_true_j.pt` by copying or
renaming it to:

```text
data/Z_true.pt
```

Then it runs estimation over offline datasets `0-100`.

The full nesting is:

```text
benchmark point j = 0,...,30
  offline replicate i = 0,...,100
    fit global B on offline_data_i
    evaluate at benchmark point j
```

## Main Configuration File

Simulation settings are in:

```text
params.yaml
```

Important fields include:

- `n_ids`, `n_timepoints`: offline data size
- `state_dim`, `reward_dim`, `action_dim`: problem dimensions
- `Z_sim.n_ids`, `Z_sim.n_timepoints`: Monte Carlo benchmark size
- `MDP`: state-transition and reward matrices
- `policy`: behavior and target policy parameters
- `gamma_val`: discount factor
- `lambda_reg`: conditional embedding regularization
- `lambda_B`: global coefficient-matrix regularization
- `target_set`: target points used in the global-X objective
- `kernel`: Matern kernel parameters
- `optimization`: optimizer settings

## Main Files

```text
params.yaml              simulation, MDP, policy, and estimator settings
sim_utils.py             synthetic MDP, policy sampling, and MC simulation
sim_eval.py              embedding metrics and aggregation helpers
main_offlinedata.py      generate offline datasets
main_MonteCarloZ.py      generate true Monte Carlo Z benchmarks
main_est.py              fit global KE-DRL and evaluate
mu_plot.py               aggregate metrics and plots
Job_data.sbatch          offline data array
Job_Z.sbatch             true Z array
Job_E_P_sa.sbatch        master benchmark loop
Job_est.sbatch           estimator array
Job_plot.sbatch          aggregation/plotting job
```

## Cluster Usage

Run from this folder:

```bash
cd /path/to/ke-drl/simulation
mkdir -p logs
```

A small test run is recommended first:

```bash
jid_data=$(sbatch --parsable --array=0-0 Job_data.sbatch)
jid_z=$(sbatch --parsable --array=0-0 --dependency=afterok:${jid_data} Job_Z.sbatch)
echo "data=$jid_data z=$jid_z"
```

If the test passes, generate all offline data and true-Z benchmarks:

```bash
jid_data=$(sbatch --parsable Job_data.sbatch)
jid_z=$(sbatch --parsable --dependency=afterok:${jid_data} Job_Z.sbatch)
echo "data=$jid_data z=$jid_z"
```

To run the full estimation pipeline after the data and true-Z files exist:

```bash
sbatch Job_E_P_sa.sbatch
```

Or submit the entire dependency chain:

```bash
jid_data=$(sbatch --parsable Job_data.sbatch)
jid_z=$(sbatch --parsable --dependency=afterok:${jid_data} Job_Z.sbatch)
jid_run=$(sbatch --parsable --dependency=afterok:${jid_z} Job_E_P_sa.sbatch)
echo "data=$jid_data z=$jid_z run=$jid_run"
```

## Outputs

Generated files are intentionally ignored by Git.

Typical outputs are:

```text
data/offline_data_*.pt
data/Z_true_*.pt
runs/run_*/data/
runs/run_*/mu/
runs/run_*/metrics/
runs/run_*/plots/
logs/
```

The simulation folder is designed so that generated data, logs, run folders,
plots, and metrics stay local to the cluster run and are not committed to the
package repository.
