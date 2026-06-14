# Simulation 2: long trajectories with an L-by-m mean-embedding basis

This folder is intentionally separate from `simulation/`.

The raw offline transition bank has

```text
n_raw = n_ids * (n_timepoints - 1)
```

rows after stacking `(S_t,A_t)` and `(S_{t+1},A_{t+1})`. The learned
coefficient matrix should not have one row per raw transition. It is now
parameterized as

```text
B_hat.shape = (L, m_grid)
```

where `L = mean_embedding_basis.n_basis` is chosen by the user.

The simulation-2 workflow in `main_est.py`:

1. load the full offline data;
2. select the training target points from the full data;
3. optionally reduce the Bellman-operator data bank for memory;
4. select `mean_embedding_basis.n_basis` current-state-action basis points;
5. fit `B in R^{L x m}` while keeping `m = num_grid_points`.

For the default setting,

```text
n_ids = 300
T = 50       -> n_raw = 14700
n_basis = 200
m_grid = 100
```

so `B_hat` is `200 x 100`, instead of `14700 x 100`.

## Cluster commands

For `T=50`, `N=300`, `m=100`, and all off-diagonal behavior/test policy pairs
among Uniform, Gaussian, and Logistic. The active pair's `L`, regularization,
kernel, and optimizer profile comes from `params_tune.yaml` under
`policy_pair_overrides`.

```bash
SIM_DIR=/work/nvme/bfez/mehrdad3/DistRL/simulation_small_N_large_T
SIM_BASE=SIM2_TIMEPOINTS=50,SIM2_N_IDS=300,SIM2_NUM_REPLICATES=100,SIM2_GRID_POINTS=100,SIM2_TARGET_MODE=all,SIM2_TARGET_POINTS=14700,SIM2_Z_IDS=10000,SIM2_Z_TIMEPOINTS=500

for PAIR in UG UL GU GL LU LG; do
  mkdir -p "$SIM_DIR/results/$PAIR/logs"
  jid_prep=$(sbatch --parsable --chdir="$SIM_DIR" --job-name="${PAIR}_prepare" --output="$SIM_DIR/results/$PAIR/logs/prepare_%j.log" --export=ALL,SIM2_POLICY_PAIR=$PAIR,SIM2_STAGE=prepare,$SIM_BASE "$SIM_DIR/Job_sim2.sbatch")
  jid_fit=$(sbatch --parsable --dependency=afterok:$jid_prep --array=0-99 --chdir="$SIM_DIR" --job-name="${PAIR}_fit" --output="$SIM_DIR/results/$PAIR/logs/fit_%A_%a.log" --export=ALL,SIM2_POLICY_PAIR=$PAIR,SIM2_STAGE=fit,$SIM_BASE "$SIM_DIR/Job_sim2.sbatch")
  sbatch --dependency=afterok:$jid_fit --chdir="$SIM_DIR" --job-name="${PAIR}_aggregate" --output="$SIM_DIR/results/$PAIR/logs/aggregate_%j.log" --export=ALL,SIM2_POLICY_PAIR=$PAIR,SIM2_STAGE=aggregate,$SIM_BASE "$SIM_DIR/Job_sim2.sbatch"
done
```

Useful overrides:

```bash
SIM2_DISABLE_PAIR_PROFILE=1
SIM2_NUM_REPLICATES=100
SIM2_GRID_POINTS=100
SIM2_Z_IDS=50000
SIM2_Z_TIMEPOINTS=500
OFFLINE_DATA_WORKERS=8
```

Each pair writes only under `results/<PAIR>/`. For example, `UL` writes
`results/UL/mu`, `results/UL/metrics`, `results/UL/plots`, and
`results/UL/shared/data`.

## Repeated Monte Carlo truth check

After the fit stage has written `results/<PAIR>/mu` and `results/<PAIR>/data`,
run the CPU post-fit repeated-MC comparison:

```bash
SIM_DIR=/work/nvme/bfez/mehrdad3/DistRL/simulation_small_N_large_T
SIM2_RESULT_ROOT=$SIM_DIR/results/6.5.2026_full_rerun

for PAIR in UG UL GU GL LU LG; do
  sbatch --chdir="$SIM_DIR" \
    --job-name="${PAIR}_mc100" \
    --export=ALL,SIM2_POLICY_PAIR=$PAIR,SIM2_RESULT_ROOT=$SIM2_RESULT_ROOT,SIM2_MC_REPEATS=100 \
    "$SIM_DIR/Job_mc_repeat_eval.sbatch"
done
```

For each fitted embedding curve, `main_mc_repeat_eval.py` generates 100
independent Monte Carlo truth samples at the same benchmark point and writes:

```text
results/<PAIR>/metrics/mc_repeat_aggregate_metrics.csv
results/<PAIR>/metrics/mc_repeat_per_benchmark_aggregate_metrics.csv
results/<PAIR>/metrics/mc_repeats/mc_repeat_metrics.csv
results/<PAIR>/metrics/mc_repeats/mc_repeat_run_summary.csv
results/<PAIR>/metrics/mc_repeats/mc_repeat_pointwise_diff_summary.csv
```

The run summary includes scalar mean/sd columns such as `RMSE_mean`,
`RMSE_sd`, `Bias_mean`, `Bias_sd`, `diff_mean_all`, and `diff_sd_all`.
The pointwise file gives `diff_mean` and `diff_sd` for
`mu_hat - mu_true` at each evaluation-grid location.
