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
T = 500      -> n_raw = 149700
T = 1000     -> n_raw = 299700
n_basis = 1500
m_grid = 400
```

so `B_hat` is `1500 x 400`, instead of `149700 x 400` or `299700 x 400`.

## Cluster commands

Run from this folder:

```bash
cd /work/nvme/bfez/mehrdad3/DistRL/simulation_2
```

For `T=500`:

```bash
jid_prep=$(sbatch --parsable --export=ALL,SIM2_STAGE=prepare,SIM2_TIMEPOINTS=500 Job_sim2.sbatch)
jid_fit=$(sbatch --parsable --dependency=afterok:$jid_prep --array=0-49 --export=ALL,SIM2_STAGE=fit,SIM2_TIMEPOINTS=500 Job_sim2.sbatch)
sbatch --dependency=afterok:$jid_fit --export=ALL,SIM2_STAGE=aggregate,SIM2_TIMEPOINTS=500 Job_sim2.sbatch
```

For `T=1000`:

```bash
jid_prep=$(sbatch --parsable --export=ALL,SIM2_STAGE=prepare,SIM2_TIMEPOINTS=1000,SIM2_TAG=T1000_N300_r1500 Job_sim2.sbatch)
jid_fit=$(sbatch --parsable --dependency=afterok:$jid_prep --array=0-49 --export=ALL,SIM2_STAGE=fit,SIM2_TIMEPOINTS=1000,SIM2_TAG=T1000_N300_r1500 Job_sim2.sbatch)
sbatch --dependency=afterok:$jid_fit --export=ALL,SIM2_STAGE=aggregate,SIM2_TIMEPOINTS=1000,SIM2_TAG=T1000_N300_r1500 Job_sim2.sbatch
```

Useful overrides:

```bash
SIM2_BASIS_N=2500
SIM2_OPERATOR_REDUCED_N=2500
SIM2_NUM_REPLICATES=100
SIM2_GRID_POINTS=600
SIM2_Z_IDS=50000
SIM2_KMEANS_ITER=30
OFFLINE_DATA_WORKERS=8
```
