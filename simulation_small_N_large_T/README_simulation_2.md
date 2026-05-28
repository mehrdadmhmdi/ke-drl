# Simulation 2: long trajectories with a reduced transition basis

This folder is intentionally separate from `simulation/`.

The raw offline transition bank has

```text
n_raw = n_ids * (n_timepoints - 1)
```

rows after stacking `(S_t,A_t)` and `(S_{t+1},A_{t+1})`. In the original
pipeline, the learned coefficient matrix has one row per raw transition,
`B_hat.shape = (n_raw, m_grid)`, which is too large when `T` is 500 or 1000.

The simulation-2 modification is in `main_est.py`:

1. load the full offline data;
2. select the training target points from the full data;
3. build `X=(S,A)` for the full transition stack;
4. select `transition_reduction.n_basis` representative observed rows in
   standardized `X` space by k-means landmarks;
5. add nearest observed rows to the training/evaluation target points;
6. call the unchanged `ke_drl` package on only the reduced transition bank.

For the default setting,

```text
n_ids = 300
T = 500      -> n_raw = 149700
T = 1000     -> n_raw = 299700
n_basis = 1500
m_grid = 400
```

so `B_hat` is roughly `1500 x 400`, instead of `149700 x 400` or
`299700 x 400`.

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
SIM2_REDUCED_N=2500
SIM2_NUM_REPLICATES=100
SIM2_GRID_POINTS=600
SIM2_Z_IDS=50000
SIM2_KMEANS_ITER=30
OFFLINE_DATA_WORKERS=8
```
