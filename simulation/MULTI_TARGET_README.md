# Multi-Target KE-DRL Simulation Architecture

## Overview

This document describes the corrected simulation architecture for KE-DRL with the following key features:

- **10 Fixed MC Evaluation Target Points**: Same across all 100 replicates
- **100 Fixed Training Target Points**: Same across all 100 replicates, different from MC points
- **Full Embedding Function Evaluation**: Curves are evaluated on a return value grid
- **100 Parallel Replicates**: Each with separate offline data
- **Multi-Subplot Visualization**: 10 independent plots (one per MC target point)
- **Aggregated Statistics**: Mean and std across 100 replicates

---

## Architecture Flow

```
┌─────────────────────────────────────────────────────┐
│ 1. CREATE FIXED TARGET POINTS                       │
│    - s_mc_targets:     10 points (fixed seed)      │
│    - a_mc_targets:     10 points (fixed seed)      │
│    - s_train_targets: 100 points (fixed seed)      │
│    - a_train_targets: 100 points (fixed seed)      │
└──────────────────┬──────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────┐
│ 2. FOR EACH OF 100 REPLICATES (PARALLEL)            │
│                                                     │
│  Phase A: MC Ground Truth                          │
│    For each of 10 MC target points:                │
│      • Sample 10K trajectories (300 timesteps)    │
│      • Collect cumulative discounted returns      │
│      • Evaluate embedding on grid: μ_truth(z)    │
│                                                     │
│  Phase B: KE-DRL Training                          │
│    • Use 100 training target points                │
│    • Fit coefficient matrix B                      │
│                                                     │
│  Phase C: Prediction                               │
│    For each of 10 MC target points:                │
│      • Predict embedding: ω = B^T k_X(target)    │
│      • Evaluate on grid: μ_pred(z) = K_Z(z) @ ω │
│      • Compute error vs ground truth               │
│                                                     │
└──────────────────┬──────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────┐
│ 3. AGGREGATE RESULTS                                │
│                                                     │
│  For each of 10 MC target points:                  │
│    • Collect predictions from 100 replicates      │
│    • μ_truth_mean = mean(μ_true[rep])             │
│    • μ_truth_std = std(μ_true[rep])               │
│    • μ_pred_mean = mean(μ_pred[rep])              │
│    • μ_pred_std = std(μ_pred[rep])                │
│    • error_mean = mean(errors[rep])               │
│    • error_std = std(errors[rep])                 │
│                                                     │
└──────────────────┬──────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────┐
│ 4. VISUALIZATION                                    │
│                                                     │
│  Create 2×5 grid (10 subplots total)               │
│  For each subplot (target point):                  │
│    • X-axis: Evaluation grid index                 │
│    • Y-axis: Kernel mean embedding value          │
│    • Bold Orange Line: μ_truth_mean (ground truth) │
│    • Orange Shaded Band: ±1 std (truth)           │
│    • Blue Line: μ_pred_mean (our estimate)        │
│    • Blue Shaded Band: ±1 std (prediction)        │
│    • Title: "Target Point i | MSE = X.XXe-0X"    │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## File Structure

### New Scripts

```
simulation/
├── target_points_config.py          # Create fixed target points
├── mc_ground_truth.py               # MC ground truth computation
├── main_est_multi_target.py         # Single replicate estimation
├── parallel_runner_multi_target.py  # Parallel execution (100 replicates)
└── aggregate_results_multi_target.py # Aggregation & visualization
```

### Configuration

Update `params.yaml` with:

```yaml
# Target points
n_mc_targets: 10          # MC evaluation target points
n_train_targets: 100      # Training target points
mc_seed: 20260512         # Fixed seed for reproducibility
train_seed: 20260513      # Different seed ensures diversity

# MC parameters
mc_n_trajectories: 10000  # Trajectories per MC target
mc_trajectory_length: 300 # Time steps per trajectory
n_eval_grid_points: 100   # Grid points for embedding evaluation

# Parallel execution
n_replicates: 100         # Number of offline data replicates
n_workers: 8              # Parallel workers (CPU cores)

# KE-DRL parameters (same as before)
nu: 3.5
length_scale: 1.0
sigma: 0.7
gamma_val: 0.9
lambda_reg: 1e-6
lambda_B: 0.0
num_steps: 5000
operator_method: "rff"
operator_num_features: 256

# Other
data_dir: "./data"
dtype: "float64"
verbose: true
```

---

## Usage Instructions

### Step 1: Generate Offline Data (if needed)

```bash
python main_offlinedata.py
# Generates 100 offline data replicates: offline_data_0.pt, ..., offline_data_99.pt
```

### Step 2: Run Parallel Estimation

```bash
python parallel_runner_multi_target.py
# Runs 100 replicates in parallel
# Creates: all_results.pt, fixed_target_points.pt
```

**Time estimate:**
- Single replicate: ~2 minutes (on GPU)
- 100 replicates with 8 workers: ~25 minutes

### Step 3: Aggregate and Visualize

```bash
python aggregate_results_multi_target.py
# Loads all_results.pt
# Generates:
#   - multi_target_comparison.png (10-subplot figure)
#   - error_summary.png (error analysis)
#   - metrics_table.csv (detailed metrics)
```

---

## Key Components

### 1. Fixed Target Points (`target_points_config.py`)

Creates deterministic target points using fixed random seeds:

```python
s_mc, a_mc, s_train, a_train = create_fixed_target_points(
    s0, a0,
    n_mc_targets=10,
    n_train_targets=100,
    mc_seed=20260512,      # Fixed!
    train_seed=20260513,   # Fixed!
)
```

**Result:** Same 10+100 target points used for all 100 replicates.

### 2. MC Ground Truth (`mc_ground_truth.py`)

For each MC target point:

```python
returns, mu_true = generate_mc_ground_truth(
    target_state=s_mc[j],
    target_action=a_mc[j],
    eval_grid=eval_grid,
    policy_name="logistic",
    policy_params={...},
    MDP_config={W_s, b_s, sigma_s, W_r, b_r, sigma_r},
    kernel_params={nu, length_scale, sigma},
    n_trajectories=10000,
    trajectory_length=300,
    gamma=0.9,
)
```

**Computation:**
1. Generate 10,000 trajectories starting from target point
2. For each trajectory: compute Z = Σ_{t=0}^{299} γ^t r_t
3. Evaluate embedding on grid: μ_true(z) = (1/10000) Σ k(z, Z_i)

**Result:** One embedding curve per MC target point

### 3. KE-DRL Training (`main_est_multi_target.py`)

```python
B_hat, history_obj, history_be, matrices = KE_DRL(
    s0=s0, a0=a0, s1=s1, a1=a1, r=r,
    s_star=s_train,        # 100 training targets
    a_star=a_train,
    ...
)
```

**Key difference from old code:**
- Training points (100) are SEPARATE from MC evaluation points (10)
- This prevents overfitting to the specific MC targets

### 4. Prediction (`main_est_multi_target.py`)

```python
mu_pred = predict_embedding_on_grid(
    s_target=s_mc[j],
    a_target=a_mc[j],
    eval_grid=eval_grid,
    X_train=torch.cat([s0, a0], dim=1),
    B_hat=B_hat,
    kernel_params={nu, length_scale, sigma},
)
```

**Computation:**
1. ω = B^T k_X(target, training_data)
2. μ_pred(z) = K_Z(z, Z_grid) @ ω
3. Evaluate on the same grid as truth

### 5. Aggregation (`aggregate_results_multi_target.py`)

```python
aggregated = {
    0: {
        'mu_truth_mean': array(n_grid,),  # Mean over 100 replicates
        'mu_truth_std': array(n_grid,),
        'mu_pred_mean': array(n_grid,),
        'mu_pred_std': array(n_grid,),
        'error_mean': scalar,
        'error_std': scalar,
    },
    1: {...},
    ...
    9: {...},
}
```

---

## Output Files

After running all steps:

```
data/
├── offline_data_0.pt           # Input
├── offline_data_1.pt
├── ...
├── offline_data_99.pt
├── fixed_target_points.pt      # Target points (saved for reference)
├── all_results.pt              # Raw results from 100 replicates
├── multi_target_comparison.png # MAIN FIGURE (10 subplots)
├── error_summary.png           # Error analysis (bar + box plots)
└── metrics_table.csv           # Detailed metrics per target
```

### Main Figure: `multi_target_comparison.png`

Example structure:

```
┌─────────────────────────────────────────────────────────┐
│  Target Point 0 | MSE = 1.23e-02   │  Target Point 1 ...│
│    (orange curve + blue curve)                          │
├─────────────────────────────────────────────────────────┤
│  Target Point 5 | MSE = 2.34e-02   │  Target Point 6 ...│
│    (orange curve + blue curve)                          │
└─────────────────────────────────────────────────────────┘
```

**Key features:**
- Bold ORANGE line: Ground truth (from MC)
- Orange shaded band: ±1 std across 100 replicates
- BLUE line: Our KE-DRL prediction
- Blue shaded band: ±1 std across 100 replicates
- Each subplot is independent and shows error for that target

---

## Reproducibility

All components use fixed seeds to ensure reproducibility:

| Component | Seed | Variability |
|-----------|------|-------------|
| Target points (MC) | 20260512 (fixed) | ✓ Same across 100 replicates |
| Target points (train) | 20260513 (fixed) | ✓ Same across 100 replicates |
| Offline data | SLURM_ARRAY_TASK_ID | ✓ Different per replicate |
| MC trajectories | Inherits from offline data | ✓ Stochastic within replicate |

**To force different target points for sensitivity analysis:**

```python
# In target_points_config.py
create_fixed_target_points(..., mc_seed=NEW_SEED_1, train_seed=NEW_SEED_2)
```

---

## Performance Notes

### Memory Usage

Per replicate:
- Offline data: ~100 MB (N=10K, D_s+D_a=10)
- MC trajectories (10K returns): ~10 MB
- B matrix: ~10 MB
- Total: ~150 MB per worker

### Computation Time

Per replicate (on GPU):
- MC ground truth: ~5 min (10K trajectories × 300 steps)
- KE-DRL training: ~2 min (5K steps)
- Predictions: ~30 sec
- **Total per replicate: ~7-8 minutes**

100 replicates with 8 workers: ~90-100 minutes wall-clock time

### Parallelization Strategy

Uses `ProcessPoolExecutor` with separate processes to avoid GPU contention:

```python
with ProcessPoolExecutor(max_workers=8) as executor:
    futures = {
        executor.submit(worker_task, replicate_id, ...): replicate_id
        for replicate_id in range(100)
    }
    # Results come back as they complete
```

---

## Common Issues & Solutions

### Issue: "No offline data found"
**Solution:** Run `main_offlinedata.py` first to generate 100 replicates

### Issue: "CUDA out of memory" with parallel workers
**Solution:** Reduce `n_workers` in params.yaml (e.g., from 8 to 4)

### Issue: "Fixed target points don't match expectations"
**Solution:** Check seeds in `target_points_config.py` - they must be identical across all uses

### Issue: "Plots show only one curve"
**Solution:** Check that MC and training target points are different (should be automatically handled)

---

## Customization

### Change Number of MC Targets

```yaml
# In params.yaml
n_mc_targets: 5  # Instead of 10
```

Result: 5-subplot figure instead of 10

### Change Grid Resolution

```yaml
n_eval_grid_points: 200  # More detailed curves
```

### Use Exact Operators (for small problems)

```yaml
operator_method: "exact"
```

### Run Fewer Replicates (for testing)

```yaml
n_replicates: 10  # Instead of 100
```

Then adjust `parallel_runner_multi_target.py`:
```python
run_parallel_estimation(params, n_replicates=10)
```

---

## Citation & References

If using this architecture in publications:

> The multi-target evaluation architecture compares kernel mean embeddings estimated via KE-DRL against ground truth embeddings obtained via Monte Carlo sampling from a target policy, with aggregated statistics across 100 offline data replicates.

---

## Questions?

Refer to the parent audit documents:
- `SIMULATION_ARCHITECTURE_FLOWCHART.md` - Visual architecture overview
- `COMPREHENSIVE_CORRECTNESS_AUDIT.md` - Technical details
- `ARCHITECTURE_AND_DATA_FLOW.md` - Data flow through modules
