#!/bin/bash
#SBATCH --account=bfez-delta-cpu
##SBATCH --partition=gpuA40x4
#SBATCH --partition=cpu
#SBATCH --job-name=d3rlpy
#SBATCH --output=logs/opt_%A.log
#SBATCH --time=48:00:00
#SBATCH --nodes=1
##SBATCH --gres=gpu:1
#SBATCH --mem=8g
#SBATCH --cpus-per-task=3
#SBATCH --mail-type=END
#SBATCH --mail-user=mehrdad3@illinois.edu



echo "Job ID:    ${SLURM_JOB_ID}"
echo "Array idx: ${SLURM_ARRAY_TASK_ID}"
echo "CPUs/node: ${SLURM_CPUS_ON_NODE}"
echo "Mem/node:  ${SLURM_MEM_PER_NODE}"

## load modules/env -- delta ai
#module purge
#module load cuda/12.6.1
#module load craype-accel-nvidia90
#module load python/miniforge3_pytorch/2.7.0

##load module/env -- delta
module load slurm-env/0.1
module load cudatoolkit/25.3_12.8
module load pytorch-conda/2.8
eval "$(conda shell.bash hook)"   # IMPORTANT
conda activate delta_env

echo $SLURM_JOB_PARTITION
echo $CUDA_VISIBLE_DEVICES


set -euo pipefail

echo "🚀 Starting KE-DRL dual policy training..."
echo "Node: $(hostname) | GPUs: ${CUDA_VISIBLE_DEVICES:-NA} | Time: $(date)"

# (optional) array-friendly seed
SEED=${SEED:-2026}
# if using job arrays, uncomment:
# SEED=$((2026 + ${SLURM_ARRAY_TASK_ID:-0}))

python3 d3rlpy_policy_opt6.py \
  --data_base ./data \
  --ckpt_dir ./policies_linear2 \
  --max_train 80000 \
  --max_test 40000 \
  --policy_steps 10000 \
  --policy_batch 512 \
  --candidate_pool_size 4096 \
  --candidate_neighbor_k 32 \
  --candidate_random_k 16 \
  --state_cols mean_hist_price,std_hist_price,prop_location_score1,srch_room_count,srch_length_of_stay,prop_review_score,comp_inv \
  --action_cols avg_price_per_night,total_promotions,std_price_usd \
  --reward_cols gross_revenue_per_night,total_clicks \
  --linear-ridge 1e-3 \
  --min-policy-std 0.05 \
  --max-policy-std 25.0 \
  --policy-std-mode heteroskedastic \
  --policy-std-ridge 1e-2 \
  --policy-std-blend 0.85 \
  --policy-std-floor 1e-6 \
  --device cuda:0 \
  --seed 2026 \
  --save_plots \
  --quiet_d3rlpy

echo "✅ DUAL POLICIES COMPLETE: $(date)"
ls -l policies_linear/iql_value_*.d3
ls -l policies_linear/linear_gaussian_policy_*.npz
ls -l policies_linear/linear_gaussian_policy_*.json
ls -l policies_linear/linear_gaussian_policy_*_summary.json
ls -l policies_linear/focus_matrix.json
ls -l policies_linear/run_summary.json


