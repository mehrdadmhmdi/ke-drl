#!/bin/bash
#SBATCH --account=bfez-delta-cpu
#SBATCH --partition=cpu
#SBATCH --job-name=expedia_iql
#SBATCH --output=logs/opt_%A.log
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mehrdad3@illinois.edu

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p logs

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${SCRIPT_DIR}:${PYTHONPATH:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Job ID:    ${SLURM_JOB_ID}"
echo "CPUs/node: ${SLURM_CPUS_ON_NODE:-NA}"
echo "Mem/node:  ${SLURM_MEM_PER_NODE:-NA}"
echo "Node:      $(hostname)"
echo "Start:     $(date)"

module load slurm-env/0.1
module load cudatoolkit/25.3_12.8
module load pytorch-conda/2.8
eval "$(conda shell.bash hook)"
conda activate delta_env

DATA_BASE="${DATA_BASE:-./Expedia_data}"
CKPT_DIR="${CKPT_DIR:-./policies_linear2}"
DEVICE="${DEVICE:-cpu}"
SEED="${SEED:-2026}"

echo "Using Python: $(which python)"
python -m pip show ke-drl || true
python -m pip show d3rlpy || true
echo "DATA_BASE=${DATA_BASE}"
echo "CKPT_DIR=${CKPT_DIR}"
echo "DEVICE=${DEVICE}"
echo "SEED=${SEED}"

python -u d3rlpy_policy_opt.py \
  --data-base "${DATA_BASE}" \
  --ckpt-dir "${CKPT_DIR}" \
  --train-blob expedia_train_timeindexed.pt \
  --test-blob expedia_test_timeindexed.pt \
  --max-train 80000 \
  --max-test 40000 \
  --policy-steps 10000 \
  --policy-steps-per-epoch 5000 \
  --policy-batch 512 \
  --candidate-pool-size 4096 \
  --candidate-neighbor-k 32 \
  --candidate-random-k 16 \
  --positive-candidate-pool-size 16384 \
  --positive-candidate-neighbor-k 96 \
  --positive-candidate-random-k 48 \
  --positive-bank-min-reward 1e-8 \
  --positive-bank-quantile 0.80 \
  --positive-bank-min-size 2000 \
  --positive-weight-boost 4.0 \
  --rare-positive-threshold 0.40 \
  --click-candidate-pool-size 16384 \
  --click-candidate-neighbor-k 128 \
  --click-candidate-random-k 64 \
  --click-bank-min-reward 1.0 \
  --click-bank-quantile 0.80 \
  --click-bank-min-size 2000 \
  --click-weight-boost 8.0 \
  --click-reward-scale 8.0 \
  --state-cols mean_hist_price,std_hist_price,prop_location_score1,srch_room_count,srch_length_of_stay,prop_review_score,comp_inv \
  --categorical-state-cols srch_room_count,srch_length_of_stay,comp_inv \
  --one-hot-categoricals 1 \
  --action-cols avg_price_per_night,total_promotions,std_price_usd \
  --integer-action-col total_promotions \
  --reward-cols gross_revenue_per_night,total_clicks \
  --linear-ridge 1e-3 \
  --min-policy-std 0.05 \
  --max-policy-std 25.0 \
  --policy-std-mode heteroskedastic \
  --policy-std-ridge 1e-2 \
  --policy-std-blend 0.85 \
  --policy-std-floor 1e-6 \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --save-plots \
  --quiet-d3rlpy

echo "Policy training complete: $(date)"
ls -lh "${CKPT_DIR}"/state_encoder.json
ls -lh "${CKPT_DIR}"/iql_value_*.d3
ls -lh "${CKPT_DIR}"/linear_gaussian_policy_*.npz
ls -lh "${CKPT_DIR}"/linear_gaussian_policy_*.json
ls -lh "${CKPT_DIR}"/linear_gaussian_policy_*_summary.json
ls -lh "${CKPT_DIR}"/focus_matrix.json
ls -lh "${CKPT_DIR}"/run_summary.json
