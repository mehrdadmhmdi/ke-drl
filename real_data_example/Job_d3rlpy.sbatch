#!/bin/bash
#SBATCH --account=bfez-delta-cpu
#SBATCH --partition=cpu-preempt
#SBATCH --job-name=expedia_gaussian
#SBATCH --output=logs/gaussian_%A.log
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mehrdad3@illinois.edu

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ -f "${SUBMIT_DIR}/d3rlpy_policy.py" ]; then
  SCRIPT_DIR="$(cd "${SUBMIT_DIR}" && pwd)"
elif [ -f "${SUBMIT_DIR}/real_data_example/d3rlpy_policy.py" ]; then
  SCRIPT_DIR="$(cd "${SUBMIT_DIR}/real_data_example" && pwd)"
else
  echo "Could not locate d3rlpy_policy.py from SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
  exit 2
fi
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

set +u
module load slurm-env/0.1 || true
module load cudatoolkit/25.3_12.8 || true
module load pytorch-conda/2.8 || true
eval "$(conda shell.bash hook)"
conda activate delta_env
set -u

DATA_BASE="${DATA_BASE:-./Expedia_data}"
CKPT_DIR="${CKPT_DIR:-./policies_linear_direct}"
DEVICE="${DEVICE:-cpu}"
SEED="${SEED:-2026}"

python -u d3rlpy_policy.py \
  --data-base "${DATA_BASE}" \
  --ckpt-dir "${CKPT_DIR}" \
  --train-blob expedia_train_timeindexed.pt \
  --test-blob expedia_test_timeindexed.pt \
  --max-train 80000 \
  --max-test 40000 \
  --state-cols mean_hist_price,std_hist_price,prop_location_score1,srch_room_count,srch_length_of_stay,prop_review_score,comp_inv \
  --categorical-state-cols srch_room_count,srch_length_of_stay,comp_inv \
  --one-hot-categoricals 1 \
  --action-cols avg_price_per_night,total_promotions,std_price_usd \
  --integer-action-col total_promotions \
  --reward-cols gross_revenue_per_night,total_clicks \
  --reward-weight-mode rank_softmax \
  --reward-temperature 3.0 \
  --click-reward-temperature 3.0 \
  --click-weight-boost 8.0 \
  --weight-max 100.0 \
  --policy-steps 10000 \
  --policy-steps-per-epoch 1000 \
  --policy-batch 512 \
  --gaussian-lr 2e-3 \
  --linear-ridge 1e-3 \
  --policy-std-ridge 1e-3 \
  --min-policy-std 0.05 \
  --max-policy-std 25.0 \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --save-plots

echo "Direct Gaussian policy training complete: $(date)"
ls -lh "${CKPT_DIR}"/state_encoder.json
ls -lh "${CKPT_DIR}"/linear_gaussian_policy_*.npz
ls -lh "${CKPT_DIR}"/linear_gaussian_policy_*.json
ls -lh "${CKPT_DIR}"/linear_gaussian_policy_*_summary.json
ls -lh "${CKPT_DIR}"/policy_difference_summary.json
ls -lh "${CKPT_DIR}"/run_summary.json
