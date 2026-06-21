#!/bin/bash
#SBATCH --account=bfez-delta-cpu
#SBATCH --partition=cpu-preempt
#SBATCH --job-name=expedia_gaussian_policy
#SBATCH --output=logs/opt_%A.log
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mehrdad3@illinois.edu

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ -f "${SUBMIT_DIR}/d3rlpy_policy_opt.py" ]; then
  SCRIPT_DIR="$(cd "${SUBMIT_DIR}" && pwd)"
elif [ -f "${SUBMIT_DIR}/real_data_example/d3rlpy_policy_opt.py" ]; then
  SCRIPT_DIR="$(cd "${SUBMIT_DIR}/real_data_example" && pwd)"
else
  echo "Could not locate real_data_example from SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
  echo "Submit from kedrl_git/real_data_example or kedrl_git." >&2
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
module load slurm-env/0.1
module load cudatoolkit/25.3_12.8
module load pytorch-conda/2.8
eval "$(conda shell.bash hook)"
conda activate delta_env
set -u

DATA_BASE="${DATA_BASE:-./Expedia_data}"
CKPT_DIR="${CKPT_DIR:-./policies_linear_overlap}"
DEVICE="${DEVICE:-cpu}"
SEED="${SEED:-2026}"

echo "Using Python: $(which python)"
python -m pip show ke-drl || true
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
  --policy-steps 20000 \
  --policy-steps-per-epoch 2000 \
  --policy-batch 512 \
  --gaussian-lr 2e-3 \
  --overlap-anchor-lambda 3.0 \
  --overlap-std-anchor-lambda 1.0 \
  --policy-improvement-mix 0.40 \
  --action-support-lower-quantile 0.02 \
  --action-support-upper-quantile 0.98 \
  --state-cols mean_hist_price,std_hist_price,prop_location_score1,srch_room_count,srch_length_of_stay,prop_review_score,comp_inv \
  --categorical-state-cols srch_room_count,srch_length_of_stay,comp_inv \
  --one-hot-categoricals 1 \
  --action-cols avg_price_per_night,total_promotions,std_price_usd \
  --integer-action-col total_promotions \
  --reward-cols gross_revenue_per_night,total_clicks \
  --reward-weight-mode rank_softmax \
  --reward-temperature 5 \
  --click-reward-temperature 10 \
  --positive-weight-boost 8 \
  --click-weight-boost 25 \
  --click-bank-min-reward 1.0 \
  --weight-max 50 \
  --reward-weight-uniform-mix 0.60 \
  --enable-policy-contrast-shift 1 \
  --contrast-top-quantile 0.92 \
  --revenue-contrast-strength 0.35 \
  --click-contrast-strength 0.70 \
  --action-shift-clip-std 0.95 \
  --revenue-price-shift-std 0.25 \
  --revenue-promotion-shift-std -0.15 \
  --revenue-spread-shift-std 0.10 \
  --click-price-shift-std -0.45 \
  --click-promotion-shift-std 0.70 \
  --click-spread-shift-std -0.15 \
  --min-abs-mean-state-coef-std 0.0 \
  --min-abs-logstd-state-coef-std 0.0 \
  --linear-ridge 5e-3 \
  --min-policy-std 0.10 \
  --max-policy-std 20.0 \
  --policy-std-ridge 1e-3 \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --save-plots

echo "Policy training complete: $(date)"
ls -lh "${CKPT_DIR}"/state_encoder.json
ls -lh "${CKPT_DIR}"/linear_gaussian_policy_*.npz
ls -lh "${CKPT_DIR}"/linear_gaussian_policy_*.json
ls -lh "${CKPT_DIR}"/linear_gaussian_policy_*_summary.json
ls -lh "${CKPT_DIR}"/policy_difference_summary.json
ls -lh "${CKPT_DIR}"/run_summary.json
