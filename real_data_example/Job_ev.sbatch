#!/bin/bash
#SBATCH --account=bfez-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=expedia_eval_main
#SBATCH --output=logs/DeltaAI_ev_%A_%a.log
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=96g
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehrdad3@illinois.edu
#SBATCH --array=0-287

# =====================================================================
# Mean-embedding + density-recovery evaluation.
# Prerequisite: run Job_d3rlpy.sbatch first so ./policies_linear3 exists.
#   sbatch Job_d3rlpy.sbatch
#   sbatch --dependency=afterok:<d3rlpy_jobid> Job_ev.sbatch
#
# The evaluator itself owns the stabilized ratio and density-recovery settings:
# bounded RuLSIF, self-normalized/clipped eta, positive-beta anchored recovery,
# spectrum-scaled ridge, and a smaller atom grid for better conditioning.
# =====================================================================

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ -f "${SUBMIT_DIR}/policy_evaluation.py" ]; then
  SCRIPT_DIR="$(cd "${SUBMIT_DIR}" && pwd)"
elif [ -f "${SUBMIT_DIR}/real_data_example/policy_evaluation.py" ]; then
  SCRIPT_DIR="$(cd "${SUBMIT_DIR}/real_data_example" && pwd)"
else
  echo "Could not locate policy_evaluation.py from SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# RuLSIF alpha-relative mixing for the density-ratio fit (0=plain uLSIF).
# 0.1 bounds the ratio by 1/0.1=10 -> removes the negative/heavy-tail blow-ups
# and raises ESS. Set to 0.0 to reproduce the old behavior.
export KEDRL_RULSIF_ALPHA="${KEDRL_RULSIF_ALPHA:-0.1}"
RULSIF_ALPHA="${RULSIF_ALPHA:-${KEDRL_RULSIF_ALPHA}}"

echo "Job ID:    ${SLURM_JOB_ID}"
echo "Array idx: ${SLURM_ARRAY_TASK_ID}"
echo "CPUs/node: ${SLURM_CPUS_ON_NODE:-NA}"
echo "Mem/node:  ${SLURM_MEM_PER_NODE:-NA}"
echo "Node:      $(hostname)"
echo "KEDRL_RULSIF_ALPHA=${KEDRL_RULSIF_ALPHA}"
echo "RULSIF_ALPHA=${RULSIF_ALPHA}"
echo "Start:     $(date)"

set +u
module purge
module load cuda/12.6.1
module load default gcc/14.2.0 craype-accel-nvidia90
module load python/miniforge3_pytorch/2.7.0
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
eval "$(conda shell.bash hook)"
set -u

DATA_BASE="${DATA_BASE:-./Expedia_data}"
POLICY_CKPT_DIR="${POLICY_CKPT_DIR:-./policies_linear3}"
STATE_ENCODER_PATH="${STATE_ENCODER_PATH:-${POLICY_CKPT_DIR}/state_encoder.json}"
PYFILE="${PYFILE:-policy_evaluation.py}"
TASK="${SLURM_ARRAY_TASK_ID}"

if [ ! -f "${STATE_ENCODER_PATH}" ]; then
  echo "ERROR: ${STATE_ENCODER_PATH} not found. Run Job_d3rlpy.sbatch first." >&2
  exit 3
fi

# Focused stable grid for the separated policies.
# The reported risk is the zero-baseline projected Bellman fit risk.
NU_Z_ARR=(3.5 4.5)
ELL_Z_ARR=(2.00 2.50 3.00)
LAM_ARR=(1e-5 2e-5 4e-5 1e-4)
SIGMA_Z_ARR=(1.0)
NU_SA_ARR=(3.5)
ELL_SA_ARR=(1.0)
SIGMA_SA_ARR=(1.0)
BAND_ARR=(0.0)
BW_PER_DIM_ARR=("30,1" "50,1" "70,1" "90,1")
LAMREC_ARR=(0.0)
GRID_ARR=(300)                       # was 1200; fewer atoms -> better-conditioned recovery
MEAN_EMBEDDING_BASIS_SIZE_ARR=(6000)
STEPS_ARR=(40000)
TARGET_POINT_ARR=("state_pc_q25" "state_pc_q50" "state_pc_q75")

N_NU_Z=${#NU_Z_ARR[@]}
N_ELL_Z=${#ELL_Z_ARR[@]}
N_LAM=${#LAM_ARR[@]}
N_SIGMA_Z=${#SIGMA_Z_ARR[@]}
N_NU_SA=${#NU_SA_ARR[@]}
N_ELL_SA=${#ELL_SA_ARR[@]}
N_SIGMA_SA=${#SIGMA_SA_ARR[@]}
N_BAND=${#BAND_ARR[@]}
N_BW_PER_DIM=${#BW_PER_DIM_ARR[@]}
N_LAMREC=${#LAMREC_ARR[@]}
N_GRID=${#GRID_ARR[@]}
N_BASIS=${#MEAN_EMBEDDING_BASIS_SIZE_ARR[@]}
N_STEPS=${#STEPS_ARR[@]}
N_TARGET_POINT=${#TARGET_POINT_ARR[@]}

TOTAL_CFG=$(( N_TARGET_POINT * N_NU_Z * N_ELL_Z * N_LAM * N_SIGMA_Z * N_NU_SA * N_ELL_SA * N_SIGMA_SA * N_BAND * N_BW_PER_DIM * N_LAMREC * N_GRID * N_BASIS * N_STEPS ))

echo "Total configs in this grid: ${TOTAL_CFG}"

if (( TASK < 0 || TASK >= TOTAL_CFG )); then
  echo "Error: TASK=${TASK} is outside valid range 0..$((TOTAL_CFG - 1))"
  exit 1
fi

IDX=$TASK
I_TARGET_POINT=$(( IDX % N_TARGET_POINT )); IDX=$(( IDX / N_TARGET_POINT ))
I_STEPS=$(( IDX % N_STEPS )); IDX=$(( IDX / N_STEPS ))
I_BASIS=$(( IDX % N_BASIS )); IDX=$(( IDX / N_BASIS ))
I_GRID=$(( IDX % N_GRID )); IDX=$(( IDX / N_GRID ))
I_LAMREC=$(( IDX % N_LAMREC )); IDX=$(( IDX / N_LAMREC ))
I_BW_PER_DIM=$(( IDX % N_BW_PER_DIM )); IDX=$(( IDX / N_BW_PER_DIM ))
I_BAND=$(( IDX % N_BAND )); IDX=$(( IDX / N_BAND ))
I_SIGMA_SA=$(( IDX % N_SIGMA_SA )); IDX=$(( IDX / N_SIGMA_SA ))
I_ELL_SA=$(( IDX % N_ELL_SA )); IDX=$(( IDX / N_ELL_SA ))
I_NU_SA=$(( IDX % N_NU_SA )); IDX=$(( IDX / N_NU_SA ))
I_SIGMA_Z=$(( IDX % N_SIGMA_Z )); IDX=$(( IDX / N_SIGMA_Z ))
I_LAM=$(( IDX % N_LAM )); IDX=$(( IDX / N_LAM ))
I_ELL_Z=$(( IDX % N_ELL_Z )); IDX=$(( IDX / N_ELL_Z ))
I_NU_Z=$(( IDX % N_NU_Z ))

NU_Z=${NU_Z_ARR[$I_NU_Z]}
ELL_Z=${ELL_Z_ARR[$I_ELL_Z]}
LAM=${LAM_ARR[$I_LAM]}
SIGMA_Z=${SIGMA_Z_ARR[$I_SIGMA_Z]}
NU_SA=${NU_SA_ARR[$I_NU_SA]}
ELL_SA=${ELL_SA_ARR[$I_ELL_SA]}
SIGMA_SA=${SIGMA_SA_ARR[$I_SIGMA_SA]}
BAND=${BAND_ARR[$I_BAND]}
BW_PER_DIM=${BW_PER_DIM_ARR[$I_BW_PER_DIM]}
LAMREC=${LAMREC_ARR[$I_LAMREC]}
GRID=${GRID_ARR[$I_GRID]}
MEAN_EMBEDDING_BASIS_SIZE=${MEAN_EMBEDDING_BASIS_SIZE_ARR[$I_BASIS]}
STEPS=${STEPS_ARR[$I_STEPS]}
TARGET_POINT=${TARGET_POINT_ARR[$I_TARGET_POINT]}

CFG_INDEX=$(( TASK / N_TARGET_POINT ))
OUTROOT="evaluation_results/expedia_encoded_10k_linear3_fixed/main_cfg_${CFG_INDEX}_${TARGET_POINT}"

echo "Resolved config:"
echo "  PYFILE       = ${PYFILE}"
echo "  DATA_BASE    = ${DATA_BASE}"
echo "  POLICY_CKPT  = ${POLICY_CKPT_DIR}"
echo "  ENCODER      = ${STATE_ENCODER_PATH}"
echo "  CFG_INDEX    = ${CFG_INDEX}"
echo "  NU_Z         = ${NU_Z}"
echo "  ELL_Z        = ${ELL_Z}"
echo "  LAM          = ${LAM}"
echo "  SIGMA_Z      = ${SIGMA_Z}"
echo "  NU_SA        = ${NU_SA}"
echo "  ELL_SA       = ${ELL_SA}"
echo "  SIGMA_SA     = ${SIGMA_SA}"
echo "  BAND         = ${BAND}"
echo "  BW_PER_DIM   = ${BW_PER_DIM}"
echo "  LAMREC       = ${LAMREC}"
echo "  GRID         = ${GRID}"
echo "  BASIS L      = ${MEAN_EMBEDDING_BASIS_SIZE}"
echo "  STEPS        = ${STEPS}"
echo "  TARGET_POINT = ${TARGET_POINT}"
echo "  OUTROOT      = ${OUTROOT}"

echo "Checking ke_drl estimate_embedding signature:"
python - <<'PY' || true
import inspect
from ke_drl.api import estimate_embedding
sig = str(inspect.signature(estimate_embedding))
for x in [
    "mean_embedding_basis_size",
    "normalize_eta",
    "eta_clip_max",
    "ratio_lambda_reg",
    "ratio_alpha_mix",
    "nu",
    "length_scale",
    "sigma",
]:
    print(f"  {x}: {x in sig}")
PY

python -u "${PYFILE}" \
  --policy-objective both \
  --cfg-index "${CFG_INDEX}" \
  --seed 2026 \
  --run-test 1 \
  --do-plots 1 \
  --out-root "${OUTROOT}" \
  --data-base "${DATA_BASE}" \
  --train-blob expedia_train_timeindexed.pt \
  --test-blob expedia_test_timeindexed.pt \
  --policy-ckpt-dir "${POLICY_CKPT_DIR}" \
  --state-encoder-path "${STATE_ENCODER_PATH}" \
  --num-steps "${STEPS}" \
  --embedding-lr 5e-5 \
  --num-grid-points "${GRID}" \
  --mean-embedding-basis-size "${MEAN_EMBEDDING_BASIS_SIZE}" \
  --mean-embedding-basis-method kmeans \
  --mean-embedding-basis-standardize 1 \
  --mean-embedding-basis-candidate-pool 0 \
  --mean-embedding-basis-max-iter 30 \
  --mean-embedding-basis-batch-size 8192 \
  --lambda-B 1e-10 \
  --max-train 10000 \
  --max-test 4000 \
  --state-cols mean_hist_price,std_hist_price,prop_location_score1,srch_room_count,srch_length_of_stay,prop_review_score,comp_inv \
  --categorical-state-cols srch_room_count,srch_length_of_stay,comp_inv \
  --one-hot-categoricals 1 \
  --action-cols avg_price_per_night,total_promotions,std_price_usd \
  --reward-cols gross_revenue_per_night,total_clicks \
  --device cuda \
  --nu-Z "${NU_Z}" \
  --ell-Z "${ELL_Z}" \
  --lambda-reg "${LAM}" \
  --sigma-Z "${SIGMA_Z}" \
  --nu-sa "${NU_SA}" \
  --ell-sa "${ELL_SA}" \
  --sigma-sa "${SIGMA_SA}" \
  --gamma 0.8 \
  --bandwidth "${BAND}" \
  --bandwidth-per-dim "${BW_PER_DIM}" \
  --lambda-rec "${LAMREC}" \
  --support-sample-n 500 \
  --clip-sampled-actions 1 \
  --std-cap-for-diagnostics 0.25 \
  --target-point-mode "${TARGET_POINT}" \
  --policy-specific-initial-action 1 \
  --discrete-reward-cols total_clicks \
  --export-subsets 1 \
  --simplex-calibrate-B 0 \
  --simplex-calib-ridge 1e-3 \
  --simplex-calib-max-rows 0 \
  --rulsif-alpha "${RULSIF_ALPHA}" \
  --normalize-eta 1 \
  --eta-clip-max 20 \
  --ratio-lambda-reg 1e-2 \
  --density-recovery-steps 20000 \
  --density-recovery-lr 1e-2 \
  --density-recovery-tol 1e-10 \
  --density-recovery-ridge 1e-5 \
  --density-recovery-init positive_beta \
  --density-recovery-anchor positive_beta \
  --density-recovery-l2-lambda 1e-2 \
  --density-recovery-kl-lambda 0.0 \
  --density-recovery-quad-points 21 \
  --density-recovery-batch-atoms 128 \
  --density-recovery-num-points 400 \
  --mean-embedding-contour-n1 120 \
  --mean-embedding-contour-n2 120 \
  --mean-embedding-contour-levels 20 \
  --mean-embedding-contour-pad 0.00

echo "End: $(date)"
