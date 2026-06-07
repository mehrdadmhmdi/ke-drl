#!/bin/bash
#SBATCH --account=bfez-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=expedia_eval
#SBATCH --output=logs/DeltaAI_ev_%A_%a.log
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8g
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehrdad3@illinois.edu
#SBATCH --array=0-15

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ -f "${SUBMIT_DIR}/policy_evaluation.py" ]; then
  SCRIPT_DIR="$(cd "${SUBMIT_DIR}" && pwd)"
elif [ -f "${SUBMIT_DIR}/real_data_example/policy_evaluation.py" ]; then
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
echo "Array idx: ${SLURM_ARRAY_TASK_ID}"
echo "CPUs/node: ${SLURM_CPUS_ON_NODE:-NA}"
echo "Mem/node:  ${SLURM_MEM_PER_NODE:-NA}"
echo "Node:      $(hostname)"
echo "Start:     $(date)"

# Conda/module hooks can reference unset backup variables under `set -u`.
set +u
module purge
module load cuda/12.6.1
module load default gcc/14.2.0 craype-accel-nvidia90
module load python/miniforge3_pytorch/2.7.0
echo $SLURM_JOB_PARTITION
echo ${CUDA_VISIBLE_DEVICES:-}
eval "$(conda shell.bash hook)"
set -u

DATA_BASE="${DATA_BASE:-./Expedia_data}"
POLICY_CKPT_DIR="${POLICY_CKPT_DIR:-./policies_linear2}"
STATE_ENCODER_PATH="${STATE_ENCODER_PATH:-${POLICY_CKPT_DIR}/state_encoder.json}"
PYFILE="${PYFILE:-policy_evaluation.py}"
TASK="${SLURM_ARRAY_TASK_ID}"

NU_ARR=(6.5 4.5)
ELL_ARR=(1.5 2.0)
LAM_ARR=(0.002 0.0005)
SIG_ARR=(1.0 0.7)
BAND_ARR=(0.0)
BW_PER_DIM_ARR=("90,1")
DISCRETE_REWARD_COLS="total_clicks"
LAMREC_ARR=(0.005)
GRID_ARR=(600)
MEAN_EMBEDDING_BASIS_SIZE_ARR=(320)
STEPS_ARR=(1000)

N_NU=${#NU_ARR[@]}
N_ELL=${#ELL_ARR[@]}
N_LAM=${#LAM_ARR[@]}
N_SIG=${#SIG_ARR[@]}
N_BAND=${#BAND_ARR[@]}
N_BW_PER_DIM=${#BW_PER_DIM_ARR[@]}
N_LAMREC=${#LAMREC_ARR[@]}
N_GRID=${#GRID_ARR[@]}
N_BASIS=${#MEAN_EMBEDDING_BASIS_SIZE_ARR[@]}
N_STEPS=${#STEPS_ARR[@]}

TOTAL_CFG=$(( N_NU * N_ELL * N_LAM * N_SIG * N_BAND * N_BW_PER_DIM * N_LAMREC * N_GRID * N_BASIS * N_STEPS ))

if (( TASK < 0 || TASK >= TOTAL_CFG )); then
  echo "Error: TASK=${TASK} is outside valid range 0..$((TOTAL_CFG - 1))"
  exit 1
fi

IDX=$TASK
I_STEPS=$(( IDX % N_STEPS )); IDX=$(( IDX / N_STEPS ))
I_BASIS=$(( IDX % N_BASIS )); IDX=$(( IDX / N_BASIS ))
I_GRID=$(( IDX % N_GRID )); IDX=$(( IDX / N_GRID ))
I_LAMREC=$(( IDX % N_LAMREC )); IDX=$(( IDX / N_LAMREC ))
I_BW_PER_DIM=$(( IDX % N_BW_PER_DIM )); IDX=$(( IDX / N_BW_PER_DIM ))
I_BAND=$(( IDX % N_BAND )); IDX=$(( IDX / N_BAND ))
I_SIG=$(( IDX % N_SIG )); IDX=$(( IDX / N_SIG ))
I_LAM=$(( IDX % N_LAM )); IDX=$(( IDX / N_LAM ))
I_ELL=$(( IDX % N_ELL )); IDX=$(( IDX / N_ELL ))
I_NU=$(( IDX % N_NU ))

NU=${NU_ARR[$I_NU]}
ELL=${ELL_ARR[$I_ELL]}
LAM=${LAM_ARR[$I_LAM]}
SIG=${SIG_ARR[$I_SIG]}
BAND=${BAND_ARR[$I_BAND]}
BW_PER_DIM=${BW_PER_DIM_ARR[$I_BW_PER_DIM]}
LAMREC=${LAMREC_ARR[$I_LAMREC]}
GRID=${GRID_ARR[$I_GRID]}
MEAN_EMBEDDING_BASIS_SIZE=${MEAN_EMBEDDING_BASIS_SIZE_ARR[$I_BASIS]}
STEPS=${STEPS_ARR[$I_STEPS]}

CFG_INDEX=${TASK}
OUTROOT="evaluation_results/expedia_encoded_10k/cfg_${CFG_INDEX}"

echo "Resolved config:"
echo "  PYFILE       = ${PYFILE}"
echo "  DATA_BASE    = ${DATA_BASE}"
echo "  POLICY_CKPT  = ${POLICY_CKPT_DIR}"
echo "  ENCODER      = ${STATE_ENCODER_PATH}"
echo "  CFG_INDEX    = ${CFG_INDEX}"
echo "  NU           = ${NU}"
echo "  ELL          = ${ELL}"
echo "  LAM          = ${LAM}"
echo "  SIG          = ${SIG}"
echo "  BAND         = ${BAND}"
echo "  BW_PER_DIM   = ${BW_PER_DIM}"
echo "  LAMREC       = ${LAMREC}"
echo "  GRID         = ${GRID}"
echo "  BASIS L      = ${MEAN_EMBEDDING_BASIS_SIZE}"
echo "  STEPS        = ${STEPS}"
echo "  OUTROOT      = ${OUTROOT}"

python -u "${PYFILE}" \
  --policy-objective both \
  --cfg-index "${CFG_INDEX}" \
  --seed 2026 \
  --run-test 1 \
  --do-plots 1 \
  --out-root "${OUTROOT}" \
  --data-base "${DATA_BASE}" \
  --train-blob expedia_train_timeindexed.pt \
  --val-blob expedia_val_timeindexed.pt \
  --test-blob expedia_test_timeindexed.pt \
  --policy-ckpt-dir "${POLICY_CKPT_DIR}" \
  --state-encoder-path "${STATE_ENCODER_PATH}" \
  --num-steps "${STEPS}" \
  --num-grid-points "${GRID}" \
  --mean-embedding-basis-size "${MEAN_EMBEDDING_BASIS_SIZE}" \
  --mean-embedding-basis-method kmeans \
  --mean-embedding-basis-standardize 1 \
  --mean-embedding-basis-max-iter 20 \
  --lambda-B 0.0 \
  --max-train 10000 \
  --max-val 4000 \
  --max-test 4000 \
  --state-cols mean_hist_price,std_hist_price,prop_location_score1,srch_room_count,srch_length_of_stay,prop_review_score,comp_inv \
  --categorical-state-cols srch_room_count,srch_length_of_stay,comp_inv \
  --one-hot-categoricals 1 \
  --action-cols avg_price_per_night,total_promotions,std_price_usd \
  --reward-cols gross_revenue_per_night,total_clicks \
  --device cuda \
  --nu-Z "${NU}" \
  --ell-Z "${ELL}" \
  --lambda-reg "${LAM}" \
  --sigma-Z "${SIG}" \
  --gamma 0.8 \
  --bandwidth "${BAND}" \
  --bandwidth-per-dim "${BW_PER_DIM}" \
  --lambda-rec "${LAMREC}" \
  --support-sample-n 500 \
  --clip-sampled-actions 1 \
  --std-cap-for-diagnostics 0.25 \
  --discrete-reward-cols "${DISCRETE_REWARD_COLS}" \
  --export-subsets 1 \
  --simplex-calibrate-B 1 \
  --simplex-calib-ridge 1e-3 \
  --simplex-calib-max-rows 0 \
  --density-recovery-steps 20000 \
  --density-recovery-lr 1e-2 \
  --density-recovery-tol 1e-10 \
  --density-recovery-ridge 1e-4 \
  --density-recovery-init positive_beta \
  --density-recovery-quad-points 256 \
  --density-recovery-batch-atoms 512 \
  --density-recovery-num-points 2200 \
  --mean-embedding-contour-n1 220 \
  --mean-embedding-contour-n2 220 \
  --mean-embedding-contour-levels 55 \
  --mean-embedding-contour-pad 0

echo "End: $(date)"
