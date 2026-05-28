#!/bin/bash
#SBATCH --account=bfez-dtai-gh
#SBATCH --partition=ghx4
##SBATCH --account=bfez-delta-gpu
##SBATCH --partition=gpuA100x8
#SBATCH --job-name=eval2
#SBATCH --output=logs/DeltaAI_ev_%A_%a.log
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=6g
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehrdad3@illinois.edu
#SBATCH --array=0-15

echo "Job ID:    ${SLURM_JOB_ID}"
echo "Array idx: ${SLURM_ARRAY_TASK_ID}"
echo "CPUs/node: ${SLURM_CPUS_ON_NODE}"
echo "Mem/node:  ${SLURM_MEM_PER_NODE}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

## load modules/env -- delta ai
module purge
module load nvidia/25.5
module load cuda/12.9
module load craype-accel-nvidia90
module load python/miniforge3_pytorch/2.11.0
eval "$(conda shell.bash hook)"
conda activate deltaAI_env

##load module/env -- delta
#module load slurm-env/0.1
#module load cudatoolkit/25.3_12.8
#module load pytorch-conda/2.8
#eval "$(conda shell.bash hook)"
#conda activate delta_env

set -euo pipefail

mkdir -p logs

# If needed, uncomment and set your project directory.
# cd /path/to/your/project

TASK=${SLURM_ARRAY_TASK_ID}
PYFILE="policy_evaluation12.py"

NU_ARR=(3.5 4.5)
ELL_ARR=(1.8 1.5)
LAM_ARR=(0.003 0.0005)
SIG_ARR=(1.0 0.7)

# Legacy scalar fallback bandwidth. It is not the active marginal smoother
# when --bandwidth-per-dim is provided, but keeping it is harmless.
BAND_ARR=(0.0)

# Actual per-dimension bandwidths for recovered marginals.
# Order must match:
#   --reward-cols gross_revenue_per_night,total_clicks
BW_PER_DIM_ARR=("90,1")

DISCRETE_REWARD_COLS="total_clicks"

LAMREC_ARR=(0.005)
GRID_ARR=(600)
STEPS_ARR=(1000)

N_NU=${#NU_ARR[@]}
N_ELL=${#ELL_ARR[@]}
N_LAM=${#LAM_ARR[@]}
N_SIG=${#SIG_ARR[@]}
N_BAND=${#BAND_ARR[@]}
N_BW_PER_DIM=${#BW_PER_DIM_ARR[@]}
N_LAMREC=${#LAMREC_ARR[@]}
N_GRID=${#GRID_ARR[@]}
N_STEPS=${#STEPS_ARR[@]}

TOTAL_CFG=$(( N_NU * N_ELL * N_LAM * N_SIG * N_BAND * N_BW_PER_DIM * N_LAMREC * N_GRID * N_STEPS ))

if (( TASK < 0 || TASK >= TOTAL_CFG )); then
  echo "Error: TASK=${TASK} is outside valid range 0..$((TOTAL_CFG - 1))"
  exit 1
fi

IDX=$TASK

I_STEPS=$(( IDX % N_STEPS ))
IDX=$(( IDX / N_STEPS ))

I_GRID=$(( IDX % N_GRID ))
IDX=$(( IDX / N_GRID ))

I_LAMREC=$(( IDX % N_LAMREC ))
IDX=$(( IDX / N_LAMREC ))

I_BW_PER_DIM=$(( IDX % N_BW_PER_DIM ))
IDX=$(( IDX / N_BW_PER_DIM ))

I_BAND=$(( IDX % N_BAND ))
IDX=$(( IDX / N_BAND ))

I_SIG=$(( IDX % N_SIG ))
IDX=$(( IDX / N_SIG ))

I_LAM=$(( IDX % N_LAM ))
IDX=$(( IDX / N_LAM ))

I_ELL=$(( IDX % N_ELL ))
IDX=$(( IDX / N_ELL ))

I_NU=$(( IDX % N_NU ))

NU=${NU_ARR[$I_NU]}
ELL=${ELL_ARR[$I_ELL]}
LAM=${LAM_ARR[$I_LAM]}
SIG=${SIG_ARR[$I_SIG]}
BAND=${BAND_ARR[$I_BAND]}
BW_PER_DIM=${BW_PER_DIM_ARR[$I_BW_PER_DIM]}
LAMREC=${LAMREC_ARR[$I_LAMREC]}
GRID=${GRID_ARR[$I_GRID]}
STEPS=${STEPS_ARR[$I_STEPS]}

CFG_INDEX=${TASK}
OUTROOT="evaluation_results/5-4-2026-5k-B/cfg_${CFG_INDEX}"

echo "Resolved config:"
echo "  PYFILE      = ${PYFILE}"
echo "  CFG_INDEX   = ${CFG_INDEX}"
echo "  NU          = ${NU}"
echo "  ELL         = ${ELL}"
echo "  LAM         = ${LAM}"
echo "  SIG         = ${SIG}"
echo "  BAND        = ${BAND}"
echo "  BW_PER_DIM  = ${BW_PER_DIM}"
echo "  DISCRETE_REWARD_COLS = ${DISCRETE_REWARD_COLS}"
echo "  LAMREC      = ${LAMREC}"
echo "  GRID        = ${GRID}"
echo "  STEPS       = ${STEPS}"
echo "  OUTROOT     = ${OUTROOT}"

python -u "${PYFILE}" \
  --policy-objective both \
  --cfg-index "${CFG_INDEX}" \
  --seed 2026 \
  --run-test 1 \
  --do-plots 1 \
  --out-root "${OUTROOT}" \
  --data-base data \
  --train-blob expedia_train_timeindexed.pt \
  --val-blob expedia_val_timeindexed.pt \
  --test-blob expedia_test_timeindexed.pt \
  --policy-ckpt-dir policies_linear2 \
  --num-steps "${STEPS}" \
  --num-grid-points "${GRID}" \
  --max-train 5000 \
  --max-val 4000 \
  --max-test 4000 \
  --state-cols mean_hist_price,std_hist_price,prop_location_score1,srch_room_count,srch_length_of_stay,prop_review_score,comp_inv \
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
  --density-recovery-quad-points 64 \
  --density-recovery-batch-atoms 128 \
  --density-recovery-num-points 1200 \
  --mean-embedding-contour-n1 220 \
  --mean-embedding-contour-n2 220 \
  --mean-embedding-contour-levels 55 \
  --mean-embedding-contour-pad 0 \

