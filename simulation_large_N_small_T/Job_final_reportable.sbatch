#!/bin/bash
##SBATCH --account=bfez-delta-gpu
#SBATCH --account=bfez-dtai-gh
#SBATCH --job-name=KEDRL_FINAL
#SBATCH --output=logs/final_%A_%a.log
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=192g
#SBATCH --cpus-per-task=64
##SBATCH --partition=gpuA100x8
#SBATCH --partition=ghx4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mehrdad3@illinois.edu

echo "Job ID:    ${SLURM_JOB_ID}"
echo "Array idx: ${SLURM_ARRAY_TASK_ID:-}"
echo "CPUs/node: ${SLURM_CPUS_ON_NODE}"
echo "Mem/node:  ${SLURM_MEM_PER_NODE}"

## load modules/env -- delta ai
module purge
module load cuda/12.6.1
module load default gcc/14.2.0 craype-accel-nvidia90
module load python/miniforge3_pytorch/2.7.0
##load module/env -- delta
#module load slurm-env/0.1
#module load cudatoolkit/25.3_12.8
#module load pytorch-conda/2.8
#eval "$(conda shell.bash hook)"   # IMPORTANT
#conda activate delta_env
echo $SLURM_JOB_PARTITION
echo ${CUDA_VISIBLE_DEVICES:-}

set -euo pipefail

BASE=$PWD
FINAL_STAGE="${FINAL_STAGE:-}"
if [ -z "$FINAL_STAGE" ]; then
  if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    FINAL_STAGE=fit
  else
    FINAL_STAGE=prepare
  fi
fi

FINAL_TAG="${KEDRL_FINAL_TAG:-sigma0p7_large}"
FINAL_RUN="$BASE/runs/final_reportable_${FINAL_TAG}"
FINAL_SHARED_RUN="${KEDRL_FINAL_SHARED_RUN:-$BASE/runs/final_reportable_shared_data_${FINAL_TAG}}"
FINAL_SHARED_DATA_DIR="$FINAL_SHARED_RUN/data"

mkdir -p "$BASE/logs"

if [ "$FINAL_STAGE" = "prepare" ]; then
  RUN="$FINAL_SHARED_RUN"
else
  RUN="$FINAL_RUN"
fi
mkdir -p "$RUN" "$RUN/data" "$RUN/logs" "$RUN/mu" "$RUN/metrics" "$RUN/plots"

KEDRL_REPO_ROOT="${KEDRL_REPO_ROOT:-}"
if [ -z "$KEDRL_REPO_ROOT" ]; then
  for candidate in "${BASE:-$PWD}" "${BASE:-$PWD}/.." "$PWD" "$PWD/.." "$PWD/../.." "$PWD/../../.." "${SLURM_SUBMIT_DIR:-$PWD}" "${SLURM_SUBMIT_DIR:-$PWD}/.."; do
    if [ -d "$candidate/src/ke_drl" ]; then
      KEDRL_REPO_ROOT="$(cd "$candidate" && pwd)"
      break
    fi
  done
fi
KEDRL_GIT_URL="${KEDRL_GIT_URL:-https://github.com/mehrdadmhmdi/ke-drl.git}"
KEDRL_GIT_REF="${KEDRL_GIT_REF:-main}"
if [ -n "$KEDRL_REPO_ROOT" ] && [ -d "$KEDRL_REPO_ROOT/src/ke_drl" ]; then
  export KEDRL_SRC="$KEDRL_REPO_ROOT/src"
  export PYTHONPATH="$KEDRL_SRC:${PYTHONPATH:-}"
  echo "Using local ke_drl source: $KEDRL_SRC"
else
  unset KEDRL_SRC
  KEDRL_GIT_URL="${KEDRL_GIT_URL:-https://github.com/mehrdadmhmdi/ke-drl.git}"
  KEDRL_GIT_REF="${KEDRL_GIT_REF:-main}"
  KEDRL_PIP_TARGET="$RUN/.kedrl_site_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
  rm -rf "$KEDRL_PIP_TARGET"
  mkdir -p "$KEDRL_PIP_TARGET"
  echo "Installing ke_drl from git: ${KEDRL_GIT_URL}@${KEDRL_GIT_REF}"
  python3 -m pip install --no-cache-dir --upgrade --force-reinstall --no-deps \
    --target "$KEDRL_PIP_TARGET" "git+${KEDRL_GIT_URL}@${KEDRL_GIT_REF}"
  export PYTHONPATH="$KEDRL_PIP_TARGET:${PYTHONPATH:-}"
fi
python3 - <<'PY'
import ke_drl
print(f"Installed ke_drl source: {ke_drl.__file__}")
PY

export VISPY_GL_BACKEND=egl
export VISPY_USE_APP=headless
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.8"

if [ "$FINAL_STAGE" = "prepare" ]; then
  export OFFLINE_DATA_WORKERS="${OFFLINE_DATA_WORKERS:-${SLURM_CPUS_PER_TASK:-64}}"
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
else
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
  export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
  export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
  export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
  export TORCH_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
fi

for f in \
  main_offlinedata.py \
  main_MonteCarloZ.py \
  main_est.py \
  main_tune_global.py \
  main_final_reportable.py \
  mu_plot.py \
  sim_utils.py \
  sim_eval.py \
  validate_sim_config.py \
  parallel_offlinedata.py \
  params_tune.yaml
do
  cp "$BASE/$f" "$RUN/"
done

cd "$RUN"
echo "Final stage: ${FINAL_STAGE}"
echo "Final run folder: ${FINAL_RUN}"
echo "Final shared data folder: ${FINAL_SHARED_DATA_DIR}"

case "$FINAL_STAGE" in
  prepare)
    python3 main_final_reportable.py --mode prepare
    echo "FINAL SHARED DATA PREP DONE"
    ;;
  fit)
    if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
      echo "FINAL_STAGE=fit requires sbatch --array=0-N." >&2
      exit 1
    fi
    if [ ! -d "$FINAL_SHARED_DATA_DIR" ]; then
      echo "Shared final data not found: $FINAL_SHARED_DATA_DIR" >&2
      echo "Run FINAL_STAGE=prepare first." >&2
      exit 1
    fi
    python3 main_final_reportable.py \
      --mode fit \
      --offline-id "${SLURM_ARRAY_TASK_ID}" \
      --shared-data-dir "$FINAL_SHARED_DATA_DIR"
    ;;
  aggregate)
    if [ ! -d "$FINAL_SHARED_DATA_DIR" ]; then
      echo "Shared final data not found: $FINAL_SHARED_DATA_DIR" >&2
      echo "Run FINAL_STAGE=prepare first." >&2
      exit 1
    fi
    python3 main_final_reportable.py \
      --mode aggregate \
      --shared-data-dir "$FINAL_SHARED_DATA_DIR"
    echo "FINAL AGGREGATION DONE"
    ;;
  commands)
    python3 main_final_reportable.py --mode commands
    ;;
  *)
    echo "Unknown FINAL_STAGE=$FINAL_STAGE. Use prepare, fit, aggregate, or commands." >&2
    exit 1
    ;;
esac
