#!/bin/bash
##SBATCH --account=bfez-delta-gpu
#SBATCH --account=bfez-dtai-gh
#SBATCH --job-name=UL_SIM2
#SBATCH --output=results/%x_%A_%a.log
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

module purge
module load cuda/12.6.1
module load default gcc/14.2.0 craype-accel-nvidia90
module load python/miniforge3_pytorch/2.7.0
echo $SLURM_JOB_PARTITION
echo ${CUDA_VISIBLE_DEVICES:-}

set -euo pipefail

BASE=$PWD
SIM2_STAGE="${SIM2_STAGE:-}"
if [ -z "$SIM2_STAGE" ]; then
  if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SIM2_STAGE=fit
  else
    SIM2_STAGE=prepare
  fi
fi
SIM2_TIMEPOINTS="${SIM2_TIMEPOINTS:-50}"
SIM2_N_IDS="${SIM2_N_IDS:-300}"
SIM2_REDUCED_N="${SIM2_REDUCED_N:-200}"
SIM2_TAG="${SIM2_TAG:-T${SIM2_TIMEPOINTS}_N${SIM2_N_IDS}_r${SIM2_REDUCED_N}}"
SIM2_POLICY_PAIR="${SIM2_POLICY_PAIR:-UL}"
case "$SIM2_POLICY_PAIR" in
  UG|UL|GU|GL|LU|LG) ;;
  *)
    echo "SIM2_POLICY_PAIR must be one of UG, UL, GU, GL, LU, LG; got ${SIM2_POLICY_PAIR}" >&2
    exit 1
    ;;
esac
if [ -n "${SLURM_JOB_ID:-}" ]; then
  scontrol update JobId="${SLURM_JOB_ID}" JobName="${SIM2_POLICY_PAIR}_${SIM2_STAGE}" >/dev/null 2>&1 || true
fi

RESULT_ROOT="${SIM2_RESULT_ROOT:-$BASE/results/$SIM2_POLICY_PAIR}"
SHARED_RUN="${SIM2_SHARED_RUN:-$RESULT_ROOT/shared}"
RUN="${SIM2_RUN:-$RESULT_ROOT}"
if [ "$SIM2_STAGE" = "prepare" ]; then
  WORK="$SHARED_RUN"
else
  WORK="$RUN"
fi
SHARED_DATA_DIR="$SHARED_RUN/data"

mkdir -p "$BASE/results" "$RESULT_ROOT" "$WORK" "$WORK/data" "$WORK/logs" "$WORK/mu" "$WORK/metrics" "$WORK/plots"

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
  KEDRL_PIP_TARGET="$WORK/.kedrl_site_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
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

if [ "$SIM2_STAGE" = "prepare" ]; then
  export OFFLINE_DATA_WORKERS="${OFFLINE_DATA_WORKERS:-8}"
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
  main_sim2.py \
  mu_plot.py \
  sim_utils.py \
  sim_eval.py \
  validate_sim_config.py \
  parallel_offlinedata.py \
  params_tune.yaml
do
  cp "$BASE/$f" "$WORK/"
done

cd "$WORK"
echo "Simulation-2 stage: ${SIM2_STAGE}"
echo "Simulation-2 tag: ${SIM2_TAG}"
echo "Simulation-2 policy pair: ${SIM2_POLICY_PAIR}"
echo "Shared data: ${SHARED_DATA_DIR}"
echo "Run folder: ${RUN}"

case "$SIM2_STAGE" in
  prepare)
    python3 main_sim2.py --mode prepare
    echo "SIMULATION_2 SHARED DATA PREP DONE"
    ;;
  fit)
    if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
      echo "SIM2_STAGE=fit requires sbatch --array=0-N." >&2
      exit 1
    fi
    if [ ! -d "$SHARED_DATA_DIR" ]; then
      echo "Shared simulation-2 data not found: $SHARED_DATA_DIR" >&2
      echo "Run SIM2_STAGE=prepare first." >&2
      exit 1
    fi
    python3 main_sim2.py \
      --mode fit \
      --offline-id "${SLURM_ARRAY_TASK_ID}" \
      --shared-data-dir "$SHARED_DATA_DIR"
    ;;
  aggregate)
    if [ ! -d "$SHARED_DATA_DIR" ]; then
      echo "Shared simulation-2 data not found: $SHARED_DATA_DIR" >&2
      echo "Run SIM2_STAGE=prepare first." >&2
      exit 1
    fi
    python3 main_sim2.py --mode aggregate --shared-data-dir "$SHARED_DATA_DIR"
    echo "SIMULATION_2 AGGREGATION DONE"
    ;;
  commands)
    python3 main_sim2.py --mode commands
    ;;
  *)
    echo "Unknown SIM2_STAGE=$SIM2_STAGE. Use prepare, fit, aggregate, or commands." >&2
    exit 1
    ;;
esac
