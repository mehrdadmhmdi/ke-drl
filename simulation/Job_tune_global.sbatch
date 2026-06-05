#!/bin/bash
##SBATCH --account=bfez-delta-gpu
#SBATCH --account=bfez-dtai-gh
#SBATCH --job-name=SIM2_TUNE
#SBATCH --output=results/tune_%A_%a.log
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128g
#SBATCH --cpus-per-task=64
##SBATCH --partition=gpuA100x8
#SBATCH --partition=ghx4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mehrdad3@illinois.edu
#SBATCH --array=0-99

echo "Job ID:    ${SLURM_JOB_ID}"
echo "Array idx: ${SLURM_ARRAY_TASK_ID}"
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
echo $CUDA_VISIBLE_DEVICES

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export TORCH_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"

set -euo pipefail

BASE=$PWD
SIM2_POLICY_PAIR="${SIM2_POLICY_PAIR:-UL}"
case "$SIM2_POLICY_PAIR" in
  UG|UL|GU|GL|LU|LG) ;;
  *)
    echo "SIM2_POLICY_PAIR must be one of UG, UL, GU, GL, LU, LG; got ${SIM2_POLICY_PAIR}" >&2
    exit 1
    ;;
esac
export SIM2_POLICY_PAIR
read COMBO_ID OFFLINE_ID COMBO_COUNT N_REP TOTAL_TASKS < <(python3 - <<'PY'
import os
import yaml

task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
with open("params_tune.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)
with open("tuning_grid.yaml", "r", encoding="utf-8") as f:
    grid = yaml.safe_load(f)
n_rep = int((params.get("experiment") or {}).get("num_replicates", 1))
combo_count = len(list(grid.get("combos") or []))
combo_id = task_id // n_rep
offline_id = task_id % n_rep
print(combo_id, offline_id, combo_count, n_rep, combo_count * n_rep)
PY
)
if [ "$COMBO_ID" -ge "$COMBO_COUNT" ]; then
  echo "Array task ${SLURM_ARRAY_TASK_ID} is outside total tuning tasks ${TOTAL_TASKS}; exiting."
  exit 0
fi
echo "Tuning combo id: ${COMBO_ID} / $((COMBO_COUNT - 1))"
echo "Offline replicate id: ${OFFLINE_ID} / $((N_REP - 1))"

RUN="$BASE/results/tune_global_${SIM2_POLICY_PAIR}_${SLURM_ARRAY_JOB_ID}_${COMBO_ID}"
mkdir -p "$BASE/results" "$RUN" "$RUN/data" "$RUN/logs" "$RUN/mu" "$RUN/metrics" "$RUN/plots"
SHARED_DATA_DIR="${KEDRL_SHARED_DATA_DIR:-$BASE/results/tune_shared_data_${SIM2_POLICY_PAIR}/data}"

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
  KEDRL_PIP_TARGET="$RUN/.kedrl_site_${SLURM_ARRAY_TASK_ID}"
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

cp "$BASE/main_offlinedata.py" "$RUN/"
cp "$BASE/main_MonteCarloZ.py" "$RUN/"
cp "$BASE/main_est.py" "$RUN/"
cp "$BASE/main_tune_global.py" "$RUN/"
cp "$BASE/sim2_profiles.py" "$RUN/"
cp "$BASE/mu_plot.py" "$RUN/"
cp "$BASE/sim_utils.py" "$RUN/"
cp "$BASE/sim_eval.py" "$RUN/"
cp "$BASE/validate_sim_config.py" "$RUN/"
cp "$BASE/parallel_offlinedata.py" "$RUN/"
cp "$BASE/params_tune.yaml" "$RUN/"
cp "$BASE/tuning_grid.yaml" "$RUN/"

cd "$RUN"
if [ ! -d "$SHARED_DATA_DIR" ]; then
  echo "Shared tuning data not found: $SHARED_DATA_DIR" >&2
  echo "Run Job_tune_prepare.sbatch first, or set KEDRL_SHARED_DATA_DIR." >&2
  exit 1
fi
python3 main_tune_global.py \
  --combo-id "${COMBO_ID}" \
  --offline-id "${OFFLINE_ID}" \
  --shared-data-dir "$SHARED_DATA_DIR"
