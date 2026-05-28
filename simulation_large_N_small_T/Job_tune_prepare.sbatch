#!/bin/bash
##SBATCH --account=bfez-delta-gpu
#SBATCH --account=bfez-dtai-gh
#SBATCH --job-name=KEDRL_TUNE_DATA
#SBATCH --output=logs/tune_data.%j.log
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128g
#SBATCH --cpus-per-task=64
##SBATCH --partition=gpuA100x8
#SBATCH --partition=ghx4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mehrdad3@illinois.edu

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

export OFFLINE_DATA_WORKERS="${SLURM_CPUS_PER_TASK:-64}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

set -euo pipefail

BASE=$PWD
RUN="${KEDRL_SHARED_RUN:-$BASE/runs/tune_shared_data}"
mkdir -p "$BASE/logs"
N_REP=$(python3 - <<'PY'
import yaml
with open("params_tune.yaml", "r", encoding="utf-8") as f:
    p = yaml.safe_load(f)
print(int((p.get("experiment") or {}).get("num_replicates", 1)))
PY
)
LAST_REP=$((N_REP - 1))
CURRENT_FINGERPRINT=$(sha256sum \
  params_tune.yaml \
  main_MonteCarloZ.py \
  sim_utils.py \
  validate_sim_config.py \
  parallel_offlinedata.py \
  main_offlinedata.py \
  | sha256sum | awk '{print $1}')
FINGERPRINT_FILE="$RUN/data/prep_fingerprint.sha256"

if [ -e "$RUN/data/Z_true.pt" ] && [ -e "$RUN/data/offline_data_0.pt" ] && [ -e "$RUN/data/offline_data_${LAST_REP}.pt" ] && [ "${KEDRL_PREP_FORCE:-0}" != "1" ]; then
  if [ -e "$FINGERPRINT_FILE" ] && [ "$(cat "$FINGERPRINT_FILE")" = "$CURRENT_FINGERPRINT" ]; then
    echo "Shared tuning data already exists at $RUN and matches current parameters/scripts."
    echo "Set KEDRL_PREP_FORCE=1 to regenerate it."
    exit 0
  fi
  echo "Shared tuning data exists but parameters/scripts changed; regenerating $RUN."
fi

if [ -d "$RUN" ] && { [ "${KEDRL_PREP_FORCE:-0}" = "1" ] || [ ! -e "$FINGERPRINT_FILE" ] || [ "$(cat "$FINGERPRINT_FILE" 2>/dev/null || true)" != "$CURRENT_FINGERPRINT" ]; }; then
  case "$(realpath "$RUN")" in
    "$(realpath "$BASE")"/runs/tune_shared_data*) rm -rf "$RUN" ;;
    *) echo "Refusing to remove unexpected path: $RUN" >&2; exit 1 ;;
  esac
fi

mkdir -p "$RUN" "$RUN/data" "$RUN/logs"

unset KEDRL_SRC
KEDRL_GIT_URL="${KEDRL_GIT_URL:-https://github.com/mehrdadmhmdi/ke-drl.git}"
KEDRL_GIT_REF="${KEDRL_GIT_REF:-main}"
KEDRL_PIP_TARGET="$RUN/.kedrl_site"
rm -rf "$KEDRL_PIP_TARGET"
mkdir -p "$KEDRL_PIP_TARGET"
echo "Installing ke_drl from git: ${KEDRL_GIT_URL}@${KEDRL_GIT_REF}"
python3 -m pip install --no-cache-dir --upgrade --force-reinstall --no-deps \
  --target "$KEDRL_PIP_TARGET" "git+${KEDRL_GIT_URL}@${KEDRL_GIT_REF}"
export PYTHONPATH="$KEDRL_PIP_TARGET:${PYTHONPATH:-}"
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
cp "$BASE/sim_utils.py" "$RUN/"
cp "$BASE/validate_sim_config.py" "$RUN/"
cp "$BASE/parallel_offlinedata.py" "$RUN/"
cp "$BASE/params_tune.yaml" "$RUN/params.yaml"

cd "$RUN"
echo "Shared tuning data folder: $(pwd)"
echo "Using params:"
python3 - <<'PY'
import yaml
with open("params.yaml", "r", encoding="utf-8") as f:
    p = yaml.safe_load(f)
print({
    "n_ids": p["n_ids"],
    "Z_sim": p["Z_sim"],
    "experiment": p["experiment"],
    "target_set": p["target_set"],
    "num_grid_points": p["num_grid_points"],
    "num_steps": p["optimization"]["num_steps"],
})
PY

python3 validate_sim_config.py --params params.yaml
python3 parallel_offlinedata.py --params params.yaml
python3 main_MonteCarloZ.py
printf '%s\n' "$CURRENT_FINGERPRINT" > data/prep_fingerprint.sha256

echo "Shared data outputs:"
ls -lh data/Z_true.pt data/offline_data_0.pt "data/offline_data_${LAST_REP}.pt"
echo "TUNING DATA PREP DONE"
