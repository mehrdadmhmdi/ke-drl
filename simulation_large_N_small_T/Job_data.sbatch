#!/bin/bash
##SBATCH --account=bfez-delta-gpu
#SBATCH --account=bfez-dtai-gh
#SBATCH --job-name=GU_data
#SBATCH --output=logs/data_%A_%a.log
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8g
#SBATCH --cpus-per-task=4
##SBATCH --partition=gpuA100x8
#SBATCH --partition=ghx4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehrdad3@illinois.edu
#SBATCH --array=0-499

# Access exported variables:
cd ${OUTDIR:-$PWD}
echo "Running in folder: $(pwd)"
mkdir -p logs data

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

set -euo pipefail

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
  KEDRL_PIP_TARGET="$PWD/.kedrl_site_${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
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

python3 validate_sim_config.py --params params.yaml
python3 main_offlinedata.py
python3 validate_sim_config.py --params params.yaml --data "data/offline_data_${SLURM_ARRAY_TASK_ID}.pt"
