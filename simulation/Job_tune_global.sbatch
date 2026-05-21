#!/bin/bash
##SBATCH --account=bfez-delta-gpu
#SBATCH --account=bfez-dtai-gh
#SBATCH --job-name=KEDRL_TUNE
#SBATCH --output=logs/tune_%A_%a.log
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
##SBATCH --partition=gpuA100x8
#SBATCH --partition=ghx4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mehrdad3@illinois.edu
#SBATCH --array=0-18

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

BASE=$PWD
RUN="$BASE/runs/tune_global_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$BASE/logs" "$RUN" "$RUN/data" "$RUN/logs" "$RUN/mu" "$RUN/metrics" "$RUN/plots"
SHARED_DATA_DIR="${KEDRL_SHARED_DATA_DIR:-$BASE/runs/tune_shared_data/data}"

if [ -z "${KEDRL_SRC:-}" ]; then
  for candidate in \
    "$BASE/../src" \
    "$BASE/../kedrl_git/src" \
    "$BASE/src" \
    "/work/nvme/bfez/${USER}/DistRL/src" \
    "/work/nvme/bfez/${USER}/DistRL/kedrl_git/src"; do
    if [ -d "$candidate/ke_drl" ]; then
      KEDRL_SRC="$candidate"
      break
    fi
  done
fi
if [ -n "${KEDRL_SRC:-}" ] && [ -d "$KEDRL_SRC" ]; then
  export KEDRL_SRC
  export PYTHONPATH="$KEDRL_SRC:${PYTHONPATH:-}"
  echo "KEDRL_SRC: $KEDRL_SRC"
else
  echo "Could not find current ke_drl source. Set KEDRL_SRC=/path/to/kedrl_git/src before sbatch." >&2
  exit 1
fi

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
python3 main_tune_global.py --combo-id "${SLURM_ARRAY_TASK_ID}" --shared-data-dir "$SHARED_DATA_DIR"
