#!/bin/bash
#SBATCH --account=bfez-delta-gpu
#SBATCH --job-name=KEDRL_TUNE
#SBATCH --output=logs/tune_%A_%a.log
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA40x4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mehrdad3@illinois.edu
#SBATCH --array=0-8

echo "Job ID:    ${SLURM_JOB_ID}"
echo "Array idx: ${SLURM_ARRAY_TASK_ID}"
echo "CPUs/node: ${SLURM_CPUS_ON_NODE}"
echo "Mem/node:  ${SLURM_MEM_PER_NODE}"

## load modules/env -- delta ai
#module purge
#module load cuda/12.6.1
#module load craype-accel-nvidia90
#module load python/miniforge3_pytorch/2.7.0

##load module/env -- delta
module load slurm-env/0.1
module load cudatoolkit/25.3_12.8
module load pytorch-conda/2.8
eval "$(conda shell.bash hook)"   # IMPORTANT
conda activate delta_env

echo $SLURM_JOB_PARTITION
echo $CUDA_VISIBLE_DEVICES

set -euo pipefail

BASE=$PWD
RUN="$BASE/runs/tune_global_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$BASE/logs" "$RUN" "$RUN/data" "$RUN/logs" "$RUN/mu" "$RUN/metrics" "$RUN/plots"

if [ -z "${KEDRL_SRC:-}" ] && [ -d "$BASE/../src" ]; then
  KEDRL_SRC="$BASE/../src"
elif [ -z "${KEDRL_SRC:-}" ] && [ -d "$BASE/../kedrl_git/src" ]; then
  KEDRL_SRC="$BASE/../kedrl_git/src"
fi
if [ -n "${KEDRL_SRC:-}" ] && [ -d "$KEDRL_SRC" ]; then
  export KEDRL_SRC
  export PYTHONPATH="$KEDRL_SRC:${PYTHONPATH:-}"
else
  echo "Warning: KEDRL_SRC=${KEDRL_SRC:-unset} not found; relying on installed ke_drl package." >&2
fi

export VISPY_GL_BACKEND=egl
export VISPY_USE_APP=headless
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.8"

cp "$BASE/main_offlinedata.py" "$RUN/"
cp "$BASE/main_MonteCarloZ.py" "$RUN/"
cp "$BASE/main_est.py" "$RUN/"
cp "$BASE/main_tune_global.py" "$RUN/"
cp "$BASE/mu_plot.py" "$RUN/"
cp "$BASE/sim_utils.py" "$RUN/"
cp "$BASE/sim_eval.py" "$RUN/"
cp "$BASE/validate_sim_config.py" "$RUN/"
cp "$BASE/params_tune.yaml" "$RUN/"
cp "$BASE/tuning_grid.yaml" "$RUN/"

cd "$RUN"
python3 main_tune_global.py --combo-id "${SLURM_ARRAY_TASK_ID}"
