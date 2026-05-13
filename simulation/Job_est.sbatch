#!/bin/bash
#SBATCH --account=bfez-dtai-gh
#SBATCH --job-name=UL_est
#SBATCH --output=logs/est_%A_%a.log
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8g
#SBATCH --cpus-per-task=4
#SBATCH --partition=ghx4
#SBATCH --mail-type=NONE
#SBATCH --mail-user=mehrdad3@illinois.edu
#SBATCH --array=0-100

set -euo pipefail

cd ${OUTDIR:-${BASE:-$PWD}}
echo "Running in folder: $(pwd)"
mkdir -p logs data mu plots metrics

echo "Job ID:    ${SLURM_JOB_ID}"
echo "Array idx: ${SLURM_ARRAY_TASK_ID}"
echo "CPUs/node: ${SLURM_CPUS_ON_NODE}"
echo "Mem/node:  ${SLURM_MEM_PER_NODE}"

## load modules/env
module load python/miniforge3_pytorch/2.7.0

export VISPY_GL_BACKEND=egl
export VISPY_USE_APP=headless
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.8"

if [ -z "${KEDRL_SRC:-}" ] && [ -d "$PWD/../kedrl_git/src" ]; then
  export KEDRL_SRC="$PWD/../kedrl_git/src"
fi
if [ -n "${KEDRL_SRC:-}" ]; then
  export PYTHONPATH="$KEDRL_SRC:${PYTHONPATH:-}"
fi

python3 main_est.py
