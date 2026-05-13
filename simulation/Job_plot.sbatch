#!/bin/bash
#SBATCH --account=bfez-delta-gpu
#SBATCH --job-name=UL_plot
#SBATCH --output=logs/plot_%A_%a.log
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=34g
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4-preempt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehrdad3@illinois.edu

cd ${OUTDIR:-${BASE:-$PWD}}
echo "Running in folder: $(pwd)"
mkdir -p logs mu plots metrics

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

export VISPY_GL_BACKEND=egl
export VISPY_USE_APP=headless
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.8"

if [ -z "${KEDRL_SRC:-}" ] && [ -d "$PWD/../src" ]; then
  export KEDRL_SRC="$PWD/../src"
elif [ -z "${KEDRL_SRC:-}" ] && [ -d "$PWD/../kedrl_git/src" ]; then
  export KEDRL_SRC="$PWD/../kedrl_git/src"
fi
if [ -n "${KEDRL_SRC:-}" ]; then
  export PYTHONPATH="$KEDRL_SRC:${PYTHONPATH:-}"
fi

python3 mu_plot.py
