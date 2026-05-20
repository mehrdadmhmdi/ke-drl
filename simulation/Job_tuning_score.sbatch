#!/bin/bash
##SBATCH --account=bfez-delta-gpu
#SBATCH --account=bfez-dtai-gh
##SBATCH --partition=gpuA100x8
#SBATCH --partition=ghx4
#SBATCH --job-name=UL_Score
#SBATCH --output=logs/data_%A_%a.log
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8g
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END
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

set -euo pipefail

BASE="${SLURM_SUBMIT_DIR:-$PWD}"
echo "BASE: $BASE"

# Run the scorer from the tunning directory; keep absolute paths
python -u "$BASE/score_combo.py"
