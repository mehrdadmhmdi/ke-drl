#!/bin/bash
#SBATCH --account=bfez-dtai-gh     #Delta AI
#SBATCH --partition=ghx4           #Delta AI
##SBATCH --partition=gpuA100x4-preempt
##SBATCH --account=bfez-delta-gpu   #Delta
#SBATCH --job-name=UL_Score
#SBATCH --output=logs/data_%A_%a.log
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8g
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END
#SBATCH --mail-user=mehrdad3@illinois.edu

set -euo pipefail

echo "Job ID:    ${SLURM_JOB_ID}"
echo "Array idx: ${SLURM_ARRAY_TASK_ID:-NA}"
echo "CPUs/node: ${SLURM_CPUS_ON_NODE}"
echo "Mem/node:  ${SLURM_MEM_PER_NODE}"

module load python/miniforge3_pytorch/2.7.0 || true

BASE="${SLURM_SUBMIT_DIR:-$PWD}"
echo "BASE: $BASE"

# Run the scorer from the tunning directory; keep absolute paths
python -u "$BASE/score_combo.py"
