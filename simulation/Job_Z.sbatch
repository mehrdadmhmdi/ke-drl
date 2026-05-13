#!/bin/bash
#####----------
### jid0=$(sbatch --array=0-0 Job_Z.sbatch | awk '{print $4}')
### sbatch --array=1-20 --dependency=afterok:$jid0 Job_Z.sbatch 
####--------------
#SBATCH --account=bfez-dtai-gh
#SBATCH --job-name=GU_Z
#SBATCH --output=logs/Z_true_%A_%a.log
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4g
#SBATCH --cpus-per-task=4
#SBATCH --partition=ghx4
#SBATCH --mail-type=NONE
#SBATCH --mail-user=mehrdad3@illinois.edu
#SBATCH --array=0-30

set -euo pipefail

cd ${OUTDIR:-$PWD}
echo "Running in folder: $(pwd)"
mkdir -p logs data

echo "Job ID:    ${SLURM_JOB_ID}"
echo "Array idx: ${SLURM_ARRAY_TASK_ID}"
echo "CPUs/node: ${SLURM_CPUS_ON_NODE}"
echo "Mem/node:  ${SLURM_MEM_PER_NODE}"

## load modules/env
module load python/miniforge3_pytorch/2.7.0

if [ -z "${KEDRL_SRC:-}" ] && [ -d "$PWD/../kedrl_git/src" ]; then
  export KEDRL_SRC="$PWD/../kedrl_git/src"
fi
if [ -n "${KEDRL_SRC:-}" ]; then
  export PYTHONPATH="$KEDRL_SRC:${PYTHONPATH:-}"
fi

python3 main_MonteCarloZ.py
