#!/bin/bash
#SBATCH --account=bfez-delta-gpu
#SBATCH --job-name=GU_data
#SBATCH --output=logs/data_%A_%a.log
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8g
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA40x4-preempt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mehrdad3@illinois.edu
#SBATCH --array=0-49

# Access exported variables:
cd ${OUTDIR:-$PWD}
echo "Running in folder: $(pwd)"
mkdir -p logs data

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

if [ -z "${KEDRL_SRC:-}" ] && [ -d "$PWD/../src" ]; then
  export KEDRL_SRC="$PWD/../src"
elif [ -z "${KEDRL_SRC:-}" ] && [ -d "$PWD/../kedrl_git/src" ]; then
  export KEDRL_SRC="$PWD/../kedrl_git/src"
fi
if [ -n "${KEDRL_SRC:-}" ]; then
  export PYTHONPATH="$KEDRL_SRC:${PYTHONPATH:-}"
fi

python3 validate_sim_config.py --params params.yaml
python3 main_offlinedata.py
python3 validate_sim_config.py --params params.yaml --data "data/offline_data_${SLURM_ARRAY_TASK_ID}.pt"
