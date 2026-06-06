#!/bin/bash
#SBATCH --account=bfez-delta-cpu
#SBATCH --partition=cpu
#SBATCH --job-name=expedia_plot
#SBATCH --output=logs/recovery_plot_%j.log
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4g
#SBATCH --mail-type=NONE
#SBATCH --mail-user=mehrdad3@illinois.edu

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p logs

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${SCRIPT_DIR}:${PYTHONPATH:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Job ID:           ${SLURM_JOB_ID}"
echo "CPUs per task:    ${SLURM_CPUS_PER_TASK}"
echo "CPUs on node:     ${SLURM_CPUS_ON_NODE:-NA}"
echo "Mem per node(MB): ${SLURM_MEM_PER_NODE:-NA}"
echo "Running on:       $(hostname)"
echo "Start time:       $(date)"

module purge
source /sw/external/python/anaconda3_gpu/etc/profile.d/conda.sh
conda activate /u/mehrdad3/.conda/envs/delta_env

INPUT_ROOT="${INPUT_ROOT:-plotting_files}"
OUT_DIR="${OUT_DIR:-plotting_files}"

echo "Using Python from: $(which python)"
python -m pip show ke-drl || true
python -m pip show torch || true

python -u plot_recovered_densities.py \
  --input-root "${INPUT_ROOT}" \
  --policy-dirs rev_policy click_policy \
  --labels rev_policy,click_policy \
  --out-dir "${OUT_DIR}" \
  --reward-cols gross_revenue_per_night,total_clicks \
  --discrete-reward-cols total_clicks \
  --nu-Z 4.5 \
  --ell-Z 1.8 \
  --sigma-Z 1.0 \
  --bandwidth-per-dim 90,1 \
  --steps 5000 \
  --lr 0.0001 \
  --quad-points 21 \
  --device cpu

echo "End time: $(date)"
echo "Done."
