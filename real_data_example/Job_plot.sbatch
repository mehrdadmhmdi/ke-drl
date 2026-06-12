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

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ -f "${SUBMIT_DIR}/plot_recovered_densities.py" ]; then
  SCRIPT_DIR="$(cd "${SUBMIT_DIR}" && pwd)"
elif [ -f "${SUBMIT_DIR}/real_data_example/plot_recovered_densities.py" ]; then
  SCRIPT_DIR="$(cd "${SUBMIT_DIR}/real_data_example" && pwd)"
else
  echo "Could not locate plot_recovered_densities.py from SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
  echo "Submit from kedrl_git/real_data_example or kedrl_git." >&2
  exit 2
fi
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
POLICY_DIRS="${POLICY_DIRS:-rev_policy click_policy}"
LABELS="${LABELS:-rev_policy,click_policy}"
BW_PER_DIM="${BW_PER_DIM:-90,1}"

echo "Using Python from: $(which python)"
python -m pip show ke-drl || true
python -m pip show torch || true

echo "Input root: ${INPUT_ROOT}"
echo "Output dir: ${OUT_DIR}"
echo "Policy dirs: ${POLICY_DIRS}"
echo "Labels: ${LABELS}"
echo "Bandwidth per dim: ${BW_PER_DIM}"
echo "Kernel params are read from each policy metrics.json unless explicit CLI overrides are added."

python -u plot_recovered_densities.py \
  --input-root "${INPUT_ROOT}" \
  --policy-dirs ${POLICY_DIRS} \
  --labels "${LABELS}" \
  --out-dir "${OUT_DIR}" \
  --reward-cols gross_revenue_per_night,total_clicks \
  --discrete-reward-cols total_clicks \
  --bandwidth-per-dim "${BW_PER_DIM}" \
  --steps 10000 \
  --lr 0.005 \
  --ridge 1e-5 \
  --init uniform \
  --anchor uniform \
  --l2-lambda 1e-3 \
  --kl-lambda 1e-3 \
  --quad-points 31 \
  --device cpu

echo "End time: $(date)"
echo "Done."
