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

# =====================================================================
# OPTIONAL standalone re-plotter (induced-embedding density recovery from
# SAVED artifacts). Job_ev.sbatch ALREADY produces the two-policy overlay
# (two_policy_reward_overlay_plots/overlay_all_mean_embedding_marginal_comparison.png),
# so you only need this if you want to re-plot saved artifacts with different
# recovery settings.
#
# Wiring: point --input-root at ONE Job_ev config dir, and --policy-dirs at the
# two per-policy subfolders inside it (each contains artifacts.pt). Example for
# the patched output tree (cfg 0, q50 target point):
#
#   export INPUT_ROOT="evaluation_results/expedia_encoded_10k_linear3_fixed/main_cfg_0_state_pc_q50"
#   export POLICY_DIRS="gross_revenue_per_night_cfg000_nuZ3.5_ellZ2.0_nuSA3.5_ellSA1.0_lam1e-05_sigmaZ1.0 total_clicks_cfg000_nuZ3.5_ellZ2.0_nuSA3.5_ellSA1.0_lam1e-05_sigmaZ1.0"
#   export LABELS="rev_policy,click_policy"
#   sbatch Job_plot.sbatch
#
# (The exact subfolder names print in the Job_ev log and depend on the cfg's
# kernel hyperparameters.)
# =====================================================================

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
echo "Running on:       $(hostname)"
echo "Start time:       $(date)"

module purge
source /sw/external/python/anaconda3_gpu/etc/profile.d/conda.sh
conda activate /u/mehrdad3/.conda/envs/delta_env

# Default to ONE patched config dir; override INPUT_ROOT/POLICY_DIRS as above.
INPUT_ROOT="${INPUT_ROOT:-evaluation_results/expedia_encoded_10k_linear3_fixed/main_cfg_0_state_pc_q50}"
POLICY_DIRS="${POLICY_DIRS:-gross_revenue_per_night_cfg000_nuZ3.5_ellZ2.0_nuSA3.5_ellSA1.0_lam1e-05_sigmaZ1.0 total_clicks_cfg000_nuZ3.5_ellZ2.0_nuSA3.5_ellSA1.0_lam1e-05_sigmaZ1.0}"
OUT_DIR="${OUT_DIR:-${INPUT_ROOT}/replot_induced}"
LABELS="${LABELS:-rev_policy,click_policy}"
BW_PER_DIM="${BW_PER_DIM:-90,1}"

echo "Using Python from: $(which python)"
echo "Input root:  ${INPUT_ROOT}"
echo "Policy dirs: ${POLICY_DIRS}"
echo "Out dir:     ${OUT_DIR}"
echo "Kernel params are read from each policy metrics.json unless overridden."

python -u plot_recovered_densities.py \
  --input-root "${INPUT_ROOT}" \
  --policy-dirs ${POLICY_DIRS} \
  --labels "${LABELS}" \
  --out-dir "${OUT_DIR}" \
  --reward-cols gross_revenue_per_night,total_clicks \
  --discrete-reward-cols total_clicks \
  --bandwidth-per-dim "${BW_PER_DIM}" \
  --steps 20000 \
  --lr 0.01 \
  --ridge 1e-4 \
  --init positive_beta \
  --anchor positive_beta \
  --l2-lambda 1e-2 \
  --kl-lambda 0.0 \
  --quad-points 21 \
  --device cpu

echo "End time: $(date)"
echo "Done."
