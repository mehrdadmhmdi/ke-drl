#!/bin/bash
#SBATCH --account=bfez-delta-cpu
#SBATCH --partition=cpu-preempt
#SBATCH --job-name=SIM2_MC100
#SBATCH --output=%x_%A.log
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mehrdad3@illinois.edu

set -euo pipefail

DEFAULT_SIM_DIR="/work/nvme/bfez/mehrdad3/DistRL/simulation_small_N_large_T"
DEFAULT_RESULT_ROOT="${DEFAULT_SIM_DIR}/results/6.5.2026_full_rerun"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ -f "${SUBMIT_DIR}/main_mc_repeat_eval.py" ]; then
  BASE="$(cd "${SUBMIT_DIR}" && pwd)"
elif [ -f "${SUBMIT_DIR}/simulation/main_mc_repeat_eval.py" ]; then
  BASE="$(cd "${SUBMIT_DIR}/simulation" && pwd)"
elif [ -f "${DEFAULT_SIM_DIR}/main_mc_repeat_eval.py" ]; then
  BASE="${DEFAULT_SIM_DIR}"
else
  echo "Could not locate simulation/main_mc_repeat_eval.py from SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
  echo "Submit from ${DEFAULT_SIM_DIR}, kedrl_git/simulation, or kedrl_git." >&2
  exit 2
fi
cd "${BASE}"
mkdir -p results

SIM2_POLICY_PAIR="${SIM2_POLICY_PAIR:-UL}"
case "$SIM2_POLICY_PAIR" in
  UG|UL|GU|GL|LU|LG) ;;
  *)
    echo "SIM2_POLICY_PAIR must be one of UG, UL, GU, GL, LU, LG; got ${SIM2_POLICY_PAIR}" >&2
    exit 1
    ;;
esac

RESULT_ROOT="${SIM2_RESULT_ROOT:-${DEFAULT_RESULT_ROOT}}"
RUN="${SIM2_RUN:-${RESULT_ROOT}/${SIM2_POLICY_PAIR}}"
if [ ! -d "${RUN}/data" ] || [ ! -d "${RUN}/mu" ]; then
  echo "Missing fitted simulation outputs under ${RUN}." >&2
  echo "Run prepare and fit first, or set SIM2_RUN to the fitted pair folder." >&2
  exit 1
fi
mkdir -p "${RUN}/logs" "${RUN}/metrics" "${RUN}/metrics/mc_repeats"

REPO_ROOT="$(cd "${BASE}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${RUN}:${BASE}:${PYTHONPATH:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TORCH_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export KEDRL_DEVICE=cpu
export KEDRL_REQUIRE_CUDA=0
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg

echo "Job ID:           ${SLURM_JOB_ID}"
echo "Array idx:        ${SLURM_ARRAY_TASK_ID:-}"
echo "Policy pair:      ${SIM2_POLICY_PAIR}"
echo "Simulation base:  ${BASE}"
echo "Result root:      ${RESULT_ROOT}"
echo "Run folder:       ${RUN}"
echo "CPUs per task:    ${SLURM_CPUS_PER_TASK:-NA}"
echo "CPUs on node:     ${SLURM_CPUS_ON_NODE:-NA}"
echo "Mem per node(MB): ${SLURM_MEM_PER_NODE:-NA}"
echo "Node:             $(hostname)"
echo "Start:            $(date)"

set +u
module load slurm-env/0.1
module load cudatoolkit/25.3_12.8
module load pytorch-conda/2.8
eval "$(conda shell.bash hook)"
conda activate delta_env
set -u

stage_file_atomic() {
  local src="$1"
  local dst="$2"
  local tmp
  tmp="${dst}.tmp.${SLURM_JOB_ID:-manual}.$$"
  cp "$src" "$tmp"
  mv -f "$tmp" "$dst"
}

stage_file_atomic "${BASE}/main_mc_repeat_eval.py" "${RUN}/main_mc_repeat_eval.py"
stage_file_atomic "${BASE}/sim_utils.py" "${RUN}/sim_utils.py"
stage_file_atomic "${BASE}/sim_eval.py" "${RUN}/sim_eval.py"

cd "${RUN}"
if [ ! -f params.yaml ]; then
  echo "Missing ${RUN}/params.yaml. Run the simulation fit or aggregate stage first." >&2
  exit 1
fi

echo "Using Python: $(which python)"
python -m pip show ke-drl || true
python -m pip show torch || true

EXTRA_ARGS=()
if [ -n "${SIM2_MC_OFFLINE_IDS:-}" ]; then
  EXTRA_ARGS+=(--offline-ids "${SIM2_MC_OFFLINE_IDS}")
fi
if [ -n "${SIM2_MC_BENCHMARK_IDS:-}" ]; then
  EXTRA_ARGS+=(--benchmark-ids "${SIM2_MC_BENCHMARK_IDS}")
fi
if [ -n "${SIM2_MC_Z_IDS:-}" ]; then
  EXTRA_ARGS+=(--z-ids "${SIM2_MC_Z_IDS}")
fi
if [ -n "${SIM2_MC_Z_TIMEPOINTS:-}" ]; then
  EXTRA_ARGS+=(--z-timepoints "${SIM2_MC_Z_TIMEPOINTS}")
fi

python -u main_mc_repeat_eval.py \
  --params params.yaml \
  --data-dir data \
  --mu-dir mu \
  --metrics-dir metrics \
  --output-dir metrics/mc_repeats \
  --mc-repeats "${SIM2_MC_REPEATS:-100}" \
  --repeat-start "${SIM2_MC_REPEAT_START:-0}" \
  --seed-offset "${SIM2_MC_SEED_OFFSET:-300000}" \
  --batch-size "${SIM2_MC_BATCH_SIZE:-2000}" \
  --progress-every "${SIM2_MC_PROGRESS_EVERY:-5}" \
  --device cpu \
  "${EXTRA_ARGS[@]}"

echo "Repeated MC comparison complete: $(date)"
ls -lh metrics/mc_repeat_aggregate_metrics.csv
ls -lh metrics/mc_repeat_per_benchmark_aggregate_metrics.csv
ls -lh metrics/mc_repeats/mc_repeat_run_summary.csv
ls -lh metrics/mc_repeats/mc_repeat_pointwise_diff_summary.csv
