#!/bin/bash
##SBATCH --account=bfez-delta-gpu
#SBATCH --account=bfez-dtai-gh
##SBATCH --partition=gpuA100x8
#SBATCH --partition=ghx4
#SBATCH --job-name=UL_PLOTS
#SBATCH --output=logs/data_%j.log
#SBATCH --error=logs/submit_plots_%j.err
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
mkdir -p "$BASE/logs" "$BASE/tunning"
KEDRL_REPO_ROOT="${KEDRL_REPO_ROOT:-}"
if [ -z "$KEDRL_REPO_ROOT" ]; then
  for candidate in "${BASE:-$PWD}" "${BASE:-$PWD}/.." "$PWD" "$PWD/.." "$PWD/../.." "$PWD/../../.." "${SLURM_SUBMIT_DIR:-$PWD}" "${SLURM_SUBMIT_DIR:-$PWD}/.."; do
    if [ -d "$candidate/src/ke_drl" ]; then
      KEDRL_REPO_ROOT="$(cd "$candidate" && pwd)"
      break
    fi
  done
fi
KEDRL_GIT_URL="${KEDRL_GIT_URL:-https://github.com/mehrdadmhmdi/ke-drl.git}"
KEDRL_GIT_REF="${KEDRL_GIT_REF:-main}"
if [ -n "$KEDRL_REPO_ROOT" ] && [ -d "$KEDRL_REPO_ROOT/src/ke_drl" ]; then
  export KEDRL_SRC="$KEDRL_REPO_ROOT/src"
  export PYTHONPATH="$KEDRL_SRC:${PYTHONPATH:-}"
  echo "Using local ke_drl source: $KEDRL_SRC"
else
  unset KEDRL_SRC
  KEDRL_GIT_URL="${KEDRL_GIT_URL:-https://github.com/mehrdadmhmdi/ke-drl.git}"
  KEDRL_GIT_REF="${KEDRL_GIT_REF:-main}"
  KEDRL_PIP_TARGET="$BASE/.kedrl_site_${SLURM_JOB_ID:-manual}"
  rm -rf "$KEDRL_PIP_TARGET"
  mkdir -p "$KEDRL_PIP_TARGET"
  echo "Installing ke_drl from git: ${KEDRL_GIT_URL}@${KEDRL_GIT_REF}"
  python3 -m pip install --no-cache-dir --upgrade --force-reinstall --no-deps \
    --target "$KEDRL_PIP_TARGET" "git+${KEDRL_GIT_URL}@${KEDRL_GIT_REF}"
  export PYTHONPATH="$KEDRL_PIP_TARGET:${PYTHONPATH:-}"
fi
python3 - <<'PY'
import ke_drl
print(f"Installed ke_drl source: {ke_drl.__file__}")
PY
:> "$BASE/tunning/agg_ids.txt"
:> "$BASE/tunning/agg_map.tsv"

MAX_PLOT_INFLIGHT=200
count_named_jobs(){ squeue -u "$USER" -h -o "%i %j" | awk -v n="PLOT_RUN" '$2==n{c++} END{print c+0}'; }

submit_with_retry() {
  local cmd="$1" tries=0 out rc=1
  while true; do
    if out=$(bash -lc "$cmd" 2>&1); then printf '%s\n' "$out"; return 0; fi
    rc=$?; tries=$((tries+1)); sleep $(( (tries<20?tries:20) * 5 ))
    if [ $tries -ge 40 ]; then echo "sbatch failed after retries: $out" >&2; return $rc; fi
  done
}

shopt -s nullglob
for RUN in "$BASE"/tunning/runs/*; do
  [ -d "$RUN" ] || continue
  PLOT_S="$RUN/Job_plot.sbatch"
  LOGDIR="$RUN/logs"; mkdir -p "$LOGDIR"
  [ -f "$PLOT_S" ] || { echo "Skip (no Job_plot.sbatch): $RUN"; continue; }

  while [ "$(count_named_jobs)" -ge "$MAX_PLOT_INFLIGHT" ]; do sleep 15; done

  OUT=$(submit_with_retry \
    "sbatch --job-name=PLOT_RUN --parsable --account=bfez-dtai-gh \
     --output=\"$LOGDIR/AGG_%j.log\" --error=\"$LOGDIR/AGG_%j.err\" \
     --time=00:15:00 --nodes=1 --mem=8g --gres=gpu:1 --cpus-per-task=4 --partition=ghx4 \
     --export=ALL,BASE=\"$RUN\",KEDRL_GIT_URL=\"$KEDRL_GIT_URL\",KEDRL_GIT_REF=\"$KEDRL_GIT_REF\" --chdir=\"$RUN\" \"$PLOT_S\"") \
    || { echo "WARN: failed submit for $RUN"; continue; }

  AGG="${OUT##* }"
  [[ "$AGG" =~ ^[0-9]+$ ]] || { echo "WARN: bad jobid: $OUT"; continue; }
  echo "$AGG" >> "$BASE/tunning/agg_ids.txt"
  printf "%s\t%s\n" "$AGG" "$RUN" >> "$BASE/tunning/agg_map.tsv"
done

echo "Done submitting plots."
