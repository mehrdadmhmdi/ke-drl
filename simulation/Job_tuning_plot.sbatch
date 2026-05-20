#!/bin/bash
#SBATCH --account=bfez-delta-gpu
#SBATCH --partition=gpuA100x8
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

BASE="${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "$BASE/logs" "$BASE/tunning"
if [ -z "${KEDRL_SRC:-}" ] && [ -d "$BASE/../src" ]; then
  KEDRL_SRC="$BASE/../src"
elif [ -z "${KEDRL_SRC:-}" ]; then
  KEDRL_SRC="$BASE/../kedrl_git/src"
fi
if [ -d "$KEDRL_SRC" ]; then
  export KEDRL_SRC
  export PYTHONPATH="$KEDRL_SRC:${PYTHONPATH:-}"
  echo "KEDRL_SRC: $KEDRL_SRC"
else
  echo "KEDRL_SRC=$KEDRL_SRC not found; using installed ke_drl package." >&2
  if [ "${KEDRL_REINSTALL_FROM_GIT:-0}" = "1" ]; then
    python -m pip install "git+https://github.com/mehrdadmhmdi/ke-drl.git"
  fi
  python - <<'PY'
import importlib.util
spec = importlib.util.find_spec("ke_drl")
if spec is None:
    raise SystemExit('ke_drl is not installed. Run: python -m pip install "git+https://github.com/mehrdadmhmdi/ke-drl.git"')
print(f"Installed ke_drl source: {spec.origin}")
PY
fi
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
    "sbatch --job-name=PLOT_RUN --parsable --account=bfez-delta-gpu \
     --output=\"$LOGDIR/AGG_%j.log\" --error=\"$LOGDIR/AGG_%j.err\" \
     --time=00:15:00 --nodes=1 --mem=8g --gres=gpu:1 --cpus-per-task=4 --partition=gpuA100x8 \
     --export=ALL,BASE=\"$RUN\",KEDRL_SRC=\"$KEDRL_SRC\" --chdir=\"$RUN\" \"$PLOT_S\"") \
    || { echo "WARN: failed submit for $RUN"; continue; }

  AGG="${OUT##* }"
  [[ "$AGG" =~ ^[0-9]+$ ]] || { echo "WARN: bad jobid: $OUT"; continue; }
  echo "$AGG" >> "$BASE/tunning/agg_ids.txt"
  printf "%s\t%s\n" "$AGG" "$RUN" >> "$BASE/tunning/agg_map.tsv"
done

echo "Done submitting plots."
