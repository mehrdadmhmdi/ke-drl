#!/bin/bash
#SBATCH --account=bfez-dtai-gh     #Delta AI
#SBATCH --partition=ghx4           #Delta AI
##SBATCH --partition=gpuA100x4-preempt
##SBATCH --account=bfez-delta-gpu   #Delta
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

set -euo pipefail
module load python/miniforge3_pytorch/2.7.0 || true

BASE="${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "$BASE/logs" "$BASE/tunning"
KEDRL_SRC="${KEDRL_SRC:-$BASE/../kedrl_git/src}"
if [ -d "$KEDRL_SRC" ]; then
  export KEDRL_SRC
  export PYTHONPATH="$KEDRL_SRC:${PYTHONPATH:-}"
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
    "sbatch --job-name=PLOT_RUN --parsable --account=bfez-dtai-gh \
     --output=\"$LOGDIR/AGG_%j.log\" --error=\"$LOGDIR/AGG_%j.err\" \
     --time=00:15:00 --nodes=1 --mem=8g --gres=gpu:1 --cpus-per-task=4 --partition=ghx4 \
     --export=ALL,BASE=\"$RUN\",KEDRL_SRC=\"$KEDRL_SRC\" --chdir=\"$RUN\" \"$PLOT_S\"") \
    || { echo "WARN: failed submit for $RUN"; continue; }

  AGG="${OUT##* }"
  [[ "$AGG" =~ ^[0-9]+$ ]] || { echo "WARN: bad jobid: $OUT"; continue; }
  echo "$AGG" >> "$BASE/tunning/agg_ids.txt"
  printf "%s\t%s\n" "$AGG" "$RUN" >> "$BASE/tunning/agg_map.tsv"
done

echo "Done submitting plots."
