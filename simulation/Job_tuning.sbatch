#!/bin/bash
#SBATCH --account=bfez-delta-gpu
#SBATCH --job-name=tuning
#SBATCH --output=logs/tune_%j.log
#SBATCH --time=30:08:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA40x4-preempt
#SBATCH --mail-type=ALL
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

export VISPY_GL_BACKEND=egl
export VISPY_USE_APP=headless
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.8"

BASE="${SLURM_SUBMIT_DIR:-$PWD}"
echo "SUBMIT_DIR: $BASE"
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

mkdir -p tunning/runs logs

# --------- Search space for the global-X estimator ----------
LREG=(2e-2 2e-3 2e-4)
LB=(2e-3 2e-4)
LREC=(1e-4 1e-2)
HULL=(1.0 1.5)
NU=(4.5 3.5)
LSE=(1.4 2)
SIG=(1.0 0.4)

# Generate all configs directly to a file to avoid large array expansion
> all_configs.txt
for lr in "${LREG[@]}"; do
  for lb in "${LB[@]}"; do
    for lrec in "${LREC[@]}"; do
      for hf in "${HULL[@]}"; do
        for nu in "${NU[@]}"; do
          for ls in "${LSE[@]}"; do
            for sg in "${SIG[@]}"; do
              echo "lambda_reg=$lr lambda_B=$lb lambda_rec=$lrec hull_expand_factor=$hf kernel.nu=$nu kernel.length_scale=$ls kernel.sigma=$sg" >> all_configs.txt
            done
          done
        done
      done
    done
  done
done

# Stage-1: random 2 configs for testing (increase to 40 later)
shuf -n 192 all_configs.txt > .configs_stage1.txt

# Clean up the full list
rm all_configs.txt

echo "Generated configs file: .configs_stage1.txt"
wc -l .configs_stage1.txt

# Check if configs file has content
if [ ! -s .configs_stage1.txt ]; then
  echo "Error: No configurations generated in .configs_stage1.txt" >&2
  exit 1
fi

# helpers must exist in $BASE
if [[ ! -f "$BASE/mk_params.py" || ! -f "$BASE/score_combo.py" ]]; then
  echo "Missing mk_params.py or score_combo.py" >&2
  exit 1
fi

:> tunning/agg_ids.txt
:> tunning/agg_map.tsv  # Ensure it's empty

# ---------- throttle + retry helpers (NAMES ONLY, no path changes) ----------
MAX_EST_INFLIGHT=200     # how many EST arrays you allow simultaneously
MAX_PLOT_INFLIGHT=200    # how many PLOT jobs you allow simultaneously
submit_with_retry () {
  # usage: submit_with_retry "<sbatch cmd...>"
  local cmd="$1"
  local tries=0
  local out rc=1
  until out=$(bash -lc "$cmd" 2>&1); do
    rc=$?
    tries=$((tries+1))
    if [ $tries -ge 10 ]; then
      echo "sbatch failed after $tries tries: $out" >&2
      return $rc
    fi
    sleep $((tries*5))
  done
  printf '%s\n' "$out"
  return 0
}

while read -r CFG; do
  echo "Processing configuration: $CFG"
  NAME=$(echo $CFG | tr ' =.' '_' | sed 's/^_//;s/_$//')
  echo "Generated NAME: $NAME"
  RUN="$BASE/tunning/runs/$NAME"
  echo "Run directory: $RUN"
  mkdir -p "$RUN"/{mu,plots,metrics,logs}

  # Copy necessary files to the run directory
  cp "$BASE/main_est.py" "$RUN/main_est.py"
  cp "$BASE/mu_plot.py" "$RUN/mu_plot.py"
  cp "$BASE/sim_utils.py" "$RUN/sim_utils.py"
  cp "$BASE/sim_eval.py" "$RUN/sim_eval.py"
  cp "$BASE/Job_est.sbatch" "$RUN/Job_est.sbatch"
  cp "$BASE/Job_plot.sbatch" "$RUN/Job_plot.sbatch"
  if [ -d "$BASE/utils" ]; then cp -r "$BASE/utils" "$RUN/utils"; fi
  cp "$BASE/params.yaml" "$RUN/params.yaml"
  cp "$BASE/mk_params.py" "$RUN/mk_params.py"
  cp "$BASE/score_combo.py" "$RUN/score_combo.py"
  cp -r "$BASE/data" "$RUN/data"  # Adjust path if data is elsewhere
  TUNE_Z_ID="${TUNE_Z_ID:-0}"
  if [ -f "$RUN/data/Z_true_${TUNE_Z_ID}.pt" ]; then
    cp -f "$RUN/data/Z_true_${TUNE_Z_ID}.pt" "$RUN/data/Z_true.pt"
  fi

  # write params.yaml with updated path
  echo "Running mk_params.py for $CFG"
  python -u "$RUN/mk_params.py" "$RUN/params.yaml" "$RUN/params.yaml" $CFG || { echo "mk_params.py failed"; exit 1; }

  # throttle number of in-flight EST arrays
  while [ "$(squeue -u "$USER" -h -n EST_RUN | wc -l)" -ge "$MAX_EST_INFLIGHT" ]; do sleep 15; done

  # submit estimation array (paths unchanged)
  echo "Submitting estimation array job"
  EST_OUTPUT=$(submit_with_retry \
    "sbatch --job-name=EST_RUN --parsable --account=bfez-delta-gpu \
     --export=ALL,BASE=\"$RUN\",KEDRL_SRC=\"$KEDRL_SRC\" --array=0-100 --chdir=\"$RUN\" \"$RUN/Job_est.sbatch\"")
  EST_RET=$?
  echo "sbatch EST output: $EST_OUTPUT"
  if [ $EST_RET -ne 0 ]; then
    echo "Failed to submit EST job: $EST_OUTPUT" >&2
    exit 1
  fi
  EST=$(echo "$EST_OUTPUT" | grep -oP '^\d+')
  if [ -z "$EST" ]; then
    echo "Failed to parse EST job ID: $EST_OUTPUT" >&2
    exit 1
  fi

  # throttle number of in-flight PLOT jobs
  while [ "$(squeue -u "$USER" -h -n PLOT_RUN | wc -l)" -ge "$MAX_PLOT_INFLIGHT" ]; do sleep 15; done

  # submit aggregation after estimate array finishes (paths unchanged)
  echo "Submitting aggregation job dependent on $EST"
  AGG_OUTPUT=$(submit_with_retry \
    "sbatch --job-name=PLOT_RUN --parsable --account=bfez-delta-gpu \
     --output=\"$RUN/logs/AGG_%j.log\" \
     --time=00:08:00 --nodes=1 --gres=gpu:1 --mem=8g --cpus-per-task=4 --partition=gpuA40x4 --mail-type=ALL \
     --mail-user=mehrdad3@illinois.edu --export=ALL,BASE=\"$RUN\",KEDRL_SRC=\"$KEDRL_SRC\" --dependency=afterany:$EST --chdir=\"$RUN\" \
     \"$RUN/Job_plot.sbatch\"")
  AGG_RET=$?
  echo "sbatch AGG output: $AGG_OUTPUT"
  if [ $AGG_RET -ne 0 ]; then
    echo "Failed to submit AGG job: $AGG_OUTPUT" >&2
    exit 1
  fi
  AGG=$(echo "$AGG_OUTPUT" | grep -oP '^\d+')
  if [ -z "$AGG" ]; then
    echo "Failed to extract AGG job ID: $AGG_OUTPUT" >&2
    exit 1
  fi
  echo $AGG >> tunning/agg_ids.txt
  echo -e "$AGG\t$RUN" >> tunning/agg_map.tsv
done < .configs_stage1.txt

echo "Verifying metrics dirs:"; ls -d "$BASE"/tunning/runs/*/metrics 2>/dev/null | head

# final scoring after all aggregations (paths unchanged)
if [ ! -s tunning/agg_ids.txt ]; then
  echo "No aggregation jobs submitted, skipping scoring" >&2
else
  ALL_DEPS=$(paste -sd: tunning/agg_ids.txt)
  echo "Submitting scoring job"
  SCORE_OUTPUT=$(
    sbatch --account=bfez-delta-gpu \
      --dependency=afterany:$ALL_DEPS \
      --chdir="$BASE/tunning" \
      --export=ALL,BASE="$BASE" \
      --job-name=TUNE_SCORE \
      --output=score_%j.log \
      --time=00:15:00 --gres=gpu:1 --mem=2g --cpus-per-task=4 --partition=gpuA40x4 \
      --wrap="python -u $BASE/score_combo.py" 2>&1
  )
  echo "sbatch SCORE output: $SCORE_OUTPUT"
  [ $? -ne 0 ] && { echo "Failed to submit scoring job: $SCORE_OUTPUT" >&2; exit 1; }
fi

echo "Orchestration submitted."
