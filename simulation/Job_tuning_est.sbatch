#!/bin/bash
#SBATCH --account=bfez-delta-gpu
#SBATCH --job-name=UL_est_tuning
#SBATCH --output=logs/tune_%j.log
#SBATCH --time=30:08:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA40x4-preempt
#SBATCH --mail-type=NONE
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
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.8"

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
else
  echo "Warning: KEDRL_SRC=$KEDRL_SRC not found; relying on installed ke_drl package." >&2
fi

mkdir -p tunning/runs logs

# --------- Search space for the global-X estimator ----------
LREG=(5e-4 2e-4)
LB=(5e-4 2e-4)
LREC=(1.5)
HULL=(1 1.3)
NU=(5.5 7.5)
LSE=(1.5 2)
SIG=(0.9)

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

# Stage-1:
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

# ---------- throttle + retry helpers ----------
MAX_EST_INFLIGHT=200
submit_with_retry () {
  local cmd="$1" tries=0 out
  until out=$(bash -lc "$cmd" 2>&1); do
    ((tries++)); [ $tries -ge 10 ] && { echo "sbatch failed: $out" >&2; exit 1; }
    sleep $((tries*5))
  done
  printf '%s\n' "$out"
}

while read -r CFG; do
  echo "Processing configuration: $CFG"
  NAME=$(echo "$CFG" | tr ' =.' '_' | sed 's/^_//;s/_$//')
  RUN="$BASE/tunning/runs/$NAME"
  mkdir -p "$RUN"/{mu,plots,metrics,logs}

  cp "$BASE/main_est.py"       "$RUN/main_est.py"
  cp "$BASE/mu_plot.py"        "$RUN/mu_plot.py"
  cp "$BASE/sim_utils.py"      "$RUN/sim_utils.py"
  cp "$BASE/sim_eval.py"       "$RUN/sim_eval.py"
  cp "$BASE/Job_est.sbatch"    "$RUN/Job_est.sbatch"
  cp "$BASE/Job_plot.sbatch"   "$RUN/Job_plot.sbatch"
  if [ -d "$BASE/utils" ]; then cp -r "$BASE/utils" "$RUN/utils"; fi
  cp "$BASE/params.yaml"       "$RUN/params.yaml"
  cp "$BASE/mk_params.py"      "$RUN/mk_params.py"
  cp "$BASE/score_combo.py"    "$RUN/score_combo.py"
  cp -r "$BASE/data"           "$RUN/data"
  TUNE_Z_ID="${TUNE_Z_ID:-0}"
  if [ -f "$RUN/data/Z_true_${TUNE_Z_ID}.pt" ]; then
    cp -f "$RUN/data/Z_true_${TUNE_Z_ID}.pt" "$RUN/data/Z_true.pt"
  fi

  echo "Running mk_params.py for $CFG"
  python -u "$RUN/mk_params.py" "$RUN/params.yaml" "$RUN/params.yaml" $CFG || { echo "mk_params.py failed"; exit 1; }

  while [ "$(squeue -u "$USER" -h -n EST_RUN | wc -l)" -ge "$MAX_EST_INFLIGHT" ]; do sleep 15; done

  echo "Submitting estimation array job"
  EST_OUTPUT=$(submit_with_retry \
    "sbatch --job-name=UL_EST_RUN --parsable --account=bfez-delta-gpu \
     --export=ALL,BASE=\"$RUN\",KEDRL_SRC=\"$KEDRL_SRC\" --array=0-100 --chdir=\"$RUN\" \"$RUN/Job_est.sbatch\"")
  EST="${EST_OUTPUT##* }"
  [[ "$EST" =~ ^[0-9]+$ ]] || { echo "Failed to parse EST job ID: $EST_OUTPUT" >&2; exit 1; }
done < .configs_stage1.txt

echo "Orchestration submitted."
