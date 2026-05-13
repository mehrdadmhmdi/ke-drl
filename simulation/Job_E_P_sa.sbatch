#!/bin/bash
#SBATCH --account=bfez-delta-gpu
#SBATCH --job-name=RUN_ALL
#SBATCH --output=logs/master-launch.%j.log
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8g
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuA40x4-preempt
#SBATCH --mail-type=END
#SBATCH --mail-user=mehrdad3@illinois.edu
#SBATCH --array=0-30                    # number of runs


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

# ============================================================ #
# Setup
# ============================================================ #
BASE=$PWD
OUTBASE=$BASE/runs
mkdir -p "$OUTBASE" logs
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

RUN="$OUTBASE/run_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RUN" "$RUN/logs"

# ------------------------------------------------------------ #
# Copy required files
# ------------------------------------------------------------ #
cp "$BASE/main_est.py"       "$RUN/"
cp "$BASE/mu_plot.py"        "$RUN/"
cp "$BASE/sim_utils.py"      "$RUN/"
cp "$BASE/sim_eval.py"       "$RUN/"
cp "$BASE/Job_est.sbatch"    "$RUN/"
cp "$BASE/Job_plot.sbatch"   "$RUN/"
if [ -d "$BASE/utils" ]; then cp -r "$BASE/utils" "$RUN/"; fi
cp "$BASE/params.yaml"       "$RUN/"
cp -r "$BASE/data"           "$RUN/"

# ------------------------------------------------------------ #
# Z true setup
# ------------------------------------------------------------ #
DATA_DIR="$RUN/data"
JID="$SLURM_ARRAY_TASK_ID"

if [ -f "${DATA_DIR}/Z_true_${JID}.pt" ]; then
  mv -f "${DATA_DIR}/Z_true_${JID}.pt" "${DATA_DIR}/Z_true.pt"
else
  echo "Warning: ${DATA_DIR}/Z_true_${JID}.pt not found." >&2
fi


# ------------------------------------------------------------ #
# Submit dependent jobs
# ------------------------------------------------------------ #
cd "$RUN"
outdir="$RUN"

jid_est=$(sbatch --parsable \
    --output=${outdir}/logs/Job_est.%j.log \
    --export=ALL,OUTDIR=${outdir},KEDRL_SRC="${KEDRL_SRC:-}" \
    Job_est.sbatch)

jid_plot=$(sbatch --parsable \
    --dependency=afterany:${jid_est} \
    --output=${outdir}/logs/Job_plot.%j.log \
    --export=ALL,OUTDIR=${outdir},KEDRL_SRC="${KEDRL_SRC:-}" \
    Job_plot.sbatch)

echo "Submitted run ${SLURM_ARRAY_TASK_ID}: Job_est=${jid_est}, Job_plot=${jid_plot}"
